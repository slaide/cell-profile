from dataclasses import dataclass
import dataclasses as dc
from pathlib import Path
import typing as tp
import os

import numpy as np
import polars as pl
import pandas as pd

import plotly.express as px

from sklearn.decomposition import PCA
import umap

from .df_util import is_meta_column, handle_outliers, remove_nans, \
        remove_highly_correlated, Sigma, Quantile, \
        df_checkNull, df_checkValue, df_checkInf, df_checkNaN

from .misc import display, print, print_time, tqdm

float_columns=[pl.col(pl.Float32),pl.col(pl.Float64)]
"""
allows to select only columns that contain float values, e.g. df.select(float_columns)

useful because columns of other types are usually just metadata, e.g. indices [int], strings etc.
"""

@dataclass
class Experiment:
    barcode:str
    db_uri:str

    # compound_layout_args:tp.Optional[dict]=None

    def retrieve_cellprofiler_pipelines(self)->pl.DataFrame:
        """
            retrieve cellprofiler pipeline information from database

            return:
                information about all plates in this experiment, more specifically:
                    - plate_id: key into plate_acquisition table that contains information about the plate/images
                    - timepoint: 0-indexed timepoint
                    - folder: path to image directory for this plate
                    - pipeline_id_feat: pipeline id of feature run for these images, is key into image_analyses table, which contains the path to the output files
                    - pipeline_id_qc: pipeline id of qc run for these images, is key into image_analyses table, which contains the path to the output files
                    
        """

        # database TODO : image_analyses.meta sometimes contains the following fields, and i would like them to exist
        #    in all entries : submitted_by, trash_analysis
        # query all json keys from a jsonb column with : SELECT DISTINCT key FROM image_analyses, LATERAL jsonb_object_keys(meta) AS key;
        
        query=f"""
            /*
            filter for relevant plates

            there may be multiple runs for the same plate, with the same timepoint (timpoint 0 or otherwise!).
                i.e. this query here may return more than one row. it is expected that the queries below
                will end up filtering for the cellprofiler runs with the longest result, meaning
                the incomplete imaging runs will be discarded then, since no cellprofiler pipeline
                has been run on them.

            somewhat related: it is nearly impossible in SQL to filter for the latest run,
                since the SQUID imaging runs have no valid timestamps saved in the database,
                only the name of the run contains this information, but it is difficult to extract
                this information from the name in SQL.
            */
            WITH FilteredPlateAcquisition AS (
                SELECT *
                FROM plate_acquisition
                WHERE plate_barcode = '{self.barcode}'
                AND hidden IS NOT TRUE
            ),
            /* precompute some things based on relevant plates */
            PreComputedLength AS (
                SELECT
                    ia.plate_acquisition_id,
                    ia.id,
                    ia.pipeline_name,
                    ia.meta,
                    ia.start,
                    ia.finish,
                    LENGTH(jsonb_pretty(ia.result)) as pretty_length
                FROM image_analyses ia
                INNER JOIN FilteredPlateAcquisition fpa
                ON ia.plate_acquisition_id = fpa.id
                WHERE ia.finish IS NOT NULL
                AND ia.start IS NOT NULL
                AND ia.meta ? 'type'
            ),
            /* select feature pipeline with longest result jsonb entry (assume this means the pipeline has finished) */
            FeatData AS (
                SELECT 
                    pcl.plate_acquisition_id, 
                    pcl.id AS feat_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY pcl.plate_acquisition_id
                        ORDER BY
                            pcl.pretty_length DESC,
                            pcl.start DESC
                    ) AS rn
                FROM PreComputedLength pcl
                WHERE pcl.meta @> '{{"type":"cp-features"}}'
            ),
            /* select qc pipeline with longest result jsonb entry (assume this means the pipeline has finished) */
            QCData AS (
                SELECT 
                    pcl.plate_acquisition_id, 
                    pcl.id AS qc_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY pcl.plate_acquisition_id
                        /*
                        order by:
                            - length to have the longest result string in the first row
                            - start timestamp to resolve cases where multiple entries have the same result length string
                                assume here that the run started last is the one with the correct data
                        */
                        ORDER BY
                            pcl.pretty_length DESC,
                            pcl.start DESC
                    ) AS rn
                FROM PreComputedLength pcl
                WHERE pcl.meta @> '{{"type":"cp-qc"}}'
            )

            /* collect into result and return */
            SELECT 
                pa.id AS plate_id, 
                pa.timepoint, 
                pa.folder as image_dir,
                fd.feat_id AS pipeline_id_feat, 
                qd.qc_id AS pipeline_id_qc
            FROM FilteredPlateAcquisition pa
            LEFT JOIN FeatData fd
                ON pa.id = fd.plate_acquisition_id
                AND fd.rn = 1
            LEFT JOIN QCData qd
                ON pa.id = qd.plate_acquisition_id
                AND qd.rn = 1
            ORDER BY pa.timepoint ASC
        """

        try:
            cellprofiler_pipelines=pl.read_database_uri(
                query=query,
                uri=self.db_uri
            )
        except Exception as e:
            print("error : exception during query", query)
            raise e

        # all columns contain essentially non-numerical data, so make sure the datatype is proper string
        cellprofiler_pipelines=cellprofiler_pipelines.cast({
            x:pl.Utf8 for x in cellprofiler_pipelines.columns
        })

        return cellprofiler_pipelines

    def retrieve_compound_layout(
        self,
        source:str="db",
        
        db_barcode_colname:str="barcode",
        db_table_name:str="plate_v1",
        db_colnames_ret:tp.Optional[tp.List[str]]=None,

        file_barcode_colname:str="Barcode",
        file_colnames_ret:tp.Optional[tp.List[str]]=None,
    )->pl.DataFrame:
        """
            if source is "db":
                - retrieve data from database references by db_uri in self
                - return columns db_colnames_ret, or a default set of columns if this argument is None
                - barcode column name in db is given by db_barcode_colname 
            if source is other string:
                - assume this is a the path of a csv file containing the relevant information
                    contains barcode and at least basic information about compound for each well
                - barcode column name is given by file_barcode_colname
                - returns columns in file_colnames_ret, or a default set if this argument is None 
                
        """
        if source=="db":
            db_default_colnames_ret:list="well_id, pert_type, solvent, compound_name, cbkid, smiles, inchi, inkey".split(", ")
            if db_colnames_ret is None:
                db_colnames_ret=db_default_colnames_ret

            compound_layout=pl.read_database_uri(
                query=f"""
                    SELECT {', '.join(db_colnames_ret)}
                    FROM "{db_table_name}"
                    WHERE "{db_barcode_colname}" = '{self.barcode}'
                """,
                uri=self.db_uri
            )
            return compound_layout
        else:
            source_file=Path(source)
            assert source_file.exists(), f"""compound layout source file '{
                source}' does not exist (current directory is {os.getcwd()})"""

            file_default_colnames_ret:list="well_id pert_type".split(" ")
            if file_colnames_ret is None:
                file_colnames_ret=file_default_colnames_ret

            layout_df=pl.read_csv(str(source))
            layout_df=layout_df.filter(pl.col(file_barcode_colname)==self.barcode)
            layout_df=layout_df.select(file_colnames_ret)

            return layout_df

    def retrieve_plates_metadata(self)->tp.List["PlateMetadata"]:
        """
        get list of all plates in this experiment
        """
    
        cellprofiler_pipelines=self.retrieve_cellprofiler_pipelines()
        
        num_images_per_timepoint=None

        valid_timepoints=[]
        for row in tqdm(cellprofiler_pipelines.rows(named=True)):
            time_point=int(row["timepoint"])
            image_plate_dir=Path(row["image_dir"])

            if not image_plate_dir.exists():
                print(f"warning - did not find dir containing images for timepoint {time_point}")
                continue

            # this line takes 0.02s-12s, and there is no good explanation on why it varies that much
            t_images=list(image_plate_dir.glob("*.tif*"))

            # use first timepoint as reference for expected number of images per timepoint
            num_images_current_timepoint=len(t_images)
            if num_images_per_timepoint is None:
                num_images_per_timepoint=num_images_current_timepoint

            # print warning if there is a mismatch
            # e.g. could be issue during imaging, microscope crashed etc.
            if num_images_current_timepoint!=num_images_per_timepoint:
                print(f"warning: timepoint {time_point} " \
                    f"contains {num_images_current_timepoint} images, " \
                    f"though {num_images_per_timepoint} were expected")

            valid_timepoints.append(PlateMetadata(
                time_point_index=time_point,
                plate_name=Path(image_plate_dir).name,
            ))

        return valid_timepoints


@dataclass
class PlateMetadata:
    
    time_point_index: int
    """ time_point index is 1-indexed (!) """
    plate_name: str
    """ name of this plate, e.g. barcode """

    time_point:tp.Optional[int]=None
    """ time_point is 0-indexed (!) """

    feature_files:tp.Dict[str,pl.DataFrame]=dc.field(default_factory=dict)
    """
    feature files

    currently, these are required:
        - 'cytoplasm'
        - 'nuclei'
        
    and these are optional:
        - 'cells'
    """

    metadata_cols:tp.List[str]=dc.field(default_factory=lambda:[
        "Metadata_AcqID",
        "Metadata_Barcode",
        "Metadata_Well",
        "Metadata_Site"
    ])

    feature_file_prefix: str='featICF_'
    """ this prefix is used in several file operations """
    feature_set_names: tp.List[str]=dc.field(default_factory=lambda:['cytoplasm','nuclei','cells'])
    """
    suffix of the feature filenames, so that the filenames are
    f'{feature_file_prefix}{feature_set_names[i]}'
    """

    @property
    def feature_filenames(self)->tp.List[str]:
        return [self.feature_file_prefix+fsn for fsn in self.feature_set_names]

    df_qc:tp.Optional[pl.DataFrame]=None
    df_qc_images:tp.Optional[pl.DataFrame]=None
    df_qc_nuclei:tp.Optional[pl.DataFrame]=None

    def read_files(self,
        cellprofiler_output_path: Path,
        cellprofiler_pipelines: pl.DataFrame,

        *,
        timeit:bool=False,
    ):

        cellprofiler_image_timepoints:tp.List[str]=cellprofiler_pipelines["timepoint"].to_list()
        
        # not sure anymore why this is necessary, but it is
        self.time_point=int(cellprofiler_image_timepoints[self.time_point_index-1])

        current_pipeline=cellprofiler_pipelines.filter(
            pl.col("timepoint")==str(self.time_point)
        ).rows(named=True)[0]

        cp_plate_out_path=cellprofiler_output_path/current_pipeline["plate_id"]

        pipeline_id_qc=None
        if current_pipeline["pipeline_id_qc"]:
            pipeline_id_qc=cp_plate_out_path/current_pipeline["pipeline_id_qc"]

            qc_raw_filepath=list(
                Path(pipeline_id_qc).glob("qcRAW_images*.parquet")
            )[0]
            self.df_qc_images=pl.read_parquet(qc_raw_filepath)
            # print(f"{self.df_qc_images.shape = }")
            
            # disabled because it filters too many images in the cleo dataset
            if False:
                # filter out images with any qc flag set:
                # qc_flag_raw*, etc. qc_flag_rawACTIN_Blurred, *_Blurry, *_Saturated
                qc_flag_cols=[
                    x for x in self.df_qc_images.columns
                    if x.startswith("qc_flag_raw")
                ]
                # print(f"{qc_flag_cols = }")
                self.df_qc_images=self.df_qc_images.filter(
                    pl.sum_horizontal(
                        pl.col([
                            c for c in qc_flag_cols
                            if c.endswith("_Blurred")])
                    ) == 0
                )
                self.df_qc_images=self.df_qc_images.filter(
                    pl.sum_horizontal(
                        pl.col([
                            c for c in qc_flag_cols
                            if c.endswith("_Blurry")])
                    ) == 0
                )
                display(self.df_qc_images.select(qc_flag_cols).sum())
                print(f"(after flag filter) {self.df_qc_images.shape = }")
                display(self.df_qc_images.head(2))

            qc_nuclei_filename_list=list(
                Path(pipeline_id_qc).glob("qcRAW_nuclei*.parquet")
            )
            assert len(qc_nuclei_filename_list)==1, f"expected exactly one nuclei file, " \
                f"but found {len(qc_nuclei_filename_list)}"
            qc_nuclei_filename=qc_nuclei_filename_list[0]
            self.df_qc_nuclei=pl.read_parquet(qc_nuclei_filename)
            # print(f"{self.df_qc_nuclei.shape = }")

            if timeit:
                print_time("read qc files")

        if self.df_qc_nuclei is not None and self.df_qc_images is not None:
            if False:
                print("self.df_qc_images")
                # list all columns with str datatype
                print("str cols")
                print("\n".join([
                    f"  {c}" for c in self.df_qc_images.columns
                    if self.df_qc_images[c].dtype==pl.Utf8]
                ))
                print("metadata cols")
                print("\n".join([
                    f"  {c}" for c in self.df_qc_images.columns
                    if c.startswith("Metadata_")]
                ))
                print("self.df_qc_nuclei")
                print("str cols")
                print("\n".join([
                    f"  {c}" for c in self.df_qc_nuclei.columns
                    if self.df_qc_nuclei[c].dtype==pl.Utf8]
                ))
                print("metadata cols")
                print("\n".join([
                    f"  {c}" for c in self.df_qc_images.columns
                    if c.startswith("Metadata_")]
                ))

            self.df_qc=self.df_qc_images.join(
                self.df_qc_nuclei,
                # join (implicitely remove all cells where the image has been filtered out)
                how="inner",
                left_on=self.metadata_cols,
                right_on=self.metadata_cols,
            )

            if False:
                print(f"after join: {self.df_qc.shape = }")
                display(self.df_qc.head(2))

            if timeit:
                print_time("joined qc files")

        pipeline_id_features=cp_plate_out_path/current_pipeline["pipeline_id_feat"]

        feature_parquet_files=list(Path(pipeline_id_features).glob("*.parquet"))
        for f in sorted(feature_parquet_files):
            if not Path(f).stem in self.feature_filenames:
                continue

            feature_set_name=Path(f).stem[len(self.feature_file_prefix):]

            # add prefix to columns names because pd.merge renames the column names if they collide
            f_df=pl.read_parquet(f)
            f_df=f_df.rename({
                x:f'{feature_set_name}_{x}' for x in f_df.columns
                if not x.startswith("Metadata_")
            })

            self.feature_files[feature_set_name]=f_df

            if timeit:
                print(f"num entries in {feature_set_name} is {f_df.shape}")

        if timeit:
            print_time("reading feature files dones")


    def process(
        self,
        
        compound_layout: pl.DataFrame,

        *,
        timeit:bool=False,
        show_non_float_columns:bool=False,
        ensure_no_nan_or_inf:bool=False,
        
        handle_unused_features:tp.Optional[tp.Literal["remove"]]=None,
        unused_feature_threshold_std:float=0.0001,
        
        remove_correlated:bool=False,
        remove_correlation_threshold:float=0.9,

        pre_normalize_clip_method:tp.Optional[dict]=None,
        post_normalize_clip_method:tp.Optional[dict]=None,
    ) -> "Plate":
        """
            process all the things, combine dataframes, clean data
            
            timeit:
                print timestamp after certain steps to find bottlenecks in program runtime
            show_non_float_columns:
                print unexpected columns (names) that have datatype other than f32/f64
            ensure_no_nan_or_inf:
                check often that no column contains NaN/inf valued float entries 
            handle_unused_features:
                what to do with features that are highly correlated or have a
                std. dev. below unused_feature_threshold_std

                can be:
                    - None : do not treat them in a special way
                    - "remove" : remove them
                
                note: in one test, this reduced the number of features from 1800 to 1100.
            
            remove_correlated:
                remove highly correlated features

                features with correlation > remove_correlation_threshold are removed

                out of the pair of columns with high correlation, the specific choice of
                column to remove is arbitrary
        """

        if timeit:
            print_time("starting")

        assert "cytoplasm" in self.feature_files, "did not find cytoplasm feature file"
        assert "nuclei" in self.feature_files, "did not find nuclei feature file"

        # step 1: Take the mean values of 'multiple nuclei' belonging to one cell
        self.feature_files['nuclei'] = self.feature_files['nuclei'].group_by(
            self.metadata_cols
            + [
                "nuclei_Parent_cells",
            ]
        ).mean()

        if timeit:
            print_time("calculated average nucleus for each cell")

        # step 2: merge nuclei and cytoplasm objects
        df = self.feature_files['cytoplasm'].join(self.feature_files['nuclei'],
                        how='inner', 
                        right_on= self.metadata_cols + ["nuclei_Parent_cells"],
                        left_on = self.metadata_cols + ["cytoplasm_ObjectNumber"])

        if timeit:
            print_time(f"joined cytoplasm and nucleus, now have {len(df)} entries")

        if "cells" in self.feature_files:
            # step 3: join cells objects
            df = df.join(self.feature_files['cells'], how='inner', 
                            left_on =  self.metadata_cols + ["cytoplasm_ObjectNumber"],
                            right_on = self.metadata_cols + ["cells_ObjectNumber"])

            if timeit:
                print_time(f"joined cytoplasm+nucleus and cells, now have {df.shape} entries")

        df=df.drop([c for c in df.columns if is_meta_column(c)])
        if self.df_qc is not None:
            self.df_qc=self.df_qc.drop([
                c for c in self.df_qc.columns
                if is_meta_column(c)
            ])

        if timeit:
            s=f"dropped unused metadata {df.shape = }"
            if self.df_qc is not None:
                s+=f" {self.df_qc.shape = }"
            print_time(s)

        # convert all *_ImageNumber columns to int
        column_names_to_convert_to_int:tp.List[str]=[]
        for col in df.columns:
            for suffix in ["_ImageNumber","_Number_Object_Number"]:
                if col.endswith(suffix):
                    column_names_to_convert_to_int.append(col)

        if len(column_names_to_convert_to_int)>0:
            df=df.with_columns(
                pl.col(column_names_to_convert_to_int).cast(pl.Int32)
            )

        if timeit:
            print_time("converted some columns from f64 to i32")
            
        if handle_unused_features=="remove":
            if timeit:
                print_time(f"num columns before feature removal: {df.shape[1]}")
                
            # remove columns with std dev <= unused_feature_threshold_std
            unused_cols=[]
            for col in df.select(float_columns).columns:
                col_std=df.select(pl.col(col).std()).to_numpy()
                assert col_std.shape==(1,1), col_std.shape
                if col_std[0][0] <= unused_feature_threshold_std:
                    unused_cols.append(col)

            df = df.drop(unused_cols)

            # remove highly correlated features
            if remove_correlated:
                highly_correlated_columns = remove_highly_correlated(
                    df.select(float_columns),
                    threshold=remove_correlation_threshold,
                    remove_inplace=False
                )
                assert type(highly_correlated_columns)==list, f"expected list, got {
                    type(highly_correlated_columns)
                }"
                df = df.drop(highly_correlated_columns)

            if timeit:
                print_time(
                    "removed columns with high correlation" \
                    f"""Number of columns after removing sigma<={
                        unused_feature_threshold_std
                    } and highly correlated: {df.shape[1]}"""
                )
        elif handle_unused_features is None:
            pass
        else:
            raise ValueError(f"handle_unused_features is {handle_unused_features} but must be None or 'remove'")
        
        if timeit:
            print_time("joining done")

        # now we have all data merged, and can start filerting, cleaning etc.
        
        # if present, use self.df_qc to filter out bad images
        if self.df_qc is not None:
            pass # TODO

        # for some reason, the site is parsed as float, even though it really should be an int
        # so convert site column to int, and check that the converted values make sense
        metadata_site_dtype=str(df['Metadata_Site'].dtype)
        if "float" in metadata_site_dtype:
            # sometimes, for some reason, site indices are inf/nan
            site_is_nan_mask=np.isnan(df['Metadata_Site'])
            site_is_inf_mask=np.isinf(df['Metadata_Site'])

            try:
                num_sites_nan=np.sum(site_is_nan_mask)
                num_sites_inf=np.sum(site_is_inf_mask)
                assert num_sites_nan==0, f"found site with value nan (n = {num_sites_nan})"
                assert num_sites_inf==0, f"found site with value inf (n = {num_sites_inf})"
            except AssertionError as e:
                print(f"info - {e}")
                df=df[~(site_is_inf_mask|site_is_nan_mask)]

            num_metadata_site_entries_nonint=np.sum(np.abs(df['Metadata_Site']%1.0)>1e-6)
            assert num_metadata_site_entries_nonint==0, f"""ERROR : {
                num_metadata_site_entries_nonint
            } imaging sites don't have integer indices. that should not be the case, and likely indicates a bug."""

            df['Metadata_Site']=df['Metadata_Site'].cast(pl.Int32)

        if timeit:
            print_time("processed some metadate")

        # [optional] investigate non-float columns
        if show_non_float_columns:
            column_dtypes=dict()
            for column in df.columns:
                dtype=df[column].dtype
                if not dtype in column_dtypes:
                    column_dtypes[dtype]=[column]
                else:
                    column_dtypes[dtype].append(column)

            for dtype,cols in column_dtypes.items():
                print(f'df has {len(cols)} columns of type {dtype.__str__()}')
                if dtype == np.dtype('O') or dtype == np.dtype('int32'):
                    for c in sorted(cols):
                        print(f"\t{c}[0] = {df[c][0]}")

            if timeit:
                print_time("check out non-float columns")

        # drop all rows that contain nan
        num_rows_before_nan_trim=df.shape[0]
        df=remove_nans(df,df.select(float_columns).columns)
        num_rows_after_nan_trim=df.shape[0]

        if timeit:
            print_time("dropped NaNs")

        # remove outliers
        df_float_cols=df.select(float_columns).columns
        if pre_normalize_clip_method is not None: 
            df=handle_outliers(df,df_float_cols,**pre_normalize_clip_method)
            
        if timeit:
            print_time("pre-normalization clipping done")

        # filter wells not treated with any drug, just DMSO
        wells_with_dmso=compound_layout.filter(pl.col('compound_pert_type')=='negcon')

        # make sure we have some wells with DMSO !
        assert wells_with_dmso.shape[0]>0, "did not find any wells 'treated' with DMSO"

        # use join to quickly select the relevant rows
        # but add unique prefix to compound information columns (to avoid name collisions)
        df_DMSO = df.join(
            wells_with_dmso.rename({
                x:f"compoundinfo_{x}"
                for x in wells_with_dmso.columns
            }),
            left_on='Metadata_Well',
            right_on='compoundinfo_compound_well_id')
        
        # then remove the compound information columns again
        df_DMSO = df_DMSO.select([
            x for x in df_DMSO.columns
            if not x.startswith('compoundinfo_')
        ])
        # ensure there is sufficient data on the DMSO wells
        assert df_DMSO.shape[0]>0, "error!"

        # calculate mean morphology features for DMSO wells
        mu = df_DMSO.select(float_columns).mean()

        if ensure_no_nan_or_inf:
            df_checkNull(mu,raise_=True)
            df_checkInf(mu,raise_=True)
            df_checkNaN(mu,raise_=True)

        # calculate stdandard deviation of DMSO morphology features
        std = df_DMSO.select(float_columns).std()
        # replace 0 with 1 (specifically not clip) to avoid div by zero
        std = std.select([
            pl.col(c).map_dict({0: 1}, default=pl.first())
            for c in std.columns
        ])
        
        if ensure_no_nan_or_inf:
            df_checkInf(std,raise_=True)
            df_checkNull(std,raise_=True)
            df_checkNaN(std,raise_=True)
            df_checkValue(std,0,raise_=True)

        if timeit:
            print_time("calculated DMSO distribution")

        # normalize plate to DMSO distribution
        df_normalized = df.with_columns(
            (pl.col(mu.columns) - mu) / std
        )

        if ensure_no_nan_or_inf:
            df_checkInf(df_normalized,raise_=True)
            df_checkNull(df_normalized,raise_=True)
            df_checkNaN(df_normalized,raise_=True)

        # write back (into dataframe containing additional columns)
        df=df.with_columns([df_normalized[c] for c in df_normalized.columns])

        if timeit:
            print_time("normalized to DMSO distribution")
            
        # clip normalized values
        df_float_cols=df.select(float_columns).columns
        if post_normalize_clip_method is not None:
            df=handle_outliers(df,df_float_cols,**post_normalize_clip_method)

        if timeit:
            print_time("post-aggregation clipping to quantiles done")

        # counts unique objects remaining (combined from feature_files, which contain overlapping data) 
        num_objects = df.shape[0]

        fraction_objects_containing_nan=1-(num_rows_after_nan_trim/num_rows_before_nan_trim)
        if timeit:
            print_time(f"""num objects (cells) {num_objects} ({
                (fraction_objects_containing_nan*100):.2f
            }% were NaN)""")

        # group/combine by well

        df=df.drop(columns=['Metadata_Site']) # should be redundant
        df_float_columns=set(df.select(float_columns).columns)
        group_by_columns=['Metadata_Well']
        other_columns=set(df.columns) \
            - df_float_columns \
            - set(group_by_columns)
        
        # group by mean for all float features, and group by first for all non-float columns (indices and string metadata)
        group_by_aggregates=[
            *[pl.mean(x) for x in list(df_float_columns)],
            *[pl.first(x) for x in list(other_columns)]
        ]

        combined_per_well=df.group_by(group_by_columns).agg(group_by_aggregates)

        if timeit:
            print_time("binned data per well")

        # add compound information
        combined_per_well=combined_per_well.join(
            compound_layout,
            how='left',
            left_on=["Metadata_Well"],
            right_on=["compound_well_id"]
        )

        if timeit:
            print_time("added compound information")

        assert self.time_point is not None, "time_point is None, but should be set"

        cpi = Plate(
            plate_name=self.plate_name,
            timepoint=self.time_point,
            combined_per_well=combined_per_well,
            qc_df=self.df_qc,
        )
        
        if timeit:
            print("first two entries of final dataframe:")
            display(combined_per_well.head(2))

            print_time(f"timepoint {self.time_point} done")

        return cpi


@dataclass
class Plate:
    timepoint:int
    """ timepoint is 0-indexed here (1-indexed in metadata, but 0-indexed in filesystem)"""

    plate_name:str
    """ barcode or something like that """
    
    combined_per_well:pl.DataFrame
    
    qc_df:tp.Optional[pl.DataFrame]=None
    
    def numeric_data(self)->pl.DataFrame:
        ret=self.combined_per_well.select(float_columns)
        ret=ret.select([c for c in ret.columns if not c.startswith("compound_")])
        return ret
    
    def plot(self,
        *,
        print_feature_pca_fraction:bool=False,
        method:str="pca",
    ):
        """
            print_feature_pca_fraction:
                print the fraction of variance explained by each dimension for each of the first two principal components
        """

        df=self.numeric_data()

        df_checkNull(df,raise_=True)
        df_checkInf(df,raise_=True)
        df_checkNaN(df,raise_=True)

        if method=="pca":
            n_components=2
            pca_red = PCA(n_components=n_components)
            reduced_data = pca_red.fit_transform(df.to_pandas())

            if print_feature_pca_fraction:
                # Get the loadings and calculate squared loadings for variance contribution
                loadings = pca_red.components_.T * np.sqrt(pca_red.explained_variance_)
                squared_loadings = pd.DataFrame(loadings**2, columns=[
                    f'PC{i+1}' for i in range(n_components)
                ], index=df.columns)

                # For each PC, get the top 5 contributing features
                top_contributors = {}
                for pc in squared_loadings.columns:
                    top_contributors[pc] = squared_loadings[pc].nlargest(5).sort_values(ascending=False)

                # Convert to DataFrame for better visualization
                top_contributors_df = pd.DataFrame(top_contributors)

                print(top_contributors_df) # type: ignore
        elif method=="umap":
            n_components=2
            umap_red = umap.UMAP(n_components=n_components) # type: ignore
            reduced_data=umap_red.fit_transform(df)
        else:
            raise ValueError(f"unknown method {method} (valid methods are [pca|umap])")

        # Visualize
        fig=px.scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            color=self.combined_per_well['compound_pert_type'],
            color_discrete_map={
                'trt':'blue',
                'poscon':'green',
                'negcon':'red'
            },
            labels={'x':'PC1','y':'PC2'},
            hover_data=[
                #self.combined_per_well['compoundinfo_name'],
                #self.combined_per_well['compound_batchid']
            ],
            title=f'{method} of {self.plate_name}'
        )

        # remove margins
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
        ).update_traces(
            marker=dict(size=10)
        ).show()

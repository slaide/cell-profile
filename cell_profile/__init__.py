from dataclasses import dataclass
import dataclasses as dc
from pathlib import Path
import typing as tp
import os

import numpy as np
import polars as pl
import pandas as pd

import plotly.express as px # type: ignore

from sklearn.decomposition import PCA # type: ignore
import umap # type: ignore

from .df_util import is_meta_column, handle_outliers, remove_nans, \
        remove_highly_correlated, Sigma, Quantile, \
        df_checkNull, df_checkValue, df_checkInf, df_checkNaN, float_columns

from .misc import display, print, print_time, tqdm

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
            assert source_file.exists(), f"compound layout source file '{source}'" \
                 f" does not exist (current directory is {os.getcwd()})"

            file_default_colnames_ret:list="well_id pert_type".split(" ")
            if file_colnames_ret is None:
                file_colnames_ret=file_default_colnames_ret

            layout_df=pl.read_csv(str(source))
            layout_df=layout_df.filter(pl.col(file_barcode_colname)==self.barcode)
            layout_df=layout_df.select(file_colnames_ret)

            return layout_df

    def retrieve_plates_metadata(self,**plate_metadata_init_kwargs)->tp.List["PlateMetadata"]:
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
                **plate_metadata_init_kwargs,
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

    keys are the feature set names (not filename! i.e. must not end in .csv etc.), values are the dataframes containing the features
    """

    metadata_cols:tp.List[str]=dc.field(default_factory=lambda:[
        "Metadata_AcqID",
        "Metadata_Barcode",
        "Metadata_Well",
        "Metadata_Site"
    ])

    root_key:tp.Dict[str,str]=dc.field(default_factory=lambda:{
        "root_file":"nuclei",
        "root_attribute_col":"ObjectNumber",
        "foreign_attribute_col":"{featureFilename}_Parent_nucleus",
    })
    """
    root_key is used to join the different feature files together

    must have a structure with exactly these keys:
        - root_file

            this feature file contains the primary objects

        - root_attribute_col

            in root_file, this column identifies a primary object, so joining will be relative to this (e.g. ObjectNumber)

        - foreign_attribute_col

            root_attribute_col will be joined with this identifier in other feature frames

            this string may contain '{featureFilename}' (e.g. '{featureFilename}_Parent_cells') which is expaneded to insert the feature filename
    """

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
        pipeline_timepoint=cellprofiler_pipelines["timepoint"]
        cellprofiler_image_timepoints:tp.List[str]=pipeline_timepoint.to_list()
        
        # not sure anymore why this is necessary, but it is
        self.time_point=int(cellprofiler_image_timepoints[self.time_point_index-1])

        current_pipeline:dict[str,str|int|float]=cellprofiler_pipelines.filter(
            pl.col("timepoint")==str(self.time_point)
        ).rows(named=True)[0]

        pipeline_plat_id=current_pipeline.get("plate_id")
        assert isinstance(pipeline_plat_id,str), f"{type(pipeline_plat_id)}"
        cp_plate_out_path=cellprofiler_output_path/pipeline_plat_id

        pipeline_id_qc_str=current_pipeline.get("pipeline_id_qc")
        if pipeline_id_qc_str:
            assert isinstance(pipeline_id_qc_str,str), f"{type(pipeline_id_qc_str)}"
            pipeline_id_qc=cp_plate_out_path/pipeline_id_qc_str

            qcraw_images_parquet_files=list(
                Path(pipeline_id_qc).glob("qcRAW_images*.parquet")
            )
            assert len(qcraw_images_parquet_files)>0
            qc_raw_filepath=qcraw_images_parquet_files[0]
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

        pipeline_id_feat=current_pipeline["pipeline_id_feat"]
        assert isinstance(pipeline_id_feat,str), f"{type(pipeline_id_feat)}"
        pipeline_id_features=cp_plate_out_path/pipeline_id_feat

        feature_parquet_files=list(Path(pipeline_id_features).glob("*.parquet"))
        for f in sorted(feature_parquet_files):
            if not Path(f).stem in self.feature_filenames:
                continue

            feature_set_name=Path(f).stem[len(self.feature_file_prefix):]

            # add prefix to columns names because pd.merge renames the column names if they collide
            f_df=pl.read_parquet(f)
            f_df=f_df.rename({
                x:f'{feature_set_name}_{x}' for x in f_df.columns
                if not (
                    x in self.metadata_cols
                    or (Path(f).stem==self.root_key["root_file"] and x==self.root_key["root_attribute_col"])
                )
            })

            self.feature_files[feature_set_name]=f_df

            if timeit:
                print(f"num entries in {feature_set_name} is {f_df.shape}")

        if timeit:
            print_time("reading feature files dones")

    @pl.StringCache() # enable polars string cache for this function to optimize conversion from string to categorical column data type
    def process(
        self,
        
        compound_layout: pl.DataFrame,

        *,
        timeit:bool=False,

        df_cols_float_to_int:tp.List[str]=[],

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

            df_cols_float_to_int:
                columns in the final dataframe to be converted from float to int

                this includes checks to ensure that the values are int(-ish)
            
            remove_correlated:
                remove highly correlated features

                features with correlation > remove_correlation_threshold are removed

                out of the pair of columns with high correlation, the specific choice of
                column to remove is arbitrary
        """

        if timeit:
            print_time("starting")

        assert self.root_key["root_file"] in self.feature_files, "did not find root feature file: "+self.root_key["root_file"]+" in "+str(self.feature_files.keys())

        # step 1: read root file
        df = self.feature_files[self.root_key["root_file"]]
        root_feature_name=self.root_key["root_attribute_col"]

        assert root_feature_name in df.columns, f"did not find {root_feature_name} of root file {self.root_key['root_file']} in {df.columns}"

        if timeit:
            print_time(f"read root file {self.root_key['root_file']} with {len(df)} entries")

        optional_feature_set_names=set(self.feature_set_names)-{self.root_key["root_file"]}

        for _feature_name in optional_feature_set_names:
            if _feature_name in self.feature_files:
                right_feature_name=self.root_key["foreign_attribute_col"].format(featureFilename=_feature_name)
                if right_feature_name not in self.feature_files[_feature_name].columns:
                    relevant_cols=[c for c in self.feature_files[_feature_name].columns if c.endswith(right_feature_name[-6:])]
                    raise ValueError(f"did not find column {right_feature_name} in {_feature_name}, some columns that may be relevant are {relevant_cols}")

                # polars re-uses the left column name!
                # e.g. on inner joining left.c1 with right.c2, the resulting column is named c1 (and c2 is lost)

                left_on_cols=self.metadata_cols+[root_feature_name]
                right_on_cols=self.metadata_cols+[right_feature_name]

                df = df.join(
                            # aggregate other dataframe and calc mean to eliminate duplicate entries
                            self.feature_files[_feature_name].group_by(right_on_cols).mean(),
                            how='inner', 
                            left_on=left_on_cols,
                            right_on=right_on_cols,
                        )

                if timeit:
                    print_time(f"joined df and {_feature_name}, now have {len(df)} entries")

        # drop unused metadata columns
        df=df.drop([
            c for c in df.columns
            if is_meta_column(c)
        ])
        
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
                assert type(highly_correlated_columns)==list, f"expected list, got " \
                    f"{type(highly_correlated_columns)}"
                df = df.drop(highly_correlated_columns)

            if timeit:
                print_time(
                    "removed columns with high correlation" \
                    f"Number of columns after removing sigma<=" \
                    f"{unused_feature_threshold_std}" \
                    f" and highly correlated: {df.shape[1]}"
                )
                
        elif handle_unused_features is None:
            pass
            
        else:
            raise ValueError(f"handle_unused_features is {handle_unused_features} but must be None or 'remove'")
        
        if timeit:
            print_time(f"joining done, {len(df)} remaining")

        # now we have all data merged, and can start filerting, cleaning etc
        
        # dataframes in use at this point:
        # - df: contains all per-cell features
        # - compound_layout: contains per-well metadata

        # convert string typed columns to categorical to save space
        if 1:
            def convert_string_cols_to_categorical(df:pl.DataFrame)->pl.DataFrame:
                converted_columns = [col for col in df.columns if df.schema[col] == pl.Utf8]
                df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in converted_columns])
                
                if timeit:
                    print_time(f"Converted string columns to categorical: {converted_columns}")
    
                return df

            df=convert_string_cols_to_categorical(df)
            compound_layout=convert_string_cols_to_categorical(compound_layout)
        
        # if present, use self.df_qc to filter out bad images
        if self.df_qc is not None:
            pass # TODO

        if len(df_cols_float_to_int)>0:
            for col_to_convert in df_cols_float_to_int:
                # for some reason, the site is parsed as float, even though it really should be an int
                # so convert site column to int, and check that the converted values make sense
                metadata_site_dtype=str(df[col_to_convert].dtype)
                if "float" in metadata_site_dtype:
                    # sometimes, for some reason, site indices are inf/nan
                    df_checkNaN(df.select(col_to_convert),raise_=True)
                    df_checkInf(df.select(col_to_convert),raise_=True)

                    num_metadata_site_entries_nonint=np.sum(np.abs(df.select(col_to_convert)%1.0)>1e-6)
                    assert num_metadata_site_entries_nonint==0, f"ERROR :" \
                        f" {num_metadata_site_entries_nonint}" \
                        f" values in {col_to_convert} are not integers!"

                    df[col_to_convert]=df[col_to_convert].cast(pl.Int32)

            if timeit:
                print_time("converted some float cols to int")

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
            print_time(f"dropped NaNs, {len(df)} remaining")

        # remove outliers
        df_float_cols=df.select(float_columns).columns
        if pre_normalize_clip_method is not None: 
            df=handle_outliers(df,df_float_cols,**pre_normalize_clip_method)
            
        if timeit:
            print_time(f"pre-normalization clipping done, {len(df)} remaining")

        # filter wells not treated with any drug, just DMSO
        wells_with_dmso=compound_layout.filter(pl.col('compound_pert_type')=='negcon')

        # make sure we have some wells with DMSO !
        assert wells_with_dmso.shape[0]>0, "did not find any wells 'treated' with DMSO"

        data_join_compound_layout_left_on=["Metadata_Well"]
        """ combined_per_well join compound_layout on this column"""
        data_join_compound_layout_right_on=["compound_well_id"]
        """ combined_per_well join compound_layout on this column"""

        # use join to quickly select the relevant rows
        # note: this statement below does not introduce new/additional columns
        df_DMSO = df.join(
            wells_with_dmso.select(data_join_compound_layout_right_on),
            how="inner",
            left_on=data_join_compound_layout_left_on,
            right_on=data_join_compound_layout_right_on
        )
        
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
            pl.col(c).replace({0: 1}, default=pl.first())
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
        df_normalized = df.with_columns([
            (pl.col(c) - mu.select(c)) / std.select(c)
            for c in mu.columns
        ])

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
            print_time(f"num objects (cells) {num_objects} (" \
                f"{(fraction_objects_containing_nan*100):.2f}%" \
                f" were NaN)")

        # group/combine by well

        df_float_columns=set(df.select(float_columns).columns)
        group_by_columns=self.metadata_cols # this might include more features than strictly necessary, but that is fine
        other_columns=set(df.columns) \
            - df_float_columns \
            - set(group_by_columns)
        
        # group by mean for all float features, and group by first for all non-float columns (indices and string metadata)
        group_by_aggregates=[
            *[pl.mean(x) for x in df_float_columns],
            *[pl.first(x) for x in other_columns]
        ]

        combined_per_well=df.group_by(group_by_columns).agg(group_by_aggregates)

        if timeit:
            print_time("binned data per well")

        # add compound information
        combined_per_well=combined_per_well.join(
            compound_layout,
            how='left',
            left_on=data_join_compound_layout_left_on,
            right_on=data_join_compound_layout_right_on,
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
        method:tp.Literal["pca","umap","pacmap"]="pca",

        file_out:tp.Optional[str]=None,

        umap_args:tp.Optional[tp.Dict[str,tp.Any]]=None,
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
            umap_default_args=dict(
                random_state=42,#for reproducible results
                n_components=2,#we usually just use 2 dimensions, for plotting
                n_neighbors=10,#according to the docs, this is a sensible default
                min_dist=0.1,#according to the docs, this is a sensible default
                metric='correlation',#just use any, correlation was not chosen for a specific reason 
            )

            if umap_args is None:
                umap_args={}

            # set default values only if not already present
            for key,value in umap_default_args.items():
                if not key in umap_args:
                    umap_args[key]=value
            
            umap_red = umap.UMAP(**umap_args)
            
            reduced_data=umap_red.fit_transform(df)
            
        elif method=="pacmap":
            # https://pypi.org/project/pacmap/
            raise NotImplementedError("pacmap")
            
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
        )

        if file_out is None:
            fig.show()
        else:
            fig.write_html(file_out)

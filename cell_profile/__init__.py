print("hello from init")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import typing as tp
import re
import time
import os

import numpy as np
import polars as pl
import pandas as pd

import plotly.express as px

from sklearn.decomposition import PCA
import umap

try:
    from IPython.display import display
except:
    def display(*args,**kwargs):
        print(*args,**kwargs)

# -- parallel map implementation
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def par_map(items,mapfunc,num_workers=None,**tqdm_args):

    start_time=time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and collect Future objects
        futures = {executor.submit(mapfunc, item): item for item in items}

        # Use tqdm to display progress
        results = []
        for future in tqdm(as_completed(futures), total=len(items)):
            results.append(future.result())

    end_time=time.time()
    print(f"time elapsed: {(end_time-start_time):.3f}s")

# example:
# par_map(range(10),lambda x:time.sleep(0.2),num_workers=1)

# -- par_map end

# chatgpt's check to see if this code is running in a jupyter notebook or not
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter Notebook or Jupyter Lab
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type, assume not a notebook
    except NameError:
        return False  # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# implement custom print function for better formatting --
def better_print(s_in:str,*args,tab_width:int=4,**kwargs):
    s=str(s_in).replace('\t',' '*tab_width)
    raw_print(s,*args,**kwargs)
    
print_already_overwritten=True
try:
    _test=raw_print
except:
    print_already_overwritten=False
    
if not print_already_overwritten:
    raw_print=print
print=better_print
# -- end custom print implementation

def print_time(msg=None,prefix=""):
    print_str=f"{prefix+' ' if prefix is not None else ''}" \
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" \
              f"{' '+msg if msg else ''}"
    print(print_str)

# this is used quite often to only select those columns that contain float values
float_columns=[pl.col(pl.Float32),pl.col(pl.Float64)]

# this is code from Dan
def is_meta_column(
    c:str,

    allowlist:tp.List[str]=["Metadata_Well","Metadata_Barcode","Metadata_AcqID","Metadata_Site"],
    denylist:tp.List[str]=[],
)->bool:
    """
        allowlist:
            the function will return False for these, no matter if they are metadata or not
        denylist:
            the function will return True for these, no matter if they are metadata or not
    """

    if c in allowlist:
        return False
    if c in denylist:
        return True

    for ex in '''
        Metadata
        ^Count
        ImageNumber
        Object
        Parent
        Children
        Plate
        Well
        Location
        _[XYZ]_
        _[XYZ]$
        BoundingBox
        Phase
        Orientation
        Angle
        Scale
        Scaling
        Width
        Height
        Group
        FileName
        PathName
        URL
        Execution
        ModuleError
        LargeBrightArtefact
        MD5Digest
    '''.split():
        if re.search(ex, c):
            return True
    return False

@dataclass
class Experiment:
    barcode:str
    db_uri:str

    # compound_layout_args:tp.Optional[dict]=None

    def retrieve_cellprofiler_pipelines(self)->pl.DataFrame:
        """
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
        cellprofiler_pipelines=cellprofiler_pipelines.cast({x:pl.Utf8 for x in cellprofiler_pipelines.columns})

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
            db_default_colnames_ret="well_id, pert_type, solvent, compound_name, cbkid, smiles, inchi, inkey".split(", ")
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
            assert source_file.exists(), f"compound layout source file '{source}' does not exist (current directory is {os.getcwd()})"

            file_default_colnames_ret="well_id pert_type".split(" ")
            if file_colnames_ret is None:
                file_colnames_ret=file_default_colnames_ret

            layout_df=pl.read_csv(str(source))
            layout_df=layout_df.filter(pl.col(file_barcode_colname)==self.barcode)
            layout_df=layout_df.select(file_colnames_ret)

            return layout_df

    def retrieve_plates_metadata(self)->tp.List["PlateMetadata"]:
    
        cellprofiler_pipelines=self.retrieve_cellprofiler_pipelines()
        
        num_images_per_timepoint=None

        valid_timepoints=[]
        for row in tqdm(cellprofiler_pipelines.rows(named=True)):
            i_path=Path(row["image_dir"])
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
                print(f"warning: timepoint {image_plate_dir.name} " \
                    f"contains {num_images_current_timepoint} images, " \
                    f"though {num_images_per_timepoint} were expected")

            valid_timepoints.append(PlateMetadata(
                time_point=time_point,
                image_plate_dir=image_plate_dir,
                cellprofiler_output_dir=i_path
            ))

        return valid_timepoints


@dataclass
class PlateMetadata:
    """ time_point is 1-indexed (!)"""
    
    time_point: int
    image_plate_dir: tp.Union[str,Path]
    cellprofiler_output_dir: tp.Union[str,Path]
    
    def process(
        self,
        
        cellprofiler_output_path: Path,
        cellprofiler_pipelines: pl.DataFrame,
        compound_layout: pl.DataFrame,
        *,
        timeit:bool=False,
        show_non_float_columns:bool=False,
        ensure_no_nan_or_inf:bool=False,
        
        handle_unused_features:tp.Optional[str]=None,
        unused_feature_threshold_std:float=0.0001,
        
        print_unused_columns:bool=False,

        pre_normalize_clip_method:tp.Optional[dict]=dict(
            level_method="sigma",
            lower_level=-3,
            upper_level=3,
            method="remove"
        ),
        post_normalize_clip_method:tp.Optional[dict]=dict(
            level_method="sigma",
            lower_level=-3,
            upper_level=3,
            method="clip"
        ),
    ):
        """
            process all the things, combine dataframes, clean data
            
            timeit:
                print timestamp after certain steps to find bottlenecks in program runtime
            show_non_float_columns:
                print unexpected columns (names) that have datatype other than f32/f64
            ensure_no_nan_or_inf:
                check often that no column contains NaN/inf valued float entries 
            handle_unused_features:
                what to do with features that are highly correlated or have a std. dev. of zero
                can be None (to not do anything) or 'remove' (to remove them)
                
                note: in one test, this reduced the number of features from 1800 to 1100.
        """

        cellprofiler_image_timepoints:tp.List[str]=cellprofiler_pipelines["timepoint"]

        # -- should be default arguments

        feature_set_names=['cytoplasm','nuclei','cells']
        # the prefix is used later on by itself
        feature_file_prefix='featICF_'

        feature_filenames=[feature_file_prefix+fsn for fsn in feature_set_names]

        # -- end default arguments

        if timeit:
            print_time("starting")
        
        i_i=self.time_point-1

        current_pipeline=cellprofiler_pipelines.filter(pl.col("timepoint")==str(self.time_point)).rows(named=True)[0]

        cp_plate_out_path=cellprofiler_output_path/current_pipeline["plate_id"]

        pipeline_id_qc=None
        qc_df=None
        qc_join_on_col_names=["Metadata_AcqID","Metadata_Barcode","Metadata_Well","Metadata_Site"]
        metadata_cols=qc_join_on_col_names
        if current_pipeline["pipeline_id_qc"]:
            pipeline_id_qc=cp_plate_out_path/current_pipeline["pipeline_id_qc"]

            qc_raw_filepath=list(Path(pipeline_id_qc).glob("qcRAW_images*.parquet"))[0]
            qc_images_df=pl.read_parquet(qc_raw_filepath)
            # print(f"{qc_images_df.shape = }")
            
            # filter out images with any qc flag set:
            # qc_flag_raw*, etc. qc_flag_rawACTIN_Blurred, *_Blurry, *_Saturated
            qc_flag_cols=[x for x in qc_images_df.columns if x.startswith("qc_flag_raw")]
            # print(f"{qc_flag_cols = }")
            #qc_images_df=qc_images_df.filter(pl.sum_horizontal(pl.col([c for c in qc_flag_cols if c.endswith("_Blurred")])) == 0)
            #qc_images_df=qc_images_df.filter(pl.sum_horizontal(pl.col([c for c in qc_flag_cols if c.endswith("_Blurry")])) == 0)
            if False:
                display(qc_images_df.select(qc_flag_cols).sum())
                print(f"(after flag filter) {qc_images_df.shape = }")
                display(qc_images_df.head(2))

            qc_nuclei_filename=list(Path(pipeline_id_qc).glob("qcRAW_nuclei*.parquet"))[0]
            qc_nuclei_df=pl.read_parquet(qc_nuclei_filename)
            # print(f"{qc_nuclei_df.shape = }")

            if False:
                print("qc_images_df")
                # list all columns with str datatype
                print("str cols")
                print("\n".join([f"  {c}" for c in qc_images_df.columns if qc_images_df[c].dtype==pl.Utf8]))
                print("metadata cols")
                print("\n".join([f"  {c}" for c in qc_images_df.columns if c.startswith("Metadata_")]))
                print("qc_nuclei_df")
                print("str cols")
                print("\n".join([f"  {c}" for c in qc_nuclei_df.columns if qc_nuclei_df[c].dtype==pl.Utf8]))
                print("metadata cols")
                print("\n".join([f"  {c}" for c in qc_images_df.columns if c.startswith("Metadata_")]))

            qc_df=qc_images_df.join(
                qc_nuclei_df,
                # join (implicitely remove all cells where the image has been filtered out)
                how="inner",
                left_on=qc_join_on_col_names,
                right_on=qc_join_on_col_names,
            )

            if False:
                print(f"after join: {qc_df.shape = }")
                display(qc_df.head(2))

        if timeit:
            print_time("read qc files")

        pipeline_id_features=cp_plate_out_path/current_pipeline["pipeline_id_feat"]

        feature_files=dict()

        feature_parquet_files=Path(pipeline_id_features).glob("*.parquet")
        feature_set_cellcount={}
        for f in sorted(feature_parquet_files):
            if not Path(f).stem in feature_filenames:
                continue

            feature_set_name=Path(f).stem[len(feature_file_prefix):]

            # add prefix to columns names because pd.merge renames the column names if they collide
            f_df=pl.read_parquet(f)
            f_df=f_df.rename({x:f'{feature_set_name}_{x}' for x in f_df.columns if not x.startswith("Metadata_")})

            feature_files[feature_set_name]=f_df

            if timeit:
                print(f"num entries in {feature_set_name} is {f_df.shape}")

            feature_set_cellcount[feature_set_name]=f_df.shape[0]

        if timeit:
            print_time("read files")

        # step 1: Take the mean values of 'multiple nuclei' belonging to one cell
        feature_files['nuclei'] = feature_files['nuclei'].group_by(
            metadata_cols
            + [
                "nuclei_Parent_cells",
            ]
        ).mean()
        # print(f"{feature_files['nuclei'].shape = }")

        if timeit:
            print_time("calculated average nucleus for each cell")

        # step 2: merge nuclei and cytoplasm objects
        df = feature_files['cytoplasm'].join(feature_files['nuclei'],
                        how='inner', 
                        right_on= metadata_cols + ["nuclei_Parent_cells"],
                        left_on = metadata_cols + ["cytoplasm_ObjectNumber"])

        if timeit:
            print_time(f"joined cytoplasm and nucleus, now have {len(df)} entries")

        # step 3: join cells objects
        df = df.join(feature_files['cells'], how='inner', 
                        left_on =  metadata_cols + ["cytoplasm_ObjectNumber"],
                        right_on = metadata_cols + ["cells_ObjectNumber"])

        if timeit:
            print_time(f"joined cytoplasm+nucleus and cells, now have {df.shape} entries")

        df=df.drop([c for c in df.columns if is_meta_column(c)])
        qc_df=qc_df.drop([c for c in qc_df.columns if is_meta_column(c)])

        if timeit:
            s=f"dropped unused metadata {df.shape = }"
            if qc_df is not None:
                s+=f" {qc_df.shape = }"
            print_time(s)

        # convert all *_ImageNumber columns to int
        for col in df.columns:
            convert_col_to_int=False
            for suffix in ["_ImageNumber","_Number_Object_Number"]:
                if col.endswith(suffix):
                    convert_col_to_int=True

            if convert_col_to_int:
                df=df.with_columns(pl.col(col).cast(pl.Int32))

        if timeit:
            print_time("converted some columns from f64 to i32")
            
        if handle_unused_features=="remove":
            if timeit:
                print_time(f"num columns before feature removal: {df.shape[1]}")
                
            # remove columns with std dev <= unused_feature_threshold_std
            unused_cols=[]
            for col in df.select(float_columns).columns:
                if df.select(pl.col(col).std()).to_numpy()[0][0] <= unused_feature_threshold_std:
                    unused_cols.append(col)

            df = df.drop(unused_cols)

            # remove highly correlated features
            highly_correlated_columns = remove_highly_correlated(df.select(float_columns),remove_inplace=False)
            df = df.drop(highly_correlated_columns)

            if timeit:
                print_time(
                    "removed columns with high correlation" \
                    f"Number of columns after removing sigma<={unused_feature_threshold_std} and highly correlated: {df.shape[1]}"
                )
        elif handle_unused_features is None:
            pass
        else:
            raise ValueError(f"handle_unused_features is {handle_unused_features} but must be None or 'remove'")
        

        # merge with qc data, if present
        qc_df=None
        if qc_df is not None:
            for c in qc_df.select(pl.col(pl.Utf8)).columns:
                print(f"qc df col '{c}'")
            for c in df.select(pl.col(pl.Utf8)).columns:
                print(f"df col '{c}'")
            
            print(f"{qc_df.shape = }  {df.shape = }")
            df=qc_df.join(
                df,
                how="inner",
                left_on=metadata_cols,
                right_on=metadata_cols,
            )

            if timeit:
                print(f"num rows after qc+feature joining: {df.shape[0]}")
                print_time("joined cytoplasm+nucleus+cells and qc data")

        if timeit:
            print_time("joining done")

        # now we have all data merged, and can start filerting, cleaning etc.
        
        # if present, use qc_df to filter out bad cells/images
        if qc_df is not None:
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
            assert num_metadata_site_entries_nonint==0, f"ERROR : {num_metadata_site_entries_nonint} imaging sites don't have integer indices. that should not be the case, and likely indicates a bug."

            df['Metadata_Site']=df['Metadata_Site'].astype(np.dtype('int32'))

        if timeit:
            print_time("processed some metadate")

        # [optional] investigate non-float columns
        if show_non_float_columns and i_i==0:
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

        # discard columns with unused information
        for col in df.columns:            
            if is_meta_column(col):
                if print_unused_columns:
                    print("unused column:",col)
                df=df.drop(col)

        if timeit:
            print_time("remove unused metadata")

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
            wells_with_dmso.rename({x:f"compoundinfo_{x}" for x in wells_with_dmso.columns}),
            left_on='Metadata_Well',
            right_on='compoundinfo_compound_well_id')
        
        # then remove the compound information columns again
        df_DMSO = df_DMSO.select([x for x in df_DMSO.columns if not x.startswith('compoundinfo_')])
        # ensure there is sufficient data on the DMSO wells
        assert df_DMSO.shape[0]>0, "error!"

        # calculate mean morphology features for DMSO wells
        mu = df_DMSO.select(float_columns).mean()

        if ensure_no_nan_or_inf:
            for col in mu.columns:
                if mu[col].is_null().any():
                    raise RuntimeError(f"some mean value in column {col,i} is nan?!")
                if mu[col].is_infinite().any():
                    raise RuntimeError(f"some mean value in column {col,i} is infinite?!")

        # calculate stdandard deviation of DMSO morphology features
        std = df_DMSO.select(float_columns).std()
        # replace 0 with 1 (specifically not clip) to avoid div by zero
        std = std.select([pl.col(c).map_dict({0: 1}, default=pl.col(c)) for c in std.columns])
        
        if ensure_no_nan_or_inf:
            for i,col in enumerate(std.columns):
                if std[col].is_null().any():
                    raise RuntimeError(f"some std value in column {col,i} is nan?!")
                if std[col].is_infinite().any():
                    raise RuntimeError(f"some std value in column {col,i} is infinite?!")
                if (std[col]==0).any():
                    raise RuntimeError(f"unexpected 0 in column {col}")

        if timeit:
            print_time("calculated DMSO distribution")

        # normalize plate to DMSO distribution
        df_normalized = df.with_columns([(pl.col(c) - mu[c]) / std[c] for c in mu.columns])

        if ensure_no_nan_or_inf:
            found_nan=False
            for i,col in enumerate(mu.columns):
                if df_normalized[col].is_null().any():
                    found_nan=True
                    print(f"some value in column {col,i} is nan")

            if found_nan:
                raise RuntimeError("found nan")

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
            print_time(f"num objects (cells) {num_objects} ({(fraction_objects_containing_nan*100):.2f}% were NaN)")

        # group/combine by well
        df=df.drop(columns=['Metadata_Site']) # should be redundant
        df_float_columns=set(list(df.select(float_columns).columns))
        # group_by_columns=['Metadata_Barcode','Metadata_Well'] # this code was here at some point, but it does not make sense because the barcode is the same for all wells...?
        group_by_columns=['Metadata_Well']
        other_columns=set(list(df.columns))-df_float_columns-set(group_by_columns)
        # group by mean for all float features, and group by first for all non-float columns (indices and string metadata)
        group_by_aggregates=[
            *[pl.mean(x) for x in list(df_float_columns)],
            *[pl.first(x) for x in list(other_columns)]
        ]
        combined_per_well=df.group_by(group_by_columns).agg(group_by_aggregates)

        if timeit:
            print_time("binned data per well")

        # add compound information
        combined_per_well=combined_per_well.join(compound_layout,how='left',left_on=["Metadata_Well"],right_on=["compound_well_id"])
        if timeit:
            print_time("added compound information")

        cpi = Plate(
            image_plate_dir=self.image_plate_dir,
            image_id=cp_plate_out_path,
            pipeline_id_qc=str(pipeline_id_qc) if pipeline_id_qc else None,
            pipeline_id_features=str(pipeline_id_features),
            # '-1' because timepoints in the metadata file are 1-indexed, but 0-indexed in the filesystem
            timepoint=str(int(cellprofiler_image_timepoints[i_i])-1),
            combined_per_well=combined_per_well,
            qc_df=qc_df,
        )
        
        if timeit:
            print("first two entries of final dataframe:")
            display(combined_per_well.head(2))

            print_time(f"timepoint {Path(cpi.image_plate_dir).name} done")

        return cpi


@dataclass
class Plate:
    timepoint:str
    image_plate_dir:tp.Union[str,Path]
    image_id:str
    pipeline_id_qc:tp.Optional[str]
    pipeline_id_features:str
    """ timepoint as in path name (i.e. 0-indexed) """
    
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
        for column in df.columns:
            column_is_null_check:bool=df.select(pl.col(column).is_null().any()).item(0,0)
            if column_is_null_check:
                raise RuntimeError(f"there is a nan value in column {column}")

        if method=="pca":
            n_components=2
            pca_red = PCA(n_components=n_components)
            reduced_data = pca_red.fit_transform(df)

            if print_feature_pca_fraction:
                # Get the loadings and calculate squared loadings for variance contribution
                loadings = pca_red.components_.T * np.sqrt(pca_red.explained_variance_)
                squared_loadings = pd.DataFrame(loadings**2, columns=[f'PC{i+1}' for i in range(n_components)], index=df.columns)

                # For each PC, get the top 5 contributing features
                top_contributors = {}
                for pc in squared_loadings.columns:
                    top_contributors[pc] = squared_loadings[pc].nlargest(5).sort_values(ascending=False)

                # Convert to DataFrame for better visualization
                top_contributors_df = pd.DataFrame(top_contributors)

                print(top_contributors_df)
        elif method=="umap":
            n_components=2
            umap_red = umap.UMAP(n_components=n_components)
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
            title=f'{method} of {Path(self.image_plate_dir).name}'
        )

        # remove margins
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
        ).update_traces(
            marker=dict(size=10)
        ).show()

def remove_highly_correlated(df, threshold=0.9, remove_inplace:bool=True)->tp.Union[tp.List[str],pl.DataFrame]:
    """
        remove_inplace:
            True : remove columns and return df
            False : return highly correlated column names
    """
    
    # Convert Polars DataFrame to NumPy for correlation calculation
    df_np = df.to_numpy()

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(df_np, rowvar=False)
    n_cols = corr_matrix.shape[0]

    # Identify columns to drop
    drop_indices = set()
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            if np.abs(corr_matrix[i, j]) > threshold:
                drop_indices.add(j)

    # Convert drop indices back to column names
    cols_to_drop = [df.columns[idx] for idx in drop_indices]

    if remove_inplace:
        # Drop columns from the original Polars DataFrame
        df_dropped = df.drop(cols_to_drop)

        return df_dropped
    else:
        return cols_to_drop
    
def handle_outliers(
    df:pl.DataFrame,
    columns:[str],
    *,
    level_method:str="sigma",
    lower_level:float=-1,
    upper_level:float=1,
    method:str="clip"
)->pl.DataFrame:
    """
    handle outliers with defined method
    
    with clip method:
        clip quantiles to provided levels in provided columns

    with remove method:
        remove outliers from dataset
    """

    valid_methods=["clip","remove"]

    method="clip"

    df_relevant_cols=df.select(columns)
    
    if level_method=="sigma":
        mean=df_relevant_cols.mean()
        sigma=df_relevant_cols.std()
        
        min_value=mean+lower_level*sigma
        max_value=mean+upper_level*sigma
    elif level_method=="quantile":
        min_value = df_relevant_cols.quantile(lower_level)
        max_value = df_relevant_cols.quantile(upper_level)
    else:
        raise ValueError(f"method '{level_method}' is invalid, must be [sigma|quantile]")
        
    # just make sure that the min is smaller than the max to avoid usage bugs
    assert (min_value>max_value).sum().to_numpy()[0,0]==0

    if method=="clip":
        for col in columns:
            df = df.with_columns(
                pl.col(col).clip(lower_bound=min_value[col],upper_bound=max_value[col])
            )

    elif method=="remove":
        for col in columns:
            df = df.filter(
                (pl.col(col) > min_value[col]) & (pl.col(col) < max_value[col])
            )

    else:
        raise RuntimeError(f"unknown method {method} (valid methods are {valid_methods})")
        
    return df

def remove_nans(df:pl.DataFrame,columns:[str])->pl.DataFrame:
    """ remove those rows that contain NaN in any of the provided columns """
    num_rows_before_nan_trim=df.shape[0]
    for col in df.select(columns).columns:
        df=df.filter(pl.col(col).is_not_null())
        
    return df

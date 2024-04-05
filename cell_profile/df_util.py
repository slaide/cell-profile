import typing as tp
import re
import pandas as pd
import polars as pl
import numpy as np

def is_meta_column(
    c:str,

    allowlist:tp.List[str]=["Metadata_Well","Metadata_Barcode","Metadata_AcqID","Metadata_Site"],
    denylist:tp.List[str]=[],
)->bool:
    """
        check if a column is a metadata column

        allowlist:
            the function will return False for these, no matter if they are metadata or not
        denylist:
            the function will return True for these, no matter if they are metadata or not

        note: this is code from Dan
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


def remove_highly_correlated(
    df:pl.DataFrame,
    threshold=0.9, 
    remove_inplace:bool=True
)->tp.Union[tp.List[str],pl.DataFrame]:
    """
        remove columns that are highly correlated with each other

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
    cols_to_drop:tp.List[str] = [df.columns[idx] for idx in drop_indices]

    if remove_inplace:
        # Drop columns from the original Polars DataFrame
        df_dropped = df.drop(cols_to_drop)

        return df_dropped
    else:
        return cols_to_drop
    
def handle_outliers(
    df:pl.DataFrame,
    columns:tp.List[str],
    *,
    level_method:tp.Literal["sigma","quantile"]="sigma",
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

def remove_nans(df:pl.DataFrame,columns:tp.List[str])->pl.DataFrame:
    """ remove those rows that contain NaN in any of the provided columns """
    num_rows_before_nan_trim=df.shape[0]
    for col in df.select(columns).columns:
        df=df.filter(pl.col(col).is_not_null())
        
    return df

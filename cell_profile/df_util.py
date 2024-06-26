import typing as tp
import re
import pandas as pd
import polars as pl
import numpy as np

float_columns=[pl.col(pl.Float32),pl.col(pl.Float64)]
"""
allows to select only columns that contain float values, e.g. df.select(float_columns)

useful because columns of other types are usually just metadata, e.g. indices [int], strings etc.
"""

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
    threshold:float=0.9,
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
    
class OutlierDetectionMethod:
    """ virtual class! """
    def __init__(self):
        pass
    def detect(self,df:pl.DataFrame,columns:tp.List[str])->tp.Tuple[pl.DataFrame,pl.DataFrame]:
        raise NotImplementedError

class Sigma(OutlierDetectionMethod):
    def __init__(self,
        lower_level:float=1,
        upper_level:tp.Optional[float]=None,
    ):
        """
        lower_level:
            sigma level for lower bound

        upper_level:
            if None, upper_level will be set to lower_level
            sigma level for upper bound
        """

        assert lower_level>0
        if upper_level is not None:
            assert upper_level>0

        self.lower_level=lower_level
        if upper_level is None:
            self.upper_level=lower_level
        else:
            self.upper_level=upper_level

    def detect(self,df:pl.DataFrame,columns:tp.List[str]) -> tp.Tuple[pl.DataFrame, pl.DataFrame]:
        selected_df=df.select(columns)
        mean=selected_df.mean()
        sigma=selected_df.std()

        min_value=mean-self.lower_level*sigma
        max_value=mean+self.upper_level*sigma

        return min_value,max_value
    
class Quantile(OutlierDetectionMethod):
    def __init__(self,
        lower_level:float=0.05,
        upper_level:float=0.95,
    ):
        """
        lower_level:
            quantile level for lower bound

        upper_level:
            quantile level for upper bound
        """

        assert 0<=lower_level<1
        assert 0<=upper_level<1

        self.lower_level=lower_level
        self.upper_level=upper_level

    def detect(self,df:pl.DataFrame,columns:tp.List[str]) -> tp.Tuple[pl.DataFrame, pl.DataFrame]:
        selected_df=df.select(columns)
        min_value=selected_df.quantile(self.lower_level)
        max_value=selected_df.quantile(self.upper_level)

        return min_value,max_value

def handle_outliers(
    df:pl.DataFrame,
    columns:tp.List[str],
    *,
    level_method:OutlierDetectionMethod=Sigma(),
    method:tp.Literal["clip","remove"]="clip",
)->pl.DataFrame:
    """
    handle outliers with defined method
    
    with clip method:
        clip quantiles to provided levels in provided columns

    with remove method:
        remove outliers from dataset
    """
    
    min_value,max_value=level_method.detect(df,columns)

    # just make sure that the min is smaller than the max to avoid usage bugs
    assert (min_value>max_value).sum().to_numpy()[0,0]==0

    if method=="clip":
        col_clip_exprs:tp.List[pl.Expr]=[]
        for col in columns:
            col_clip_exprs.append(
                pl.col(col).clip(lower_bound=min_value[col],upper_bound=max_value[col])
            )
        df = df.with_columns(col_clip_exprs)

    elif method=="remove":
        col_remove_exprs:tp.List[pl.Expr]=[]
        for col in columns:
            col_remove_exprs.append(
                (pl.col(col) > min_value[col]) & (pl.col(col) < max_value[col])
            )
        df = df.filter(col_remove_exprs)

    else:
        raise RuntimeError(f"unknown method {method} (valid methods are {handle_outliers.__annotations__['method'].__args__})")
        
    return df

def df_checkNull(df:pl.DataFrame,raise_:bool=False)->tp.List[str]:
    null_check=df.select(pl.col(df.columns).is_null().any())
    assert null_check.shape == (1,df.shape[1]), f"expected one row, got {null_check.shape[0]}"

    ret=[]
    for i,col in enumerate(df.columns):
        if null_check.item(row=0,column=col)==True:
            ret.append(col)

    if raise_ and len(ret)>0:
        for col in ret:
            print(f"some value in column {col} is null")
        raise RuntimeError("some value is null")
        
    return ret

def df_checkValue(df:pl.DataFrame,value:float,raise_:bool=False)->tp.List[str]:
    df=df.select(float_columns)

    value_check=df.select((pl.col(df.columns)==value).any())
    assert value_check.shape == (1,df.shape[1]), f"expected one row, got {value_check.shape[0]}"
    
    ret=[]
    for i,col in enumerate(df.columns):
        if value_check.item(row=0,column=col)==True:
            ret.append(col)

    if raise_ and len(ret)>0:
        for col in ret:
            print(f"some value in column {col} has target value ({value})")
        raise RuntimeError(f"some value is {value}")
        
    return ret

def df_checkInf(df:pl.DataFrame,raise_:bool=False)->tp.List[str]:
    df=df.select(float_columns)
    
    inf_check=df.select(pl.col(df.columns).is_infinite().any())
    assert inf_check.shape == (1,df.shape[1]), f"expected one row, got {inf_check.shape[0]}"
    
    ret=[]
    for i,col in enumerate(df.columns):
        if inf_check.item(row=0,column=col)==True:
            ret.append(col)

    if raise_ and len(ret)>0:
        for col in ret:
            print(f"some value in column {col} is inf")
        raise RuntimeError("some value is inf")
        
    return ret

def df_checkNaN(df:pl.DataFrame,raise_:bool=False)->tp.List[str]:
    df=df.select(float_columns)
    
    nan_check=df.select(pl.col(df.columns).is_nan().any())
    assert nan_check.shape == (1,df.shape[1]), f"expected one row, got {nan_check.shape[0]}"
    
    ret=[]
    for i,col in enumerate(df.columns):
        if nan_check.item(row=0,column=col)==True:
            ret.append(col)

    if raise_ and len(ret)>0:
        for col in ret:
            print(f"some value in column {col} is nan")
        raise RuntimeError("some value is nan")
        
    return ret

def remove_nans(df:pl.DataFrame,columns:tp.List[str])->pl.DataFrame:
    """ remove those rows that contain NaN in any of the provided columns """
    filter_exprs:tp.List[pl.Expr]=[]
    for col in df.select(columns).columns:
        filter_exprs.append(pl.col(col).is_not_null())

    df=df.filter(filter_exprs)
        
    return df

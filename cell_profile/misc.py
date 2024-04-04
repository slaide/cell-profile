import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# chatgpt's check to see if this code is running in a jupyter notebook or not
def is_notebook():
    try:
        # get_ipython not defined in standard python interpreter, i.e. call will throw
        shell = get_ipython().__class__.__name__ # type: ignore
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
    _test=raw_print # type: ignore
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

try:
    from IPython.display import display # type: ignore
except:
    def display(*args,**kwargs):
        print(*args,**kwargs)


# -- parallel map implementation
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

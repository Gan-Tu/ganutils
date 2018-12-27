"""concurrency
Core Concurrent Modules
"""

from tqdm import tqdm
from multiprocessing import Pool, Lock


###############################################################
# Quick, basic multi-processing tasks
# ===================================
#

def multiProcess(fn, iterable, num_threads, pbar=False, desc=None):
    iterable = tqdm(iterable, desc=desc) if pbar else iterable
    pool = Pool(num_threads)
    res = pool.map(fn, iterable)
    pool.close()
    pool.join() 
    return res


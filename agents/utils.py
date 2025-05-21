import numpy as np

def get_dkw_quantile_index(n_samples: int, alpha: float, delta: float) -> int:
    q = 1 - alpha + np.sqrt((1.0/(2*n_samples))*np.log(2.0/delta))
    index = int(np.ceil(n_samples * q) - 1)
    if index < 0:
        return 0
    elif index >= n_samples:
        return n_samples - 1
    return index

def get_conformal_quantile_index(n_samples: int, alpha: float) -> int:
    q = 1 - alpha
    index = int(np.ceil(n_samples * q) - 1)
    if index < 0:
        return 0
    elif index >= n_samples:
        return n_samples - 1
    return index

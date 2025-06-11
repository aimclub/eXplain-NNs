import torch
import numpy as np


def integral_diff(scales1, values1, scales2, values2):
    
    check1 = (scales1.shape == scales2.shape) and torch.all(scales1 == scales2)
    err_scales = "Scales must be the same for diagrams to be comparable"
    assert check1, err_scales
    
    err_shapes = f"""Both values arrays and scales arrays must be of the same size, got:
                     \n{scales1.shape}, {scales2.shape}, {values1.shape}, {values2.shape}"""

    assert (values1.shape == values2.shape) and (values1.shape == scales1.shape), err_shapes
    
    vals_diff = np.array(values1) - np.array(values2)
    
    res_val = 0
    for i in range(len(vals_diff)-1):
        res_val += np.abs(vals_diff[i])*(scales1[i+1] - scales1[i])
    
    return res_val/(scales1[-1] - scales2[0])
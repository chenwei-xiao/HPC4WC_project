import gt4py as gt
from gt4py import gtscript
import numpy as np

########################### USEFUL FUNCTIONS ###########################
# These should be moved in a separate file to avoid cluttering and be 
# reused in other places!
########################################################################

@gtscript.function
def sqrt(x):
    return x**0.5


@gtscript.function
def exp(x):
    return np.e**x
        

@gtscript.function
def min(x, y):
    if x <= y:
        return x
    else:
        return y
        

@gtscript.function
def max(x, y):
    if x >= y:
        return x
    else:
        return y
        

@gtscript.function
def log(x, a):
	return a * (x**(1.0/a)) - a
        

def slice_to_3d(slice):
    return slice[np.newaxis, :, :]
    
    
def exit_routine(cnvflg, im):
    totflg = True
    for i in range(0, im):
        totflg = totflg and cnvflg[0, i, 0] == 0
        
    return totflg
    
########################################################################
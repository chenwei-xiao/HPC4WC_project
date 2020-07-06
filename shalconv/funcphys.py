import numpy as np
from shalcnv.physcons import (
    con_ttp,
    con_psat,
    con_xponal,
    con_xponbl,
    con_xponai,
    con_xponbi
)


### Global variables used in fpvs ###
c1xpvs = None
c2xpvs = None

### Look-up table for saturation vapor pressure ###
nxpvs = 7501                  # Size of look-up table
tbpvs = np.empty(nxpvs)       # Look-up table stored as a 1D numpy array


# Computes saturation vapor pressure table as a function of temperature 
# for the table lookup function fpvs.
def gpvs():
    global c1xpvs
    global c2xpvs
    
    xmin   = 180.0
    xmax   = 330.0
    xinc   = (xmax - xmin)/(nxpvs - 1)
    c2xpvs = 1./xinc
    c1xpvs = 1. - xmin * c2xpvs
    
    for jx in range(0, nxpvs):
        x = xmin + jx * xinc
        tbpvs[jx] = fpvsx(x)
        
    
# Compute saturation vapor pressure from the temperature. A linear 
# interpolation is done between values in a lookup table computed in 
# gpvs.
def fpvs(t):
    xj = min(max(c1xpvs + c2xpvs * t, 1.), nxpvs)
    jx = min(xj, nxpvs - 1)
    
    return tbpvs[jx] + (xj - jx) * (tbpvs[jx+1] - tbpvs[jx])
    

# Compute exact saturation vapor pressure from temperature.
def fpvsx(t):
    tr = con_ttp/t
    tliq = con_ttp
    tice = con_ttp - 20.
    
    if t >= tliq:
        return con_psat * (tr**xponal) * np.exp(xponbl * (1. - tr))
    elif t < tice:
        return con_psat * (tr**xponai) * np.exp(xponbi * (1. - tr))
    else:
        w = (t - tice)/(tliq - tice)
        pvl = con_psat * (tr**xponal) * np.exp(xponbl * (1. - tr))
        pvi = con_psat * (tr**xponai) * np.exp(xponbi * (1. - tr))
        
        return w * pvl + (1. - w) * pvi

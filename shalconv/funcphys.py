import numpy as np
from shalconv.physcons import (
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
# for the table lookup function fpvs
def gpvs():
    global c1xpvs
    global c2xpvs
    
    xmin   = 180.0
    xmax   = 330.0
    xinc   = (xmax - xmin)/(nxpvs - 1)
    c2xpvs = 1./xinc
    c1xpvs = -xmin * c2xpvs
    
    for jx in range(0, nxpvs):
        x = xmin + jx * xinc
        tbpvs[jx] = fpvsx(x)


# Compute exact saturation vapor pressure from temperature
def fpvsx(t):
    tr = con_ttp/t
    tliq = con_ttp
    tice = con_ttp - 20.
    
    if t >= tliq:
        return con_psat * (tr**con_xponal) * np.exp(con_xponbl * (1. - tr))
    elif t < tice:
        return con_psat * (tr**con_xponai) * np.exp(con_xponbi * (1. - tr))
    else:
        w = (t - tice)/(tliq - tice)
        pvl = con_psat * (tr**con_xponal) * np.exp(con_xponbl * (1. - tr))
        pvi = con_psat * (tr**con_xponai) * np.exp(con_xponbi * (1. - tr))
        
        return w * pvl + (1. - w) * pvi


# Compute saturation vapor pressure from the temperature. A linear 
# interpolation is done between values in a lookup table computed in 
# gpvs.
def fpvs(t):
    xj = min(max(c1xpvs + c2xpvs * t, 0.), nxpvs - 1)
    jx = int(min(xj, nxpvs - 2))
    
    return tbpvs[jx] + (xj - jx) * (tbpvs[jx+1] - tbpvs[jx])


# Function fpvs as gtscript.function, to be used in stencils
# ~ @gtscript.function
# ~ def fpvs_gtfunc(t):
    # ~ xj = min(max(c1xpvs + c2xpvs * t, 0.), nxpvs - 1)
    # ~ jx = int(min(xj, nxpvs - 2))
    
    # ~ return tbpvs[jx] + (xj - jx) * (tbpvs[jx+1] - tbpvs[jx])

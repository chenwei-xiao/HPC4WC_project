import numpy as np
from gt4py import gtscript

BACKEND     = "gtx86"
REBUILD     = True

DTYPE_INT   = np.int32
DTYPE_FLOAT = np.float64
INT_FIELD   = gtscript.Field[DTYPE_INT]
FLT_FIELD   = gtscript.Field[DTYPE_FLT]

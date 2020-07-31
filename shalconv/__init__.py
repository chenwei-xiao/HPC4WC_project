import numpy as np
from gt4py import gtscript

DATAPATH    = "/data"
BACKEND     = "numpy"#"gtx86"
REBUILD     = True
BACKEND_OPTS = {'verbose': True} if BACKEND.startswith('gt') else {}
default_origin = (0, 0, 0)

DTYPE_INT   = np.int32
DTYPE_FLOAT = np.float64

FIELD_INT   = gtscript.Field[DTYPE_INT]
FIELD_FLOAT = gtscript.Field[DTYPE_FLOAT]

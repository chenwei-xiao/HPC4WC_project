import numpy as np
from gt4py import gtscript

ISDOCKER       = True
if ISDOCKER:
    DATAPATH   = "/data"
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    DATAPATH       = "/scratch/snx3000/course20/physics_standalone/shalconv/data"
    SERIALBOX_DIR  = "/project/c14/install/daint/serialbox2_master/gnu_debug"
BACKEND        = "numpy"#"numpy"#"gtx86"#debug
REBUILD        = False
BACKEND_OPTS   = {}#{'verbose': True} if BACKEND.startswith('gt') else {}
default_origin = (0, 0, 0)

DTYPE_INT   = np.int32
DTYPE_FLOAT = np.float64

FIELD_INT   = gtscript.Field[DTYPE_INT]
FIELD_FLOAT = gtscript.Field[DTYPE_FLOAT]

def change_backend(backend):
    global BACKEND
    BACKEND = backend
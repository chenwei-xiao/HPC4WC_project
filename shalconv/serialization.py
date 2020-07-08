import numpy as np
import gt4py as gt
from . import BACKEND

#SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"
#sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

int_vars = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud"]

IN_VARS = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud", \
        "clam", "c0s", "c1", "asolfac", "pgcon", "delt", "islimsk", \
           "psp", "delp", "prslp", "garea", "hpbl", "dot", \
           "phil", "fscav", \
           "kcnv", "kbot", "ktop", "qtr", "q1", "t1", "u1", \
           "v1", "rn", "cnvw", "cnvc", "ud_mf", "dt_mf"]
OUT_VARS = ["kcnv", "kbot", "ktop", "qtr", "q1", "t1", "u1", \
            "v1", "rn", "cnvw", "cnvc", "ud_mf", "dt_mf"]


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        d[var] = serializer.read(var, savepoint)
    return d
    
def numpy_dict_to_gt4py_dict(data_dict, backend = BACKEND):
    """
    Transform dict of numpy arrays into dict of gt4py storages
    1d array of shape (nx) will be transformed into storage of shape (1, nx, 1)
    2d array of shape (nx, nz) will be transformed into storage of shape (1, nx, nz)
    3d array is kept the same (numpy arrays), doing slices later
    0d array will be transformed into a scalar
    """
    ix = data_dict["ix"]#im <= ix
    km = data_dict["km"]
    for var in data_dict:
        data = data_dict[var]
        ndim = len(data.shape)
        #if var == "fscav":
        #    data_dict["fscav"] = data # shape = (number of tracers)
        if (ndim > 0) and (ndim <= 2):
            default_origin = (0, 0, 0)
            arrdata = np.zeros((1,ix,km))
            if ndim == 1: # 1D array (horizontal dimension)
                arrdata[...] = data[np.newaxis, :, np.newaxis]
            elif ndim == 2: #2D array (horizontal dimension, vertical dimension)
                arrdata[...] = data[np.newaxis, :, :]
            data_dict[var] = gt.storage.from_array(arrdata, backend, default_origin)
        #elif ndim == 3: #3D array qntr(horizontal dimension, vertical dimension, number of tracers)
        #    data_dict[var] = data
        elif var in int_vars:
            data_dict[var] = int(data)
        else:
            data_dict[var] = float(data)
    
def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(ref_data.keys()), \
        "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), \
            "Data from exp and ref does not match for field " + key

def read_data(tile, is_in):
    """
    Read serialbox2 format data under `./data` folder with prefix of `Generator_rank{tile}`
    :param tile: specify the number of tile in data
    :type tile: int
    :param is_in: true means in, false means out
    :type is_in: boolean
    """
    #TODO: read_async and readbuffer
    serializer = ser.Serializer(ser.OpenModeKind.Read, "./data", "Generator_rank" + str(tile))
    inoutstr = "in" if is_in else "out"
    sp = [sp for sp in serializer.savepoint_list() if sp.name.startswith("samfshalcnv-"+inoutstr)][0]
    vars = IN_VARS if is_in else OUT_VARS
    data = data_dict_from_var_list(vars, serializer, sp)
    return data
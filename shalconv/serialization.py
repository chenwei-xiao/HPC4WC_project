import numpy as np
import gt4py as gt

#SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"
#sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

IN_VARS = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud", \ #input
           "clam", "c0s", "c1", "asolfac", "pgcon", "delt", "islimsk", \
           "psp", "delp", "prslp", "garea", "hpbl", "dot", \
           "phil", "fscav", \
           "kcnv", "kbot", "ktop", "qtr", "q1", "t1", "u1", \ #inout
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
    0d array will be transformed into a scalar
    """
    for var in data_dict:
        data = data_dict[var]
        ndim = len(data.shape)
        if ndim > 0:
            default_origin = (0, 0, 0)
            if ndim == 1: # 1D array (horizontal dimension)
                data = data[np.newaxis, :, np.newaxis]
            else if ndim == 2: #2D array (horizontal dimension, vertical dimension)
                data = data[np.newaxis, :, :]
            data_dict[var] = gt.storage.from_array(data, backend, default_origin)
        else:
            data_dict[var] = float(data)
    
def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(ref_data.keys()), \
        "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), \
            "Data from exp and ref does not match for field " + key

def read_data(tile, inout):
    """
    Read serialbox2 format data under `./data` folder with prefix of `Generator_rank{tile}`
    :param tile: specify the number of tile in data
    :type tile: int
    :param inout: either "in" or "out" meaning input or output data
    :type inout: string
    """
    #TODO: read_async and readbuffer
    serializer = ser.Serializer(ser.OpenModeKind.Read, "./data", "Generator_rank" + str(tile))
    sp = [sp for sp in serializer.savepoint_list() if sp.name.startswith("samfshalcnv-"+inout)]
    data = data_dict_from_var_list(IN_VARS, serializer, sp)
    return data
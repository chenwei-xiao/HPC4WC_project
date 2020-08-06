import numpy as np
import gt4py as gt
from . import BACKEND, DTYPE_FLOAT, DTYPE_INT, SERIALBOX_DIR
from copy import deepcopy

import sys
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

int_vars = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud"]
int_arrs = ['islimsk', 'kcnv', 'kbot', 'ktop', 'kpbl', 'kb', 'kbcon',
            'kbcon1', 'ktcon', 'ktcon1', 'ktconn', 'kbm', 'kmax',
            'cnvflg', 'flg']

IN_VARS = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud",
           "clam", "c0s", "c1", "asolfac", "pgcon", "delt", "islimsk",
           "psp", "delp", "prslp", "garea", "hpbl", "dot",
           "phil", "fscav",
           "kcnv", "kbot", "ktop", "qtr", "q1", "t1", "u1",
           "v1", "rn", "cnvw", "cnvc", "ud_mf", "dt_mf"]
OUT_VARS = ["kcnv", "kbot", "ktop", "qtr", "q1", "t1", "u1",
            "v1", "rn", "cnvw", "cnvc", "ud_mf", "dt_mf"]

def clean_numpy_dict(data_dict):
    for var in data_dict:
        if var in int_vars:
            data_dict[var] = DTYPE_INT(data_dict[var][0])
        elif data_dict[var].size <= 1:
            data_dict[var] = DTYPE_FLOAT(data_dict[var][0])

def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        d[var] = serializer.read(var, savepoint)
    clean_numpy_dict(d)
    return d
    
def numpy_dict_to_gt4py_dict(data_dict, backend = BACKEND):
    """
    Transform dict of numpy arrays into dict of gt4py storages, return new dict
    1d array of shape (nx) will be transformed into storage of shape (1, nx, nz)
    2d array of shape (nx, nz) will be transformed into storage of shape (1, nx, nz)
    3d array is kept the same (numpy arrays), doing slices later
    0d array will be transformed into a scalar
    """
    ix = int(data_dict["ix"])#im <= ix
    km = int(data_dict["km"])
    new_data_dict = {}
    for var in data_dict:
        data = data_dict[var]
        ndim = len(data.shape)
        #if var == "fscav":
        #    data_dict["fscav"] = data # shape = (number of tracers)
        if (ndim > 0) and (ndim <= 2) and (data.size >= 2):
            default_origin = (0, 0, 0)
            arrdata = np.zeros((1,ix,km))
            if ndim == 1: # 1D array (horizontal dimension)
                arrdata[...] = data[np.newaxis, :, np.newaxis]
            elif ndim == 2: #2D array (horizontal dimension, vertical dimension)
                arrdata[...] = data[np.newaxis, :, :]
            dtype = DTYPE_INT if var in int_arrs else DTYPE_FLOAT
            new_data_dict[var] = gt.storage.from_array(arrdata, backend, default_origin, dtype = dtype)
        elif ndim == 3: #3D array qntr(horizontal dimension, vertical dimension, number of tracers)
            new_data_dict[var] = deepcopy(data)
        else: # scalars
            new_data_dict[var] = deepcopy(data)
    return new_data_dict
    
def compare_data(exp_data, ref_data):
    wrong = []
    flag = True
    for key in exp_data:
        mask = ~np.isnan(ref_data[key])
        if not np.allclose(exp_data[key][mask], ref_data[key][mask]):
            wrong.append(key)
            flag = False
        else:
            print(f"Succefully validate {key}!")
    assert flag, f"Data from exp and ref does not match for field {wrong}"

def read_data(tile, is_in, path = "./data"):
    """
    Read serialbox2 format data under `./data` folder with prefix of `Generator_rank{tile}`
    :param tile: specify the number of tile in data
    :type tile: int
    :param is_in: true means in, false means out
    :type is_in: boolean
    """
    #TODO: read_async and readbuffer
    #TODO: multiple savepoints
    serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Generator_rank" + str(tile))
    inoutstr = "in" if is_in else "out"
    sp = [sp for sp in serializer.savepoint_list() if sp.name.startswith("samfshalcnv-"+inoutstr)][0]
    vars = IN_VARS if is_in else OUT_VARS
    data = data_dict_from_var_list(vars, serializer, sp)
    return data

def view_gt4pystorage(data_dict):
    new_data_dict = {}
    for key in data_dict:
        data = data_dict[key]
        new_data_dict[key] = data.view(np.ndarray)
    return new_data_dict
import pytest
import gt4py as gt
from gt4py import gtscript
#import sys
#sys.path.append("..")
from tests.read_serialization import read_serialization_part3, read_serialization_part4
from shalconv.kernels.stencils_part34 import *
from shalconv.serialization import read_data, compare_data, OUT_VARS, numpy_dict_to_gt4py_dict
from shalconv import *
from shalconv.physcons import (
    con_g     as grav,
    con_cp    as cp,
    con_hvap  as hvap,
    con_rv    as rv,
    con_fvirt as fv,
    con_t0c   as t0c,
    con_rd    as rd,
    con_cvap  as cvap,
    con_cliq  as cliq,
    con_eps   as eps,
    con_epsm1 as epsm1,
    con_e     as e
)

def samfshalcnv_part3(input_dict, data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection
    
    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """
    ix = input_dict["ix"]
    km = input_dict["km"]
    shape = (1, ix, km)
    g = grav
    betaw = 0.03
    dtmin = 600.0
    dt2 = input_dict["delt"]
    dtmax = 10800.0
    dxcrt = 15.0e3
    cnvflg = data_dict["cnvflg"]
    k_idx = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    kmax = data_dict["kmax"]
    kb = data_dict["kb"]
    ktcon = data_dict["ktcon"]
    ktcon1 = data_dict["ktcon1"]
    kbcon1 = data_dict["kbcon1"]
    kbcon = data_dict["kbcon"]
    dellah = data_dict["dellah"]
    dellaq = data_dict["dellaq"]
    dellau = data_dict["dellau"]
    dellav = data_dict["dellav"]
    del0 = data_dict["del"]
    zi = data_dict["zi"]
    zi_ktcon1 = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_kbcon1 = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heo = data_dict["heo"]
    qo = data_dict["qo"]
    xlamue = data_dict["xlamue"]
    xlamud = data_dict["xlamud"]
    eta = data_dict["eta"]
    hcko = data_dict["hcko"]
    qrcko = data_dict["qrcko"]
    uo = data_dict["uo"]
    ucko = data_dict["ucko"]
    vo = data_dict["vo"]
    vcko = data_dict["vcko"]
    qcko = data_dict["qcko"]
    dellal = data_dict["dellal"]
    qlko_ktcon = data_dict["qlko_ktcon"]
    wc = data_dict["wc"]
    gdx = data_dict["gdx"]
    dtconv = data_dict["dtconv"]
    u1 = data_dict["u1"]
    v1 = data_dict["v1"]
    po = data_dict["po"]
    to = data_dict["to"]
    tauadv = data_dict["tauadv"]
    xmb = data_dict["xmb"]
    sigmagfm = data_dict["sigmagfm"]
    garea = data_dict["garea"]
    scaldfunc = data_dict["scaldfunc"]
    xmbmax = data_dict["xmbmax"]
    sumx = data_dict["sumx"]
    umean = data_dict["umean"]
    
    # Calculate the tendencies of the state variables (per unit cloud base 
    # mass flux) and the cloud base mass flux
    comp_tendencies( g, betaw, dtmin, dt2, dtmax, dxcrt, cnvflg, k_idx,
                     kmax, kb, ktcon, ktcon1, kbcon1, kbcon, dellah,
                     dellaq, dellau, dellav, del0, zi, zi_ktcon1,
                     zi_kbcon1, heo, qo, xlamue, xlamud, eta, hcko,
                     qrcko, uo, ucko, vo, vcko, qcko, dellal, 
                     qlko_ktcon, wc, gdx, dtconv, u1, v1, po, to, 
                     tauadv, xmb, sigmagfm, garea, scaldfunc, xmbmax,
                     sumx, umean)
                     
    return dellah, dellaq, dellau, dellav, dellal

def test_part3():
    input_dict = read_data(0, True, path = DATAPATH)
    data_dict = read_serialization_part3()
    out_dict = read_serialization_part4()
    
    dellah, dellaq, dellau, dellav, dellal = samfshalcnv_part3(input_dict, data_dict)
    
    compare_data({"dellah":dellah.view(np.ndarray),"dellaq":dellaq.view(np.ndarray),
                  "dellau":dellau.view(np.ndarray),"dellav":dellav.view(np.ndarray),
                  "dellal":dellal.view(np.ndarray)},
                 {"dellah":out_dict["dellah"].view(np.ndarray),"dellaq":out_dict["dellaq"].view(np.ndarray),
                  "dellau":out_dict["dellau"].view(np.ndarray),"dellav":out_dict["dellav"].view(np.ndarray),
                  "dellal":out_dict["dellal"].view(np.ndarray)})

if __name__ == "__main__":
    test_part3()

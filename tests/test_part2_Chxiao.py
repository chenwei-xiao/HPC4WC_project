import gt4py as gt
#import sys
#sys.path.append("..")
from tests.read_serialization import *
from shalconv.kernels.utils import get_1D_from_index, exit_routine
from shalconv.kernels.stencils_part2 import *
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

def samfshalcnv_part2(ix,km,clam,pgcon,delt,c1,
                      kpbl,kb,kbcon,kbcon1,ktcon,ktcon1,kbm,kmax,
                      po,to,qo,uo,vo,qeso,dbyo,zo,
                      heo,heso,hcko,ucko,vcko,qcko,
                      aa1,cina,clamt,del0,wu2,buo,drag,wc,
                      pdot,hmax,xlamue,xlamud,pfld,
                      eta,zi,c0t,sumx,cnvflg,flg,islimsk,dot,
                      k_idx,heo_kb,dot_kbcon,pfld_kbcon,pfld_kb,pfld_kbcon1,
                      cnvwt,dellal,ktconn,pwo,qlko_ktcon,qrcko,xmbmax):

    ### Search in the PBL for the level of maximum moist static energy to start the ascending parcel.
    stencil_static0(cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, qeso, qo, po, uo, vo, heso, pfld)
    
    ### Search below the index "kbm" for the level of free convection (LFC) where the condition.
    get_1D_from_index(heo, heo_kb, kb, k_idx)
    stencil_static1(cnvflg, flg, kbcon, kmax, k_idx, kbm, kb, heo_kb, heso)

    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, ix): return

    ### Determine the vertical pressure velocity at the LFC.
    get_1D_from_index(dot, dot_kbcon, kbcon, k_idx)
    get_1D_from_index(pfld, pfld_kbcon, kbcon, k_idx)
    get_1D_from_index(pfld, pfld_kb, kb, k_idx)
    stencil_static2(cnvflg, pdot, dot_kbcon, islimsk, k_idx, kbcon, kb, pfld_kb, pfld_kbcon)
    
    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, ix): return

    ### turbulent entrainment rate assumed to be proportional to subcloud mean TKE
    #if(ntk > 0):
    #    qtr_ntr = gt.storage.from_array(slice_to_3d(qtr[:, :, ntr]), BACKEND, default_origin)
    #    stencil_static3(sumx, tkemean, cnvflg, k_idx, kb, kbcon, zo, qtr_ntr,
    #                     clamt, clam, dtke)
    #    qtr[:,:,ntr] = qtr_ntr.view(np.ndarray)[0,:,:]
    #else:
    stencil_static4(cnvflg, clamt, clam )
    
    ### assume updraft entrainment rate is an inverse function of height
    stencil_static5(cnvflg, xlamue, clamt, zi, xlamud, k_idx, kbcon, kb,
                     eta, ktconn, kmax, kbm, hcko, ucko, vcko, heo, uo, vo)

    stencil_static7(cnvflg, k_idx, kb, kmax, zi, xlamue, xlamud, hcko, heo, dbyo,
                     heso, pgcon, ucko, uo, vcko, vo)

    stencil_update_kbcon1_cnvflg(dbyo, cnvflg, kmax, kbm, kbcon, kbcon1, flg, k_idx)
    get_1D_from_index(pfld, pfld_kbcon1, kbcon1, k_idx)
    stencil_static9(cnvflg, pfld_kbcon, pfld_kbcon1)
    
    if exit_routine(cnvflg, ix): return

    ### calculate convective inhibition
    stencil_static10(cina, cnvflg, k_idx, kb, kbcon1, zo, el2orc, qeso, to,
                      dbyo, qo, pdot, islimsk)
    
    if exit_routine(cnvflg, ix): return

    dt2 = delt
    stencil_static11(flg, cnvflg, ktcon, kbm, kbcon1, dbyo, kbcon, del0, xmbmax,
                     dt2, aa1, kb, qcko, qo, qrcko, zi, el2orc, qeso, to, xlamue, 
                     xlamud, eta, c0t, c1, dellal, buo, drag, zo, k_idx, pwo,
                     cnvwt)
    
    if exit_routine(cnvflg, ix): return

    stencil_static12(cnvflg, aa1, flg, ktcon1, kbm, k_idx, ktcon, zo, qeso, 
                     to, dbyo, zi, xlamue, xlamud, qcko, qrcko, qo, eta, del0,
                     c0t, c1, pwo, cnvwt, buo, wu2, wc, sumx, kbcon1, drag, dellal)

    
    #if(ncloud > 0):
    stencil_static13(cnvflg, k_idx, ktcon, qeso, to, dbyo, qcko, qlko_ktcon)
    # else:
    #     stencil_static14(cnvflg, vshear, k_idx, kb, ktcon, uo, vo, zi, edt)

def apply_arguments_part2(input_dict, data_dict):
    clam = input_dict['clam']
    pgcon = input_dict['pgcon']
    delt = input_dict['delt']
    c1 = input_dict['c1']
    ix = data_dict['ix']
    km = data_dict['km']
    islimsk = data_dict['islimsk']
    dot = data_dict['dot']
    qtr = data_dict['qtr']
    kpbl = data_dict['kpbl']
    kb = data_dict['kb']
    kbcon = data_dict['kbcon']
    kbcon1 = data_dict['kbcon1']
    ktcon = data_dict['ktcon']
    ktcon1 = data_dict['ktcon1']
    kbm = data_dict['kbm']
    kmax = data_dict['kmax']
    aa1 = data_dict['aa1']
    cina = data_dict['cina']
    tkemean = data_dict['tkemean']
    clamt = data_dict['clamt']
    del0 = data_dict['del']
    edt = data_dict['edt']
    pdot = data_dict['pdot']
    po = data_dict['po']
    hmax = data_dict['hmax']
    vshear = data_dict['vshear']
    xlamud = data_dict['xlamud']
    pfld = data_dict['pfld']
    to = data_dict['to']
    qo = data_dict['qo']
    uo = data_dict['uo']
    vo = data_dict['vo']
    qeso = data_dict['qeso']
    ctro = data_dict['ctro']
    wu2 = data_dict['wu2']
    buo = data_dict['buo']
    drag = data_dict['drag']
    wc = data_dict['wc']
    dbyo = data_dict['dbyo']
    zo = data_dict['zo']
    xlamue = data_dict['xlamue']
    heo = data_dict['heo']
    heso = data_dict['heso']
    hcko = data_dict['hcko']
    ucko = data_dict['ucko']
    vcko = data_dict['vcko']
    qcko = data_dict['qcko']
    ecko = data_dict['ecko']
    eta = data_dict['eta']
    zi = data_dict['zi']
    c0t = data_dict['c0t']
    sumx = data_dict['sumx']
    cnvflg = data_dict['cnvflg']
    flg = data_dict['flg']
    cnvwt = data_dict['cnvwt']
    dellal = data_dict['dellal']
    ktconn = data_dict['ktconn']
    pwo = data_dict['pwo']
    qlko_ktcon = data_dict['qlko_ktcon']
    qrcko = data_dict['qrcko']
    xmbmax = data_dict['xmbmax']
    shape = (1, ix, km)
    k_idx = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    heo_kb = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dot_kbcon = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kb = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon1 = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    samfshalcnv_part2(ix, km, clam, pgcon, delt, c1,
                      kpbl, kb, kbcon, kbcon1, ktcon, ktcon1, kbm, kmax,
                      po, to, qo, uo, vo, qeso, dbyo, zo,
                      heo, heso, hcko, ucko, vcko, qcko,
                      aa1, cina, clamt, del0, wu2, buo, drag, wc,
                      pdot, hmax, xlamue, xlamud, pfld,
                      eta, zi, c0t, sumx, cnvflg, flg, islimsk, dot,
                      k_idx, heo_kb, dot_kbcon, pfld_kbcon, pfld_kb, pfld_kbcon1,
                      cnvwt, dellal, ktconn, pwo, qlko_ktcon, qrcko, xmbmax)
    return cnvwt, dellal, pwo, qlko_ktcon, qrcko, xmbmax

def view_gt4pystorage(data_dict):
    new_data_dict = {}
    for key in data_dict:
        data = data_dict[key]
        new_data_dict[key] = data.view(np.ndarray)
    return new_data_dict

def test_part2():
    input_dict = read_data(0, True, path = DATAPATH)
    data_dict = read_serialization_part2()
    out_dict_p3 = read_serialization_part3()
    out_dict_p4 = read_serialization_part4()
    cnvwt, dellal, pwo, qlko_ktcon, qrcko, xmbmax = apply_arguments_part2(input_dict, data_dict)
    exp_data = view_gt4pystorage({"cnvwt":cnvwt, "dellal":dellal, "pwo":pwo,
                "qlko_ktcon":qlko_ktcon, "qrcko":qrcko, "xmbmax":xmbmax})
    ref_data = view_gt4pystorage({"cnvwt":out_dict_p4["cnvwt"], "dellal":out_dict_p3["dellal"],
                "pwo": out_dict_p4["pwo"], "qlko_ktcon": out_dict_p3["qlko_ktcon"],
                "qrcko":out_dict_p3["qrcko"],"xmbmax":out_dict_p3["xmbmax"]})
    compare_data(exp_data, ref_data)

if __name__ == "__main__":
    test_part2()
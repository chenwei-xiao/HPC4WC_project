import pytest
import gt4py as gt
from gt4py import gtscript
#import sys
#sys.path.append("..")
from shalconv.kernels.stencils_part1 import *
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

def samfshalcnv_part2(data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection
    
    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """

############################################Static control###############################################
    ### Search in the PBL for the level of maximum moist static energy to start the ascending parcel.
    stencil_static0( cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, qeso, qo, po, uo, vo, heso, pfld)
    
    ### Search below the index "kbm" for the level of free convection (LFC) where the condition.
    stencil_static1( cnvflg, flg, kbcon, kmax, k_idx, kbm, kb, heo, heso)

    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, im): return

    ### Determine the vertical pressure velocity at the LFC.
    stencil_static2( cnvflg, pdot, dot, islimsk, k_idx, kbcon, kb, pfld)
    
    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, im): return

    ### turbulent entrainment rate assumed to be proportional to subcloud mean TKE
    if(ntk > 0): 
        qtr_ntr = gt.storage.from_array(slice_to_3d(qtr[:, :, ntr]), BACKEND, default_origin)
        stencil_static3( sumx, tkemean, cnvflg, k_idx, kb, kbcon, zo, qtr_ntr,
                         clamt, clam, dtke)
        qtr[:,:,ntr] = qtr_ntr.view(np.ndarray)[0,:,:]
    else:
        stencil_static4( cnvflg, clamt, clam
                         )
    
    ### assume updraft entrainment rate is an inverse function of height
    stencil_static5( cnvflg, xlamue, clamt, zi, xlamud, k_idx, kbcon, kb,
                     eta, ktconn, kmax, kbm, hcko, ucko, vcko, heo, uo, vo)

    stencil_static7( cnvflg, k_idx, kb, kmax, zi, xlamue, xlamud, hcko, heo, dbyo,
                     heso, pgcon, ucko, uo, vcko, vo)

    stencil_static9( cnvflg, flg, kbcon1, kmax, k_idx, kbm, kbcon, dbyo,
                     pfld)
    
    if exit_routine(cnvflg, im): return

    ### calculate convective inhibition
    stencil_static10( cina, cnvflg, k_idx, kb, kbcon1, zo, el2orc, qeso, to,
                      dbyo, qo, pdot, islimsk)
    
    if exit_routine(cnvflg, im): return

    stencil_static11( flg, cnvflg, ktcon, kbm, kbcon1, dbyo, kbcon, del0, xmbmax,
                     delt, aa1, kb, qcko, qo, qrcko, zi, el2orc, qeso, to, xlamue, 
                     xlamud, eta, c0t, c1, dellal, buo, drag, zo, k_idx, pwo,
                     cnvwt)
    
    if exit_routine(cnvflg, im): return

    stencil_static12( cnvflg, aa1, flg, ktcon1, kbm, k_idx, ktcon, zo, qeso, 
                     to, dbyo, zi, xlamue, xlamud, qcko, qrcko, qo, eta, del0,
                     c0t, c1, pwo, cnvwt, buo, wu2, wc, sumx, kbcon1, drag, dellal)

    
    if(ncloud > 0):
        stencil_static13( cnvflg, k_idx, ktcon, qeso, to, dbyo, qcko, qlko_ktcon)
    else:
        stencil_static14( cnvflg, vshear, k_idx, kb, ktcon, uo, vo, zi, edt)
    




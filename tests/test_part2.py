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
    stencil_static0( cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, fact1,
                     fact2, el2orc, qeso, qo, po, uo, vo, heso, pfld)
    
    ### Search below the index "kbm" for the level of free convection (LFC) where the condition.
    stencil_static1( cnvflg, flg, kbcon, kmax, k_idx, kbm, kb, heo, heso)

    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, im): return

    ### Determine the vertical pressure velocity at the LFC.
    stencil_static2( cnvflg, pdot, dot, islimsk, k_idx, kbcon, kb, w1l, w2l, 
                     w3l, w4l, w1s, w2s, w3s, w4s, cinpcrmx, cinpcrmn, pfld)
    
    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, im): return

    ### turbulent entrainment rate assumed to be proportional to subcloud mean TKE
    if(ntk > 0): 
        qtr_ntr = gt.storage.from_array(slice_to_3d(qtr[:, :, ntr]), BACKEND, default_origin)
        stencil_static3( sumx, tkemean, cnvflg, k_idx, kb, kbcon, zo, qtr_ntr,
                         tkemn, tkemx, clamt, clam, clamd, dtke)
        qtr[:,:,ntr] = qtr_ntr ## is this permitted?
    else:
        stencil_static4( cnvflg, clamt, clam
                         )
    
    ### assume updraft entrainment rate is an inverse function of height
    stencil_static5( cnvflg, xlamue, clamt, zi, xlamud, k_idx, kbcon, kb,
                     eta, ktconn, kmax, kbm, hcko, ucko, vcko, heo, uo, vo)
    




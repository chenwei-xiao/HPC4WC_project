from .physcons import con_g as g
from . import DTYPE_FLOAT, DTYPE_INT, BACKEND, INT_FIELD, FLT_FIELD
import gt4py as gt
from gt4py import gtscript
from __gtscript__ import PARALLEL, FORWARD, BACKWARD, computation, interval
import numpy as np

@gtscript.stencil(backend = BACKEND)
def calc_dellah_q_u_v(
    delp: gtscript.Field[DTYPE_FLOAT],
    ktcon: gtscript.Field[int],
    index_k: gtscript.Field[int],
    dtime_max_arr: gtscript.Field[DTYPE_FLOAT],
    *,
    delt: float
):
    with computation(FORWARD):
        with interval(0, 1):
            dtime_max_arr = delt
        with interval(1, -1):
            if index_k - 1 < ktcon:
                dtime_max_arr = min(dtime_max_arr[0, 0, -1], 0.5 * delp[0, 0, -1])
            else:
                dtime_max_arr = dtime_max_arr[0, 0, -1]

def samfshalcnv_part3(im, ix, km, ntr, cnvflg, kmax, kb, ktcon,
                      del_val, zi, heo, qo, xlamue, xlamud, eta,
                      hcko, qrcko, qcko, uo, ucko, vo, vcko,
                      ctro, ecko, qlko_ktcon, ktcon1, u1, v1,
                      umean, sumx, tauadv, gdx, dtconv, po, wc, xmb,
                      sigmagfm, garea, scaldfunc, xmbmax,
                      dellah, dellaq, dellau, dellav, dellae_np, dellal):
    """
    Scale-Aware Mass-Flux Shallow Convection PART 3 line 1304 - 1513
    :param im: IN
    :param km: IN
    :param ntr: IN
    :param cnvflg: IN
    :param del: IN, renamed to del_val to avoid conflict in keywords
    ...
    :param dtconv: OUT, needed by xmb
    :param sumx: OUT, needed by umean
    :param umean: OUT, needed by tauadv
    :param tauadv: OUT, needed by xmb
    :param xmb: OUT, used in PART4
    :param sigmagfm: OUT, used in PART4
    :param scaldfunc: OUT, needed by xmb
    :param dellah: OUT, used in PART4, initialized in PART3
    :param dellaq: OUT, used in PART4, initialized in PART3
    :param dellau: OUT, used in PART4, initialized in PART3
    :param dellav: OUT, used in PART4, initialized in PART3
    :param dellal: OUT, used in PART4
    :param dellae_np: OUT, used in PART4, initialized in PART3, (im,km,ntr), as numpy array
    Dependency
    - della*: del, zi, heo, qo, xlamue, xlamud, eta, hcko, 
              qrcko, qcko, uo, ucko, vo, vcko, ctro, ecko,
              ktcon, qlko_ktcon
    - xmb: kbcon, po, to, *tauadv*, *dtconv*, wc, *scaldfunc*
        + dtconv: zi, ktcon1, kbcon1, wc, gdx, dtmin, dt2, dtmax
        + tauadv: gdx, *umean*
            + umean: *smux*, u1, v1, zi
                + smux: zi
        + scaldfunc: gdx, dxcrt, *sigmagfm*
    - sigmagfm: garea, xlamue, kbcon
    Reference in PART4
    - dellae: delebar, ctr
        + delebar: not referenced by output vars
        + ctr: *qtr*
    """
    shape_2d = (1, ix, km)
    default_origin = (0, 0, 0)
    index_ijk_np = np.indices(shape_2d) 
    index_k = gt.storage.from_array(index_ijk_np[2] + 1, BACKEND, default_origin, shape_2d, dtype = DTYPE_INT) # index STARTING FROM 1
    
    km1 = km - 1

    # what would the change be, that a cloud with unit mass
    # will do to the environment?
    # PART3: Calculate the tendencies of the state variables (per unit cloud base mass flux) and the cloud base mass flux.
    # Calculate the change in moist static energy, moisture mixing ratio, and horizontal winds per unit cloud base mass flux for all layers below cloud top from equations B.14 and B.15 from Grell (1993) \cite grell_1993, and for the cloud top from B.16 and B.17.
    dellah[...] = 0.0
    dellaq[...] = 0.0
    dellau[...] = 0.0
    dellav[...] = 0.0
    dellae_np[...] = 0.0
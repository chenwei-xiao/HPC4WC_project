from .physcons import con_g, qamin
from . import DTYPE_FLOAT, BACKEND
import gt4py as gt
from gt4py import gtscript
from __gtscript__ import PARALLEL, FORWARD, computation, interval
import numpy as np

## Constants
epsil = 1e-22 # prevent division by zero
escav = 0.8   # wet scavenging efficiency

@gtscript.function
def set_qaero(qtr, kmax, index_k):
    qaero = max(qamin, qtr) if index_k <= kmax else 0.0
    return qaero

@gtscript.function
def set_xmbp(xmb, delp):
    xmbp = con_g * xmb / delp
    return xmbp

@gtscript.function
def set_ctro2(qaero, kmax, index_k):
    if index_k + 1 <= kmax:
        ctro2 = 0.5 * (qaero[0, 0, 0] + qaero[0, 0, 1])
    else:
        ctro2 = qaero # boundary already set in qaero
        #if index_k == kmax:
        #    ctro2 = qaero
        #else:
        #    ctro2 = 0.0
    return ctro2

@gtscript.function
def set_ecko2(ctro2, cnvflg, kb, index_k):
    if cnvflg and (index_k <= kb):
        ecko2 = ctro2
    else:
        ecko2 = 0.0
    return ecko2

@gtscript.stencil(backend = BACKEND)
def set_work_arrays(
    qtr: gtscript.Field[DTYPE_FLOAT],
    xmb: gtscript.Field[DTYPE_FLOAT],
    delp: gtscript.Field[DTYPE_FLOAT],
    kmax: gtscript.Field[int],
    kb: gtscript.Field[int],
    cnvflg: gtscript.Field[int],
    index_k: gtscript.Field[int],
    qaero: gtscript.Field[DTYPE_FLOAT],
    xmbp: gtscript.Field[DTYPE_FLOAT],
    ctro2: gtscript.Field[DTYPE_FLOAT],
    ecko2: gtscript.Field[DTYPE_FLOAT]
):
    with computation(PARALLEL), interval(...):
        qaero = set_qaero(qtr, kmax, index_k)
        xmbp = set_xmbp(xmb, delp)
    with computation(PARALLEL), interval(...):
        ctro2 = set_ctro2(qaero, kmax, index_k)
        ecko2 = set_ecko2(ctro2, cnvflg, kb, index_k)

@gtscript.stencil(backend = BACKEND)
def calc_ecko2_chem_c_dellae2(
    zi: gtscript.Field[DTYPE_FLOAT],
    xlamue: gtscript.Field[DTYPE_FLOAT],
    xlamud: gtscript.Field[DTYPE_FLOAT],
    ctro2: gtscript.Field[DTYPE_FLOAT],
    c0t: gtscript.Field[DTYPE_FLOAT],
    eta: gtscript.Field[DTYPE_FLOAT],
    xmbp: gtscript.Field[DTYPE_FLOAT],
    kb: gtscript.Field[int],
    ktcon: gtscript.Field[int],
    cnvflg: gtscript.Field[int],
    index_k: gtscript.Field[int],
    ecko2: gtscript.Field[DTYPE_FLOAT],
    chem_c: gtscript.Field[DTYPE_FLOAT],
    # chem_pw: gtscript.Field[DTYPE_FLOAT],
    dellae2: gtscript.Field[DTYPE_FLOAT],
    *,
    fscav: float
):
    with computation(FORWARD), interval(1, -1):
        if cnvflg and (index_k > kb) and (index_k < ktcon):
            dz   = zi[0, 0, 0] - zi[0, 0, -1]
            tem  = 0.5  * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
            tem1 = 0.25 * (xlamud[0, 0, 0] + xlamud[0, 0,  0]) * dz
            factor = 1.0 + tem - tem1

            # if conserved (not scavenging) then
            ecko2 = ((1.0 - tem1) * ecko2[0, 0, -1] +
                     0.5 * tem * (ctro2[0, 0, 0] + ctro2[0, 0, -1])) / factor
            #    how much will be scavenged
            #    this choice was used in GF, and is also described in a
            #    successful implementation into CESM in GRL (Yu et al. 2019),
            #    it uses dimesnsionless scavenging coefficients (fscav),
            #    but includes henry coeffs with gas phase chemistry
            #    fraction fscav is going into liquid
            chem_c = escav * fscav * ecko2
            # of that part is going into rain out (chem_pw)
            tem2 = chem_c / (1.0 + c0t * dz)
            # chem_pw = c0t * dz * tem2 * eta # etah
            ecko2 = tem2 + ecko2 - chem_c
    with computation(PARALLEL), interval(0, -1):
        if index_k >= ktcon:
            ecko2 = ctro2
        if cnvflg and (index_k == ktcon):
            #for the subsidence term already is considered
            dellae2 = eta[0, 0, -1] * ecko2[0, 0, -1] * xmbp



def samfshalcnv_aerosols(im, ix, km, itc, ntc, ntr, delt,
                         cnvflg, kb, kmax, kbcon, ktcon, fscav_np,
                         xmb, c0t, eta, zi, xlamue, xlamud, delp,
                         qtr_np, qaero_np):
    """
    Aerosol process in shallow convection
    :param im: horizontal loop extent
    :param ix: horizontal dimension (im <= ix)
    :param km: vertical layer dimension
    :param itc: number of aerosol tracers transported/scavenged by convection
    :param ntc: number of chemical tracers
    :param ntr: number of tracers for scale-aware mass flux schemes
    :param delt: physics time step
    :param cnvflg: (im) flag of convection
    :param kb: (im)
    :param kmax: (im)
    :param kbcon: (im)
    :param ktcon: (im)
    :param fscav_np: (ntc) numpy array of aerosol scavenging coefficients
    :param xmb: (im)
    :param c0t: (im,km) Cloud water parameters
    :param eta: (im,km)
    :param zi: (im,km) height
    :param xlamue: (im,km)
    :param xlamud: (im)
    :param delp: (ix,km) pressure?
    :param qtr_np: (ix,km,ntr+2) numpy array
    :param qaero_np: (im,km,ntc) numpy array
    """
    shape_2d = (1, ix, km)
    default_origin = (0, 0, 0)
    # Initialization
    xmbp = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Chemical transport variables (2D slices)
    qtr     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    qaero   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ctro2   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ecko2   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ecdo2   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    dellae2 = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Additional variables for tracers for wet deposition (2D slices)
    chem_c  = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    # chem_pw = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    # wet_dep = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Additional variables for fct
    flx_lo  = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    totlout = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    clipout = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Misc
    #kmax_np = kmax[0, :, 0].view(np.ndarray)
    index_ijk_np = np.indices(shape_2d) 
    index_k = gt.storage.from_array(index_ijk_np[2] + 1, BACKEND, default_origin, shape_2d, dtype = int) # index STARTING FROM 1

    # Begin
    ## Check if aerosols are present
    if (ntc <= 0) or (itc <= 0) or (ntr <= 0): return
    if (ntr < itc + ntc - 3): return

    # km1 = km - 1

    ## Tracer loop
    for n in range(ntc):
        ## Initialize work variables
        chem_c[...]  = 0.0
        # chem_pw[...] = 0.0
        ctro2[...]   = 0.0
        dellae2[...] = 0.0
        ecdo2[...]   = 0.0
        ecko2[...]   = 0.0
        qaero[...]   = 0.0

        it = n + itc - 1
        qtr[...] = qtr_np[np.newaxis, :, :, it]
        set_work_arrays(qtr, xmb, delp, kmax, kb, cnvflg, 
                        index_k, qaero, xmbp, ctro2, ecko2)
        # do chemical tracers, first need to know how much reevaporates
        # aerosol re-evaporation is set to zero for now
        # calculate include mixing ratio (ecko2), how much goes into
        # rainwater to be rained out (chem_pw), and total scavenged,
        # if not reevaporated (pwav)
        fscav = fscav_np[n]
        calc_ecko2_chem_c_dellae2(zi, xlamue, xlamud, ctro2, c0t, eta, xmbp, kb,
                                   ktcon, cnvflg, index_k, ecko2, chem_c,
                                   dellae2, fscav = fscav)

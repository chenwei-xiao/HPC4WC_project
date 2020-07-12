import gt4py as gt
from gt4py import gtscript
from shalconv.funcphys import fpvs
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


def samfshalcnv(data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection
    
    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """
    
    ### Input variables and arrays ###
    im         = data_dict["im"]
    ix         = data_dict["ix"]
    km         = data_dict["km"]
    itc        = data_dict["itc"]
    ntc        = data_dict["ntc"]
    ntk        = data_dict["ntk"]
    ntr        = data_dict["ntr"]
    ncloud     = data_dict["ncloud"]
    clam       = data_dict["clam"]
    c0s        = data_dict["c0s"]
    c1         = data_dict["c1"]
    asolfac    = data_dict["asolfac"]
    pgcon      = data_dict["pgcon"]
    delt       = data_dict["delt"]
    islimsk    = data_dict["islimsk"]
    psp        = data_dict["psp"]
    delp       = data_dict["delp"]
    prslp      = data_dict["prslp"]
    garea      = data_dict["garea"]
    hpbl       = data_dict["hpbl"]
    dot        = data_dict["dot"]
    phil       = data_dict["phil"]
    fscav      = data_dict["fscav"]
    
    ### Output buffers ###
    kcnv       = data_dict["kcnv"]
    kbot       = data_dict["kbot"]
    ktop       = data_dict["ktop"]
    qtr        = data_dict["qtr"]
    q1         = data_dict["q1"]
    t1         = data_dict["t1"]
    u1         = data_dict["u1"]
    v1         = data_dict["v1"]
    rn         = data_dict["rn"]
    cnvw       = data_dict["cnvw"]
    cnvc       = data_dict["cnvc"]
    ud_mf      = data_dict["ud_mf"]
    dt_mf      = data_dict["dt_mf"]
    
    ### Local storages for 1D arrays (integer) ###
    kpbl       = gt.storage.ones (BACKEND, default_origin, shape, dtype=INT)
    kb         = gt.storage.empty(BACKEND, default_origin, shape, dtype=INT)
    kbcon      = gt.storage.empty(BACKEND, default_origin, shape, dtype=INT)
    kbcon1     = gt.storage.empty(BACKEND, default_origin, shape, dtype=INT)
    ktcon      = gt.storage.ones (BACKEND, default_origin, shape, dtype=INT)
    ktcon1     = gt.storage.empty(BACKEND, default_origin, shape, dtype=INT)
    ktconn     = gt.storage.ones (BACKEND, default_origin, shape, dtype=INT)
    kbm        = gt.storage.empty(BACKEND, default_origin, shape, dtype=INT)
    kmax       = gt.storage.empty(BACKEND, default_origin, shape, dtype=INT)
    
    ### Local storages for 1D arrays ("bool") ###
    cnvflg     = gt.storage.ones (BACKEND, default_origin, shape, dtype=INT)
    flg        = gt.storage.empty(BACKEND, default_origin, shape, dtype=INT)
    
    ### Local storages for 1D arrays (float) ###
    aa1        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=FLT)
    cina       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=FLT)
    tkemean    = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    clamt      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    ps         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    del        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    prsl       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    umean      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    tauadv     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    gdx        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    delhbar    = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    delq       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    delq2      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    delqbar    = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    delqev     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    deltbar    = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    deltv      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dtconv     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    edt        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=FLT)
    pdot       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=FLT)
    po         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qcond      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qevap      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    hmax       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    rntot      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    vshear     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=FLT)
    xlamud     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    xmb        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    xmbmax     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    delubar    = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    delvbar    = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    c0         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    wc         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    scaldfunc  = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    sigmagfm   = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qlko_ktcon = gt.storage.zeros(BACKEND, default_origin, shape, dtype=FLT)
    sumx       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    tx1        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    
    ### Local storages for 2D arrays (float) ###
    pfld       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    to         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    uo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    vo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qeso       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    wu2        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    buo        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    drag       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dellal     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dbyo       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    zo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    xlamue     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    heo        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    heso       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dellah     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dellaq     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dellau     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dellav     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    hcko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    ucko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    vcko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qcko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qrcko      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    eta        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    zi         = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    pwo        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    c0t        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    cnvwt      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    
    ### Local storages for 2D arrays (float, tracers) ###
    delebar    = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    
    ### Local storages for 3D arrays (float, tracers), this we'll contain slices ###
    ctr        = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    ctro       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    dellae     = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    ecko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    qaero      = gt.storage.empty(BACKEND, default_origin, shape, dtype=FLT)
    
    ### K-indices field ###
    k_idx      = gt.storage.from_array(np.tile(np.linspace(0, km, km + 1, dtype=INT), shape), BACKEND, default_origin)
    
    ### Local Parameters ###
    g          = grav
    elocp      = hvap/cp
    el2orc     = hvap * hvap/(rv * cp)
    d0         = 0.001
    cm         = 1.0
    delta      = fv
    fact1      = (cvap - cliq)/rv
    fact2      = hvap/rv - fact1 * t0c
    clamd      = 0.1
    tkemx      = 0.65
    tkemn      = 0.05
    dtke       = tkemx - tkemn
    dthk       = 25.0
    cinpcrmx   = 180.0
    cinpcrmn   = 120.0
    cinacrmx   = -120.0
    cinacrmn   = -80.0
    crtlamd    = 3.0e-4
    dtmax      = 10800.0
    dtmin      = 600.0
    bet1       = 1.875.0
    cd1        = 0.506
    f1         = 2.0
    gam1       = 0.5
    betaw      = 0.03
    dxcrt      = 15.0e3
    h1         = 0.33333333
    tf         = 233.16
    tcr        = 263.16
    tcrf       = 1.0/(tcr - tf)
    
    
    ### Determine whether to perform aerosol transport ###
    do_aerosols = (itc > 0) and (ntc > 0) and (ntr > 0)
    if (do_aerosols):
        do_aerosols = (ntr >= itc)
    
    ### Compute preliminary quantities needed for the static and feedback control portions of the algorithm ###
    
    # Convert input Pa terms to Cb terms
    pa_to_cb( psp, 
              prslp,
              delp,
              ps,
              prsl,
              del,
              origin=origin,
              domain=domain )
    
    km1 = km - 1
    
    ### Initialize storages (simple initializations already done above with gt4py functionalities) ###
    
    # Initialize column-integrated and other single-value-per-column 
    # variable arrays
    init_col_arr( kcnv,
                  cnvflg,
                  kbot,
                  ktop,
                  kbcon,
                  kb,
                  rn, 
                  origin=origin,
                  domain=domain )
    
    # Return to the calling routine if deep convection is present or the 
    # surface buoyancy flux is negative
    totflg = True
    for i in range(0, im):
        totflg = totflg and (not cnvflg[0, i, 0] == 1)
    if totflg: return
    
    # Initialize further parameters and arrays
    init_par_and_arr( islimsk, 
                      c0,
                      c0s,
                      asolfac,
                      t1,
                      c0t,
                      tem, 
                      tem1, 
                      cnvw,
                      cnvc,
                      ud_mf,
                      dt_mf,
                      origin=origin,
                      domain=domain )
                      
    dt2   = delt
    
    # Model tunable parameters are all here
    aafac   = 0.05
    evfact  = 0.3
    evfactl = 0.3
    w1l     = -8.0e-3 
    w2l     = -4.0e-2
    w3l     = -5.0e-3 
    w4l     = -5.0e-4
    w1s     = -2.0e-4
    w2s     = -2.0e-3
    w3s     = -1.0e-3
    w4s     = -2.0e-5
    
    # Initialize the rest
    init_final( km, kbm, k_idx, kmax, flg, cnvflg, kpbl, tx1, 
                ps, prsl, zo, phil, zi, pfld, eta, hcko, qcko, 
                qrcko, ucko, vcko, dbyo, pwo, dellal, to, qo, 
                uo, vo, wu2, buo, drag, cnvwt, qeso, heo, heso )
                
    
    # Tracer loop (this goes at the end!)
    for n in range(0, ntr):
        
        if n >= 2 and n < ntr + 2:
            kk = n-2
            
            qtr_shift = gt.storage.from_array(slice_to_3d(qtr[:, :, kk]), BACKEND, default_origin)
            
            init_tracers( cnvflg,
                          k_idx,
                          kmax,
                          ctr, 
                          ctro, 
                          ecko,
                          qtr_shift )
                          

############################### STENCILS ###############################

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def pa_to_cb( psp  : FLT_FIELD, 
              prslp: FLT_FIELD,
              delp : FLT_FIELD,
              ps   : FLT_FIELD,
              prsl : FLT_FIELD,
              del  : FLT_FIELD ):
    
    from __gtscript__ import PARALLEL, computation, interval
    
    with computation(PARALLEL), interval(...):
        
        # Convert input Pa terms to Cb terms
        ps   = psp   * 0.001
        prsl = prslp * 0.001
        del  = delp  * 0.001
        

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"sqrt": sqrt})
def init_col_arr( km    : INT,
                  kcnv  : INT_FIELD, 
                  cnvflg: INT_FIELD,
                  kbot  : INT_FIELD,
                  ktop  : INT_FIELD,
                  kbcon : INT_FIELD,
                  kb    : INT_FIELD,
                  rn    : FLT_FIELD,
                  gdx   : FLT_FIELD ):
    
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import sqrt
    
    with computation(PARALLEL), interval(...):
        
        # Initialize column-integrated and other single-value-per-column 
        # variable arrays
        if (kcnv == 1):
            cnvflg = 0
            
        if (cnvflg == 1):
            kbot = km + 1
            ktop = 0
            
        rn    = 0.0
        kbcon = km
        kb    = km
        gdx   = sqrt(garea)


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"exp": exp})
def init_par_and_arr( c0s    : FLT,
                      asolfac: FLT,
                      d0     : FLT,
                      islimsk: INT_FIELD,
                      c0     : FLT_FIELD,
                      t1     : FLT_FIELD,
                      c0t    : FLT_FIELD,
                      cnvw   : FLT_FIELD,
                      cnvc   : FLT_FIELD,
                      ud_mf  : FLT_FIELD,
                      dt_mf  : FLT_FIELD ):
    
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import exp
    
    with computation(PARALLEL), interval(...):
        
        # Determine aerosol-aware rain conversion parameter over land
        if islimsk == 1:
            c0 = c0s * asolfac
        else:
            c0 = c0s
            
        # Determine rain conversion parameter above the freezing level 
        # which exponentially decreases with decreasing temperature 
        # from Han et al.'s (2017) \cite han_et_al_2017 equation 8
        if t1 > 273.16:
            c0t = c0
        else:
            tem = exp(d0 * (t1 - 273.16))    # Cannot use functions in conditionals?
            c0t  = c0 * tem
            
        # Initialize convective cloud water and cloud cover to zero
        cnvw = 0.0
        cnvc = 0.0
        
        # Initialize updraft mass fluxes to zero
        ud_mf = 0.0
        dt_mf = 0.0


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"min": min, "max": max})
def init_final( km    : INT,
                kbm   : INT_FIELD,
                k_idx : INT_FIELD,
                kmax  : INT_FIELD,
                flg   : INT_FIELD,
                cnvflg: INT_FIELD,
                kpbl  : INT_FIELD,
                tx1   : FLT_FIELD,
                ps    : FLT_FIELD,
                prsl  : FLT_FIELD,
                zo    : FLT_FIELD,
                phil  : FLT_FIELD,
                zi    : FLT_FIELD,
                pfld  : FLT_FIELD,
                eta   : FLT_FIELD,
                hcko  : FLT_FIELD,
                qcko  : FLT_FIELD,
                qrcko : FLT_FIELD,
                ucko  : FLT_FIELD,
                vcko  : FLT_FIELD,
                dbyo  : FLT_FIELD,
                pwo   : FLT_FIELD,
                dellal: FLT_FIELD,
                to    : FLT_FIELD,
                qo    : FLT_FIELD,
                uo    : FLT_FIELD,
                vo    : FLT_FIELD,
                wu2   : FLT_FIELD,
                buo   : FLT_FIELD,
                drag  : FLT_FIELD,
                cnvwt : FLT_FIELD,
                qeso  : FLT_FIELD,
                heo   : FLT_FIELD,
                heso  : FLT_FIELD ):
    
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import min, max
    
    with computation(PARALLEL), interval(...):
        
        # Determine maximum indices for the parcel starting point (kbm) 
        # and cloud top (kmax)
        kbm  = km
        kmax = km
        tx1  = 1.0/ps
        
        if prsl * tx1(i) > 0.7: kbm  = k_idx + 1
        if prsl * tx1(i) > 0.6: kmax = k_idx + 1
        
        kbm = min(kbm, kmax)
        
        # Calculate hydrostatic height at layer centers assuming a flat 
        # surface (no terrain) from the geopotential
        zo = phil/g
        
    with computation(PARALLEL), interval(0, -1):
        
        # Calculate interface height
        zi = 0.5 * (zo[0, 0, 0] + zo[0, 0, +1])
        
    with computation(PARALLEL), interval(...):
        
        # Find the index for the PBL top using the PBL height; enforce 
        # that it is lower than the maximum parcel starting level
        flg = cnvflg
        
        # ~ do k = 2, km1
          # ~ do i=1,im
            # ~ if (flg(i) .and. zo(i,k) <= hpbl(i)) then
              # ~ kpbl(i) = k
            # ~ else
              # ~ flg(i) = .false.
            # ~ endif
          # ~ enddo
        # ~ enddo
        
        kpbl = min(kpbl, kbm)
        
        if cnvflg == 1 and k_idx <= kmax:
            
            # Convert prsl from centibar to millibar, set normalized mass 
            # flux to 1, cloud properties to 0, and save model state 
            # variables (after advection/turbulence)
            pfld   = prsl * 10.0
            eta    = 1.0
            hcko   = 0.0
            qcko   = 0.0
            qrcko  = 0.0
            ucko   = 0.0
            vcko   = 0.0
            dbyo   = 0.0
            pwo    = 0.0
            dellal = 0.0
            to     = t1
            qo     = q1
            uo     = u1
            vo     = v1
            wu2    = 0.0
            buo    = 0.0
            drag   = 0.0
            cnvwt  = 0.0
            
            # Calculate saturation specific humidity and enforce minimum 
            # moisture values
            qeso = (0.01 * eps * fpvs(to))/(pfld + epsm1 * qeso)    # fpsv is a function (can't be called inside conditional), also how to access lookup table?
            qeso = max(qeso, 1.0e-8 )
            qo   = max(qo  , 1.0e-10)
            
            # Calculate moist static energy (heo) and saturation moist 
            # static energy (heso)
            tem  = phil + cp * to
            heo  = tem + hvap * qo
            heso = tem + hvap * qeso
            


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def init_tracers( cnvflg: INT_FIELD,
                  k_idx : INT_FIELD,
                  kmax  : INT_FIELD,
                  ctr   : FLT_FIELD,    # Remember to pass in the 
                  ctro  : FLT_FIELD,    # correct slices (kk=n-2 and 
                  ecko  : FLT_FIELD,    # n=3, ntr+2)
                  qtr   : FLT_FIELD ):
    
    from __gtscript__ import PARALLEL, computation, interval
    
    with computation(PARALLEL), interval(...):
        
        # Initialize tracer variables
        if cnvflg == 1 and k_idx <= kmax:
            ctr  = qtr
            ctro = qtr
            ecko = 0.0


@gtscript.function
def sqrt(x):
    return x**0.5


@gtscript.function
def exp(x):
    return e**x
        

@gtscript.function
def min(x, y):
    if x <= y:
        return x
    else:
        return y
        

@gtscript.function
def max(x, y):
    if x >= y:
        return x
    else:
        return y
        

def slice_to_3d(slice):
    return slice[np.newaxis, :, :]

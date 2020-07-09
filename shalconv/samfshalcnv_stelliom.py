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
    con_epsm1 as epsm1
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
    
    ### Local storages 1D (integer) ###
    kpbl       = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    kb         = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    kbcon      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    kbcon1     = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    ktcon      = gt.storage.ones (BACKEND, default_origin, shape_1d, dtype=int)
    ktcon1     = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    ktconn     = gt.storage.ones (BACKEND, default_origin, shape_1d, dtype=int)
    kbm        = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    kmax       = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    
    ### Local storages 1D ("bool") ###
    cnvflg     = gt.storage.ones (BACKEND, default_origin, shape_1d, dtype=int)
    flg        = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=int)
    
    ### Local storages 1D (float) ###
    aa1        = gt.storage.zeros(BACKEND, default_origin, shape_1d, dtype=float)
    cina       = gt.storage.zeros(BACKEND, default_origin, shape_1d, dtype=float)
    tkemean    = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    clamt      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    ps         = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    del        = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    prsl       = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    umean      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    tauadv     = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    gdx        = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    delhbar    = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    delq       = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    delq2      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    delqbar    = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    delqev     = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    deltbar    = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    deltv      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    dtconv     = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    edt        = gt.storage.zeros(BACKEND, default_origin, shape_1d, dtype=float)
    pdot       = gt.storage.zeros(BACKEND, default_origin, shape_1d, dtype=float)
    po         = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    qcond      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    qevap      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    hmax       = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    rntot      = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    vshear     = gt.storage.zeros(BACKEND, default_origin, shape_1d, dtype=float)
    xlamud     = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    xmb        = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    xmbmax     = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    delubar    = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    delvbar    = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    c0         = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    wc         = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    scaldfunc  = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    sigmagfm   = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    qlko_ktcon = gt.storage.zeros(BACKEND, default_origin, shape_1d, dtype=float)
    sumx       = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    tx1        = gt.storage.empty(BACKEND, default_origin, shape_1d, dtype=float)
    
    ### Local storages 2D (float) ###
    pfld       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    to         = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    qo         = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    uo         = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    vo         = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    qeso       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    wu2        = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    buo        = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    drag       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    dellal     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    dbyo       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    zo         = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    xlamue     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    heo        = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    heso       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    dellah     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    dellaq     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    dellau     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    dellav     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    hcko       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    ucko       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    vcko       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    qcko       = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    qrcko      = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    eta        = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    zi         = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    pwo        = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    c0t        = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    cnvwt      = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype=float)
    
    ### Local storages 2D (float, tracers) ###
    delebar    = gt.storage.empty(BACKEND, default_origin, shape_2d_ntr, dtype=float) # shape (im, ntr) -> (1, im, ntr)
    
    ### Local storages 3D (float, tracers) ###
    ctr        = gt.storage.empty(BACKEND, default_origin, shape_3d_ntr, dtype=float) # shape (im, km, ntr)
    ctro       = gt.storage.empty(BACKEND, default_origin, shape_3d_ntr, dtype=float) # shape (im, km, ntr)
    dellae     = gt.storage.empty(BACKEND, default_origin, shape_3d_ntr, dtype=float) # shape (im, km, ntr)
    ecko       = gt.storage.empty(BACKEND, default_origin, shape_3d_ntr, dtype=float) # shape (im, km, ntr)
    qaero      = gt.storage.empty(BACKEND, default_origin, shape_3d_ntc, dtype=float) # shape (im, km, ntc)
    
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
    ps   = psp   * 0.001
    prsl = prslp * 0.001
    del  = delp  * 0.001
    
    km1 = km - 1
    
    ### Initialize storages (simple initializations already done above with gt4py functionalities) ###
    # ~ do i=1,im
        # ~ cnvflg(i) = .true.                      # Done
        # ~ if(kcnv(i) == 1) cnvflg(i) = .false.    # 
        # ~ if(cnvflg(i)) then                      # 
          # ~ kbot(i)=km+1                          # 
          # ~ ktop(i)=0                             # 
        # ~ endif                                   # 
        # ~ rn(i)=0.                                # 
        # ~ kbcon(i)=km                             # 
        # ~ ktcon(i)=1                              # Done
        # ~ ktconn(i)=1                             # Done
        # ~ kb(i)=km                                # 
        # ~ pdot(i) = 0.                            # Done
        # ~ qlko_ktcon(i) = 0.                      # Done
        # ~ edt(i)  = 0.                            # Done
        # ~ aa1(i)  = 0.                            # Done
        # ~ cina(i) = 0.                            # Done
        # ~ vshear(i) = 0.                          # Done
        # ~ gdx(i) = sqrt(garea(i))                 # Done
    # ~ enddo
    

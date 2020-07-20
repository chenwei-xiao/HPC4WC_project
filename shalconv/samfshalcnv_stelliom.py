import gt4py as gt
from gt4py import gtscript
from shalconv.funcphys import fpvsx_gt as fpvs
from shalconv.samfaerosols import samfshalcnv_aerosols 
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
    

############################ INITIALIZATION ############################

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
    kpbl       = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kb         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon1     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon      = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon1     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktconn     = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbm        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kmax       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
    ### Local storages for 1D arrays ("bool") ###
    cnvflg     = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    flg        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
    ### Local storages for 1D arrays (float) ###
    aa1        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cina       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tkemean    = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    clamt      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ps         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    del        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    prsl       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    umean      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tauadv     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    gdx        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delhbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq2      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqev     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltv      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dtconv     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    edt        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pdot       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    po         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcond      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qevap      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hmax       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    rntot      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vshear     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamud     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmb        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmbmax     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delubar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delvbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wc         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    scaldfunc  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sigmagfm   = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qlko_ktcon = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sumx       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tx1        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_ktcon1  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_kbcon1  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 2D arrays (float) ###
    pfld       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    to         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    uo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qeso       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wu2        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    buo        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    drag       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellal     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dbyo       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zo         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamue     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heo        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heso       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellah     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellaq     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellau     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellav     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hcko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ucko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vcko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qrcko      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    eta        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi         = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pwo        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0t        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cnvwt      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 2D arrays (float, tracers), this will contain slices along n-axis ###
    delebar    = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 3D arrays (float, tracers), this will contain slices along n-axis ###
    ctr        = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ctro       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellae     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ecko       = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qaero      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### K-indices field ###
    k_idx      = gt.storage.from_array(np.tile(np.linspace(0, km, km + 1, dtype=DTYPE_INT), shape), BACKEND, default_origin)
    
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
    init_col_arr( km,
                  kcnv,
                  cnvflg,
                  kbot,
                  ktop,
                  kbcon,
                  kb,
                  rn, 
                  gdx, 
                  garea, 
                  origin=origin,
                  domain=domain )
    
    # Return to the calling routine if deep convection is present or the 
    # surface buoyancy flux is negative
    if exit_routine(cnvflg, im): return
    
    # Initialize further parameters and arrays
    init_par_and_arr( c0s,
                      asolfac,
                      d0,
                      islimsk,
                      c0,
                      t1,
                      c0t,
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
                uo, vo, wu2, buo, drag, cnvwt, qeso, heo, heso,
                origin=origin, domain=domain )
                
    
    # Tracers loop (THIS GOES AT THE END AND POSSIBLY MERGED WITH OTHER 
    # TRACER LOOPS!) --> better use version below
    for n in range(2, ntr+2):
        
        #kk = n-2
        
        qtr_shift = gt.storage.from_array(slice_to_3d(qtr[:, :, n]), BACKEND, default_origin)
        
        # Initialize tracers. Keep in mind that, qtr slice is for the 
        # n-th tracer, while the other storages are slices representing 
        # the (n-2)-th tracer.
        init_tracers( cnvflg,
                      k_idx,
                      kmax,
                      ctr, 
                      ctro, 
                      ecko,
                      qtr_shift,
                      origin=origin, 
                      domain=domain )
                      
    # Alternative to merge with main tracer loop
    #for n in range(0, ntr):
    #    
    #    kk = n+2
    #    
    #    qtr_shift = gt.storage.from_array(slice_to_3d(qtr[:, :, kk]), BACKEND, default_origin)
    #    
    #    # Initialize tracers. Keep in mind that, qtr slice is for the 
    #    # (n+2)-th tracer, while the other storages are slices 
    #    # representing the n-th tracer.
    #    init_tracers( cnvflg,
    #                  k_idx,
    #                  kmax,
    #                  ctr, 
    #                  ctro, 
    #                  ecko,
    #                  qtr_shift,
    #                  origin=origin, 
    #                  domain=domain )

########################################################################


######################### FROM LINES 1304-1513 #########################
# Calculate the tendencies of the state variables (per unit cloud base 
# mass flux) and the cloud base mass flux
########################################################################
    
    # Calculate the tendencies of the state variables (per unit cloud base 
    # mass flux) and the cloud base mass flux
    comp_tendencies( g, betaw, dtmin, dt2, dtmax, dxcrt, cnvflg, k_idx,
                     kmax, kb, ktcon, ktcon1, kbcon1, kbcon, dellah,
                     dellaq, dellau, dellav, del, zi, zi_ktcon1,
                     zi_kbcon1, heo, qo, xlamue, xlamud, eta, hcko,
                     qrcko, uo, ucko, vo, vcko, qcko, dellal, 
                     qlko_ktcon, wc, gdx, dtconv, u1, v1, po, to, 
                     tauadv, xmb, sigmagfm, garea, scaldfunc, xmbmax,
                     origin=origin, domain=domain )
    
    # Tracers loop (THIS GOES AT THE END AND POSSIBLY MERGED WITH OTHER 
    # TRACER LOOPS!)
    for n in range(0, ntr):
        
        # Calculate the tendencies of the state variables (tracers part)
        comp_tendencies_tr( g, 
                            cnvflg, 
                            k_idx, 
                            kmax, 
                            kb, 
                            ktcon, 
                            dellae, 
                            del, 
                            eta, 
                            ctro, 
                            ecko, 
                            origin=origin, 
                            domain=domain )
                            
    
    # Transport aerosols if present (THIS HAS DEPENDENCIES ON THE 
    # TRACERS LOOP, SO WE'LL PROBABLY NEED TO SPLIT IT HERE, THUS ALSO 
    # HAVING TO UPDATE THE ORIGINAL TRACER'S FIELDS WITH THE PER SLICE 
    # COMPUTATION!)
    if do_aerosols:
        samfshalcnv_aerosols( im, ix, km, itc, ntc, ntr, delt, cnvflg, 
                              kb, kmax, kbcon, ktcon, fscav, xmb, c0t, 
                              eta, zi, xlamue, xlamud, delp, qtr, qaero,
                              origin=origin, domain=domain )
    
########################################################################


######################### FROM LINES 1514-1806 #########################
# For the "feedback control", calculate updated values of the state 
# variables by multiplying the cloud base mass flux and the tendencies 
# calculated per unit cloud base mass flux from the static control
########################################################################

    # For the "feedback control", calculate updated values of the state 
    # variables by multiplying the cloud base mass flux and the 
    # tendencies calculated per unit cloud base mass flux from the 
    # static control
    feedback_control_update( dt2, g, evfact, evfactl, el2orc, elocp, 
                             cnvflg, k_idx, kmax, kb, ktcon, flg, 
                             islimsk, ktop, kbot, kbcon, kcnv, qeso, 
                             pfld, delhbar, delqbar, deltbar, delubar, 
                             delvbar, qcond, dellah, dellaq, t1, xmb, 
                             q1, u1, dellau, v1, dellav, del, rntot, 
                             delqev, delq2, pwo, deltv, delq, qevap, rn, 
                             edt, cnvw, cnvwt, cnvc, ud_mf, dt_mf,
                             origin=origin, domain=domain )
    
    # Tracers loop                     
    for n in range(0, ntr):
        
        # Use qtr_shift already defined at the beginning of the ntr 
        # tracers loop (see alternative above)
        # ~ kk = n+2
        # ~ qtr_shift = gt.storage.from_array(slice_to_3d(qtr[:, :, kk]), BACKEND, default_origin)
        
        # Calculate updated values of the state variables (tracers part)
        feedback_control_upd_trr( dt2,
                                  g,
                                  cnvflg,
                                  k_idx,
                                  kmax,
                                  ktcon,
                                  delebar,
                                  ctr,
                                  dellae,
                                  xmb,
                                  qtr_shift, 
                                  origin=origin,
                                  domain=domain )
                                  
        # UPDATE QTR BASED ON RESULTS IN THE SLICE QTR_SHIFT!
        
        # UPDATE OTHER TRACERS' FIELDS BASED ON THE RESULTS IN THE 
        # SLICES!
    
    # Separate detrained cloud water into liquid and ice species as a 
    # function of temperature only
    if ncloud > 0:
        
        qtr_0 = gt.storage.from_array(slice_to_3d(qtr[:, :, 0]), BACKEND, default_origin)
        qtr_1 = gt.storage.from_array(slice_to_3d(qtr[:, :, 1]), BACKEND, default_origin)
        
        separate_detrained_cw( dt2,
                               tcr,
                               tcrf,
                               cnvflg,
                               k_idx,
                               kbcon,
                               ktcon,
                               dellal,
                               xmb,
                               t1,
                               qtr_1,
                               qtr_0,
                               origin=origin,
                               domain=domain )
        
        # UPDATE QTR BASED ON RESULTS IN THE SLICE QTR_0 AND QTR_1!
    
    if do_aerosols: 
           
        # Tracers loop (aerosols)
        for n in range(0, ntc):
            
            kk = n + itc - 1
            
            qtr_shift = gt.storage.from_array(slice_to_3d(qtr[:, :, kk]), BACKEND, default_origin)
            
            # Store aerosol concentrations if present                           
            store_aero_conc( cnvflg,
                             k_idx,
                             kmax,
                             rn,
                             qtr_shift,
                             qaero,
                             origin=origin,
                             domain=domain )
                             
            # UPDATE QTR BASED ON RESULTS IN THE SLICE QTR_SHIFT!
            
            # UPDATE OTHER TRACERS' FIELDS BASED ON THE RESULTS IN THE 
            # SLICES!
            
    # Include TKE contribution from shallow convection
    if ntk > 0:
        
        qtr_ntk = gt.storage.from_array(slice_to_3d(qtr[:, :, ntk]), BACKEND, default_origin)
        
        tke_contribution( betaw,
                          cnvflg,
                          k_idx,
                          kb,
                          ktop,
                          eta,
                          xmb,
                          pfld,
                          t1, 
                          sigmagfm,
                          qtr_ntk,
                          origin=origin,
                          domain=domain )
        
        # UPDATE QTR BASED ON RESULTS IN THE SLICE QTR_SHIFT!

########################################################################


############################### STENCILS ###############################
# All the gtscript stencil used in this file
########################################################################

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def pa_to_cb( psp  : FIELD_FLOAT, 
              prslp: FIELD_FLOAT,
              delp : FIELD_FLOAT,
              ps   : FIELD_FLOAT,
              prsl : FIELD_FLOAT,
              del  : FIELD_FLOAT ):
    
    from __gtscript__ import PARALLEL, computation, interval
    
    with computation(PARALLEL), interval(...):
        
        # Convert input Pa terms to Cb terms
        ps   = psp   * 0.001
        prsl = prslp * 0.001
        del  = delp  * 0.001
        

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"sqrt": sqrt})
def init_col_arr( km    : DTYPE_INT,
                  kcnv  : FIELD_INT, 
                  cnvflg: FIELD_INT,
                  kbot  : FIELD_INT,
                  ktop  : FIELD_INT,
                  kbcon : FIELD_INT,
                  kb    : FIELD_INT,
                  rn    : FIELD_FLOAT,
                  gdx   : FIELD_FLOAT,
                  garea : FIELD_FLOAT ):
    
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
def init_par_and_arr( c0s    : DTYPE_FLOAT,
                      asolfac: DTYPE_FLOAT,
                      d0     : DTYPE_FLOAT,
                      islimsk: FIELD_INT,
                      c0     : FIELD_FLOAT,
                      t1     : FIELD_FLOAT,
                      c0t    : FIELD_FLOAT,
                      cnvw   : FIELD_FLOAT,
                      cnvc   : FIELD_FLOAT,
                      ud_mf  : FIELD_FLOAT,
                      dt_mf  : FIELD_FLOAT ):
    
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
            c0t = c0 * tem
            
        # Initialize convective cloud water and cloud cover to zero
        cnvw = 0.0
        cnvc = 0.0
        
        # Initialize updraft mass fluxes to zero
        ud_mf = 0.0
        dt_mf = 0.0


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"min": min, "max": max, "fpvs": fpvs})
def init_final( km    : DTYPE_INT,
                kbm   : FIELD_INT,
                k_idx : FIELD_INT,
                kmax  : FIELD_INT,
                flg   : FIELD_INT,
                cnvflg: FIELD_INT,
                kpbl  : FIELD_INT,
                tx1   : FIELD_FLOAT,
                ps    : FIELD_FLOAT,
                prsl  : FIELD_FLOAT,
                zo    : FIELD_FLOAT,
                phil  : FIELD_FLOAT,
                zi    : FIELD_FLOAT,
                pfld  : FIELD_FLOAT,
                eta   : FIELD_FLOAT,
                hcko  : FIELD_FLOAT,
                qcko  : FIELD_FLOAT,
                qrcko : FIELD_FLOAT,
                ucko  : FIELD_FLOAT,
                vcko  : FIELD_FLOAT,
                dbyo  : FIELD_FLOAT,
                pwo   : FIELD_FLOAT,
                dellal: FIELD_FLOAT,
                to    : FIELD_FLOAT,
                qo    : FIELD_FLOAT,
                uo    : FIELD_FLOAT,
                vo    : FIELD_FLOAT,
                wu2   : FIELD_FLOAT,
                buo   : FIELD_FLOAT,
                drag  : FIELD_FLOAT,
                cnvwt : FIELD_FLOAT,
                qeso  : FIELD_FLOAT,
                heo   : FIELD_FLOAT,
                heso  : FIELD_FLOAT ):
    
    from __gtscript__ import PARALLEL, FORWARD, BACKWARD, computation, interval
    from __externals__ import min, max, fpvs
    
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
        
        # Initialize flg in parallel computation block
        flg = cnvflg
        
    with computation(PARALLEL), interval(0, -1):
        
        # Calculate interface height
        zi = 0.5 * (zo[0, 0, 0] + zo[0, 0, +1])
    
    with computation(FORWARD),interval(1,-1):
        
        # Find the index for the PBL top using the PBL height; enforce 
        # that it is lower than the maximum parcel starting level
        if flg[0, 0, -1] and (zo <= hpbl):
            kbpl = k_idx
            flg  = flg[0, 0, -1]
        else:
            kbpl = kbpl[0, 0, -1]
            flg  = False
        
    with computation(BACKWARD),interval(1,-1):
        
        # Propagate results back to update whole field
        kbpl = kbpl[0, 0, 1]
        flg  = flg[0, 0, 1]
        
    with computation(PARALLEL), interval(...):
        
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
            val1 = 1.0e-8
            val2 = 1.0e-10
            qeso = max(qeso, val1 )
            qo   = max(qo  , val2)
            
            # Calculate moist static energy (heo) and saturation moist 
            # static energy (heso)
            tem  = phil + cp * to
            heo  = tem + hvap * qo
            heso = tem + hvap * qeso
            


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def init_tracers( cnvflg: FIELD_INT,
                  k_idx : FIELD_INT,
                  kmax  : FIELD_INT,
                  ctr   : FIELD_FLOAT, 
                  ctro  : FIELD_FLOAT, 
                  ecko  : FIELD_FLOAT,
                  qtr   : FIELD_FLOAT ):
    
    from __gtscript__ import PARALLEL, computation, interval
    
    with computation(PARALLEL), interval(...):
        
        # Initialize tracer variables
        if cnvflg == 1 and k_idx <= kmax:
            ctr  = qtr
            ctro = qtr
            ecko = 0.0
            

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"min": min, "max": max, "sqrt": sqrt})
def comp_tendencies( g         : DTYPE_FLOAT,
                     betaw     : DTYPE_FLOAT,
                     dtmin     : DTYPE_FLOAT,
                     dt2       : DTYPE_FLOAT,
                     dtmax     : DTYPE_FLOAT,
                     dxcrt     : DTYPE_FLOAT,
                     cnvflg    : FIELD_INT,
                     k_idx     : FIELD_INT,
                     kmax      : FIELD_INT,
                     kb        : FIELD_INT,
                     ktcon     : FIELD_INT,
                     ktcon1    : FIELD_INT,
                     kbcon1    : FIELD_INT,
                     kbcon     : FIELD_INT,
                     dellah    : FIELD_FLOAT,
                     dellaq    : FIELD_FLOAT,
                     dellau    : FIELD_FLOAT,
                     dellav    : FIELD_FLOAT,
                     del       : FIELD_FLOAT,
                     zi        : FIELD_FLOAT,
                     zi_ktcon1 : FIELD_FLOAT,
                     zi_kbcon1 : FIELD_FLOAT,
                     heo       : FIELD_FLOAT,
                     qo        : FIELD_FLOAT,
                     xlamue    : FIELD_FLOAT,
                     xlamud    : FIELD_FLOAT,
                     eta       : FIELD_FLOAT,
                     hcko      : FIELD_FLOAT,
                     qrcko     : FIELD_FLOAT,
                     uo        : FIELD_FLOAT,
                     ucko      : FIELD_FLOAT,
                     vo        : FIELD_FLOAT,
                     vcko      : FIELD_FLOAT,
                     qcko      : FIELD_FLOAT,
                     dellal    : FIELD_FLOAT,
                     qlko_ktcon: FIELD_FLOAT,
                     wc        : FIELD_FLOAT,
                     gdx       : FIELD_FLOAT,
                     dtconv    : FIELD_FLOAT,
                     u1        : FIELD_FLOAT,
                     v1        : FIELD_FLOAT,
                     po        : FIELD_FLOAT,
                     to        : FIELD_FLOAT,
                     tauadv    : FIELD_FLOAT,
                     xmb       : FIELD_FLOAT,
                     sigmagfm  : FIELD_FLOAT,
                     garea     : FIELD_FLOAT,
                     scaldfunc : FIELD_FLOAT,
                     xmbmax    : FIELD_FLOAT ):
    
    from __gtscript__ import PARALLEL, FORWARD, BACKWARD, computation, interval
    from __externals__ import min, max, sqrt
    
    # Calculate the change in moist static energy, moisture 
    # mixing ratio, and horizontal winds per unit cloud base mass 
    # flux for all layers below cloud top from equations B.14 
    # and B.15 from Grell (1993) \cite grell_1993, and for the 
    # cloud top from B.16 and B.17
    
    # Initialize zi_ktcon1 and zi_kbcon1 fields (propagate forward)
    with computation(FORWARD), interval(...):
            
        if k_idx == ktcon1: 
            zi_ktcon1 = zi
        elif k_idx > 0:
            zi_ktcon1 = zi_ktcon1[0, 0, -1]
            
        if k_idx == kbcon1: 
            zi_kbcon1 = zi
        elif k_idx > 0:
            zi_kbcon1 = zi_kbcon1[0, 0, -1]
    
    # Initialize zi_ktcon1 and zi_kbcon1 fields (propagate backward)    
    with computation(BACKWARD), interval(0, -1):
        
        zi_ktcon1 = zi_ktcon1[0, 0, 1]
        zi_kbcon1 = zi_kbcon1[0, 0, 1]
    
    with computation(PARALLEL), interval(...):
        
        if cnvflg == 1 and k_idx <= kmax:
            dellah = 0.0
            dellaq = 0.0
            dellau = 0.0
            dellav = 0.0
    
    with computation(PARALLEL), interval(1, -1):
        
        # Changes due to subsidence and entrainment
        if cnvflg == 1 and k_idx > kb and k_idx < ktcon:
                
            dp  = 1000.0 * del
            dz  = zi[0, 0, 0] - zi[0, 0, -1]
            gdp = g/dp
            
            dv1h = heo[0, 0,  0]
            dv3h = heo[0, 0, -1]
            dv2h = 0.5 * (dv1h + dv3h)
            
            dv1q = qo[0, 0,  0]
            dv3q = qo[0, 0, -1]
            dv2q = 0.5 * (dv1q + dv3q)
            
            tem  = 0.5 * (xlamue[0, 0, 0] + xlamue[0, 0, -1])
            tem1 = xlamud
            
            eta_curr = eta[0, 0,  0]
            eta_prev = eta[0, 0, -1]
            
            dellah = dellah + ( eta_curr * dv1h - \
                                eta_prev * dv3h - \
                                eta_prev * dv2h * tem * dz + \
                                eta_prev * tem1 * 0.5 * dz * \
                                  (hcko[0, 0, 0] + hcko[0, 0, -1]) ) * gdp
            
            dellaq = dellaq + ( eta_curr * dv1q - \
                                eta_prev * dv3q - \
                                eta_prev * dv2q * tem * dz + \
                                eta_prev * tem1 * 0.5 * dz * \
                                  (qrcko[0, 0, 0] + qrcko[0, 0, -1]) ) * gdp
                                  
            tem1   = eta_curr * (uo[0, 0,  0] - ucko[0, 0,  0])
            tem2   = eta_prev * (uo[0, 0, -1] - ucko[0, 0, -1])
            dellau = dellau + (tem1 - tem2) * gdp
            
            tem1   = eta_curr * (vo[0, 0,  0] - vcko[0, 0,  0])
            tem2   = eta_prev * (vo[0, 0, -1] - vcko[0, 0, -1])
            dellav = dellav + (tem1 - tem2) * gdp
            
    with computation(PARALLEL), interval(...):
        
        # Cloud top
        if cnvflg == 1:
            
            if ktcon == k_idx:
                
                dp   = 1000.0 * del
                gdp  = g/dp
                
                dv1h   = heo[0, 0, -1]
                dellah = eta[0, 0, -1] * (hcko[0, 0, -1] - dv1h) * gdp
                
                dv1q   = qo [0, 0, -1]
                dellaq = eta[0, 0, -1] * (qcko[0, 0, -1] - dv1q) * gdp
                
                dellau = eta[0, 0, -1] * (ucko[0, 0, -1] - uo[0, 0, -1]) * gdp
                dellav = eta[0, 0, -1] * (vcko[0, 0, -1] - vo[0, 0, -1]) * gdp
                
                # Cloud water
                dellal = eta[0, 0, -1] * qlko_ktcon * gdp
                
            # Following Bechtold et al. (2008) \cite 
            # bechtold_et_al_2008, calculate the convective turnover 
            # time using the mean updraft velocity (wc) and the cloud 
            # depth. It is also proportional to the grid size (gdx).
            tem = zi_ktcon1 - zi_kbcon1
            
            tfac   = 1.0 + gdx/75000.0
            dtconv = tfac * tem/wc
            dtconv = max(dtconv, dtmin)
            dtconv = max(dtconv, dt2)
            dtconv = min(dtconv, dtmax)
            
            # Initialize field for advective time scale computation
            sumx  = 0.0
            umean = 0.0
    
    # Calculate advective time scale (tauadv) using a mean cloud layer 
    # wind speed (propagate forward)
    with computation(FORWARD), interval(1, -1):
        
        if cnvflg == 1:
            if k_idx >= kbcon1 and k_idx < ktcon1:
                dz    = zi[0, 0, 0] - zi[0, 0, -1]
                tem   = sqrt(u1*u1 + v1*v1)
                umean = umean[0, 0, -1] + tem * dz
                sumx  = sumx [0, 0, -1] + dz
            else:
                umean = umean[0, 0, -1]
                sumx  = sumx [0, 0, -1]
     
     # Calculate advective time scale (tauadv) using a mean cloud layer 
     # wind speed (propagate backward)           
     with computation(BACKWARD), interval(1, -2):
         
         if cnvflg == 1:
             umean = umean[0, 0, 1]
             sumx  = sumx [0, 0, 1]
            
    with computation(PARALLEL), interval(...):
        
        if cnvflg == 1:
            umean  = umean/sumx
            val    = 1.0
            umean  = max(umean, val)  # Passing literals (e.g. 1.0) to functions might cause errors in conditional statements
            tauadv = gdx/umean
            
            if k_idx == kbcon:
                
                # From Han et al.'s (2017) \cite han_et_al_2017 equation 
                # 6, calculate cloud base mass flux as a function of the 
                # mean updraft velocity
                rho  = po * 100.0 / (rd * to)
                tfac = tauadv/dtconv
                val  = 1.0
                tfac = min(tfac, val)  # Same as above: literals
                xmb  = tfac * betaw * rho * wc
                
                # For scale-aware parameterization, the updraft fraction 
                # (sigmagfm) is first computed as a function of the 
                # lateral entrainment rate at cloud base (see Han et 
                # al.'s (2017) \cite han_et_al_2017 equation 4 and 5), 
                # following the study by Grell and Freitas (2014) \cite 
                # grell_and_freitus_2014
                val1 = 2.0e-4
                val2 = 6.0e-4
                tem  = max(xlamue, val1)
                tem  = 0.2/min(tem, val2)
                tem1 = 3.14 * tem * tem
                
                sigmagfm = tem1/garea
                val3     = 0.001
                val4     = 0.999
                sigmagfm = max(sigmagfm, val3)
                sigmagfm = min(sigmagfm, val4)
            
            # Then, calculate the reduction factor (scaldfunc) of the 
            # vertical convective eddy transport of mass flux as a 
            # function of updraft fraction from the studies by Arakawa 
            # and Wu (2013) \cite arakawa_and_wu_2013 (also see Han et 
            # al.'s (2017) \cite han_et_al_2017 equation 1 and 2). The 
            # final cloud base mass flux with scale-aware 
            # parameterization is obtained from the mass flux when 
            # sigmagfm << 1, multiplied by the reduction factor (Han et 
            # al.'s (2017) \cite han_et_al_2017 equation 2).
            if gdx < dxcrt:
                scaldfunc = (1.0 - sigmagfm) * (1.0 - sigmagfm)
                val1      = 1.0
                val2      = 0.0
                scaldfunc = min(scaldfunc, val1)
                scaldfunc = max(scaldfunc, val2)
            else:
                scaldfunc = 1.0
            
            xmb = xmb * scaldfunc
            xmb = min(xmb, xmbmax)


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def comp_tendencies_tr( g      : DTYPE_FLOAT,
                        cnvflg : FIELD_INT,
                        k_idx  : FIELD_INT,
                        kmax   : FIELD_INT,
                        kb     : FIELD_INT,
                        ktcon  : FIELD_INT,
                        dellae : FIELD_FLOAT,
                        del    : FIELD_FLOAT,
                        eta    : FIELD_FLOAT,
                        ctro   : FIELD_FLOAT,
                        ecko   : FIELD_FLOAT ):
    
    from __gtscript__ import PARALLEL, computation, interval
    
    with computation(PARALLEL), interval(...):
        
        if cnvflg == 1 and k_idx <= kmax:
            
            dellae = 0.0
            
    with computation(PARALLEL), interval(1, -1):
        
        if cnvflg == 1 and k_idx > kb and k_idx < ktcon:
            
            # Changes due to subsidence and entrainment
            dp = 1000.0 * del
            
            tem1 = eta[0, 0,  0] * (ctro[0, 0,  0] - ecko[0, 0,  0])
            tem2 = eta[0, 0, -1] * (ctro[0, 0, -1] - ecko[0, 0, -1])
            
            dellae = dellae + (tem1 - tem2) * g/dp
            
    with computation(PARALLEL), interval(...):
        
        # Cloud top
        if cnvflg == 1:
            if ktcon == k_idx:
                
                dp = 1000.0 * del
                
                dellae = eta[0, 0, -1] * (ecko[0, 0, -1] - ctro[0, 0, -1]) * g/dp
            

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"fpvs": fpvs, "min": min, "max": max})
def feedback_control_update( dt2    : DTYPE_FLOAT,
                             g      : DTYPE_FLOAT,
                             evfact : DTYPE_FLOAT,
                             evfactl: DTYPE_FLOAT,
                             el2orc : DTYPE_FLOAT,
                             elocp  : DTYPE_FLOAT,
                             cnvflg : FIELD_INT,
                             k_idx  : FIELD_INT,
                             kmax   : FIELD_INT,
                             kb     : FIELD_INT,
                             ktcon  : FIELD_INT,
                             flg    : FIELD_INT,
                             islimsk: FIELD_INT,
                             ktop   : FIELD_INT,
                             kbot   : FIELD_INT,
                             kbcon  : FIELD_INT,
                             kcnv   : FIELD_INT,
                             qeso   : FIELD_FLOAT,
                             pfld   : FIELD_FLOAT,
                             delhbar: FIELD_FLOAT,
                             delqbar: FIELD_FLOAT,
                             deltbar: FIELD_FLOAT,
                             delubar: FIELD_FLOAT,
                             delvbar: FIELD_FLOAT,
                             qcond  : FIELD_FLOAT,
                             dellah : FIELD_FLOAT,
                             dellaq : FIELD_FLOAT,
                             t1     : FIELD_FLOAT,
                             xmb    : FIELD_FLOAT,
                             q1     : FIELD_FLOAT,
                             u1     : FIELD_FLOAT,
                             dellau : FIELD_FLOAT,
                             v1     : FIELD_FLOAT,
                             dellav : FIELD_FLOAT,
                             del    : FIELD_FLOAT,
                             rntot  : FIELD_FLOAT,
                             delqev : FIELD_FLOAT,
                             delq2  : FIELD_FLOAT,
                             pwo    : FIELD_FLOAT,
                             deltv  : FIELD_FLOAT,
                             delq   : FIELD_FLOAT,
                             qevap  : FIELD_FLOAT,
                             rn     : FIELD_FLOAT,
                             edt    : FIELD_FLOAT,
                             cnvw   : FIELD_FLOAT,
                             cnvwt  : FIELD_FLOAT,
                             cnvc   : FIELD_FLOAT,
                             ud_mf  : FIELD_FLOAT,
                             dt_mf  : FIELD_FLOAT ):
    
    from __gtscript__ import PARALLEL, FORWARD, BACKWARD, computation, interval                         
    from __externals__ import fpvs, min, max
    
    with computation(PARALLEL), interval(...):
        
        # Initialize flg
        flg = cnvflg
        
        # Recalculate saturation specific humidity
        qeso = 0.01 * fpvs(t1)    # fpvs is in Pa
        qeso = eps * qeso/(pfld + epsm1 * qeso)
        val  = 1.0e-8
        qeso = max(qeso, val)
        
        if cnvflg == 1 and k_idx > kb and k_idx <= ktcon:
                
                # - Calculate the temperature tendency from the moist 
                #   static energy and specific humidity tendencies
                # - Update the temperature, specific humidity, and 
                #   horizontal wind state variables by multiplying the 
                #   cloud base mass flux-normalized tendencies by the 
                #   cloud base mass flux
                dellat = (dellah - hvap * dellaq)/cp
                t1     = t1 + dellat * xmb * dt2
                q1     = q1 + dellaq * xmb * dt2
                u1     = u1 + dellau * xmb * dt2
                v1     = v1 + dellav * xmb * dt2
                
                # Recalculate saturation specific humidity using the 
                # updated temperature
                qeso = 0.01 * fpvs(t1)    # fpvs is in Pa
                qeso = eps * qeso/(pfld + epsm1 * qeso)
                val  = 1.0e-8
                qeso = max(qeso, val)
    
    # Accumulate column-integrated tendencies (propagate forward)
    with computation(FORWARD):
        
        # To avoid conditionals in the full interval
        with interval(0, 1):
            
            if cnvflg == 1 and k_idx > kb and k_idx <= ktcon:
                
                    dp  = 1000.0 * del
                    dpg = dp/g
                    
                    delhbar = delhbar + dellah * xmb * dpg
                    delqbar = delqbar + dellaq * xmb * dpg
                    deltbar = deltbar + dellat * xmb * dpg
                    delubar = delubar + dellau * xmb * dpg
                    delvbar = delvbar + dellav * xmb * dpg
        
        with interval(1, None):
            
            if cnvflg == 1:
                if k_idx > kb and k_idx <= ktcon:
                
                    dp  = 1000.0 * del
                    dpg = dp/g
                    
                    delhbar = delhbar[0, 0, -1] + dellah * xmb * dpg
                    delqbar = delqbar[0, 0, -1] + dellaq * xmb * dpg
                    deltbar = deltbar[0, 0, -1] + dellat * xmb * dpg
                    delubar = delubar[0, 0, -1] + dellau * xmb * dpg
                    delvbar = delvbar[0, 0, -1] + dellav * xmb * dpg
                    
                else:
                    
                    delhbar = delhbar[0, 0, -1]
                    delqbar = delqbar[0, 0, -1]
                    deltbar = deltbar[0, 0, -1]
                    delubar = delubar[0, 0, -1]
                    delvbar = delvbar[0, 0, -1]
        
    with computation(BACKWARD):
        
        # To avoid conditionals in the full interval
        with interval(-1, None):
            
            rntot = rntot + pwo * xmb * 0.001 * dt2
        
        with interval(0, -1):
            if cnvflg == 1:
                
                # Accumulate column-integrated tendencies (propagate backward)
                delhbar = delhbar[0, 0, -1]
                delqbar = delqbar[0, 0, -1]
                deltbar = deltbar[0, 0, -1]
                delubar = delubar[0, 0, -1]
                delvbar = delvbar[0, 0, -1]
                
                # Add up column-integrated convective precipitation by 
                # multiplying the normalized value by the cloud base 
                # mass flux (propagate backward)
                if k_idx > kb and k_idx < ktcon:
                
                    rntot = rntot[0, 0, 1] + pwo * xmb * 0.001 * dt2
                
                else:
                    
                    rntot = rntot[0, 0, 1]
    
    # Add up column-integrated convective precipitation by 
    # multiplying the normalized value by the cloud base 
    # mass flux (propagate forward)                
    with computation(FORWARD), interval(1, None):
        
        if cnvflg == 1:
            rntot = rntot[0, 0, -1]
    
    # - Determine the evaporation of the convective precipitation 
    #   and update the integrated convective precipitation
    # - Update state temperature and moisture to account for 
    #   evaporation of convective precipitation
    # - Update column-integrated tendencies to account for 
    #   evaporation of convective precipitation
    
    # TODO:
    # ~ with computation(BACKWARD), interval(...):
    
        # ~ if k_idx <= kmax:
            
            # ~ deltv = 0.0
            # ~ delq  = 0.0
            # ~ qevap = 0.0
            
            # ~ if cnvflg == 1 and k_idx > kb and k_idx < ktcon:
                
                # ~ rn = rn + pwo * xmb * 0.001 * dt2
                
    
    with computation(PARALLEL), interval(...)
        
        if cnvflg == 1:
            
            if rn < 0.0 or flg == 0: rn = 0.0
           
            ktop = ktcon
            kbot = kbcon
            kcnv = 2
            
            if k_idx >= kbcon and k < ktcon:
                
                # Calculate shallow convective cloud water
                cnvw = cnvwt * xmb * dt2
                
                # Calculate convective cloud cover, which is used when 
                # pdf-based cloud fraction is used
                val1 = 1.0 + 675.0 * eta * xmb
                cnvc = 0.04 * log(val1)    # How to implement log?
                val2 = 0.2
                val3 = 0.0
                cnvc = min(cnvc, val2)
                cnvc = max(cnvc, val3)
                
            # Calculate the updraft convective mass flux
            if k_idx >= kb and k_idx < ktop:
                ud_mf = eta * xmb * dt2
            
            # Save the updraft convective mass flux at cloud top
            if k_idx == ktop - 1:
                dt_mf = ud_mf
                

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def feedback_control_upd_trr( dt2    : DTYPE_FLOAT,
                              g      : DTYPE_FLOAT,
                              cnvflg : FIELD_INT,
                              k_idx  : FIELD_INT,
                              kmax   : FIELD_INT,
                              ktcon  : FIELD_INT,
                              delebar: FIELD_FLOAT,
                              ctr    : FIELD_FLOAT,
                              dellae : FIELD_FLOAT,
                              xmb    : FIELD_FLOAT,
                              qtr    : FIELD_FLOAT ):
                                  
    from __gtscript__ import PARALLEL, FORWARD, BACKWARD, computation, interval 
    
    with computation(PARALLEL), interval(...)
        delebar = 0.0
        
        if cnvflg == 1 and k_idx <= kmax and k_idx <= ktcon:
            
            ctr = ctr + dellae * xmb * dt2
            qtr = ctr
            
    # Propagate forward delebar values
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if cnvflg == 1 and k_idx <= kmax and k_idx <= ktcon:
                delebar = delebar + dellae * xmb * dp/g    # Where does dp come from? Is it correct to use the last value at line 1559 of samfshalcnv.F?
                
        with interval(1, None):
            
            if cnvflg == 1:
                if k_idx <= kmax and k_idx <= kcon:
                    delebar = delebar[0, 0, -1] + dellae * xmb * dp/g
                else:
                    delebar = delebar[0, 0, -1] + dellae * xmb * dp/g
    
    # Propagate backward delebar values                
    with computation(BACKWARD), interval(0, -1):
        
        if cnvflg == 1:
            delebar = delebar[0, 0, 1]
        

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def store_aero_conc( cnvflg: FIELD_INT,
                     k_idx : FIELD_INT,
                     kmax  : FIELD_INT,
                     rn    : FIELD_FLOAT,
                     qtr   : FIELD_FLOAT,
                     qaero : FIELD_FLOAT ):
                         
    from __gtscript__ import PARALLEL, computation, interval 
    
    with computation(PARALLEL), interval(...):
        
        # Store aerosol concentrations if present
        if cnvflg == 1 and rn > 0.0 and k <= kmax:
            qtr = qaero
            

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"max": max, "min": min})
def separate_detrained_cw( dt2   : DTYPE_FLOAT,
                           tcr   : DTYPE_FLOAT,
                           tcrf  : DTYPE_FLOAT,
                           cnvflg: FIELD_INT,
                           k_idx : FIELD_INT,
                           kbcon : FIELD_INT,
                           ktcon : FIELD_INT,
                           dellal: FIELD_FLOAT,
                           xmb   : FIELD_FLOAT,
                           t1    : FIELD_FLOAT,
                           qtr_1 : FIELD_FLOAT,
                           qtr_0 : FIELD_FLOAT ):
                               
    from __gtscript__ import PARALLEL, computation, interval 
    
    with computation(PARALLEL), interval(...):
        
        # Separate detrained cloud water into liquid and ice species as 
        # a function of temperature only
        if cnvflg == 1 and k_idx >= kbcon and k_idx <= ktcon:
            
            tem  = dellal * xmb * dt2
            val1 = 1.0
            val2 = 0.0
            tem1 = min(val1, (tcr - t1) * tcrf)
            tem1 = max(val2, tem1)
            
            if qtr_1 > -999.0:
                qtr_0 = qtr_0 + tem * tem1
                qtr_1 = qtr_1 + tem * (1.0 - tem1)
            else:
                qtr_0 = qtr_0 + tem
                

@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, externals={"max": max})
def tke_contribution( betaw   : DTYPE_FLOAT,
                      cnvflg  : FIELD_INT,
                      k_idx   : FIELD_INT,
                      kb      : FIELD_INT,
                      ktop    : FIELD_INT,
                      eta     : FIELD_FLOAT,
                      xmb     : FIELD_FLOAT,
                      pfld    : FIELD_FLOAT,
                      t1      : FIELD_FLOAT, 
                      sigmagfm: FIELD_FLOAT,
                      qtr_ntk : FIELD_FLOAT ):
                          
    from __gtscript__ import PARALLEL, computation, interval 
    
    with computation(PARALLEL), interval(1, -1):
        
        # Include TKE contribution from shallow convection
        if cnvflg == 1 and k_idx > kb and k < ktop:
            
            tem      = 0.5 * (eta[0, 0, -1] + eta[0, 0, 0]) * xmb
            tem1     = pfld * 100.0/(rd * t1)
            sigmagfm = max(sigmagfm, betaw)
            ptem     = tem/(sigmagfm * tem1)
            qtr_ntk  = qtr_ntk + 0.5 * sigmagfm * ptem * ptem

########################################################################


########################### USEFUL FUNCTIONS ###########################
# These should be moved in a separate file to avoid cluttering and be 
# reused in other places!
########################################################################

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
    
    
def exit_routine(cnvflg, im):
    totflg = True
    for i in range(0, im):
        totflg = totflg and cnvflg[0, i, 0] == 0
        
    return totflg
    
########################################################################

import pytest
import gt4py as gt
from gt4py import gtscript
#import sys
#sys.path.append("..")
from shalconv.kernels.stencils_part1 import *
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

def samfshalcnv_part1(data_dict):
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

    shape = (1, ix, km)
    
    ### Local storages for 1D arrays (integer) ###
    kpbl       = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kb         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon1     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon      = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon1     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktconn     = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbm        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kmax       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
    ### Local storages for 1D arrays ("bool") ###
    cnvflg     = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    flg        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
    ### Local storages for 1D arrays (float) ###
    aa1        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cina       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tkemean    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    clamt      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ps         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    del0       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    prsl       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    umean      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tauadv     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    gdx        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delhbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq2      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqev     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltv      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dtconv     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    edt        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pdot       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    po         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcond      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qevap      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hmax       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    rntot      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vshear     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamud     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmb        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmbmax     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delubar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delvbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wc         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    scaldfunc  = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sigmagfm   = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qlko_ktcon = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sumx       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tx1        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_ktcon1  = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_kbcon1  = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 2D arrays (float) ###
    pfld       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    to         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    uo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qeso       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wu2        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    buo        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    drag       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellal     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dbyo       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamue     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heo        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heso       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellah     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellaq     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellau     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellav     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hcko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ucko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vcko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qrcko      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    eta        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pwo        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0t        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cnvwt      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 2D arrays (float, tracers), this will contain slices along n-axis ###
    delebar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 3D arrays (float, tracers), this will contain slices along n-axis ###
    ctr        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ctro       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellae     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ecko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qaero      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### K-indices field ###
    k_idx      = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)

    ### State buffer for 1D-2D interactions
    state_buf1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    state_buf2 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
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
    bet1       = 1.875
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
    pa_to_cb( psp, prslp, delp, ps, prsl, del0)
    
    km1 = km - 1
    
    ### Initialize storages (simple initializations already done above with gt4py functionalities) ###
    
    # Initialize column-integrated and other single-value-per-column 
    # variable arrays
    init_col_arr( km, kcnv, cnvflg, kbot, ktop,
                  kbcon, kb, rn, gdx, garea)
    
    # Return to the calling routine if deep convection is present or the 
    # surface buoyancy flux is negative
    if exit_routine(cnvflg, im): return
    
    # Initialize further parameters and arrays
    init_par_and_arr( c0s, asolfac, d0, islimsk, c0,
                      t1, c0t, cnvw, cnvc, ud_mf, dt_mf)
                      
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
    init_kbm_kmax(km, kbm, k_idx, kmax, state_buf1, state_buf2, tx1, ps, prsl)
    init_final( km, kbm, k_idx, kmax, flg, cnvflg, kpbl, tx1, 
                ps, prsl, zo, phil, zi, pfld, eta, hcko, qcko, 
                qrcko, ucko, vcko, dbyo, pwo, dellal, to, qo, 
                uo, vo, wu2, buo, drag, cnvwt, qeso, heo, heso, hpbl,
                t1, q1, u1, v1)
                
    
    # Tracers loop (THIS GOES AT THE END AND POSSIBLY MERGED WITH OTHER 
    # TRACER LOOPS!) --> better use version below
    for n in range(2, ntr+2):
        
        #kk = n-2
        
        qtr_shift = gt.storage.from_array(slice_to_3d(qtr[:, :, n]), BACKEND, default_origin)
        
        # Initialize tracers. Keep in mind that, qtr slice is for the 
        # n-th tracer, while the other storages are slices representing 
        # the (n-2)-th tracer.
        init_tracers( cnvflg, k_idx, kmax, ctr, ctro, ecko, qtr_shift)
    return heo, heso, qo, qeso

def call_fort_part1(fort_fun, data_dict):
    im = data_dict["im"]
    ix = data_dict["ix"]
    km = data_dict["km"]
    delt = data_dict["delt"]
    itc = data_dict["itc"]
    ntc = data_dict["ntc"]
    ntk = data_dict["ntk"]
    ntr = data_dict["ntr"]
    delp = data_dict["delp"]
    prslp = data_dict["prslp"]
    psp = data_dict["psp"]
    phil = data_dict["phil"]
    qtr = data_dict["qtr"]
    q1 = data_dict["q1"]
    t1 = data_dict["t1"]
    u1 = data_dict["u1"]
    v1 = data_dict["v1"]
    fscav = data_dict["fscav"]
    rn = data_dict["rn"]
    kbot = data_dict["kbot"]
    ktop = data_dict["ktop"]
    kcnv = data_dict["kcnv"]
    islimsk = data_dict["islimsk"]
    garea = data_dict["garea"]
    dot = data_dict["dot"]
    ncloud = data_dict["ncloud"]
    hpbl = data_dict["hpbl"]
    ud_mf = data_dict["ud_mf"]
    dt_mf = data_dict["dt_mf"]
    cnvw = data_dict["cnvw"]
    cnvc = data_dict["cnvc"]
    clam = data_dict["clam"]
    c0s = data_dict["c0s"]
    c1 = data_dict["c1"]
    pgcon = data_dict["pgcon"]
    asolfac = data_dict["asolfac"]
    shape2d = (im, km)
    #heo = np.zeros(shape2d)
    #heso = np.zeros(shape2d)
    #qo = np.zeros(shape2d)
    #qeso = np.zeros(shape2d)
    heo, heso, qo, qeso = fort_fun(im = im, ix = ix, km = km, delt = delt, itc = itc,
                                   ntc = ntc, ntk = ntk, ntr = ntr, delp = delp,
                                   prslp = prslp, psp = psp, phil = phil, qtr = qtr[:,:,:ntr+2],
                                   q1 = q1, t1 = t1, u1 = u1, v1 = v1, fscav = fscav,
                                   rn = rn, kbot = kbot, ktop = ktop, kcnv = kcnv,
                                   islimsk = islimsk, garea = garea, dot = dot,
                                   ncloud = ncloud, hpbl = hpbl, ud_mf = ud_mf,
                                   dt_mf = dt_mf, cnvw = cnvw, cnvc = cnvc, clam = clam,
                                   c0s = c0s, c1 = c1, pgcon = pgcon, asolfac = asolfac)
    return heo, heso, qo, qeso


def test_part1():
    data_dict = read_data(0,"in", path = "/data")
    gt4py_dict = numpy_dict_to_gt4py_dict(data_dict)
    heo, heso, qo, qeso = samfshalcnv_part1(gt4py_dict)
    import numpy.f2py, os
    os.system("f2py --f2cmap fortran/.f2py_f2cmap -c -m part1 fortran/part1.f90")
    #numpy.f2py.run_main(['--f2cmap', 'fortran/.f2py_f2cmap', '-c', '-m', 'part1', 'fortran/part1.f90'])
    import part1
    from shalconv.funcphys import fpvsx
    t = 20.0
    assert abs(part1.mod.fpvsx(t) - fpvsx(t)) < 1e-6, "Fortran impl and numpy impl don't match!"
    heo_np, heso_np, qo_np, qeso_np = call_fort_part1(part1.mod.part1, data_dict)
    compare_data({"heo":heo.view(np.ndarray)[0,:,:],"heso":heso.view(np.ndarray)[0,:,:],"qo":qo.view(np.ndarray)[0,:,:],"qeso":qeso.view(np.ndarray)[0,:,:]},
                 {"heo":heo_np,"heso":heso_np,"qo":qo_np,"qeso":qeso_np})

if __name__ == "__main__":
    test_part1()

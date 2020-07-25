import gt4py as gt
from gt4py import gtscript
import numpy as np

backend = "numpy"  # "debug", "numpy", "gtx86", "gtmc", "gtcuda" (not working)
backend_opts = {'verbose': True} if backend.startswith('gt') else {}
dtype = np.float64
rebuild = False

## need write fpvs
@gtscript.function
def fpvs(x):
    return x

externals = {
    "fpvs":fpvs
}

@gtscript.stencil(backend=backend,externals=externals, verbose=True)
def stencil_static0(
    cnvflg: gtscript.Field[dtype],
    hmax: gtscript.Field[dtype],
    heo: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kpbl: gtscript.Field[dtype],
    kmax: gtscript.Field[dtype],
    dz: gtscript.Field[dtype],
    zo: gtscript.Field[dtype],
    dp: gtscript.Field[dtype],
    es: gtscript.Field[dtype],
    to: gtscript.Field[dtype],
    pprime: gtscript.Field[dtype],
    epsm1: gtscript.Field[dtype],
    eps: gtscript.Field[dtype],
    qs: gtscript.Field[dtype],
    dqsdp: gtscript.Field[dtype],
    desdt: gtscript.Field[dtype],
    dqsdt: gtscript.Field[dtype],
    fact1: gtscript.Field[dtype],
    fact2: gtscript.Field[dtype],
    gamma: gtscript.Field[dtype],
    el2orc: gtscript.Field[dtype],
    qeso: gtscript.Field[dtype],
    g: gtscript.Field[dtype],
    hvap: gtscript.Field[dtype],
    cp: gtscript.Field[dtype],
    dt: gtscript.Field[dtype],
    dq: gtscript.Field[dtype],
    qo: gtscript.Field[dtype],
    po: gtscript.Field[dtype],
    uo: gtscript.Field[dtype],
    vo: gtscript.Field[dtype],
    heso: gtscript.Field[dtype],
    pfld: gtscript.Field[dtype]
):
    """
    Scale-Aware Mass-Flux Shallow Convection
    :to use the k[1,0:im,0:km] as storage of 1 to k index.
    """
    with computation(PARALLEL), interval(...):
        if cnvflg[0,0,0]:
            hmax=heo[0,0,0]
            kb=1

    with computation(FORWARD), interval(1,None):
        hmax = hmax[0,0,-1]
        if (cnvflg[0,0,0] and k[0,0,0] <= kpbl[0,0,0]):
                if(heo[0,0,0] > hmax[0,0,0]):
                    kb   = k[0,0,0]
                    hmax = heo[0,0,0]
# to make all slice like the final slice    
    with computation(BACKWARD), interval(0,-1):
        kb = kb[0,0,1]
        hmax = hmax[0,0,1]

    with computation(FORWARD), interval(0,-1):
        tmp = fpvs(to[0,0,1])
        if (cnvflg[0,0,0] and k[0,0,0] <= kmax[0,0,0]-1):
            dz      = .5 * (zo[0,0,1] - zo[0,0,0])
            dp      = .5 * (pfld[0,0,1] - pfld[0,0,0])
            es      = 0.01 * tmp     # fpvs is in pa
            pprime  = pfld[0,0,1] + epsm1 * es
            qs      = eps * es / pprime
            dqsdp   = - qs / pprime
            desdt   = es * (fact1 / to[0,0,1] + fact2 / (to[0,0,1]**2))
            dqsdt   = qs * pfld[0,0,1] * desdt / (es * pprime)
            gamma   = el2orc * qeso[0,0,1] / (to[0,0,1]**2)
            dt      = (g * dz + hvap * dqsdp * dp) / (cp * (1. + gamma))
            dq      = dqsdt * dt + dqsdp * dp
            to = to[0,0,1] + dt
            qo = qo[0,0,1] + dq
            po = .5 * (pfld[0,0,0] + pfld[0,0,1])
    
    with computation(FORWARD), interval(0,-1):
        tmp = fpvs(to)
        if (cnvflg[0,0,0] and k[0,0,0] <= kmax[0,0,0]-1):
            qeso = 0.01 * tmp     # fpvs is in pa
            qeso = eps * qeso[0,0,0] / (po[0,0,0] + epsm1*qeso[0,0,0])
#            val1      =    1.e-8         
            qeso = qeso[0,0,0] if (qeso[0,0,0]>1.e-8) else 1.e-8
#            val2      =    1.e-10        
            qo   = qo[0,0,0] if (qo[0,0,0]>1.e-10) else 1.e-10
#           qo   = min(qo[0,0,0],qeso[0,0,0])
            heo  = .5 * g * (zo[0,0,0] + zo[0,0,1]) + \
                    cp * to[0,0,0] + hvap * qo[0,0,0]
            heso = .5 * g * (zo[0,0,0] + zo[0,0,1]) + \
                    cp * to[0,0,0] + hvap * qeso[0,0,0]
            uo   = .5 * (uo[0,0,0] + uo[0,0,1])
            vo   = .5 * (vo[0,0,0] + vo[0,0,1])



## ntr stencil

@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_ntrstatic0(
    cnvflg: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kmax: gtscript.Field[dtype],
    ctro: gtscript.Field[dtype]
):
    with computation(FORWARD), interval(0,-1):
        if(cnvflg and k <= (kmax-1)):
            ctro = .5 * (ctro + ctro[0,0,1])


@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static1(
    cnvflg: gtscript.Field[dtype],
    flg: gtscript.Field[dtype],
    kbcon: gtscript.Field[dtype],
    kmax: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kbm: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    heo_kb: gtscript.Field[dtype],
    heso: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(...):
        flg = cnvflg
        if(flg):
            kbcon = kmax
    
    with computation(FORWARD), interval(...):
        if(k != 1):
            heo_kb = heo_kb[0,0,-1]
        if(k == kb):
            heo_kb = heo
    
    with computation(BACKWARD), interval(0,-1):
        heo_kb = heo_kb[0,0,1]

    with computation(FORWARD), interval(1,-1):
        kbcon = kbcon[0,0,-1]
        flg = flg[0,0,-1]
        if (flg and k < kbm):
#to use heo_kb to represent heo(i,kb(i))
            if(k[0,0,0] > kb[0,0,0] and heo_kb > heso[0,0,0]):
                kbcon = k
                flg   = False

## to make all slice like the final slice
    with computation(FORWARD), interval(-1,None):
        kbcon = kbcon[0,0,-1]
        flg = flg[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        kbcon = kbcon[0,0,1]
        flg = flg[0,0,1]
            
    with computation(PARALLEL), interval(...):
        if(cnvflg):
            if(kbcon == kmax):
                cnvflg = False
    
## judge LFC and return 553-558

@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static2(
    cnvflg: gtscript.Field[dtype],
    pdot: gtscript.Field[dtype],
    dotkbcon: gtscript.Field[dtype],
    islimsk: gtscript.Field[dtype],
    w1: gtscript.Field[dtype],
    w2: gtscript.Field[dtype],
    w3: gtscript.Field[dtype],
    w4: gtscript.Field[dtype],
    w1l: gtscript.Field[dtype],
    w2l: gtscript.Field[dtype],
    w3l: gtscript.Field[dtype],
    w4l: gtscript.Field[dtype],
    w1s: gtscript.Field[dtype],
    w2s: gtscript.Field[dtype],
    w3s: gtscript.Field[dtype],
    w4s: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    ptem: gtscript.Field[dtype],
    ptem1: gtscript.Field[dtype],
    cinpcr: gtscript.Field[dtype],
    cinpcrmx: gtscript.Field[dtype],
    cinpcrmn: gtscript.Field[dtype],
    tem1: gtscript.Field[dtype],
    pfld_kb: gtscript.Field[dtype],
    pfld_kbcon: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(...):
        if(cnvflg):
#to use dotkbcon to represent dot(i,kbcon(i))
#            pdot(i)  = 10.* dotkbcon
            pdot[0,0,0]  = 0.01 * dotkbcon # Now dot is in Pa/s

    with computation(FORWARD), interval(...):
        if(k != 1):
            pfld_kb = pfld_kb[0,0,-1]
            pfld_kbcon = pfld_kbcon[0,0,-1]
        if(k == kb):
            pfld_kb = pfld
        if(k == kbcon):
            pfld_kbcon = pfld
    
    with computation(BACKWARD), interval(0,-1):
        pfld_kb = pfld_kb[0,0,1]
        pfld_kbcon = pfld_kbcon[0,0,1]

    with computation(PARALLEL), interval(...):
        if(cnvflg):
            if(islimsk == 1):
                w1 = w1l # require constant to be also 3d shape
                w2 = w2l
                w3 = w3l
                w4 = w4l
            else:
                w1 = w1s
                w2 = w2s
                w3 = w3s
                w4 = w4s
            if(pdot <= w4):
                tem = (pdot - w4) / (w3 - w4)
            elif(pdot >= -w4):
                tem = - (pdot + w4) / (w4 - w3)
            else:
                tem = 0.
            tem = tem if (tem>-1) else -1
            tem = tem if (tem<1) else 1
            ptem = 1. - tem
            ptem1= .5*(cinpcrmx-cinpcrmn)
            cinpcr = cinpcrmx - ptem * ptem1
#to use pfld_kb and pfld_kbcon to represent pfld(i,kb(i))
            tem1 = pfld_kb - pfld_kbcon
            if(tem1 > cinpcr):
                cnvflg = False

## do totflg judgement and return
## if ntk >0 : also need to define ntk dimension to 1
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static3(
    sumx: gtscript.Field[dtype],
    tkemean: gtscript.Field[dtype],
    cnvflg: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    kbcon: gtscript.Field[dtype],
    dz: gtscript.Field[dtype],
    zo: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    qtr: gtscript.Field[dtype],
    tkemn: gtscript.Field[dtype],
    tkemx: gtscript.Field[dtype],
    clamt: gtscript.Field[dtype],
    clam: gtscript.Field[dtype],
    clamd: gtscript.Field[dtype],
    tem1: gtscript.Field[dtype],
    dtke: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(...):
        sumx = 0.
        tkemean = 0.
    
    with computation(FORWARD), interval(0,-1):
        if(cnvflg):
            if(k >= kb and k < kbcon):
                dz = zo[0,0,1] - zo[0,0,0]
                tem = 0.5 * (qtr[0,0,0]+qtr[0,0,1])
                tkemean = tkemean + tem * dz #dz, tem to be 3d
                sumx = sumx + dz

    with computation(PARALLEL), interval(...):
        if(cnvflg):
            tkemean = tkemean / sumx
            if(tkemean > tkemx): #tkemx, clam, clamd, tkemnm, dtke to be 3d
               clamt = clam + clamd 
            elif(tkemean < tkemn):
               clamt = clam - clamd
            else:
               tem = tkemx - tkemean
               tem1 = 1. - 2. *  tem / dtke
               clamt = clam + clamd * tem1

## else :
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static4(
    cnvflg: gtscript.Field[dtype],
    clamt: gtscript.Field[dtype],
    clam: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        if(cnvflg):
            clamt  = clam
##
## start updraft entrainment rate.
## pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static5(
    cnvflg: gtscript.Field[dtype],
    xlamue: gtscript.Field[dtype],
    clamt: gtscript.Field[dtype],
    zi: gtscript.Field[dtype],
    xlamud: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kbcon: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    dz: gtscript.Field[dtype],
    ptem: gtscript.Field[dtype],
    eta: gtscript.Field[dtype],
    ktconn: gtscript.Field[dtype],
    kmax: gtscript.Field[dtype],
    kbm: gtscript.Field[dtype],
    hcko: gtscript.Field[dtype],
    ucko: gtscript.Field[dtype],
    vcko: gtscript.Field[dtype],
    heo: gtscript.Field[dtype],
    uo: gtscript.Field[dtype],
    vo: gtscript.Field[dtype]
):
    with computation(FORWARD), interval(0,-1):
        if(cnvflg):
            xlamue = clamt / zi
    
    with computation(BACKWARD), interval(-1,None):
        if(cnvflg):
            xlamue[0,0,0] = xlamue[0,0,-1]
    
    with computation(PARALLEL), interval(...):
        if(cnvflg):
#           xlamud(i) = xlamue(i,kbcon(i))
#           xlamud(i) = crtlamd
            xlamud = 0.001 * clamt

    with computation(BACKWARD), interval(0,-1):
        if (cnvflg):
            if( k < kbcon and k >= kb):
                dz    = zi[0,0,1] - zi[0,0,0]
                ptem  = 0.5*(xlamue[0,0,0]+xlamue[0,0,1])-xlamud[0,0,0]
                eta   = eta[0,0,1] / (1. + ptem[0,0,0] * dz[0,0,0])
    
    with computation(PARALLEL), interval(...):
        flg = cnvflg
    
    with computation(FORWARD), interval(1,-1):
        flg = flg[0,0,-1]
        kmax = kmax[0,0,-1]
        ktconn = ktconn[0,0,-1]
        kbm = kbm[0,0,-1]
        if(flg):
            if(k > kbcon and k < kmax):
                dz       = zi[0,0,0] - zi[0,0,-1]
                ptem     = 0.5*(xlamue[0,0,0]+xlamue[0,0,-1])-xlamud[0,0,0]
                eta = eta[0,0,-1] * (1 + ptem * dz)
                if(eta <= 0.):
                    kmax = k
                    ktconn = k
                    kbm = kbm if (kbm<kmax) else kmax
                    flg = False
 ## to make all slice same as final slice
    with computation(FORWARD), interval(-1,None):
        flg = flg[0,0,-1]
        kmax = kmax[0,0,-1]
        ktconn = ktconn[0,0,-1]
        kbm = kbm[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        flg = flg[0,0,1]
        kmax = kmax[0,0,1]
        ktconn = ktconn[0,0,1]
        kbm = kbm[0,0,1]
    
    with computation(PARALLEL), interval(...):
        if(cnvflg):
          #indx         = kb
          if(k==kb):
            hcko = heo
            ucko = uo
            vcko = vo

## for tracers do n = 1, ntr: use ecko, ctro [n] => [1,i,k]
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_ntrstatic1(
    cnvflg: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    ecko: gtscript.Field[dtype],
    ctro: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(...):
        if(cnvflg and k == kb):
            ecko = ctro


## not pass
# @gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
# def stencil_static6(
#     cnvflg: gtscript.Field[dtype],
#     k: gtscript.Field[dtype],
#     kb: gtscript.Field[dtype],
#     ecko: gtscript.Field[dtype],
#     ctro: gtscript.Field[dtype]
# ):
#     with computation(PARALLEL), interval(...):
#         if(cnvflg):
#             #indx = kb(i)
#             if(k==kb):
#                 ecko = ctro

## end do

## Line 769
## Calculate the cloud properties as a parcel ascends, modified by entrainment and detrainment. Discretization follows Appendix B of Grell (1993) \cite grell_1993 . Following Han and Pan (2006) \cite han_and_pan_2006, the convective momentum transport is reduced by the convection-induced pressure gradient force by the constant "pgcon", currently set to 0.55 after Zhang and Wu (2003) \cite zhang_and_wu_2003 .
## pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static7(
    cnvflg: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    kmax: gtscript.Field[dtype],
    dz: gtscript.Field[dtype],
    zi: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    xlamue: gtscript.Field[dtype],
    tem1: gtscript.Field[dtype],
    xlamud: gtscript.Field[dtype],
    factor: gtscript.Field[dtype],
    hcko: gtscript.Field[dtype],
    heo: gtscript.Field[dtype],
    dbyo: gtscript.Field[dtype],
    heso: gtscript.Field[dtype],
    cm: gtscript.Field[dtype],
    ptem: gtscript.Field[dtype],
    pgcon: gtscript.Field[dtype],
    ptem1: gtscript.Field[dtype],
    ucko: gtscript.Field[dtype],
    uo: gtscript.Field[dtype],
    vcko: gtscript.Field[dtype],
    vo: gtscript.Field[dtype]
):
    with computation(FORWARD), interval(1,-1):
        if(cnvflg):
            if(k > kb and k < kmax):
                dz   = zi[0,0,0] - zi[0,0,-1]
                tem  = 0.5 * (xlamue[0,0,0]+xlamue[0,0,-1]) * dz[0,0,0]
                tem1 = 0.5 * xlamud * dz
                factor = 1. + tem - tem1
                hcko = ((1.-tem1)*hcko[0,0,-1]+tem*0.5* \
                            (heo+heo[0,0,-1]))/factor
                dbyo = hcko - heso
#
                tem  = 0.5 * cm * tem
                factor = 1. + tem
                ptem = tem + pgcon
                ptem1= tem - pgcon
                ucko = ((1.-tem)*ucko[0,0,-1]+ptem*uo \
                            +ptem1*uo[0,0,-1])/factor
                vcko = ((1.-tem)*vcko[0,0,-1]+ptem*vo \
                            +ptem1*vo[0,0,-1])/factor

## for n = 1, ntr:
## pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_ntrstatic2(
    cnvflg: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    kmax: gtscript.Field[dtype],
    zi: gtscript.Field[dtype],
    dz: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    xlamue: gtscript.Field[dtype],
    factor: gtscript.Field[dtype],
    ecko: gtscript.Field[dtype],
    ctro: gtscript.Field[dtype]
):
    with computation(FORWARD), interval(1,-1):
        if (cnvflg):
            if(k > kb and k < kmax):
                dz   = zi - zi[0,0,-1]
                tem  = 0.25 * (xlamue+xlamue[0,0,-1]) * dz
                factor = 1. + tem
                ecko = ((1.-tem)*ecko[0,0,-1]+tem* \
                                (ctro+ctro[0,0,-1]))/factor
## enddo 
## not pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static9(
    cnvflg: gtscript.Field[dtype],
    flg: gtscript.Field[dtype],
    kbcon1: gtscript.Field[dtype],
    kmax: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kbm: gtscript.Field[dtype],
    kbcon: gtscript.Field[dtype],
    dbyo: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    dthk: gtscript.Field[dtype],
    pfld_kbcon: gtscript.Field[dtype],
    pfld_kbcon1: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        flg = cnvflg
        kbcon1 = kmax

    with computation(FORWARD), interval(1,-1):
        flg = flg[0,0,-1]
        if (flg and k < kbm):
            if(k >= kbcon and dbyo > 0.):
                kbcon1 = k
                flg    = False

## to make all slice like the final slice
    with computation(FORWARD), interval(-1,None):
        flg = flg[0,0,-1]
        kbcon1 = kbcon1[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        flg = flg[0,0,1]
        kbcon1 = kbcon1[0,0,1]


    with computation(PARALLEL),interval(...):
        if(cnvflg):
            if(kbcon1 == kmax):
                cnvflg = False

    with computation(FORWARD), interval(...):
        if(k != 1):
            pfld_kbcon = pfld_kbcon[0,0,-1]
            pfld_kbcon1 = pfld_kbcon1[0,0,-1]
        if(k == kbcon):
            pfld_kbcon = pfld
        if(k == kbcon1):
            pfld_kbcon1 = pfld
    
    with computation(BACKWARD), interval(0,-1):
        pfld_kbcon = pfld_kbcon[0,0,1]
        pfld_kbcon1 = pfld_kbcon1[0,0,1]

    with computation(PARALLEL),interval(...):
        if(cnvflg):
# use pfld_kbcon and pfld_kbcon1 to represent
#           tem = pfld(i,kbcon(i)) - pfld(i,kbcon1(i))
            tem = pfld_kbcon - pfld_kbcon1
            if(tem > dthk):
                    cnvflg = False

## judge totflg return

## calculate convective inhibition
## not pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static10(
    cina: gtscript.Field[dtype],
    cnvflg: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    kbcon1: gtscript.Field[dtype],
    dz1: gtscript.Field[dtype],
    zo: gtscript.Field[dtype],
    gamma: gtscript.Field[dtype],
    g: gtscript.Field[dtype],
    el2orc: gtscript.Field[dtype],
    qeso: gtscript.Field[dtype],
    to: gtscript.Field[dtype],
    rfact: gtscript.Field[dtype],
    delta: gtscript.Field[dtype],
    cp: gtscript.Field[dtype],
    hvap: gtscript.Field[dtype],
    dbyo: gtscript.Field[dtype],
    qo: gtscript.Field[dtype],
    w1: gtscript.Field[dtype],
    w2: gtscript.Field[dtype],
    w3: gtscript.Field[dtype],
    w4: gtscript.Field[dtype],
    w1l: gtscript.Field[dtype],
    w2l: gtscript.Field[dtype],
    w3l: gtscript.Field[dtype],
    w4l: gtscript.Field[dtype],
    w1s: gtscript.Field[dtype],
    w2s: gtscript.Field[dtype],
    w3s: gtscript.Field[dtype],
    w4s: gtscript.Field[dtype],
    pdot: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    tem1: gtscript.Field[dtype],
    cinacrmx: gtscript.Field[dtype],
    cinacrmn: gtscript.Field[dtype],
    cinacr: gtscript.Field[dtype],
    *,
    islimsk: dtype
):
    with computation(FORWARD), interval(1,-1):
        cina = cina[0,0,-1]
        if (cnvflg):
            if(k > kb and k < kbcon1):
                dz1 = zo[0,0,1] - zo
                gamma = el2orc * qeso / (to**2)
                rfact =  1. + delta * cp * gamma \
                        * to / hvap
                cina = cina + dz1 * (g / (cp * to)) \
                        * dbyo / (1. + gamma) \
                        * rfact 
#               val = 0.
                cina = (cina + \
#    &                 dz1 * eta(i,k) * g * delta *
                        dz1 * g * delta * \
                        (qeso - qo)) if ((qeso - qo)>0.) else cina

## to make all slices like the final slice    
    with computation(FORWARD), interval(-1,None):
        cina = cina[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        cina = cina[0,0,1]

    with computation(PARALLEL), interval(...):
        if(cnvflg):
#
            if(islimsk == 1):
                w1 = w1l
                w2 = w2l
                w3 = w3l
                w4 = w4l
            else:
                w1 = w1s
                w2 = w2s
                w3 = w3s
                w4 = w4s
            
            if(pdot <= w4):
                tem = (pdot - w4) / (w3 - w4)
            elif(pdot >= -w4):
                tem = - (pdot + w4) / (w4 - w3)
            else:
                tem = 0.
    
#            val1    =            -1.
            tem = tem if (tem > -1.) else -1.
#            val2    =             1.
            tem = tem if (tem < 1.) else 1.
            tem = 1. - tem
            tem1= .5*(cinacrmx-cinacrmn)
            cinacr = cinacrmx - tem * tem1
    #
    #         cinacr = cinacrmx
            if(cina < cinacr):
                cnvflg = False

## totflag and return

##  determine first guess cloud top as the level of zero buoyancy
##    limited to the level of P/Ps=0.7
## not pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static11(
    flg: gtscript.Field[dtype],
    cnvflg: gtscript.Field[dtype],
    ktcon: gtscript.Field[dtype],
    kbm: gtscript.Field[dtype],
    kbcon1: gtscript.Field[dtype],
    dbyo: gtscript.Field[dtype],
    kbcon: gtscript.Field[dtype],
    dp: gtscript.Field[dtype],
    del0: gtscript.Field[dtype],
    xmbmax: gtscript.Field[dtype],
    g: gtscript.Field[dtype],
    dt2: gtscript.Field[dtype],
    aa1: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    qcko: gtscript.Field[dtype],
    qo: gtscript.Field[dtype],
    qrcko: gtscript.Field[dtype],
    dz: gtscript.Field[dtype],
    zi: gtscript.Field[dtype],
    gamma: gtscript.Field[dtype],
    el2orc: gtscript.Field[dtype],
    qeso: gtscript.Field[dtype],
    to: gtscript.Field[dtype],
    hvap: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    tem1: gtscript.Field[dtype],
    xlamue: gtscript.Field[dtype],
    xlamud: gtscript.Field[dtype],
    factor: gtscript.Field[dtype],
    eta: gtscript.Field[dtype],
    dq: gtscript.Field[dtype],
    qrch: gtscript.Field[dtype],
    etah: gtscript.Field[dtype],
    ptem: gtscript.Field[dtype],
    c0t: gtscript.Field[dtype],
    c1: gtscript.Field[dtype],
    qlk: gtscript.Field[dtype],
    dellal: gtscript.Field[dtype],
    buo: gtscript.Field[dtype],
    rfact: gtscript.Field[dtype],
    drag: gtscript.Field[dtype],
    zo: gtscript.Field[dtype],
    delta: gtscript.Field[dtype],
    cp: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    dz1: gtscript.Field[dtype],
    pwo: gtscript.Field[dtype],
    cnvwt: gtscript.Field[dtype],
    *,
    ncloud: dtype
):
    with computation(PARALLEL), interval(...):
        flg = cnvflg
        if(flg):
            ktcon = kbm
    
    with computation(FORWARD), interval(1,-1):
        flg = flg[0,0,-1]
        ktcon = ktcon[0,0,-1]
        if (flg and k < kbm):
            if(k > kbcon1 and dbyo < 0.):
                ktcon = k
                flg   = False

## to make all slices ilke final slice
    with computation(FORWARD), interval(-1,None):
        flg = flg[0,0,-1]
        ktcon = ktcon[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        flg = flg[0,0,1]
        ktcon = ktcon[0,0,1]


##  specify upper limit of mass flux at cloud base

    with computation(PARALLEL), interval(...):
        if(cnvflg):
#         xmbmax(i) = .1
#
            k = kbcon
            # change del name to del0
            dp = 1000. * del0[0,0,0]
            xmbmax = dp / (2. * g * dt2)
#
#         xmbmax(i) = dp / (g * dt2)
#
#         tem = dp / (g * dt2)
#         xmbmax(i) = min(tem, xmbmax(i))

##  compute cloud moisture property and precipitation
    with computation(PARALLEL), interval(...):
        if (cnvflg):
            aa1 = 0.
            if (k == kb):
                qcko = qo
                qrcko = qo

##  Calculate the moisture content of the entraining/detraining parcel (qcko) and the value it would have if just saturated (qrch), according to equation A.14 in Grell (1993) \cite grell_1993 . Their difference is the amount of convective cloud water (qlk = rain + condensate). Determine the portion of convective cloud water that remains suspended and the portion that is converted into convective precipitation (pwo). Calculate and save the negative cloud work function (aa1) due to water loading. Above the level of minimum moist static energy, some of the cloud water is detrained into the grid-scale cloud water from every cloud layer with a rate of 0.0005 \f$m^{-1}\f$ (dellal).
    with computation(FORWARD), interval(1,-1):
        if (cnvflg):
            if(k > kb and k < ktcon):
                dz    = zi - zi[0,0,-1]
                gamma = el2orc * qeso / (to**2)
                qrch = qeso \
                    + gamma * dbyo / (hvap * (1. + gamma))
    #j
                tem  = 0.5 * (xlamue+xlamue[0,0,-1]) * dz
                tem1 = 0.5 * xlamud * dz
                factor = 1. + tem - tem1
                qcko = ((1.-tem1)*qcko[0,0,-1]+tem*0.5* \
                            (qo+qo[0,0,-1]))/factor
                qrcko = qcko
    #j
                dq = eta * (qcko - qrch)
#
#             rhbar(i) = rhbar(i) + qo(i,k) / qeso(i,k)
##  below lfc check if there is excess moisture to release latent heat
#
                if(k >= kbcon and dq > 0.):
                    etah = .5 * (eta + eta[0,0,-1])
                    dp = 1000. * del0
                    if(ncloud > 0):
                        ptem = c0t + c1
                        qlk = dq / (eta + etah * ptem * dz)
                        dellal = etah * c1 * dz * qlk * g / dp
                    else:
                        qlk = dq / (eta + etah * c0t * dz)
                        
                    buo = buo - g * qlk
                    qcko= qlk + qrch
                    pwo = etah * c0t * dz * qlk
                    cnvwt = etah * qlk * g / dp

                if(k >= kbcon):
                    rfact =  1. + delta * cp * gamma \
                          * to / hvap
                    buo = buo + (g / (cp * to)) \
                           * dbyo / (1. + gamma) \
                           * rfact
#                    val = 0.
                    buo = (buo + g * delta * (qeso - qo)) if((qeso - qo)>0.) else buo
                    drag = xlamue if(xlamue > xlamud) else xlamud
## L1064: Calculate the cloud work function according to Pan and Wu (1995) \cite pan_and_wu_1995 equation 4        
    with computation(PARALLEL), interval(...):
        if (cnvflg):
            aa1 = 0.
    
    with computation(FORWARD), interval(1,-1):
        aa1=aa1[0,0,-1]
        if (cnvflg):
            if(k >= kbcon and k < ktcon):
                dz1 = zo[0,0,1] - zo
                aa1 = aa1 + buo * dz1
    
## to make all slices like final slice
    with computation(FORWARD), interval(-1,None):
        aa1 = aa1[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        aa1 = aa1[0,0,1]

    with computation(PARALLEL), interval(...):
        if(cnvflg and aa1 <= 0.):
            cnvflg = False

## totflg and return

## estimate the onvective overshooting as the level
##   where the [aafac * cloud work function] becomes zero,
##   which is the final cloud top
##   limited to the level of P/Ps=0.7

## Continue calculating the cloud work function past the point of neutral buoyancy to represent overshooting according to Han and Pan (2011) \cite han_and_pan_2011 . Convective overshooting stops when \f$ cA_u < 0\f$ where \f$c\f$ is currently 10%, or when 10% of the updraft cloud work function has been consumed by the stable buoyancy force. Overshooting is also limited to the level where \f$p=0.7p_{sfc}\f$.
## not pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static12(
    cnvflg: gtscript.Field[dtype],
    aa1: gtscript.Field[dtype],
    flg: gtscript.Field[dtype],
    ktcon1: gtscript.Field[dtype],
    kbm: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    ktcon: gtscript.Field[dtype],
    dz1: gtscript.Field[dtype],
    zo: gtscript.Field[dtype],
    gamma: gtscript.Field[dtype],
    el2orc: gtscript.Field[dtype],
    qeso: gtscript.Field[dtype],
    to: gtscript.Field[dtype],
    rfact: gtscript.Field[dtype],
    delta: gtscript.Field[dtype],
    cp: gtscript.Field[dtype],
    hvap: gtscript.Field[dtype],
    dbyo: gtscript.Field[dtype],
    g: gtscript.Field[dtype],
    zi: gtscript.Field[dtype],
    qrch: gtscript.Field[dtype],
    tem: gtscript.Field[dtype],
    tem1: gtscript.Field[dtype],
    dz: gtscript.Field[dtype],
    xlamue: gtscript.Field[dtype],
    xlamud: gtscript.Field[dtype],
    factor: gtscript.Field[dtype],
    qcko: gtscript.Field[dtype],
    qrcko: gtscript.Field[dtype],
    qo: gtscript.Field[dtype],
    eta: gtscript.Field[dtype],
    dq: gtscript.Field[dtype],
    etah: gtscript.Field[dtype],
    dp: gtscript.Field[dtype],
    del0: gtscript.Field[dtype],
    ptem: gtscript.Field[dtype],
    c0t: gtscript.Field[dtype],
    c1: gtscript.Field[dtype],
    qlk: gtscript.Field[dtype],
    pwo: gtscript.Field[dtype],
    cnvwt: gtscript.Field[dtype],
    ptem1: gtscript.Field[dtype],
    buo: gtscript.Field[dtype],
    wu2: gtscript.Field[dtype],
    wc: gtscript.Field[dtype],
    sumx: gtscript.Field[dtype],
    kk: gtscript.Field[dtype],
    kbcon1: gtscript.Field[dtype],
    drag: gtscript.Field[dtype],
    dellal: gtscript.Field[dtype],
    *,
    aafac: dtype,
    ncloud: dtype
    # bb1: dtype,
    # bb2: dtype 
):
    with computation(PARALLEL), interval(...):
        if (cnvflg):
            aa1 = aafac * aa1
        flg = cnvflg
        ktcon1 = kbm
    
    with computation(FORWARD), interval(1,-1):
        aa1 = aa1[0,0,-1]
        ktcon1 = ktcon1[0,0,-1]
        flg = flg[0,0,-1]
        if (flg):
            if(k >= ktcon and k < kbm):
                dz1 = zo[0,0,1] - zo
                gamma = el2orc * qeso / (to**2)
                rfact =  1. + delta * cp * gamma \
                     * to / hvap
                aa1 = aa1 + \
                      dz1 * (g / (cp * to)) \
                     * dbyo / (1. + gamma) \
                     * rfact
#               val = 0.
#               aa1(i) = aa1(i) +
##   &                 dz1 * eta(i,k) * g * delta *
#    &                 dz1 * g * delta *
#    &                 max(val,(qeso(i,k) - qo(i,k)))
                if(aa1 < 0.):
                    ktcon1 = k
                    flg = False

## to make all slice like final slice
    with computation(FORWARD), interval(-1,None):
        aa1 = aa1[0,0,-1]
        ktcon1 = ktcon1[0,0,-1]
        flg = flg[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        aa1 = aa1[0,0,1]
        ktcon1 = ktcon1[0,0,1]
        flg = flg[0,0,1]

## compute cloud moisture property, detraining cloud water
##    and precipitation in overshooting layers

## For the overshooting convection, calculate the moisture content of the entraining/detraining parcel as before. Partition convective cloud water and precipitation and detrain convective cloud water in the overshooting layers.
    with computation(FORWARD), interval(1,-1):
        if (cnvflg):
            if(k >= ktcon and k < ktcon1):
                dz    = zi - zi[0,0,-1]
                gamma = el2orc * qeso / (to**2)
                qrch = qeso \
                     + gamma * dbyo / (hvap * (1. + gamma))
#j
                tem  = 0.5 * (xlamue+xlamue[0,0,-1]) * dz
                tem1 = 0.5 * xlamud * dz
                factor = 1. + tem - tem1
                qcko = ((1.-tem1)*qcko[0,0,-1]+tem*0.5* \
                            (qo+qo[0,0,-1]))/factor
                qrcko = qcko
#j
                dq = eta * (qcko - qrch)
#
##  check if there is excess moisture to release latent heat
#
                if(dq > 0.):
                    etah = .5 * (eta + eta[0,0,-1])
                    dp = 1000. * del0
                    if(ncloud > 0):
                        ptem = c0t + c1
                        qlk = dq / (eta + etah * ptem * dz)
                        dellal = etah * c1 * dz * qlk * g / dp
                    else:
                        qlk = dq / (eta + etah * c0t * dz)
                    
                    qcko = qlk + qrch
                    pwo = etah * c0t * dz * qlk
                    cnvwt = etah * qlk * g / dp

##  compute updraft velocity square(wu2)
## Calculate updraft velocity square(wu2) according to Han et al.'s (2017) \cite han_et_al_2017 equation 7.
    with computation(FORWARD), interval(1,-1):
        # bb1 = 4.0
        # bb2 = 0.8
        if (cnvflg):
            if(k > kbcon1 and k < ktcon):
                dz    = zi - zi[0,0,-1]
                tem  = 0.25 * 4.0 * (drag+drag[0,0,-1]) * dz
                tem1 = 0.5 * 0.8 * (buo+buo[0,0,-1]) * dz
                ptem = (1. - tem) * wu2[0,0,-1]
                ptem1 = 1. + tem
                wu2 = (ptem + tem1) / ptem1
                wu2 = wu2 if(wu2 > 0.) else 0.

##  compute updraft velocity averaged over the whole cumulus
    with computation(PARALLEL), interval(...):
        wc = 0.
        sumx = 0.
    
    with computation(FORWARD), interval(1,-1):
        wc = wc[0,0,-1]
        sumx = sumx[0,0,-1]
        if (cnvflg):
            if(k > kbcon1 and k < ktcon):
                dz = zi - zi[0,0,-1]
                tem = 0.5 * ((wu2)**0.5 + (wu2[0,0,-1])**0.5)
                wc = wc + tem * dz
                sumx = sumx + dz
    
## to make all slice like final slice
    with computation(FORWARD), interval(-1,None):
        wc = wc[0,0,-1]
        sumx = sumx[0,0,-1]
    with computation(BACKWARD), interval(0,-1):
        wc = wc[0,0,1]
        sumx = sumx[0,0,1]
        
    
    with computation(PARALLEL), interval(...):
        if(cnvflg):
            if(sumx == 0.):
                cnvflg=False
            else:
                wc = wc / sumx
#            val = 1.e-4
            if (wc < 1.e-4):
                cnvflg=False

## exchange ktcon with ktcon1
    with computation(PARALLEL), interval(...):
        if(cnvflg):
            kk = ktcon
            ktcon = ktcon1
            ktcon1 = kk

## this section is ready for cloud water
##  if(ncloud > 0):
## pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static13(
    cnvflg: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    ktcon: gtscript.Field[dtype],
    gamma: gtscript.Field[dtype],
    el2orc: gtscript.Field[dtype],
    qeso: gtscript.Field[dtype],
    to: gtscript.Field[dtype],
    qrch: gtscript.Field[dtype],
    dbyo: gtscript.Field[dtype],
    hvap: gtscript.Field[dtype],
    dq: gtscript.Field[dtype],
    qcko: gtscript.Field[dtype],
    qlko_ktcon: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(...):
        if(cnvflg):
            k = ktcon - 1
            gamma = el2orc * qeso / (to**2)
            qrch = qeso \
                + gamma * dbyo / (hvap * (1. + gamma))
            dq = qcko - qrch
#  check if there is excess moisture to release latent heat
            if(dq > 0.):
                qlko_ktcon = dq
                qcko = qrch

## endif

## compute precipitation efficiency in terms of windshear
## pass
@gtscript.stencil(backend=backend,externals=externals, rebuild=rebuild, **backend_opts)
def stencil_static14(
    cnvflg: gtscript.Field[dtype],
    vshear: gtscript.Field[dtype],
    k: gtscript.Field[dtype],
    kb: gtscript.Field[dtype],
    ktcon: gtscript.Field[dtype],
    uo: gtscript.Field[dtype],
    vo: gtscript.Field[dtype],
    ziktcon: gtscript.Field[dtype],
    zikb: gtscript.Field[dtype],
    edt: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(...):
        if(cnvflg):
            vshear = 0.
    
    with computation(FORWARD), interval(1,None):
        vshear = vshear[0,0,-1]
        if (cnvflg):
            if(k > kb and k <= ktcon):
#                shear= ((uo-uo[0,0,-1]) ** 2 \
#                      + (vo-vo[0,0,-1]) ** 2)**0.5
                vshear = vshear + ((uo-uo[0,0,-1]) ** 2 \
                      + (vo-vo[0,0,-1]) ** 2)**0.5

## to make all slice like final slice
    with computation(BACKWARD), interval(0,-1):
        vshear = vshear[0,0,1]
    
    with computation(PARALLEL), interval(...):
        if(cnvflg):
#use ziktcon and zikb to represent zi(ktcon) and zi(kb)          
            vshear = 1.e3 * vshear / (ziktcon-zikb)
#            e1=1.591-.639*vshear \
#            +.0953*(vshear**2)-.00496*(vshear**3)
            edt=1.-(1.591-.639*vshear \
            +.0953*(vshear**2)-.00496*(vshear**3))
#           val =         .9
            edt = edt if(edt < .9) else .9
#           val =         .0
            edt = edt if(edt > .0) else .0


@gtscript.stencil(name="test.stencil", backend="numpy")
def copy_stencil(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
    from __gtscript__ import computation, interval, PARALLEL
    with computation(PARALLEL), interval(...):
        out_field = in_field


def samfshalcnv_static(data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection
    
    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """
    ## initializaion
    
    
    ## "static"
    ## Perform calculations related to the updraft of the entraining/detraining cloud model ("static control").
    ## Search in the PBL for the level of maximum moist static energy to start the ascending parcel.
    origin = (0,0,0)
    shape = (1,im,km)
    stencil_static0(
        cnvflg = cnvflg ,
        hmax = hmax,
        heo = heo,
        kb = kb ,
        k = k,
        kpbl = kpbl ,
        kmax = kmax ,
        dz = dz ,
        zo = zo,
        dp = dp,
        es = es,
        to = to,
        pprime = pprime,
        epsm1 = epsm1 ,
        eps = eps,
        qs = qs,
        dqsdp = dqsdp,
        desdt = desdt,
        dqsdt = dqsdt,
        fact1 = fact1,
        fact2 = fact2,
        gamma = gamma,
        el2orc = el2orc,
        qeso = qeso,
        g = g,
        hvap = hvap,
        cp = cp,
        dt = dt,
        dq = dq,
        qo = qo,
        po = po,
        uo = uo,
        vo = vo,
        heso = heso,
        pfld = pfld,
        origin = origin,
        domain = shape
    )

    ##ntr loop
    for n in range(ntr):
        ctro_n = ctro_np[:,:,n].reshape(1,im,km) 
        ctro = gt.storage.from_array(ctro_n, backend = 'numpy', (0, 0,0)) 
        stencil_ntrstatic0(
            cnvflg = cnvflg,
            k = k,
            kmax = kmax,
            ctro = ctro,
            origin = origin,
            domain = shape
        )
        ctro_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
        copy_stencil(ctro, ctro_out)
        ctro_np[:,:,n] = np.array(ctro_out.data)
    
    heo_kb = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    stencil_static1(
        cnvflg  = cnvflg,
        flg  = flg,
        kbcon  = kbcon,
        kmax  = kmax,
        k  = k,
        kbm  = kbm,
        kb  = kb,
        heo_kb  = heo_kb,
        heso  = heso,
        origin = origin,
        domain = shape
    )

    cnvflg_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    copy_stencil(cnvflg, cnvflg_out)
    cnvflg_np = np.array(cnvflg_out.data)
    totflg = True
    for i in range(im):
        totflg = totflg and (not cnvflg_np[0,i,0])
    if(totflg):
         return
    
    pfld_kb = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    pfld_kbcon = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    stencil_static2(
        cnvflg  = cnvflg,
        pdot  = pdot,
        dotkbcon  = dotkbcon,
        islimsk  = islimsk,
        w1  = w1,
        w2  = w2,
        w3  = w3,
        w4  = w4,
        w1l  = w1l,
        w2l  = w2l,
        w3l  = w3l,
        w4l  = w4l,
        w1s  = w1s,
        w2s  = w2s,
        w3s  = w3s,
        w4s  = w4s,
        tem  = tem,
        ptem  = ptem,
        ptem1  = ptem1,
        cinpcr  = cinpcr,
        cinpcrmx  = cinpcrmx,
        cinpcrmn  = cinpcrmn,
        tem1  = tem1,
        pfld_kb  = pfld_kb,
        pfld_kbcon  = pfld_kbcon,
        origin = origin,
        domain = shape
    )

    cnvflg_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    copy_stencil(cnvflg, cnvflg_out)
    cnvflg_np = np.array(cnvflg_out.data)
    totflg = True
    for i in range(im):
        totflg = totflg and (not cnvflg_np[0,i,0])
    if(totflg):
         return

    if(ntk > 0): 
        stencil_static3(
            sumx  = sumx,
            tkemean  = tkemean,
            cnvflg  = cnvflg,
            k  = k,
            kb  = kb,
            kbcon  = kbcon,
            dz  = dz,
            zo  = zo,
            tem  = tem,
            qtr  = qtr,
            tkemn  = tkemn,
            tkemx  = tkemx,
            clamt  = clamt,
            clam  = clam,
            clamd  = clamd,
            tem1  = tem1,
            dtke  = dtke,
            origin = origin,
            domain = shape
        )
    else:
        stencil_static4(
            cnvflg = cnvflg,
            clamt = clamt,
            clam = clam,
            origin = origin,
            domain = shape
        )

    stencil_static5(
        cnvflg = cnvflg,
        xlamue = xlamue,
        clamt = clamt,
        zi = zi,
        xlamud = xlamud,
        k = k,
        kbcon = kbcon,
        kb = kb,
        dz = dz,
        ptem = ptem,
        eta = eta,
        ktconn = ktconn,
        kmax = kmax,
        kbm = kbm,
        hcko = hcko,
        ucko = ucko,
        vcko = vcko,
        heo = heo,
        uo = uo,
        vo = vo,
        origin = origin,
        domain = shape
    )

    for n in range(ntr): 
        ecko_n = ecko_np[:,:,n].reshape(1,im,km) 
        ecko = gt.storage.from_array(ecko_n, backend = 'numpy', (0, 0,0)) 
        stencil_ntrstatic1(
            cnvflg = cnvflg,
            k = k,
            kb = kb,
            ecko = ecko,
            ctro = ctro,
            origin = origin,
            domain = shape
        )
        ecko_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
        copy_stencil(ecko, ecko_out)
        ecko_np[:,:,n] = np.array(ecko_out.data)
        

    # stencil_static6(
    #     cnvflg = cnvflg,
    #     k = k,
    #     kb = kb,
    #     ecko = ecko,
    #     ctro = ctro,
    #     origin = origin,
    #     domain = shape
    # )

    stencil_static7(
        cnvflg = cnvflg,
        k = k,
        kb = kb,
        kmax = kmax,
        dz = dz,
        zi = zi,
        tem = tem,
        xlamue = xlamue,
        tem1 = tem1,
        xlamud = xlamud,
        factor = factor,
        hcko = hcko,
        heo = heo,
        dbyo = dbyo,
        heso = heso,
        cm = cm,
        ptem = ptem,
        pgcon = pgcon,
        ptem1 = ptem1,
        ucko = ucko,
        uo = uo,
        vcko = vcko,
        vo = vo,
        origin = origin,
        domain = shape
    )
    for n in range(ntr):
        ecko_n = ecko_np[:,:,n].reshape(1,im,km) 
        ecko = gt.storage.from_array(ecko_n, backend = 'numpy', (0, 0,0)) 
        ctro_n = ctro_np[:,:,n].reshape(1,im,km) 
        ctro = gt.storage.from_array(ctro_n, backend = 'numpy', (0, 0,0)) 
        
        stencil_ntrstatic2(
            cnvflg = cnvflg,
            k = k,
            kb = kb,
            kmax = kmax,
            zi = zi,
            dz = dz,
            tem = tem,
            xlamue = xlamue,
            factor = factor,
            ecko = ecko,
            ctro = ctro,
            origin = origin,
            domain = shape
        )

        ecko_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
        copy_stencil(ecko, ecko_out)
        ecko_np[:,:,n] = np.array(ecko_out.data)

    # stencil_static8(
    #     cnvflg = cnvflg,
    #     k = k,
    #     kb = kb,
    #     kmax = kmax,
    #     zi = zi,
    #     dz = dz,
    #     tem = tem,
    #     xlamue = xlamue,
    #     factor = factor,
    #     ecko = ecko,
    #     ctro = ctro,
    #     origin = origin,
    #     domain = shape
    # )
    pfld_kbcon = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    pfld_kbcon1 = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    stencil_static9(
        cnvflg = cnvflg,
        flg = flg,
        kbcon1 = kbcon1,
        kmax = kmax,
        k = k,
        kbm = kbm,
        kbcon = kbcon,
        dbyo = dbyo,
        tem = tem,
        dthk = dthk,
        pfld_kbcon = pfld_kbcon,
        pfld_kbcon1 = pfld_kbcon1,
        origin = origin,
        domain = shape
    )

    cnvflg_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    copy_stencil(cnvflg, cnvflg_out)
    cnvflg_np = np.array(cnvflg_out.data)
    totflg = True
    for i in range(im):
        totflg = totflg and (not cnvflg_np[0,i,0])
    if(totflg):
         return


    stencil_static10(
        cina = cina,
        cnvflg = cnvflg,
        k = k,
        kb = kb,
        kbcon1 = kbcon1,
        dz1 = dz1,
        zo = zo,
        gamma = gamma,
        g = g,
        el2orc = el2orc,
        qeso = qeso,
        to = to,
        rfact = rfact,
        delta = delta,
        cp = cp,
        hvap = hvap,
        dbyo = dbyo,
        qo = qo,
        w1 = w1,
        w2 = w2,
        w3 = w3,
        w4 = w4,
        w1l = w1l,
        w2l = w2l,
        w3l = w3l,
        w4l = w4l,
        w1s = w1s,
        w2s = w2s,
        w3s = w3s,
        w4s = w4s,
        pdot = pdot,
        tem = tem,
        tem1 = tem1,
        cinacrmx = cinacrmx,
        cinacrmn = cinacrmn,
        cinacr = cinacr,
        islimsk = islimsk, # dtype
        origin = origin,
        domain = shape
    )

    cnvflg_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    copy_stencil(cnvflg, cnvflg_out)
    cnvflg_np = np.array(cnvflg_out.data)
    totflg = True
    for i in range(im):
        totflg = totflg and (not cnvflg_np[0,i,0])
    if(totflg):
         return


    stencil_static11(
        flg = flg,
        cnvflg = cnvflg,
        ktcon = ktcon,
        kbm = kbm,
        kbcon1 = kbcon1,
        dbyo = dbyo,
        kbcon = kbcon,
        dp = dp,
        del0 = del0,
        xmbmax = xmbmax,
        g = g,
        dt2 = dt2,
        aa1 = aa1,
        kb = kb,
        qcko = qcko,
        qo = qo,
        qrcko = qrcko,
        dz = dz,
        zi = zi,
        gamma = gamma,
        el2orc = el2orc,
        qeso = qeso,
        to = to,
        hvap = hvap,
        tem = tem,
        tem1 = tem1,
        xlamue = xlamue,
        xlamud = xlamud,
        factor = factor,
        eta = eta,
        dq = dq,
        qrch = qrch,
        etah = etah,
        ptem = ptem,
        c0t = c0t,
        c1 = c1,
        qlk = qlk,
        dellal = dellal,
        buo = buo,
        rfact = rfact,
        drag = drag,
        zo = zo,
        delta = delta,
        cp = cp,
        k = k,
        dz1 = dz1,
        pwo = pwo,
        cnvwt = cnvwt,
        ncloud = ncloud, # dtype
        origin = origin,
        domain = shape
    )

    cnvflg_out = gt4py.storage.zeros(shape=shape, default_origin=(0, 0, 0), dtype=float, backend="numpy")
    copy_stencil(cnvflg, cnvflg_out)
    cnvflg_np = np.array(cnvflg_out.data)
    totflg = True
    for i in range(im):
        totflg = totflg and (not cnvflg_np[0,i,0])
    if(totflg):
         return


    stencil_static12(
        cnvflg = cnvflg,
        aa1 = aa1,
        flg = flg,
        ktcon1 = ktcon1,
        kbm = kbm,
        k = k,
        ktcon = ktcon,
        dz1 = dz1,
        zo = zo,
        gamma = gamma,
        el2orc = el2orc,
        qeso = qeso,
        to = to,
        rfact = rfact,
        delta = delta,
        cp = cp,
        hvap = hvap,
        dbyo = dbyo,
        g = g,
        zi = zi,
        qrch = qrch,
        tem = tem,
        tem1 = tem1,
        dz = dz,
        xlamue = xlamue,
        xlamud = xlamud,
        factor = factor,
        qcko = qcko,
        qrcko = qrcko,
        qo = qo,
        eta = eta,
        dq = dq,
        etah = etah,
        dp = dp,
        del0 = del0,
        ptem = ptem,
        c0t = c0t,
        c1 = c1,
        qlk = qlk,
        pwo = pwo,
        cnvwt = cnvwt,
        ptem1 = ptem1,
        buo = buo,
        wu2 = wu2,
        wc = wc,
        sumx = sumx,
        kk = kk,
        kbcon1 = kbcon1,
        drag = drag,
        dellal = dellal,
        aafac = aafac, #: dtype,
        ncloud = ncloud, #: dtype,
        origin = origin,
        domain = shape
    )
    if(ncloud > 0):
        stencil_static13(
            cnvflg = cnvflg,
            k = k,
            ktcon = ktcon,
            gamma = gamma,
            el2orc = el2orc,
            qeso = qeso,
            to = to,
            qrch = qrch,
            dbyo = dbyo,
            hvap = hvap,
            dq = dq,
            qcko = qcko,
            qlko_ktcon = qlko_ktcon,
            origin = origin,
            domain = shape
        )
    else:
        stencil_static14(
            cnvflg = cnvflg,
            vshear = vshear,
            k = k,
            kb = kb,
            ktcon = ktcon,
            uo = uo,
            vo = vo,
            ziktcon = ziktcon,
            zikb = zikb,
            edt = edt
        )
























    




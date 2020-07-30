import gt4py as gt
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval
from shalconv.funcphys import fpvs
from . import *

externals = {
    "fpvs":fpvs
}

@gtscript.stencil(backend=BACKEND,externals=externals, verbose=True)
def stencil_static0(
    cnvflg: FIELD_FLOAT,
    hmax: FIELD_FLOAT,
    heo: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kpbl: FIELD_FLOAT,
    kmax: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    zo: FIELD_FLOAT,
    dp: FIELD_FLOAT,
    es: FIELD_FLOAT,
    to: FIELD_FLOAT,
    pprime: FIELD_FLOAT,
    epsm1: FIELD_FLOAT,
    eps: FIELD_FLOAT,
    qs: FIELD_FLOAT,
    dqsdp: FIELD_FLOAT,
    desdt: FIELD_FLOAT,
    dqsdt: FIELD_FLOAT,
    fact1: FIELD_FLOAT,
    fact2: FIELD_FLOAT,
    gamma: FIELD_FLOAT,
    el2orc: FIELD_FLOAT,
    qeso: FIELD_FLOAT,
    g: FIELD_FLOAT,
    hvap: FIELD_FLOAT,
    cp: FIELD_FLOAT,
    dt: FIELD_FLOAT,
    dq: FIELD_FLOAT,
    qo: FIELD_FLOAT,
    po: FIELD_FLOAT,
    uo: FIELD_FLOAT,
    vo: FIELD_FLOAT,
    heso: FIELD_FLOAT,
    pfld: FIELD_FLOAT
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

@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_ntrstatic0(
    cnvflg: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kmax: FIELD_FLOAT,
    ctro: FIELD_FLOAT
):
    with computation(FORWARD), interval(0,-1):
        if(cnvflg and k <= (kmax-1)):
            ctro = .5 * (ctro + ctro[0,0,1])


@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static1(
    cnvflg: FIELD_FLOAT,
    flg: FIELD_FLOAT,
    kbcon: FIELD_FLOAT,
    kmax: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kbm: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    heo_kb: FIELD_FLOAT,
    heso: FIELD_FLOAT
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

@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static2(
    cnvflg: FIELD_FLOAT,
    pdot: FIELD_FLOAT,
    dotkbcon: FIELD_FLOAT,
    islimsk: FIELD_FLOAT,
    w1: FIELD_FLOAT,
    w2: FIELD_FLOAT,
    w3: FIELD_FLOAT,
    w4: FIELD_FLOAT,
    w1l: FIELD_FLOAT,
    w2l: FIELD_FLOAT,
    w3l: FIELD_FLOAT,
    w4l: FIELD_FLOAT,
    w1s: FIELD_FLOAT,
    w2s: FIELD_FLOAT,
    w3s: FIELD_FLOAT,
    w4s: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    ptem: FIELD_FLOAT,
    ptem1: FIELD_FLOAT,
    cinpcr: FIELD_FLOAT,
    cinpcrmx: FIELD_FLOAT,
    cinpcrmn: FIELD_FLOAT,
    tem1: FIELD_FLOAT,
    pfld_kb: FIELD_FLOAT,
    pfld_kbcon: FIELD_FLOAT
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static3(
    sumx: FIELD_FLOAT,
    tkemean: FIELD_FLOAT,
    cnvflg: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    kbcon: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    zo: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    qtr: FIELD_FLOAT,
    tkemn: FIELD_FLOAT,
    tkemx: FIELD_FLOAT,
    clamt: FIELD_FLOAT,
    clam: FIELD_FLOAT,
    clamd: FIELD_FLOAT,
    tem1: FIELD_FLOAT,
    dtke: FIELD_FLOAT
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static4(
    cnvflg: FIELD_FLOAT,
    clamt: FIELD_FLOAT,
    clam: FIELD_FLOAT,
):
    with computation(PARALLEL), interval(...):
        if(cnvflg):
            clamt  = clam
##
## start updraft entrainment rate.
## pass
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static5(
    cnvflg: FIELD_FLOAT,
    xlamue: FIELD_FLOAT,
    clamt: FIELD_FLOAT,
    zi: FIELD_FLOAT,
    xlamud: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kbcon: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    ptem: FIELD_FLOAT,
    eta: FIELD_FLOAT,
    ktconn: FIELD_FLOAT,
    kmax: FIELD_FLOAT,
    kbm: FIELD_FLOAT,
    hcko: FIELD_FLOAT,
    ucko: FIELD_FLOAT,
    vcko: FIELD_FLOAT,
    heo: FIELD_FLOAT,
    uo: FIELD_FLOAT,
    vo: FIELD_FLOAT
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_ntrstatic1(
    cnvflg: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    ecko: FIELD_FLOAT,
    ctro: FIELD_FLOAT
):
    with computation(PARALLEL), interval(...):
        if(cnvflg and k == kb):
            ecko = ctro


## not pass
# @gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
# def stencil_static6(
#     cnvflg: FIELD_FLOAT,
#     k: FIELD_FLOAT,
#     kb: FIELD_FLOAT,
#     ecko: FIELD_FLOAT,
#     ctro: FIELD_FLOAT
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static7(
    cnvflg: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    kmax: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    zi: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    xlamue: FIELD_FLOAT,
    tem1: FIELD_FLOAT,
    xlamud: FIELD_FLOAT,
    factor: FIELD_FLOAT,
    hcko: FIELD_FLOAT,
    heo: FIELD_FLOAT,
    dbyo: FIELD_FLOAT,
    heso: FIELD_FLOAT,
    cm: FIELD_FLOAT,
    ptem: FIELD_FLOAT,
    pgcon: FIELD_FLOAT,
    ptem1: FIELD_FLOAT,
    ucko: FIELD_FLOAT,
    uo: FIELD_FLOAT,
    vcko: FIELD_FLOAT,
    vo: FIELD_FLOAT
):
    with computation(FORWARD), interval(1,-1):
        if(cnvflg):
            if(k > kb and k < kmax):
                dz   = zi[0,0,0] - zi[0,0,-1]
                tem  = 0.5 * (xlamue[0,0,0]+xlamue[0,0,-1]) * dz[0,0,0]
                tem1 = 0.5 * xlamud * dz
                factor = 1. + tem - tem1
                hcko = ((1.-tem1)*hcko[0,0,-1]+tem*0.5*
                            (heo+heo[0,0,-1]))/factor
                dbyo = hcko - heso
#
                tem  = 0.5 * cm * tem
                factor = 1. + tem
                ptem = tem + pgcon
                ptem1= tem - pgcon
                ucko = ((1.-tem)*ucko[0,0,-1]+ptem*uo
                            +ptem1*uo[0,0,-1])/factor
                vcko = ((1.-tem)*vcko[0,0,-1]+ptem*vo
                            +ptem1*vo[0,0,-1])/factor

## for n = 1, ntr:
## pass
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_ntrstatic2(
    cnvflg: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    kmax: FIELD_FLOAT,
    zi: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    xlamue: FIELD_FLOAT,
    factor: FIELD_FLOAT,
    ecko: FIELD_FLOAT,
    ctro: FIELD_FLOAT
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static9(
    cnvflg: FIELD_FLOAT,
    flg: FIELD_FLOAT,
    kbcon1: FIELD_FLOAT,
    kmax: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kbm: FIELD_FLOAT,
    kbcon: FIELD_FLOAT,
    dbyo: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    dthk: FIELD_FLOAT,
    pfld_kbcon: FIELD_FLOAT,
    pfld_kbcon1: FIELD_FLOAT,
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static10(
    cina: FIELD_FLOAT,
    cnvflg: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    kbcon1: FIELD_FLOAT,
    dz1: FIELD_FLOAT,
    zo: FIELD_FLOAT,
    gamma: FIELD_FLOAT,
    g: FIELD_FLOAT,
    el2orc: FIELD_FLOAT,
    qeso: FIELD_FLOAT,
    to: FIELD_FLOAT,
    rfact: FIELD_FLOAT,
    delta: FIELD_FLOAT,
    cp: FIELD_FLOAT,
    hvap: FIELD_FLOAT,
    dbyo: FIELD_FLOAT,
    qo: FIELD_FLOAT,
    w1: FIELD_FLOAT,
    w2: FIELD_FLOAT,
    w3: FIELD_FLOAT,
    w4: FIELD_FLOAT,
    w1l: FIELD_FLOAT,
    w2l: FIELD_FLOAT,
    w3l: FIELD_FLOAT,
    w4l: FIELD_FLOAT,
    w1s: FIELD_FLOAT,
    w2s: FIELD_FLOAT,
    w3s: FIELD_FLOAT,
    w4s: FIELD_FLOAT,
    pdot: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    tem1: FIELD_FLOAT,
    cinacrmx: FIELD_FLOAT,
    cinacrmn: FIELD_FLOAT,
    cinacr: FIELD_FLOAT,
    *,
    islimsk: dtype
):
    with computation(FORWARD), interval(1,-1):
        cina = cina[0,0,-1]
        if (cnvflg):
            if(k > kb and k < kbcon1):
                dz1 = zo[0,0,1] - zo
                gamma = el2orc * qeso / (to**2)
                rfact =  1. + delta * cp * gamma * to / hvap
                cina = cina + dz1 * (g / (cp * to)) * \
                       dbyo / (1. + gamma) * rfact
#               val = 0.
                cina = (cina +
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static11(
    flg: FIELD_FLOAT,
    cnvflg: FIELD_FLOAT,
    ktcon: FIELD_FLOAT,
    kbm: FIELD_FLOAT,
    kbcon1: FIELD_FLOAT,
    dbyo: FIELD_FLOAT,
    kbcon: FIELD_FLOAT,
    dp: FIELD_FLOAT,
    del0: FIELD_FLOAT,
    xmbmax: FIELD_FLOAT,
    g: FIELD_FLOAT,
    dt2: FIELD_FLOAT,
    aa1: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    qcko: FIELD_FLOAT,
    qo: FIELD_FLOAT,
    qrcko: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    zi: FIELD_FLOAT,
    gamma: FIELD_FLOAT,
    el2orc: FIELD_FLOAT,
    qeso: FIELD_FLOAT,
    to: FIELD_FLOAT,
    hvap: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    tem1: FIELD_FLOAT,
    xlamue: FIELD_FLOAT,
    xlamud: FIELD_FLOAT,
    factor: FIELD_FLOAT,
    eta: FIELD_FLOAT,
    dq: FIELD_FLOAT,
    qrch: FIELD_FLOAT,
    etah: FIELD_FLOAT,
    ptem: FIELD_FLOAT,
    c0t: FIELD_FLOAT,
    c1: FIELD_FLOAT,
    qlk: FIELD_FLOAT,
    dellal: FIELD_FLOAT,
    buo: FIELD_FLOAT,
    rfact: FIELD_FLOAT,
    drag: FIELD_FLOAT,
    zo: FIELD_FLOAT,
    delta: FIELD_FLOAT,
    cp: FIELD_FLOAT,
    k: FIELD_FLOAT,
    dz1: FIELD_FLOAT,
    pwo: FIELD_FLOAT,
    cnvwt: FIELD_FLOAT,
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static12(
    cnvflg: FIELD_FLOAT,
    aa1: FIELD_FLOAT,
    flg: FIELD_FLOAT,
    ktcon1: FIELD_FLOAT,
    kbm: FIELD_FLOAT,
    k: FIELD_FLOAT,
    ktcon: FIELD_FLOAT,
    dz1: FIELD_FLOAT,
    zo: FIELD_FLOAT,
    gamma: FIELD_FLOAT,
    el2orc: FIELD_FLOAT,
    qeso: FIELD_FLOAT,
    to: FIELD_FLOAT,
    rfact: FIELD_FLOAT,
    delta: FIELD_FLOAT,
    cp: FIELD_FLOAT,
    hvap: FIELD_FLOAT,
    dbyo: FIELD_FLOAT,
    g: FIELD_FLOAT,
    zi: FIELD_FLOAT,
    qrch: FIELD_FLOAT,
    tem: FIELD_FLOAT,
    tem1: FIELD_FLOAT,
    dz: FIELD_FLOAT,
    xlamue: FIELD_FLOAT,
    xlamud: FIELD_FLOAT,
    factor: FIELD_FLOAT,
    qcko: FIELD_FLOAT,
    qrcko: FIELD_FLOAT,
    qo: FIELD_FLOAT,
    eta: FIELD_FLOAT,
    dq: FIELD_FLOAT,
    etah: FIELD_FLOAT,
    dp: FIELD_FLOAT,
    del0: FIELD_FLOAT,
    ptem: FIELD_FLOAT,
    c0t: FIELD_FLOAT,
    c1: FIELD_FLOAT,
    qlk: FIELD_FLOAT,
    pwo: FIELD_FLOAT,
    cnvwt: FIELD_FLOAT,
    ptem1: FIELD_FLOAT,
    buo: FIELD_FLOAT,
    wu2: FIELD_FLOAT,
    wc: FIELD_FLOAT,
    sumx: FIELD_FLOAT,
    kk: FIELD_FLOAT,
    kbcon1: FIELD_FLOAT,
    drag: FIELD_FLOAT,
    dellal: FIELD_FLOAT,
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static13(
    cnvflg: FIELD_FLOAT,
    k: FIELD_FLOAT,
    ktcon: FIELD_FLOAT,
    gamma: FIELD_FLOAT,
    el2orc: FIELD_FLOAT,
    qeso: FIELD_FLOAT,
    to: FIELD_FLOAT,
    qrch: FIELD_FLOAT,
    dbyo: FIELD_FLOAT,
    hvap: FIELD_FLOAT,
    dq: FIELD_FLOAT,
    qcko: FIELD_FLOAT,
    qlko_ktcon: FIELD_FLOAT
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
@gtscript.stencil(backend=BACKEND,externals=externals, rebuild=REBUILD, **BACKEND_OPTS)
def stencil_static14(
    cnvflg: FIELD_FLOAT,
    vshear: FIELD_FLOAT,
    k: FIELD_FLOAT,
    kb: FIELD_FLOAT,
    ktcon: FIELD_FLOAT,
    uo: FIELD_FLOAT,
    vo: FIELD_FLOAT,
    ziktcon: FIELD_FLOAT,
    zikb: FIELD_FLOAT,
    edt: FIELD_FLOAT
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
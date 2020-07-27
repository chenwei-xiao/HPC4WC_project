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
    stencil_static0(cnvflg, hmax, heo, kb, k, kpbl, kmax,
                    dz, zo, dp, es, to, pprime, epsm1 , eps,
                    qs, dqsdp, desdt, dqsdt, fact1, fact2,
                    gamma, el2orc, qeso, g, hvap, cp, dt, dq,
                    qo, po, uo, vo, heso, pfld,
                    origin = origin, domain = shape)

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
























    




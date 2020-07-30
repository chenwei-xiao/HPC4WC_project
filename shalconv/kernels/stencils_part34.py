import gt4py as gt
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval
from shalconv.funcphys import fpvsx_gt as fpvs
from shalconv.physcons import (
    con_g     as g,
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
from .utils import *
from . import *

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
                     del0      : FIELD_FLOAT,
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
                     xmbmax    : FIELD_FLOAT,
                     sumx      : FIELD_FLOAT,
                     umean     : FIELD_FLOAT):
    
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
                
            dp  = 1000.0 * del0
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
                
                dp   = 1000.0 * del0
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
            dtconv = dtconv if(dtconv > dtmin) else dtmin #max(dtconv, dtmin)
            dtconv = dtconv if(dtconv > dt2) else dt2     #max(dtconv, dt2)
            dtconv = dtconv if(dtconv < dtmax) else dtmax #min(dtconv, dtmax)
            
            # Initialize field for advective time scale computation
            sumx  = 0.0
            umean = 0.0
    
    # Calculate advective time scale (tauadv) using a mean cloud layer 
    # wind speed (propagate forward)
    with computation(FORWARD), interval(1, -1):
        
        if cnvflg == 1:
            if k_idx >= kbcon1 and k_idx < ktcon1:
                dz    = zi[0, 0, 0] - zi[0, 0, -1]
                tem   = (u1*u1 + v1*v1)**0.5 #sqrt(u1*u1 + v1*v1)
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
            umean  = umean if(umean > 1.0) else 1.0 #max(umean, val)  # Passing literals (e.g. 1.0) to functions might cause errors in conditional statements
            tauadv = gdx/umean
            
            if k_idx == kbcon:
                
                # From Han et al.'s (2017) \cite han_et_al_2017 equation 
                # 6, calculate cloud base mass flux as a function of the 
                # mean updraft velocity
                rho  = po * 100.0 / (rd * to)
                tfac = tauadv/dtconv
                tfac = tfac if(tfac<1.0) else 1.0 #min(tfac, val)  # Same as above: literals
                xmb  = tfac * betaw * rho * wc
                
                # For scale-aware parameterization, the updraft fraction 
                # (sigmagfm) is first computed as a function of the 
                # lateral entrainment rate at cloud base (see Han et 
                # al.'s (2017) \cite han_et_al_2017 equation 4 and 5), 
                # following the study by Grell and Freitas (2014) \cite 
                # grell_and_freitus_2014
                tem  = xlamue if(xlamue > 2.0e-4) else 2.0e-4 #max(xlamue, 2.0e-4)
                tem  = tem if(xlamue < 6.0e-4) else 6.0e-4 #0.2/min(tem, 6.0e-4)
                tem  = 0.2 / tem
                tem1 = 3.14 * tem * tem
                
                sigmagfm = tem1/garea
                sigmagfm = sigmagfm if(sigmagfm > 0.001) else 0.001 #max(sigmagfm, 0.001)
                sigmagfm = sigmagfm if(sigmagfm > 0.999) else 0.999 #min(sigmagfm, 0.999)
            
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
                scaldfunc = scaldfunc if (scaldfunc < 1.) else 1.  # min(scaldfunc, 1.0)
                scaldfunc = scaldfunc if (scaldfunc > 0.) else 0.  # max(scaldfunc, 0.0)
            else:
                scaldfunc = 1.0
            
            xmb = xmb * scaldfunc
            xmb = xmb if(xmb < xmbmax) else xmbmax #min(xmb, xmbmax)


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def comp_tendencies_tr( g      : DTYPE_FLOAT,
                        cnvflg : FIELD_INT,
                        k_idx  : FIELD_INT,
                        kmax   : FIELD_INT,
                        kb     : FIELD_INT,
                        ktcon  : FIELD_INT,
                        dellae : FIELD_FLOAT,
                        del0   : FIELD_FLOAT,
                        eta    : FIELD_FLOAT,
                        ctro   : FIELD_FLOAT,
                        ecko   : FIELD_FLOAT):

    with computation(PARALLEL), interval(...):
        
        if cnvflg == 1 and k_idx <= kmax:
            
            dellae = 0.0
            
    with computation(PARALLEL), interval(1, -1):
        
        if cnvflg == 1 and k_idx > kb and k_idx < ktcon:
            
            # Changes due to subsidence and entrainment
            dp = 1000.0 * del0
            
            tem1 = eta[0, 0,  0] * (ctro[0, 0,  0] - ecko[0, 0,  0])
            tem2 = eta[0, 0, -1] * (ctro[0, 0, -1] - ecko[0, 0, -1])
            
            dellae = dellae + (tem1 - tem2) * g/dp
            
    with computation(PARALLEL), interval(...):
        
        # Cloud top
        if cnvflg == 1:
            if ktcon == k_idx:
                
                dp = 1000.0 * del0
                
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
                             del0    : FIELD_FLOAT,
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
                             dt_mf  : FIELD_FLOAT,
                             eta    : FIELD_FLOAT):
    
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
                
                    dp  = 1000.0 * del0
                    dpg = dp/g
                    
                    delhbar = delhbar + dellah * xmb * dpg
                    delqbar = delqbar + dellaq * xmb * dpg
                    deltbar = deltbar + dellat * xmb * dpg
                    delubar = delubar + dellau * xmb * dpg
                    delvbar = delvbar + dellav * xmb * dpg
        
        with interval(1, None):
            
            if cnvflg == 1:
                if k_idx > kb and k_idx <= ktcon:
                
                    dp  = 1000.0 * del0
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
    
    # TODO: LAST MISSING LOOP
    # ~ with computation(BACKWARD), interval(...):
    
        # ~ if k_idx <= kmax:
            
            # ~ deltv = 0.0
            # ~ delq  = 0.0
            # ~ qevap = 0.0
            
            # ~ if cnvflg == 1 and k_idx > kb and k_idx < ktcon:
                
                # ~ rn = rn + pwo * xmb * 0.001 * dt2
                
    
    with computation(PARALLEL), interval(...):
        
        if cnvflg == 1:
            
            if rn < 0.0 or flg == 0: rn = 0.0
           
            ktop = ktcon
            kbot = kbcon
            kcnv = 2
            
            if k_idx >= kbcon and k_idx < ktcon:
                
                # Calculate shallow convective cloud water
                cnvw = cnvwt * xmb * dt2
                
                # Calculate convective cloud cover, which is used when 
                # pdf-based cloud fraction is used
                val1 = 1.0 + 675.0 * eta * xmb
                cnvc = 0.04 * log(val1, 1.0e4)  # 1.0e4 seems to get reasonable results, since val1 is on average ~50
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
                              qtr    : FIELD_FLOAT,
                              kcon   : FIELD_FLOAT):
    
    with computation(PARALLEL), interval(...):
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
    
    with computation(PARALLEL), interval(...):
        
        # Store aerosol concentrations if present
        if cnvflg == 1 and rn > 0.0 and k_idx <= kmax:
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
    
    with computation(PARALLEL), interval(...):
        
        # Separate detrained cloud water into liquid and ice species as 
        # a function of temperature only
        if cnvflg == 1 and k_idx >= kbcon and k_idx <= ktcon:
            
            tem  = dellal * xmb * dt2
            tem1 = (tcr - t1) * tcrf if(((tcr - t1) * tcrf)<1.0) else 1.0 #min(1.0, (tcr - t1) * tcrf)
            tem1 = tem1 if(tem1 > 0.0) else 0.0 #max(0.0, tem1)
            
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
    
    with computation(PARALLEL), interval(1, -1):
        
        # Include TKE contribution from shallow convection
        if cnvflg == 1 and k_idx > kb and k_idx < ktop:
            
            tem      = 0.5 * (eta[0, 0, -1] + eta[0, 0, 0]) * xmb
            tem1     = pfld * 100.0/(rd * t1)
            sigmagfm = sigmagfm if(sigmagfm > betaw) else betaw #max(sigmagfm, betaw)
            ptem     = tem/(sigmagfm * tem1)
            qtr_ntk  = qtr_ntk + 0.5 * sigmagfm * ptem * ptem

########################################################################
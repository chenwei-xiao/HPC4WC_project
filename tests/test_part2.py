import numpy as np
import os
from .read_serialization import read_serialization_part2

def part2(ix,km,islimsk,dot,qtr,kpbl,kb,kbcon,kbcon1,ktcon,ktcon1,
          kbm,kmax,aa1,cina,tkemean,clamt,del0,edt,pdot,po,hmax,
          vshear,xlamud,pfld,to,qo,uo,vo,qeso,ctro,wu2,buo,drag,
          wc,dbyo,zo,xlamue,heo,heso,hcko,ucko,vcko,qcko,ecko,eta,
          zi,c0t,sumx,cnvflg,flg):
    return

def test_part2():
    data_dict = read_serialization_part2()
    ix = data_dict['ix']
    km = data_dict['km']
    islimsk = data_dict['islimsk']
    dot = data_dict['dot']
    qtr = data_dict['qtr']
    kpbl = data_dict['kpbl']
    kb = data_dict['kb']
    kbcon = data_dict['kbcon']
    kbcon1 = data_dict['kbcon1']
    ktcon = data_dict['ktcon']
    ktcon1 = data_dict['ktcon1']
    kbm = data_dict['kbm']
    kmax = data_dict['kmax']
    aa1 = data_dict['aa1']
    cina = data_dict['cina']
    tkemean = data_dict['tkemean']
    clamt = data_dict['clamt']
    del0 = data_dict['del']
    edt = data_dict['edt']
    pdot = data_dict['pdot']
    po = data_dict['po']
    hmax = data_dict['hmax']
    vshear = data_dict['vshear']
    xlamud = data_dict['xlamud']
    pfld = data_dict['pfld']
    to = data_dict['to']
    qo = data_dict['qo']
    uo = data_dict['uo']
    vo = data_dict['vo']
    qeso = data_dict['qeso']
    ctro = data_dict['ctro']
    wu2 = data_dict['wu2']
    buo = data_dict['buo']
    drag = data_dict['drag']
    wc = data_dict['wc']
    dbyo = data_dict['dbyo']
    zo = data_dict['zo']
    xlamue = data_dict['xlamue']
    heo = data_dict['heo']
    heso = data_dict['heso']
    hcko = data_dict['hcko']
    ucko = data_dict['ucko']
    vcko = data_dict['vcko']
    qcko = data_dict['qcko']
    ecko = data_dict['ecko']
    eta = data_dict['eta']
    zi = data_dict['zi']
    c0t = data_dict['c0t']
    sumx = data_dict['sumx']
    cnvflg = data_dict['cnvflg']
    flg = data_dict['flg']
    part2(ix, km, islimsk, dot, qtr, kpbl, kb, kbcon, kbcon1, ktcon, ktcon1,
          kbm, kmax, aa1, cina, tkemean, clamt, del0, edt, pdot, po, hmax,
          vshear, xlamud, pfld, to, qo, uo, vo, qeso, ctro, wu2, buo, drag,
          wc, dbyo, zo, xlamue, heo, heso, hcko, ucko, vcko, qcko, ecko, eta,
          zi, c0t, sumx, cnvflg, flg)

if __name__ == "__main__":
    test_part2()
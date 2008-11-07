#!/usr/bin/python
import numpy as N
from string import split
from string import replace

def sodar(filename):
#    '''
#        Read in SODAR datasets
#    '''
#
#   Open File
#
    try:
        f = open(filename)
        txtdata = f.readlines()
        f.close()
        while txtdata[0][0]=="#":
            txtdata.pop(0)

        date=[];time=[];height=[];ws=[];wd=[];u=[];v=[];w=[]
        sigw=[];bs=[];T=[];ec=[];ws2=[];wd2=[];ws_t=[];wd_t=[]
        wind_base=[];wsh=[];wshd=[];sigu_r=[];sigu=[];sigu_b=[]
        sigv_r=[];sigv=[];sigv_b=[];sigw_b=[];sigtheta=[];sigphi=[]
        pg_stab=[];bs_b=[];ct2=[];ct2b=[];ecu=[];ecv=[];ecw=[]
        ecT=[];confu=[];confv=[];confw=[];sgnu=[];sgnv=[];sgnw=[];sgnT=[]
        numdopimu=[];numdopimv=[];numdopimw=[];delt_u=[];delt_v=[]
        delt_w=[];delz_u=[];delz_v=[];delz_w=[]

        for line in txtdata:
            tmp = split(line,',');date.append(split(tmp[0],' ')[0]);time.append(split(tmp[0],' ')[1])
            for i in range(1,49):
                if((tmp[i]=='') | (tmp[i]=='\r\n') | (tmp[i]=='\n')):
                    tmp[i]=-999
                elif i==49:
                    tmp[49]=replace(tmp[49],'\r\n','')
            height.append(tmp[1]);ws.append(tmp[2]);wd.append(tmp[3]);u.append(tmp[4])
            v.append(tmp[5]);w.append(tmp[6]);sigw.append(tmp[7]);bs.append(tmp[8])
            T.append(tmp[9]);ec.append(tmp[10]);ws2.append(tmp[11]);wd2.append(tmp[12])
            ws_t.append(tmp[13]);wd_t.append(tmp[14]);wind_base.append(tmp[15]);wsh.append(tmp[16])
            wshd.append(tmp[17]);sigu_r.append(tmp[18]);sigu.append(tmp[19]);sigu_b.append(tmp[20])
            sigv_r.append(tmp[21]);sigv.append(tmp[22]);sigv_b.append(tmp[23]);sigw_b.append(tmp[24])
            sigtheta.append(tmp[25]);sigphi.append(tmp[26]);pg_stab.append(tmp[27]);bs_b.append(tmp[28])
            ct2.append(tmp[29]);ct2b.append(tmp[30]);ecu.append(tmp[31]);ecv.append(tmp[32])
            ecw.append(tmp[33]);ecT.append(tmp[34]);confu.append(tmp[35]);confv.append(tmp[36])
            confw.append(tmp[37]);sgnu.append(tmp[38]);sgnv.append(tmp[39]);sgnw.append(tmp[40])
            numdopimu.append(tmp[41]);numdopimv.append(tmp[42]);numdopimw.append(tmp[43]);delt_u.append(tmp[44])
            delt_v.append(tmp[45]);delt_w.append(tmp[46]);delz_u.append(tmp[47]);delz_v.append(tmp[48])
            delz_w.append(tmp[49])

        rec_length=N.array(height).astype('float').size/48

        ha=N.ma.MaskedArray(N.array(height).astype('float'),mask=N.array(height).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wsa=N.ma.MaskedArray(N.array(ws).astype('float'),mask=N.array(ws).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wda=N.ma.MaskedArray(N.array(wd).astype('float'),mask=N.array(wd).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ua=N.ma.MaskedArray(N.array(u).astype('float'),mask=N.array(u).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        va=N.ma.MaskedArray(N.array(v).astype('float'),mask=N.array(v).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wa=N.ma.MaskedArray(N.array(w).astype('float'),mask=N.array(w).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigwa=N.ma.MaskedArray(N.array(sigw).astype('float'),mask=N.array(sigw).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        bsa=N.ma.MaskedArray(N.array(bs).astype('float'),mask=N.array(bs).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        eca=N.ma.MaskedArray(N.array(ec).astype('int'),mask=N.array(ec).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wsha=N.ma.MaskedArray(N.array(wsh).astype('float'),mask=N.array(wsh).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wshda=N.ma.MaskedArray(N.array(wshd).astype('float'),mask=N.array(wshd).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        Ta=N.ma.MaskedArray(N.array(T).astype('float'),mask=N.array(T).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ws2a=N.ma.MaskedArray(N.array(ws2).astype('float'),mask=N.array(ws2).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wd2a=N.ma.MaskedArray(N.array(wd2),mask=N.array(wd2)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ws_ta=N.ma.MaskedArray(N.array(ws_t),mask=N.array(ws_t)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wd_ta=N.ma.MaskedArray(N.array(wd_t),mask=N.array(wd_t)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        wind_basea=N.ma.MaskedArray(N.array(wind_base),mask=N.array(wind_base)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigu_ra=N.ma.MaskedArray(N.array(sigu_r),mask=N.array(sigu_r)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigua=N.ma.MaskedArray(N.array(sigu),mask=N.array(sigu)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigu_ba=N.ma.MaskedArray(N.array(sigu_b),mask=N.array(sigu_b)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigv_ra=N.ma.MaskedArray(N.array(sigv_r),mask=N.array(sigv_r)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigva=N.ma.MaskedArray(N.array(sigv),mask=N.array(sigv)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigv_ba=N.ma.MaskedArray(N.array(sigv_b),mask=N.array(sigv_b)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigw_ba=N.ma.MaskedArray(N.array(sigw_b),mask=N.array(sigw_b)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigthetaa=N.ma.MaskedArray(N.array(sigtheta),mask=N.array(sigtheta)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sigphia=N.ma.MaskedArray(N.array(sigphi),mask=N.array(sigphi)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        pg_staba=N.ma.MaskedArray(N.array(pg_stab),mask=N.array(pg_stab)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        bs_ba=N.ma.MaskedArray(N.array(bs_b),mask=N.array(bs_b)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ct2a=N.ma.MaskedArray(N.array(ct2),mask=N.array(ct2)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ct2ba=N.ma.MaskedArray(N.array(ct2b),mask=N.array(ct2b)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ecua=N.ma.MaskedArray(N.array(ecu),mask=N.array(ecu)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ecva=N.ma.MaskedArray(N.array(ecv),mask=N.array(ecv)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ecwa=N.ma.MaskedArray(N.array(ecw),mask=N.array(ecw)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        ecTa=N.ma.MaskedArray(N.array(ecT),mask=N.array(ecT)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        confua=N.ma.MaskedArray(N.array(confu),mask=N.array(confu)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        confva=N.ma.MaskedArray(N.array(confv),mask=N.array(confv)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        confwa=N.ma.MaskedArray(N.array(confw),mask=N.array(confw)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sgnua=N.ma.MaskedArray(N.array(sgnu),mask=N.array(sgnu)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sgnva=N.ma.MaskedArray(N.array(sgnv),mask=N.array(sgnv)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sgnwa=N.ma.MaskedArray(N.array(sgnw),mask=N.array(sgnw)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        sgnTa=N.ma.MaskedArray(N.array(sgnT),mask=N.array(sgnT)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        numdopimua=N.ma.MaskedArray(N.array(numdopimu),mask=N.array(numdopimu)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        numdopimva=N.ma.MaskedArray(N.array(numdopimv),mask=N.array(numdopimv)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        numdopimwa=N.ma.MaskedArray(N.array(numdopimw),mask=N.array(numdopimw)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        delt_ua=N.ma.MaskedArray(N.array(delt_u),mask=N.array(delt_u)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        delt_va=N.ma.MaskedArray(N.array(delt_v),mask=N.array(delt_v)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        delt_wa=N.ma.MaskedArray(N.array(delt_w),mask=N.array(delt_w)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        delz_ua=N.ma.MaskedArray(N.array(delz_u),mask=N.array(delz_u)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        delz_va=N.ma.MaskedArray(N.array(delz_v),mask=N.array(delz_v)==-999,fill_value=-999).reshape(-1,rec_length).transpose()
        delz_wa=N.ma.MaskedArray(N.array(delz_w),mask=N.array(delz_w).astype('float')==-999,fill_value=-999).reshape(-1,rec_length).transpose()

        date=N.array(date).reshape(-1,rec_length).transpose();time=N.array(time).reshape(-1,rec_length).transpose()
    except IOError:
        print '%s does not exist\n'%filename
        raise

    return (ha,wsa,wda,ua,va,wa,sigwa,bsa,Ta,eca,ws2a,wd2a,ws_ta,wd_ta,wind_basea,wsha,wshda,sigu_ra,sigua,sigu_ba,sigv_r,sigv,sigv_ba,sigw_b,sigtheta,sigphi,pg_staba,bs_ba,ct2a,ct2ba,ecua,ecva,ecwa,confu,confv,confw,sgnu,sgnv,sgnw,sgnT,numdopimu,numdopimv,numdopimw,delt_u,delt_v,delt_w,delz_u,delz_v,delz_w,date,time)

if __name__=='__main__':
    import sys
    filename = sys.argv[1]

    data=sodar(filename)

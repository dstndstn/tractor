import numpy as np
import pyfits as pyf
import os
import sys
from tractor.rc3 import *
from astrometry.util.file import *
from astrometry.util.starutil_numpy import *
import matplotlib
import pylab as plt
from astropysics.obstools import *

def mu_50(i,r):
    return i+2.5*(log10(pi*r**2))

def extinction(pos):
    #where pos is the position
    
    #extinction values by filter for Sloan
    sloanu=5.155
    sloang=3.793
    sloanr=2.751
    sloani=2.086
    sloanz=1.479

    galactic=radectolb(pos[0],pos[1])
    print 'galactic',galactic
    x=get_SFD_dust(galactic[0], galactic[1],dustmap='ebv',interpolate=True)
    correction=[x*sloanu,x*sloang,x*sloanr,x*sloani,x*sloanz]
    correctu=float(correction[0])
    correctg=float(correction[1])
    correctr=float(correction[2])
    correcti=float(correction[3])
    correctz=float(correction[4])
    return correctu,correctg,correctr,correcti,correctz

def makeNSAtlastable():
    data=pyf.open("nsa-short.fits.gz")[1].data
    nsa_name=[]
    nsa_ra=[]
    nsa_dec=[]
    nsa_sersic_th50=[]
    nsa_nsid=[]
    cg_ra=[]
    cg_dec=[]
    cg_totmags=[]
    cg_devmags=[]
    cg_devre=[]
    cg_devab=[]
    cg_devphi=[]
    cg_expmags=[]
    cg_expre=[]
    cg_expab=[]
    cg_expphi=[]
    cg_r50s=[]
    cg_r90s=[]
    cg_conc=[]
    cg_extinction=[]
    cg_mu50=[]
    sloanu=5.155
    sloang=3.793
    sloanr=2.751
    sloani=2.086
    sloanz=1.479


    os.chdir("NSAtlas_Output/")
    
    
    #print extinction('NGC_3884-updated.pickle')
    


    for files in os.listdir("."):
        if files.endswith("-updated.pickle"):
            print files
            # this gives the tractor results 
            #data from rc3
            strip=files.rstrip('-updated.pickle')
            replace=strip.replace('_',' ')
            nsa_name.append(replace)
            nsid = float (replace.split()[2])
            e=data.field('NSAID')
            mask = e ==nsid
            record = data[mask]
            ra = float(data['RA'][0])
            dec = float(data['DEC'][0])
            sersic_th50 = float(data['SERSIC_TH50'][0])
            nsa_ra.append(ra)
            nsa_dec.append(dec)
            nsa_nsid.append(nsid)
            nsa_sersic_th50.append(sersic_th50)
            print ra,dec,nsid,sersic_th50
            cg_ra=[]
            cg_dec=[]
            cg_totmags=[]
            cg_devmags=[]
            cg_devre=[]
            cg_devab=[]
            cg_devphi=[]
            cg_expmags=[]
            cg_expre=[]
            cg_expab=[]
            cg_expphi=[]
            cg_r50s=[]
            cg_r90s=[]
            cg_conc=[]
            cg_extinction=[]
            cg_mu50=[]
            CG,r50s,r90s,concs=unpickle_from_file(files)
            pos=CG.getPosition()
            tot=CG.getBrightness()
            print tot, 'tot mags'

            dev=CG.brightnessDev
            dev_re=CG.shapeDev.re
            dev_ab=CG.shapeDev.ab
            dev_phi=CG.shapeDev.phi
            
            exp=CG.brightnessExp
            exp_re=CG.shapeExp.re
            exp_ab=CG.shapeExp.ab
            exp_phi=CG.shapeExp.phi
            
            cg_ra.append(pos[0])
            cg_dec.append(pos[1])
            cg_totmags.append(tot)
            cg_devmags.append(dev)
            cg_devre.append(dev_re)
            cg_devab.append(dev_ab)
            cg_devphi.append(dev_phi)
            cg_expmags.append(exp)
            cg_expre.append(exp_re)
            cg_expab.append(exp_ab)
            cg_expphi.append(exp_phi)
            cg_r50s.append(r50s)
            cg_r90s.append(r90s)
            cg_conc.append(concs)

            #get extinction from SFD
            galactic=radectolb(pos[0],pos[1])
            print 'galactic',galactic
            x=get_SFD_dust(galactic[0], galactic[1],dustmap='ebv',interpolate=True)
            correction=[x*sloanu,x*sloang,x*sloanr,x*sloani,x*sloanz]
            correctu=float(correction[0])
            correctg=float(correction[1])
            correctr=float(correction[2])
            correcti=float(correction[3])
            correctz=float(correction[4])
            cg_extinctiontemp=[]
            cg_extinctiontemp.append(correctu)
            cg_extinctiontemp.append(correctg)
            cg_extinctiontemp.append(correctr)
            cg_extinctiontemp.append(correcti)
            cg_extinctiontemp.append(correctz)
            cg_extinction.append(cg_extinctiontemp)
            #print cg_extinction
            
            imag_corrected=tot[3]-correcti
            print tot[3],correcti,imag_corrected
            print r50s[3]
            y=mu_50(imag_corrected,r50s[3])
    
            cg_mu50.append(mu_50(imag_corrected,r50s[3]))

            
    # print rc3_name
    # print rc3_ra
    # print rc3_dec
    # print rc3_log_ae
    # print rc3_log_d25
    print cg_ra, cg_dec
    print cg_totmags
    # print cg_devmags
    # print cg_devre
    # print cg_devab
    # print cg_devphi
    # print cg_conc
    print cg_extinction 
    n=np.arange(100)
    hdu=pyf.PrimaryHDU(n)
    a1=np.array(nsa_name)
    a2=np.array(nsa_ra)
    a3=np.array(nsa_dec)
    a4=np.array(nsa_nsid)
    a5=np.array(nsa_sersic_th50)
    a6=np.array(cg_ra)
    a7=np.array(cg_dec)
    a8=np.array(cg_r50s)
    a9=np.array(cg_r90s)
    a10=np.array(cg_conc)
    a11=np.array(cg_totmags)
    a12=np.array(cg_devre)
    a13=np.array(cg_devab)
    a14=np.array(cg_devphi)
    a15=np.array(cg_devmags)
    a16=np.array(cg_expre)
    a17=np.array(cg_expab)
    a18=np.array(cg_expphi)
    a19=np.array(cg_expmags)
    a20=np.array(cg_extinction)
    a21=np.array(cg_mu50)
    col1=pyf.Column(name='NSA_NAME',format='10A',array=a1)
    col2=pyf.Column(name='NSA_RA',format='1E',array=a2)
    col3=pyf.Column(name='NSA_DEC',format='1E',array=a3)
    col4=pyf.Column(name='NSA_NSID',format='1E',array=a4)
    col5=pyf.Column(name='NSA_SESIC_TH50',format='1E',array=a5)
    col6=pyf.Column(name='CG_RA',format='1E',array=a6)
    col7=pyf.Column(name='CG_DEC',format='1E',array=a7)
    col8=pyf.Column(name='CG_R50S',format='5E',array=a8)
    col9=pyf.Column(name='CG_R90S',format='5E',array=a9)
    col10=pyf.Column(name='CG_CONC',format='5E',array=a10)
    col11=pyf.Column(name='CG_TOTMAGS',format='5E',array=a11)
    col12=pyf.Column(name='CG_DEVRE',format='1E',array=a12)
    col13=pyf.Column(name='CG_DEVAB',format='1E',array=a13)
    col14=pyf.Column(name='CG_DEVPHI',format='1E',array=a14)
    col15=pyf.Column(name='CG_DEVMAGS',format='5E',array=a15)
    col16=pyf.Column(name='CG_EXPRE',format='1E',array=a16)
    col17=pyf.Column(name='CG_EXPAB',format='1E',array=a17)
    col18=pyf.Column(name='CG_EXPPHI',format='1E',array=a18)
    col19=pyf.Column(name='CG_EXPMAGS',format='5E',array=a19)
    col20=pyf.Column(name='CG_EXTINCTION',format='5E',array=a20)
    col21=pyf.Column(name='CG I-SB', format='1E',array=a21)
    cols=pyf.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21])
    tbhdu=pyf.new_table(cols)
    tbhdulist=pyf.HDUList([hdu,tbhdu])
    os.chdir("../")
    tbhdulist.writeto('nsa_galaxies.fits',clobber=True)
    print os.pwd()



def makeRC3table():
    rc3=pyf.open('newrc3limited.fits')
    sloanu=5.155
    sloang=3.793
    sloanr=2.751
    sloani=2.086
    sloanz=1.479


    rc3_name=[]
    rc3_ra=[]
    rc3_dec=[]
    rc3_log_ae=[]
    rc3_log_d25=[]
    cg_ra=[]
    cg_dec=[]
    cg_totmags=[]
    cg_devmags=[]
    cg_devre=[]
    cg_devab=[]
    cg_devphi=[]
    cg_expmags=[]
    cg_expre=[]
    cg_expab=[]
    cg_expphi=[]
    cg_r50s=[]
    cg_r90s=[]
    cg_conc=[]
    cg_extinction=[]
    cg_mu50=[]

    os.chdir("RC3_Output/updated_pickle/")
    
    
    #print extinction('NGC_3884-updated.pickle')
    


    for files in os.listdir("."):
        if files.endswith("-updated.pickle"):
            print files
            # this gives the tractor results 
            #data from rc3
            strip=files.rstrip('-updated.pickle')
            replace=strip.replace('_',' ')
            rc3_name.append(replace)
            rc3=getName(replace)
            ra = float(rc3['RA'][0])
            dec = float(rc3['DEC'][0])
            log_ae = float(rc3['LOG_AE'][0])
            log_d25 = float(rc3['LOG_D25'][0])
            rc3_ra.append(ra)
            rc3_dec.append(dec)
            rc3_log_ae.append(log_ae)
            rc3_log_d25.append(log_d25)
            print ra,dec,log_ae,log_d25
            cg_ra=[]
            cg_dec=[]
            cg_totmags=[]
            cg_devmags=[]
            cg_devre=[]
            cg_devab=[]
            cg_devphi=[]
            cg_expmags=[]
            cg_expre=[]
            cg_expab=[]
            cg_expphi=[]
            cg_r50s=[]
            cg_r90s=[]
            cg_conc=[]
            cg_extinction=[]
            cg_mu50=[]
            CG,r50s,r90s,concs=unpickle_from_file(files)
            pos=CG.getPosition()
            tot=CG.getBrightness()
            print tot, 'tot mags'

            dev=CG.brightnessDev
            dev_re=CG.shapeDev.re
            dev_ab=CG.shapeDev.ab
            dev_phi=CG.shapeDev.phi
            
            exp=CG.brightnessExp
            exp_re=CG.shapeExp.re
            exp_ab=CG.shapeExp.ab
            exp_phi=CG.shapeExp.phi
            
            cg_ra.append(pos[0])
            cg_dec.append(pos[1])
            cg_totmags.append(tot)
            cg_devmags.append(dev)
            cg_devre.append(dev_re)
            cg_devab.append(dev_ab)
            cg_devphi.append(dev_phi)
            cg_expmags.append(exp)
            cg_expre.append(exp_re)
            cg_expab.append(exp_ab)
            cg_expphi.append(exp_phi)
            cg_r50s.append(r50s)
            cg_r90s.append(r90s)
            cg_conc.append(concs)

            #get extinction from SFD
            galactic=radectolb(pos[0],pos[1])
            print 'galactic',galactic
            x=get_SFD_dust(galactic[0], galactic[1],dustmap='ebv',interpolate=True)
            correction=[x*sloanu,x*sloang,x*sloanr,x*sloani,x*sloanz]
            correctu=float(correction[0])
            correctg=float(correction[1])
            correctr=float(correction[2])
            correcti=float(correction[3])
            correctz=float(correction[4])
            cg_extinctiontemp=[]
            cg_extinctiontemp.append(correctu)
            cg_extinctiontemp.append(correctg)
            cg_extinctiontemp.append(correctr)
            cg_extinctiontemp.append(correcti)
            cg_extinctiontemp.append(correctz)
            cg_extinction.append(cg_extinctiontemp)
            #print cg_extinction
            
            imag_corrected=tot[3]-correcti
            print tot[3],correcti,imag_corrected
            print r50s[3]
            y=mu_50(imag_corrected,r50s[3])
    
            cg_mu50.append(mu_50(imag_corrected,r50s[3]))

            
    # print rc3_name
    # print rc3_ra
    # print rc3_dec
    # print rc3_log_ae
    # print rc3_log_d25
    # print cg_ra, cg_dec
    # print cg_totmags
    # print cg_devmags
    # print cg_devre
    # print cg_devab
    # print cg_devphi
    # print cg_conc
    print cg_extinction 
    n=np.arange(100)
    hdu=pyf.PrimaryHDU(n)
    a1=np.array(rc3_name)
    a2=np.array(rc3_ra)
    a3=np.array(rc3_dec)
    a4=np.array(rc3_log_ae)
    a5=np.array(rc3_log_d25)
    a6=np.array(cg_ra)
    a7=np.array(cg_dec)
    a8=np.array(cg_r50s)
    a9=np.array(cg_r90s)
    a10=np.array(cg_conc)
    a11=np.array(cg_totmags)
    a12=np.array(cg_devre)
    a13=np.array(cg_devab)
    a14=np.array(cg_devphi)
    a15=np.array(cg_devmags)
    a16=np.array(cg_expre)
    a17=np.array(cg_expab)
    a18=np.array(cg_expphi)
    a19=np.array(cg_expmags)
    a20=np.array(cg_extinction)
    a21=np.array(cg_mu50)
    col1=pyf.Column(name='RC3_NAME',format='10A',array=a1)
    col2=pyf.Column(name='RC3_RA',format='1E',array=a2)
    col3=pyf.Column(name='RC3_DEC',format='1E',array=a3)
    col4=pyf.Column(name='RC3_LOG_AE',format='1E',array=a4)
    col5=pyf.Column(name='RC3_LOG_D25',format='1E',array=a5)
    col6=pyf.Column(name='CG_RA',format='1E',array=a6)
    col7=pyf.Column(name='CG_DEC',format='1E',array=a7)
    col8=pyf.Column(name='CG_R50S',format='5E',array=a8)
    col9=pyf.Column(name='CG_R90S',format='5E',array=a9)
    col10=pyf.Column(name='CG_CONC',format='5E',array=a10)
    col11=pyf.Column(name='CG_TOTMAGS',format='5E',array=a11)
    col12=pyf.Column(name='CG_DEVRE',format='1E',array=a12)
    col13=pyf.Column(name='CG_DEVAB',format='1E',array=a13)
    col14=pyf.Column(name='CG_DEVPHI',format='1E',array=a14)
    col15=pyf.Column(name='CG_DEVMAGS',format='5E',array=a15)
    col16=pyf.Column(name='CG_EXPRE',format='1E',array=a16)
    col17=pyf.Column(name='CG_EXPAB',format='1E',array=a17)
    col18=pyf.Column(name='CG_EXPPHI',format='1E',array=a18)
    col19=pyf.Column(name='CG_EXPMAGS',format='5E',array=a19)
    col20=pyf.Column(name='CG_EXTINCTION',format='5E',array=a20)
    col21=pyf.Column(name='CG I-SB', format='1E',array=a21)
    cols=pyf.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21])
    tbhdu=pyf.new_table(cols)
    tbhdulist=pyf.HDUList([hdu,tbhdu])
    os.chdir("../../")
    tbhdulist.writeto('large_galaxies.fits',clobber=True)

#try to run on all types of pickle files with differents names while going back one directory

if __name__ == '__main__':
    makeNSAtlastable()

import os
import numpy as np
import pylab as plt
import pyfits
import sys


#Work in progress...
def main():
    rc3 = pyfits.open('rc3limited.fits')
    entries=[]
    for entry in rc3[1].data:
        print (10**entry['LOG_D25'])/10.
        fn = '%s.pickle' % (entry['NAME'])
        if os.path.exists(fn):
            print '%s has run successfully already' %entry['NAME'] 
            continue 
        else:
            print 'run %s through tractor' %entry['NAME']
            entries.append('%s' %entry['NAME'])
    things=[str(x) for x in entries]
    
    for entry in things:
        print 'entry'
        newentry=entry.replace(' ', '')
        print newentry
        #assert(False)
        os.system("python -u general.py '%s' --threads 4 --itune1 6 --itune2 6 --nocache 1>%s.log 2>%s_err.log" % (entry,entry,entry))
        print 'running tractor for %s' %entry
        assert(False)
        os.system('cp flip-%s.pdf RC3_Output' % entry)
        os.system('cp ngc-%s.png RC3_Output' % entry)
        os.system('cp flip-%s.tex RC3_Output' % entry)
        
            
#some rc3 entries dont have a name in the 'NAME' data field. what to do with those? Also, the space in the name NGC 1234 is causing problems with the os.system commands. run this file and you will see what i am talking about. thats all that needs to be fixed and everything else should be good to go

#3053 galaxies
if __name__ == '__main__':
    main()

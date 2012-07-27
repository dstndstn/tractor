import os
import numpy as np
import pylab as plt
import pyfits

#Work in progress...
def main():
    rc3 = pyfits.open('rc3limited.fits')
    for entry in rc3[1].data:
        fn = '%s.pickle' % (entry['NAME'])
        if os.path.exists(fn):
            continue
        else:
            pass


if __name__ == '__main__':
    main()

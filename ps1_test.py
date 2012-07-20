import ps1, bigboss_test

def test(imfn, radecroi=(334.1, 334.4, 0.1, 0.4)):
    basedir = '/project/projectdirs/bigboss/data/cfht/w4/megapipe'
    field = 'W4+1-1.I'
    filtermap = { 'i.MP9701': 'i' }
    
    im = ps1.get_ps1_chip_image(imfn, offset=(2000,2500), npix=(1000,1000))
    import matplotlib
    matplotlib.use('Agg')
    bigboss_test.make_plots('ps1-', im, mags=['PS1_i'], radecroi_in=radecroi)

if __name__ == "__main__":
    import matplotlib
    import cPickle as pickle
    imlist = pickle.load(open('/global/u1/s/schlafly/imlist.pkl'))
    matplotlib.use('Agg')
    test(imlist[1][28764])

# some image: imlist[1][24], ROI: 334.0, 334.1, 0.1, 0.2)
# i band image: imlist[1][28764], ROI: (334.15, 334.25, 0.1, 0.2)

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import pyfits

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import *

ps = PlotSequence('w4')
T = fits_table('wisew4phot.fits')
print 'Read', len(T)

T.cut(T.done > 0)
print 'Cut to', len(T), 'done'

print 'RA,Dec', T.ra.min(), T.ra.max(), T.dec.min(), T.dec.max()

print 'fit_ok_r:', np.unique(T.fit_ok_r)
T.cut(T.fit_ok_r)
print len(T), 'fit_ok_r'

ha = dict(bins=50, histtype='step', range=(-2,5))
plt.clf()
for bb,cc in zip('ugriz', 'bgrmk'):
    nm = T.get('sdss_%s_nanomaggies' % bb)
    #nm = nm[(nm != 0) * (nm != 1)]
    nm = nm[(nm != 0)]
    plt.hist(nm, color=cc, **ha)
ps.savefig()

ha = dict(bins=50, histtype='step', range=(18,27))
plt.clf()
for bb,cc in zip('ugriz', 'bgrmk'):
    # nm = T.get('sdss_%s_nanomaggies' % bb)
    # nm = nm[(nm != 0) * (nm != 1)]
    # nm = nm[nm > 0.]
    # plt.hist(-2.5 * (np.log10(nm) - 9.), color=cc, **ha)
    plt.hist(T.get('sdss_%s_mag' % bb), color=cc, **ha)
    plt.xlabel('mag')
ps.savefig()

for bb,cc in zip('ugriz', 'bgrmk'):
    nm = T.get('sdss_%s_nanomaggies' % bb)
    I = (nm > 0) * (nm != 1)
    nm = nm[I]
    #print sum(nm == 1), '== 1'
    w4 = T.w4mpro[I]
    mag = -2.5 * (np.log10(nm) - 9.)
    plt.clf()
    loghist(w4, mag - w4, 200, range=((0,9),(10,25)))
    plt.xlabel('W4 (mag)')
    plt.ylabel('%s - W4 (mag)' % (bb))
    plt.title('SDSS forced photometry: %s band' % bb)
    xl,xh = plt.xlim()
    plt.xlim(xh,xl)
    ps.savefig()


I = match_radec(T.ra, T.dec, T.ra, T.dec, 0.5, notself=True,
                indexlist=True)
N = np.zeros(len(T), int)
for i,jj in enumerate(I):
    if jj is None:
        continue
    N[i] = len(jj)

# plt.clf()
# plt.hist(N)
# plt.xlabel('N within 1/2 degree')
# ps.savefig()

plt.clf()
loghist(T.ra, T.dec, 200)
plt.xlabel('RA')
plt.ylabel('Dec')
ps.savefig()

plt.clf()
plt.hist(np.minimum(N, 50), 50)
plt.xlabel('N within 1/2 degree')
ps.savefig()

T.cut(N < 10)
print 'Removing close groups:', len(T)

plt.clf()
loghist(T.ra, T.dec, 200)
plt.xlabel('RA')
plt.ylabel('Dec')
ps.savefig()
    
if False:
    # Bright cloud of r-W4 ~ 12
    J = (T.sdss_r_mag < 22.)
    print sum(J), 'in blue cloud'
    I = np.argsort(T.sdss_r_mag)
    Tj = T[I[:100]]
    write_file('\n'.join([
        '<img src="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f">' % (r,d)
        for r,d in zip(Tj.ra, Tj.dec)]),
        'blue.html')

T.sdss_r_mag[np.logical_not(np.isfinite(T.sdss_r_mag))] = 30.

#T.cut(np.logical_or(T.sdss_r_nanomaggies <= 0.,
#                     T.sdss_r_mag > 23.))
#print len(T), 'faint in r'

T.cut(T.sdss_r_mag > 23.)
print len(T), 'faint in r'

#T.cut(np.logical_or(T.sdss_r_nanomaggies <= 0.,
#                     T.sdss_r_mag - T.w4mpro > 14.))

T.cut(T.sdss_r_mag - T.w4mpro > 14.)
print len(T), 'after r-W4 > 14 cut'

T.cut(T.prochi2_r < 5.)
print 'Cut to', len(T), 'sources on prochi2_r < 5'


T.writeto('w4targets-cut.fits')

plt.clf()
loghist(T.ra, T.dec, 200)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('%i sources' % len(T))
ps.savefig()


# plt.clf()
# ha = dict(bins=50, histtype='step', range=(18,25))
# plt.hist(T.sdss_r_mag[J], color='b', **ha)
# plt.hist(T.sdss_r_mag[np.logical_not(J)], color='r', **ha)
# plt.xlabel('r mag')
# plt.title('"red clump" sources')
# ps.savefig()

#T.cut(np.logical_not(J)) # * (T.r > 23))
#print 'sdss_r_mag:', np.min(T.sdss_r_mag), np.max(T.sdss_r_mag)

plt.clf()
loghist(T.w4mpro, T.sdss_r_mag - T.w4mpro, 200, range=((0,9),(10,25)))
plt.xlabel('W4 (mag)')
plt.ylabel('r - W4 (mag)')
xl,xh = plt.xlim()
plt.xlim(xh,xl)
ps.savefig()

I = np.argsort(T.sdss_r_mag)
Tj = T[I[:100]]

write_file('<table>' + '\n'.join([
    '<tr><td>%f<td><img src="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f"></tr>' % (f, r,d)
    for f, r,d in zip(Tj.sdss_r_mag, Tj.ra, Tj.dec)]) + '</table>',
    'red.html')

# for rcut in [23, 24]:
#     I = np.flatnonzero(T.sdss_r_mag > rcut)
#     J = I[np.argsort(T.sdss_r_mag[I])]
#     Tj = T[J[:100]]
# 
#     write_file('<table>' + '\n'.join([
#         '<tr><td>%f<td><img src="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f"></tr>' % (f, r,d)
#         for f, r,d in zip(Tj.sdss_r_mag, Tj.ra, Tj.dec)]) + '</table>',
#         'red-%i.html' % rcut)
#T.cut(T.sdss_r_mag > 23.)

# plt.clf()
# plt.hist(T.profracflux_r, 50)
# plt.xlabel('pro frac flux')
# ps.savefig()

#plt.clf()
#plt.hist(np.log10(T.proflux_r), 50)
#plt.xlabel('log pro flux')
#ps.savefig()

plt.clf()
plt.hist(np.log10(T.w4rchi2), 50)
plt.xlabel('log w4 rchi2')
ps.savefig()

plt.clf()
plt.hist(T.npix_r, 50)
plt.xlabel('npix r')
ps.savefig()

plt.clf()
plt.hist(T.prochi2_r, 50)
plt.xlabel('pro chi2 r')
ps.savefig()

plt.clf()
plt.hist(T.sdss_r_mag_err, 50, range=(0,1))
plt.xlabel('r mag err')
ps.savefig()

I = np.argsort(T.prochi2_r)
Tj = T[I[:100]]
write_file('\n'.join([
    '<img src="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f">' % (r,d)
    for r,d in zip(Tj.ra, Tj.dec)]),
    'goodchi.html')

Tj = T[I[-1:-100:-1]]

write_file('<table>' + '\n'.join([
    '<tr><td>%f<td><img src="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f"></tr>' % (f, r,d)
    for f, r,d in zip(Tj.prochi2_r, Tj.ra, Tj.dec)]) + '</table>',
    'badchi.html')

# write_file('\n'.join([
#     '<img src="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f">' % (r,d)
#     for r,d in zip(Tj.ra, Tj.dec)]),
#     'badchi.html')


# plt.clf()
# loghist(T.ra, T.dec, 200)
# plt.xlabel('RA')
# plt.ylabel('Dec')
# ps.savefig()




plt.clf()
plt.plot(np.vstack([
    T.sdss_u_mag, T.sdss_g_mag, T.sdss_r_mag, T.sdss_i_mag,
    T.sdss_z_mag,
    T.w1mpro, T.w2mpro, T.w3mpro, T.w4mpro]), 'k-',
    alpha=0.01)
plt.xticks(np.arange(9), ['u','g','r','i','z','W1','W2','W3','W4'])
plt.ylim(5, 30)
plt.ylabel('Mag')
ps.savefig()

I = np.argsort(T.sdss_r_mag)
Tj = T[I[:100]]
for t in Tj[:20]:
    print t.ra, t.dec

I = np.argsort(T.sdss_r_mag)
Tj = T[I[:400]]

write_file('\n'.join([
    '<img src="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f">' % (r,d)
    for r,d in zip(Tj.ra, Tj.dec)]),
    'good.html')

urls = ['http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?w=100&h=100&scale=0.4&ra=%.4f&dec=%.4f' % (r,d)
       for r,d in zip(Tj.ra, Tj.dec)]
fns = ['sdss-rgb/sdss-rgb-%.4f_%.4f.jpg' % (r,d) for r,d in zip(Tj.ra, Tj.dec)]

for i,(fn,url) in enumerate(zip(fns, urls)):
    if os.path.exists(fn):
        print 'got', i, fn
        continue
    cmd = 'wget "%s" -O %s' % (url, fn)
    os.system(cmd)

ims = []
for fn in fns:
    print 'reading', fn
    try:
        ims.append(plt.imread(fn).astype(np.float32) / 255.)
    except:
        pass
ims = np.array(ims)
print ims.shape, ims.dtype, ims.max()
medim = np.median(ims, axis=0)
print 'median', medim.shape, medim.dtype, medim.max()

meanim = np.mean(ims, axis=0)
print 'mean', meanim.shape, meanim.dtype, meanim.max()

plt.clf()
plt.imshow(medim / medim.max(), interpolation='nearest', origin='lower')
ps.savefig()

plt.clf()
plt.imshow(meanim / meanim.max(), interpolation='nearest', origin='lower')
ps.savefig()


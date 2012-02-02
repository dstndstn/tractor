if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import logging
import numpy as np
import pylab as plt

import pyfits

from astrometry.util.file import *

from tractor import *
from tractor import sdss as st

def save(idstr, tractor, zr,debug=False,plotAll=False,imgi=0):
	print "Index: ", imgi
	mod = tractor.getModelImages()[imgi]
	chi = tractor.getChiImages()[imgi]

	synthfn = 'synth-%s.fits' % idstr
	print 'Writing synthetic image to', synthfn
	pyfits.writeto(synthfn, mod, clobber=True)

	pfn = 'tractor-%s.pickle' % idstr
	print 'Saving state to', pfn
	pickle_to_file(tractor, pfn)

	timg = tractor.getImage(imgi)
	data = timg.getImage()
	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])
	imchi = ima.copy()
	imchi.update(vmin=-10, vmax=10)
	
	if debug:
		sources = tractor.getCatalog()
		wcs = timg.getWcs()
		allobjx = []
		allobjy = []
		allobjc = []
		pointx = []
		pointy = []
		xplotx = []
		xploty = []

		for obj in sources:
			if (isinstance(obj,PointSource)):
				xt,yt = wcs.positionToPixel(None,obj.getPosition())
				pointx.append(xt)
				pointy.append(yt)
				continue
			print type(obj)
			shapes = []
			attrType = []
			if (isinstance(obj,st.CompositeGalaxy)):
				for attr in 'shapeExp', 'shapeDev':
					shapes.append(getattr(obj, attr))
					attrType.append(attr)
			else:
				shapes.append(getattr(obj,'shape'))
				attrType.append(' ')
			x0,y0 = wcs.positionToPixel(None,obj.getPosition())
			
			cd = timg.getWcs().cdAtPixel(x0,y0)
			print "CD",cd
			for i,shape in enumerate(shapes):
				xplotx.append(x0)
				xploty.append(y0)
				T=np.linalg.inv(shape.getTensor(cd))
				print "Inverted tensor:",T
				print obj.getPosition()
				print i

				x,y = [],[]
				for theta in np.linspace(0,2*np.pi,100):
					ux = np.cos(theta)
					uy = np.sin(theta)
					dx,dy = np.dot(T,np.array([ux,uy]))
					x.append(x0+dx)
					y.append(y0+dy)
				allobjx.append(x)
				allobjy.append(y)
				if (attrType[i] == 'shapeExp'):
					allobjc.append('b')
				elif attrType[i] == 'shapeDev':
					allobjc.append('g')
				else:
					allobjc.append('r')

	# Make a non-linear stretched map using image "I" to set the limits:

	ss = np.sort(data.ravel())
	mn,mx = [ss[int(p*len(ss))] for p in [0.1, 0.99]]
	q1,q2,q3 = [ss[int(p*len(ss))] for p in [0.25, 0.5, 0.75]]

	def nlmap(X):
		Y = (X - q2) / ((q3-q1)/2.)
		return np.arcsinh(Y * 10.)/10.
	def myimshow(x, *args, **kwargs):
		mykwargs = kwargs.copy()
		if 'vmin' in kwargs:
			mykwargs['vmin'] = nlmap(kwargs['vmin'])
		if 'vmax' in kwargs:
			mykwargs['vmax'] = nlmap(kwargs['vmax'])
		return plt.imshow(nlmap(x), *args, **mykwargs)

	def savepng(pre, img, title=None,**kwargs):
		fn = '%s-%s.png' % (pre, idstr)
		print 'Saving', fn
		plt.clf()

		#Raises an error otherwise... no idea why --dm
		np.seterr(under='print')
		

		if kwargs['vmin'] == -10:
			plt.imshow(img, **kwargs)
		else:
			myimshow(img,**kwargs)
		ax = plt.axis()
		if debug:
			print len(xplotx),len(allobjx)
			for i,(objx,objy,objc) in enumerate(zip(allobjx,allobjy,allobjc)):
				plt.plot(objx,objy,'-',c=objc)
				tempx = []
				tempx.append(xplotx[i])
				tempx.append(objx[0])
				tempy = []
				tempy.append(xploty[i])
				tempy.append(objy[0])
				plt.plot(tempx,tempy,'-',c='purple')
			plt.plot(pointx,pointy,'y.')
			plt.plot(xplotx,xploty,'xg')
		plt.axis(ax)
		if title is not None:
			plt.title(title)
		plt.colorbar()
		plt.gray()
		plt.savefig(fn)

	sky = np.median(mod)
	savepng('data', data - sky, title='Data '+timg.name, **ima)
	savepng('model', mod - sky, title='Model', **ima)
	savepng('diff', data - mod, title='Data - Model', **ima)
	savepng('chi',chi,title='Chi',**imchi)
	print "Chi mean: ", np.mean(chi)
	print "Chi median: ", np.median(chi)
	if plotAll:
		debug = False
		for i,src in enumerate(tractor.getCatalog()):
			savepng('data-s%i'%(i+1),data - sky, title='Data '+timg.name,**ima)
			modelimg = tractor.getModelImage(timg, srcs=[src])
			savepng('model-s%i'%(i+1), modelimg - sky, title='Model-s%i'%(i+1),**ima) 
			savepng('diff-s%i'%(i+1), data - modelimg, title='Model-s%i'%(i+1),**ima)
			savepng('chi-s%i'%(i+1),tractor.getChiImage(imgi,srcs=[src]),title='Chi',**imchi)


def main():
    run = 2863
    field = 180
    camcol = 4
    x0 = 200
    x1 = 900
    y0 = 400
    y1 = 850

    roi = [x0,x1,y0,y1]
    
    bands=['r','g','i','u','z']
    bandname = 'r'

    rerun = 0

    TI = []
    TI.extend([st.get_tractor_image(run, camcol, field, bandname,roi=roi,useMags=True) for bandname in bands])
    sources = st.get_tractor_sources(run, camcol, field,bandname, bands=bands,roi=roi)

    timg,info = TI[0]
    photocal = timg.getPhotoCal()

    wcs = timg.getWcs()
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)

    tims = [timg for timg,tinf in TI]
    tractor = st.SDSSTractor(tims)
    tractor.addSources(sources)

    zr = np.array([-5.,+5.]) * info['skysig']

    print bands

    for src in sources:
        if isinstance(src,st.CompositeGalaxy):
            x,y = wcs.positionToPixel(src,src.getPosition())
            if (80 < x < 100 and 275 < y < 310):
                print src,x,y
                tractor.removeSource(src)


    for i,band in enumerate(bands):
        save('initial-%s-' % (band), tractor,zr,debug=True,imgi=i)

    for i in range(4):
        for src in tractor.getCatalog():
            print src
            tractor.optimizeCatalogLoop(nsteps=4,srcs=[src],sky=False)
        for j,band in enumerate(bands):
            save('tune-%d-%s-' % (i+1,band),tractor,zr,debug=True)
        tractor.clearCache()
if __name__ == '__main__':
    import cProfile
    import sys
    from datetime import tzinfo, timedelta, datetime
    cProfile.run('main()','prof-%s.dat' % (datetime.now().isoformat()))
    sys.exit(0)

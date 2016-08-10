from __future__ import print_function
import os
import logging
import numpy as np
import pylab as plt
import pyfits

from astrometry.util.file import *
from astrometry.util.plotutils import ArcsinhNormalize

from tractor import *
from tractor import sdss as st


def save(idstr, tractor, nlscale=1.,debug=False,plotAll=False,imgi=0,chilo=-10.,chihi=10.):
	print("Index: ", imgi)
	mod = tractor.getModelImage(imgi)
	chi = tractor.getChiImage(imgi=imgi)
	synthfn = 'synth-%s.fits' % idstr
	print('Writing synthetic image to', synthfn)
	pyfits.writeto(synthfn, mod, clobber=True)

	#pfn = 'tractor-%s.pickle' % idstr
	#print 'Saving state to', pfn
	#pickle_to_file(tractor, pfn)

	plt.clf()
	plt.hist(chi.ravel(),range=(-10,10), bins=100)
	plt.savefig('chi2.png')

	timg = tractor.getImage(imgi)
	data = timg.getImage()
	print('Mod type:', mod.dtype)
	print('Chi type:', chi.dtype)
	print('Data type:', data.dtype)
	zr = timg.zr
	print('zr', zr)
	# Set up nonlinear mapping based on the statistics of the data image.
	#sigma = np.median(timg.getInvError())
	#print 'sigma', sigma
	ima = dict(interpolation='nearest', origin='lower')
	if nlscale == 0.:
		ima.update(vmin=zr[0], vmax=zr[1])
	else:
		q1,q2,q3 = np.percentile(data.ravel(), [25, 50, 75])
		print('Data quartiles:', q1, q2, q3)
		ima.update(norm = ArcsinhNormalize(mean=q2, std=(1./nlscale) * (q3-q1)/2., 
						   vmin=zr[0], vmax=zr[1]))

	imchi = ima.copy()
	if nlscale ==0. or True:
		imchi.update(vmin=chilo, vmax=chihi, norm=None)
	else:
		imchi.update(norm=ArcsinhNormalize(mean=0., std=1./nlscale, vmin=chilo, vmax=chihi))

	imdiff = ima.copy()
	dzr = (zr[1] - zr[0])/2.
	if nlscale == 0.:
		imdiff.update(vmin=-dzr, vmax=+dzr, norm=None)
	else:
		imdiff.update(norm= ArcsinhNormalize(mean=0., std=1./nlscale, vmin=-dzr, vmax=dzr))

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
				xt,yt = wcs.positionToPixel(obj.getPosition(), obj)
				pointx.append(xt)
				pointy.append(yt)
				continue
			shapes = []
			attrType = []
			if (isinstance(obj,st.CompositeGalaxy)):
				for attr in 'shapeExp', 'shapeDev':
					shapes.append(getattr(obj, attr))
					attrType.append(attr)
			else:
				shapes.append(getattr(obj,'shape'))
				attrType.append(' ')
			x0,y0 = wcs.positionToPixel(obj.getPosition(), obj)
			
			cd = timg.getWcs().cdAtPixel(x0,y0)
			for i,shape in enumerate(shapes):
				xplotx.append(x0)
				xploty.append(y0)
				T=np.linalg.inv(shape.getTensor(cd))

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

	def savepng(pre, img, title=None,**kwargs):
		fn = '%s-%s.png' % (pre, idstr)
		print('Saving', fn)
		plt.clf()
		plt.imshow(img, **kwargs)
		ax = plt.axis()
		if debug:
			print(len(xplotx),len(allobjx))
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

	savepng('data', data, title='Data '+timg.name, **ima)
	savepng('model', mod, title='Model '+timg.name, **ima)
	savepng('diff', data - mod, title='Data - Model, ' + timg.name, **imdiff)
	savepng('chi',chi,title='Chi ' + timg.name, **imchi)
	print("Chi mean: ", np.mean(chi))
	print("Chi median: ", np.median(chi))
	if plotAll:
		debug = False
		for i,src in enumerate(tractor.getCatalog()):
			savepng('data-s%i'%(i+1),data - sky, title='Data '+timg.name,**ima)
			modelimg = tractor.getModelImage(timg, srcs=[src])
			savepng('model-s%i'%(i+1), modelimg - sky, title='Model-s%i'%(i+1),**ima) 
			savepng('diff-s%i'%(i+1), data - modelimg, title='Model-s%i'%(i+1),**ima)
			savepng('chi-s%i'%(i+1),tractor.getChiImage(imgi,srcs=[src]),
				title='Chi',**imchi)

def saveBands(idstr, tractor, zr,bands, debug=False,plotAll=False):
    for i,band in enumerate(bands):
        save(idstr+'-%s-' % (band), tractor,zr,debug=debug,plotAll=plotAll,imgi=i)


def saveAll(idstr, tractor, nlscale=1.,debug=False,plotAll=False,plotBands=False,chilo=-10.,chihi=10.):
    for i,img in enumerate(tractor.getImages()):
	    if i % 5 == 2 or plotBands: #Only print 'r' band
		    save(idstr+'-%d' % (i), tractor,nlscale=nlscale,debug=debug, plotAll=plotAll, 
			 imgi=i, chilo=chilo, chihi=chihi)


def plotInvvar(idstr,tractor):
	models = tractor.getModelImages()
	timgs = tractor.getImages()
	for i,(timg,mod) in enumerate(zip(timgs,models)):
		if i % 5 ==2: #Only print 'r' band HACK
			data = timg.getImage()
			plt.clf()
			plt.plot(mod.flatten(),(data-mod).flatten()**2,'x')
			plt.savefig(idstr+'invvar-%d' % (i))

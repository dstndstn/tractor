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


# Assumes one image.
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
	if plotAll:
		debug = False
		for i,src in enumerate(tractor.getCatalog()):
			savepng('data-s%i'%(i+1),data - sky, title='Data '+timg.name,**ima)
			modelimg = tractor.getModelImage(timg, srcs=[src])
			savepng('model-s%i'%(i+1), modelimg - sky, title='Model-s%i'%(i+1),**ima) 
			savepng('diff-s%i'%(i+1), data - modelimg, title='Model-s%i'%(i+1),**ima)
			savepng('chi-s%i'%(i+1),tractor.getChiImage(imgi,srcs=[src]),title='Chi',**imchi)
		

def main():
	from optparse import OptionParser
	import sys


	tune = []
	def store_value (option, opt, value, parser):
		if opt == '--ntune':
			tune.append(('n',value))
		elif opt == '--itune':
			tune.append(('i',value))

	parser = OptionParser(usage=('%prog'))
	parser.add_option('-r', '--run', dest='run', type='int')
	parser.add_option('-c', '--camcol', dest='camcol', type='int')
	parser.add_option('-f', '--field', dest='field', type='int')
	parser.add_option('-b', '--band', dest='band', help='SDSS Band (u, g, r, i, z)')
	parser.add_option('--curl', dest='curl', action='store_true', default=False, help='Use "curl", not "wget", to download files')
	parser.add_option('--ntune', action='callback', callback=store_value, type=int,  help='Improve synthetic image over DR7 by locally optimizing likelihood for nsteps iterations')
	parser.add_option('--itune', action='callback', callback=store_value, type=int, nargs=2, help='Optimizes each source individually')
	parser.add_option('--roi', dest='roi', type=int, nargs=4, help='Select an x0,x1,y0,y1 subset of the image')
	parser.add_option('--prefix', dest='prefix', help='Set output filename prefix; default is the SDSS  RRRRRR-BC-FFFF string (run, band, camcol, field)')
	parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
					  help='Make more verbose')
	parser.add_option('-d','--debug',dest='debug',action='store_true',default=False,help="Trigger debug images")
	parser.add_option('--plotAll',dest='plotAll',action='store_true',default=False,help="Makes a plot for each source")
	opt,args = parser.parse_args()
	print tune

	if opt.verbose == 0:
		lvl = logging.INFO
	else:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	run = opt.run
	field = opt.field
	camcol = opt.camcol
	bands = []
	for char in opt.band:
		bands.append(char)
		if not char in ['u','g','r','i','z']:
			parser.print_help()
			print
			print 'Must supply band (u/g/r/i/z)'
			sys.exit(-1)
	rerun = 0
	if run is None or field is None or camcol is None or len(bands)==0:
		parser.print_help()
		print 'Must supply --run, --camcol, --field, --band'
		sys.exit(-1)
	bandname = bands[0] #initial position and shape
	prefix = opt.prefix
	if prefix is None:
		prefix = '%06i-%i-%04i' % (run,camcol, field)


	TI = []
	TI.extend([st.get_tractor_image(run, camcol, field, bandname,
					 curl=opt.curl, roi=opt.roi,useMags=True) for bandname in bands])
	sources = st.get_tractor_sources(run, camcol, field,bandname, bands=bands,
					 curl=opt.curl, roi=opt.roi)
	
	timg,info = TI[0]
	photocal = timg.getPhotoCal()
	for source in sources:
		print 'source', source
		#print 'brightness', source.getBrightness()
		#print 'counts', photocal.brightnessToCounts(source.getBrightness())
		#assert(photocal.brightnessToCounts(source.getBrightness()) >= 0.)

	### DEBUG
	wcs = timg.getWcs()
	for i,s in enumerate(sources):
		x,y = wcs.positionToPixel(s, s.getPosition())
		print i, ('(%.1f, %.1f): ' % (x,y)), s

	tims = [timg for timg.tinf in TI]
	tractor = st.SDSSTractor(tims)
	tractor.addSources(sources)

	zr = np.array([-5.,+5.]) * info['skysig']
	for j,band in enumerate(bands):
		save('initial-%s-' % (band) + prefix, tractor, zr,opt.debug,opt.plotAll,imgi=j)

	for count, each in enumerate(tune):
		if each[0]=='n':
			for i in range(each[1]):
				tractor.optimizeCatalogLoop(nsteps=1)
				for j,band in enumerate(bands):
					save('tune-%d-%d-%s-' % (count+1, i+1,band) + prefix, tractor, zr, opt.debug,opt.plotAll,imgi=j)
	
		elif each[0]=='i':
			for i in range(each[1][0]):
				for src in tractor.getCatalog():
					tractor.optimizeCatalogLoop(nsteps=each[1][1],srcs=[src])
				for j,band in enumerate(bands):
					save('tune-%d-%d-%s-' % (count+1,i+1,band) + prefix, tractor, zr, opt.debug,opt.plotAll,imgi=j)

	makeflipbook(opt, prefix,tune,len(tractor.getCatalog()),bands)
	print
	print 'Created flip-book flip-%s.pdf' % prefix



def makeflipbook(opt, prefix,tune,numSrcs,bands):
	# Create a tex flip-book of the plots
	tex = r'''
	\documentclass[compress]{beamer}
	\usepackage{helvet}
	\newcommand{\plot}[1]{\includegraphics[width=0.5\textwidth]{#1}}
	\begin{document}
	'''
	if len(tune) != 0:
		tex += r'''\part{Tuning steps}\frame{\partpage}''' + '\n'
	page = r'''
	\begin{frame}{%s}
	\plot{data-%s}
	\plot{model-%s} \\
	\plot{diff-%s}
        \plot{chi-%s} \\
	\end{frame}'''
	for j,band in enumerate(bands):
		tex += page % (('Initial model, Band: %s' % (band),) + ('initial-%s-' % (band) + prefix,)*4)
	if opt.plotAll:
		for i in range(numSrcs):
			for j,band in enumerate(bands):
				tex += page % (('Source: %i, Band: %s' % (i+1,band),)+ ('s%d-initial-%s-' % (i+1,band) + prefix,)*4)
	for count,step in enumerate(tune):
		if step[0] == 'n':
			for i in range (step[1]):
				for k,band in enumerate(bands):
					tex += page % (('Tuning set %i, Tuning step %i, Band: %s' % (count+1,i+1,band),) +
					   ('tune-%d-%d-%s-' % (count+1,i+1,band) + prefix,)*4)
				if opt.plotAll:
					for j in range(numSrcs):
						for k,band in enumerate(bands):
							tex += page % (('Source: %i, Band: %s' % (j+1,band),)+ ('s%d-tune-%d-%d-%s-' % (j+1,count+1,i+1,band) + prefix,)*4)
		elif step[0] == 'i': 
			for i in range(step[1][0]):
				for band in bands:
					tex += page % (('Tuning set %i, Individual tuning step %i, Band: %s' % (count+1,i+1,band),) + ('tune-%d-%d-%s-' % (count+1,i+1,band) + prefix,)*4)
				if opt.plotAll:
					for j in range(numSrcs):
						for band in bands:
							tex += page % (('Source: %i, Band: %s' % (j+1,band),) + ('s%d-tune-%d-%d-%s-' % (j+1,count+1,i+1,band) + prefix,)*4)
	if len(tune) != 0: 
		last = tune[len(tune)-1]
		lastSet = len(tune) - 1
		if last[0] == 'n':
			final = last[1]
		elif last[0] == 'i':
			final = last[1][0]
		lBand = bands[0]
		# Finish with a 'blink'
		tex += r'''\part{Before-n-after}\frame{\partpage}''' + '\n'
		tex += (r'''
		\begin{frame}{Data}
		\plot{data-%s}
		\plot{data-%s} \\
		\plot{diff-%s}
		\plot{diff-%s}
		\end{frame}
		\begin{frame}{Before (left); After (right)}
		\plot{model-%s}
		\plot{model-%s} \\
		\plot{diff-%s}
		\plot{diff-%s}
		\end{frame}
		''' % (('initial-%s-' % (lBand) + prefix,)*2 +
			   ('initial-%s-' %(lBand) +prefix, 'tune-%d-%d-%s-' % (lastSet+1,final,lBand)+ prefix)*3))
	tex += r'\end{document}' + '\n'
	fn = 'flip-' + prefix + '.tex'
	print 'Writing', fn
	open(fn, 'wb').write(tex)
	os.system("pdflatex '%s'" % fn)

if __name__ == '__main__':
	import cProfile
	import sys
	from datetime import tzinfo, timedelta, datetime
	cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	sys.exit(0)
	#main()

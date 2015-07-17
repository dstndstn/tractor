all: mix emfit # ceres tsnnls

FORCE:

ceres: FORCE
	$(MAKE) -C tractor ceres
mix: FORCE
	$(MAKE) -C tractor mix
emfit: FORCE
	$(MAKE) -C tractor emfit

cython:
	python setup-cython.py build_ext --inplace
.PHONY: cython

doc:
	$(MAKE) -C doc -f Makefile.sphinx html PYTHONPATH=$(shell pwd):${PYTHONPATH}
	cp -a doc/_build/html .
.PHONY: doc

_denorm.so: denorm.i
	swig -python $<
	gcc -fPIC -c denorm_wrap.c $$(python-config --includes)
	gcc -o $@ -shared denorm_wrap.o -L$$(python-config --prefix)/lib $$(python-config --libs --ldflags)

_refcnt.so: refcnt.i
	swig -python $<
	gcc -fPIC -c refcnt_wrap.c $$(python-config --includes)
	gcc -o $@ -shared refcnt_wrap.o -L$$(python-config --prefix)/lib $$(python-config --libs --ldflags)

_callgrind.so: callgrind.i
	swig -python $<
	gcc -fPIC -c callgrind_wrap.c $$(python-config --includes) -I/usr/include/valgrind
	gcc -o _callgrind.so -shared callgrind_wrap.o -L$$(python-config --prefix)/lib $$(python-config --libs --ldflags)

refcnt: _refcnt.so refcnt.py
.PHONY: refcnt

INSTALL_DIR ?= /usr/local/tractor

PY_INSTALL_DIR ?= $(INSTALL_DIR)/lib/python

TRACTOR_INSTALL_DIR := $(PY_INSTALL_DIR)/tractor

# emfit.py mix.py
# mp_fourier.py 

# cfht.py compiled_profiles.py diesel.py galaxy_profiles.py galex.py hst.py 
# integral_image.py mpcache.py nasasloan.py ordereddict.py overview.py 
# rc3.py saveImg.py sdss-main-old.py sdss_galaxy_old.py source_extractor.py
# total_ordering.py tychodata.py

TRACTOR_INSTALL := __init__.py basics.py cache.py ducks.py ellipses.py engine.py \
	fitpsf.py galaxy.py imageutils.py mixture_profiles.py motion.py \
	patch.py psfex.py sfd.py sdss.py sersic.py splinesky.py source_extractor.py utils.py \
	emfit.py mix.py _emfit.so _mix.so

WISE_INSTALL_DIR := $(PY_INSTALL_DIR)/wise
WISE_INSTALL := __init__.py allwisecat.py forcedphot.py unwise.py wise_psf.py \
	wisecat.py \
	allsky-atlas.fits wise-psf-avg.fits

CERES_INSTALL := ceres.py _ceres.so

install:
	-($(MAKE) ceres && $(MAKE) install-ceres)
	$(MAKE) mix emfit
	mkdir -p $(TRACTOR_INSTALL_DIR)
	@for x in $(TRACTOR_INSTALL); do \
		echo cp tractor/$$x '$(TRACTOR_INSTALL_DIR)/'$$x; \
		cp tractor/$$x '$(TRACTOR_INSTALL_DIR)/'$$x; \
	done
	mkdir -p $(WISE_INSTALL_DIR)
	@for x in $(WISE_INSTALL); do \
		echo cp wise/$$x '$(WISE_INSTALL_DIR)/'$$x; \
		cp wise/$$x '$(WISE_INSTALL_DIR)/'$$x; \
	done

install-ceres:
	mkdir -p $(TRACTOR_INSTALL_DIR)
	@for x in $(CERES_INSTALL); do \
		echo cp tractor/$$x '$(TRACTOR_INSTALL_DIR)/'$$x; \
		cp tractor/$$x '$(TRACTOR_INSTALL_DIR)/'$$x; \
	done


.PHONE: install
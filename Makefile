all: setup

FORCE:

PYTHON ?= python3
setup: FORCE
	$(PYTHON) setup.py build_ext --inplace --with-ceres --with-cython

cov:
	coverage erase
	coverage run -a test/test_galaxy.py
	coverage html
# coverage run test/test_psfex.py
# coverage run -a test/test_sdss.py
# coverage run -a test/test_tractor.py
# coverage run -a examples/tractor-sdss-synth.py --roi 100 200 100 200 --no-flipbook
.PHONY: cov

cython-clean:
	@for x in basics brightness ceres_optimizer ducks ellipses engine galaxy \
		image imageutils lsqr_optimizer mixture_profiles motion optimize patch \
		pointsource psf psfex sersic sfd shifted sky splinesky tractortime \
		utils wcs constrained_optimizer dense_optimizer; do \
		rm tractor/$$x.c tractor/$$x.cpython*.so; \
	done

doc:
	$(MAKE) -C doc -f Makefile.sphinx html PYTHONPATH=$(shell pwd):${PYTHONPATH}
	cp -a doc/_build/html .
.PHONY: doc


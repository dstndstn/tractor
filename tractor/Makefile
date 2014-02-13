all: mix ceres

ceres: _ceres.so ceres.py
.PHONY: ceres

mix: _mix.so mix.py
.PHONY: mix

emfit: _emfit.so emfit.py
.PHONY: emfit


mix.py mix_wrap.c: mix.i
	swig -python -I. mix.i

_mix.so: mix_wrap.c setup-mix.py
	python setup-mix.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_mix.so $@

emfit.py emfit_wrap.c: emfit.i
	cd tractor && swig -python -I. emfit.i

_emfit.so: emfit_wrap.c setup-emfit.py
	python setup-emfit.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_emfit.so $@




NUMPY_INC := $(shell python -c "from numpy.distutils.misc_util import get_numpy_include_dirs as d; print ' '.join('-I'+x for x in d())")

PYMOD_LIB ?= $(shell python-config --libs) #-lpython
PYMOD_INC ?= $(shell python-config --includes)

EIGEN_INC ?= $(shell pkg-config --cflags eigen3)

CERES_INC ?= 
# ceres 1.6.0 was so easy...
#CERES_LIB ?= -L/usr/local/lib -lceres_shared -lglog

# ceres 1.7.0 doesn't have the shared lib on my homebrew install...
CERES_LIB ?= /usr/local/lib/libceres.a -L/usr/local/lib -lglog \
/usr/local/lib/libcxsparse.a \
/usr/local/lib/libcholmod.a \
/usr/local/lib/libcamd.a \
/usr/local/lib/libcolamd.a \
/usr/local/lib/libamd.a \
-framework Accelerate

# On Riemann:
#SSlib=~/software/suitespares-4.1.2/lib
#PYMOD_LIB="" CERES_LIB :="${HOME}/software/ceres-solver-1.7.0/lib/libceres.a $SSlib/lib{cholmod,amd,camd,colamd,ccolamd,suitesparseconfig,metis}.a ${ATLAS_DIR}/lib/lib{lapack,f77blas,atlas,cblas}.a -L${HOME}/software/glog-0.3.3/lib -lglog -lgfortran -lrt" CERES_INC="-I${HOME}/software/glog-0.3.3/include -I${HOME}/software/ceres-solver-1.7.0/include -I${HOME}/software/eigen" make _ceres.so

# On BBQ (an ubuntu box):
#
# -install ceres-solver-1.8.0:
#
# wget "https://ceres-solver.googlecode.com/files/ceres-solver-1.8.0.tar.gz"
# tar xzf ceres-solver-1.8.0.tar.gz
# mkdir ceres-build
# cd ceres-build
# cmake ../ceres-solver-1.8.0 -DCMAKE_INSTALL_PREFIX=/usr/local/ceres-solver-1.8.0 -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC"
# make
# sudo make install
#
# -in ~/.bashrc:
#
# export CERES_LIB="/usr/local/ceres-solver-1.8.0/lib/libceres.a -lglog /usr/local/lib/libcxsparse.a /usr/local/lib/libcholmod.a /usr/local/lib/libcamd.a /usr/local/lib/libcolamd.a /usr/local/lib/libamd.a -lgomp /usr/lib/liblapack.a /usr/lib/libf77blas.a /usr/lib/libatlas.a /usr/lib/libcblas.a -lgfortran"
# export CERES_INC="-I/usr/local/ceres-solver-1.8.0/include"

_ceres.so: ceres.i ceres-tractor.h ceres-tractor.cc
	swig -python -c++ $(NUMPY_INC) $(CERES_INC) $(EIGEN_INC) $<
	g++ -Wall -fPIC -c ceres_wrap.cxx $(PYMOD_INC) $(NUMPY_INC) $(CERES_INC) $(EIGEN_INC)
	g++ -Wall -fPIC -c ceres-tractor.cc $(CERES_INC) $(EIGEN_INC)
	g++ -Wall -fPIC -o _ceres.so -shared ceres_wrap.o ceres-tractor.o $(CERES_LIB) $(PYMOD_LIB)

LIBTSNNLS_INC ?=
LIBTSNNLS_LIB ?= -ltsnnls

_tsnnls.so: tsnnls.i
	swig -python $(NUMPY_INC) $(LIBTSNNLS_INC) $<
	gcc -Wall -fPIC -c tsnnls_wrap.c $$(python-config --includes) $(NUMPY_INC) $(LIBTSNNLS_INC)
	gcc -Wall -fPIC -o _tsnnls.so -shared tsnnls_wrap.o $(LIBTSNNLS_LIB)

#	gcc -Wall -fPIC -o $@ -shared tsnnls_wrap.o $$(python-config --ldflags) $(LIBTSNNLS_LIB)
#	gcc -Wall -fPIC -o _tsnnls.so -shared tsnnls_wrap.o $(LIBTSNNLS_LIB) -lpython
#	gcc -Wall -fPIC -o $@ -shared tsnnls_wrap.o -L$$(python-config --prefix)/lib $$(python-config --libs --ldflags) $(LIBTSNNLS_LIB)




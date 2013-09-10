all: mix emfit #refcnt

doc: html
	cp -a _build/html .
.PHONE: doc

NUMPY_INC := $(shell python -c "from numpy.distutils.misc_util import get_numpy_include_dirs as d; print ' '.join('-I'+x for x in d())")

LIBTSNNLS_INC ?=
LIBTSNNLS_LIB ?= -ltsnnls

_tsnnls.so: tsnnls.i
	swig -python $(NUMPY_INC) $(LIBTSNNLS_INC) $<
	gcc -Wall -fPIC -c tsnnls_wrap.c $$(python-config --includes) $(NUMPY_INC) $(LIBTSNNLS_INC)
	gcc -Wall -fPIC -o _tsnnls.so -shared tsnnls_wrap.o $(LIBTSNNLS_LIB) -lpython
#	gcc -Wall -fPIC -o $@ -shared tsnnls_wrap.o -L$$(python-config --prefix)/lib $$(python-config --libs --ldflags) $(LIBTSNNLS_LIB)

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

tractor/mix.py tractor/mix_wrap.c: tractor/mix.i
	cd tractor && swig -python -I. mix.i

tractor/_mix.so: tractor/mix_wrap.c tractor/setup-mix.py
	cd tractor && python setup-mix.py build --force --build-base build --build-platlib build/lib
	cp tractor/build/lib/_mix.so $@

mix: tractor/_mix.so tractor/mix.py
.PHONY: mix


tractor/emfit.py tractor/emfit_wrap.c: tractor/emfit.i
	cd tractor && swig -python -I. emfit.i

tractor/_emfit.so: tractor/emfit_wrap.c tractor/setup-emfit.py
	cd tractor && python setup-emfit.py build --force --build-base build --build-platlib build/lib
	cp tractor/build/lib/_emfit.so $@

emfit: tractor/_emfit.so tractor/emfit.py
.PHONY: emfit


include Makefile.sphinx

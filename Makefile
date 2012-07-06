all: mix emfit

doc: html
	cp -a _build/html .
.PHONE: doc

_denorm.so: denorm.i
	swig -python $<
	gcc -fPIC -c denorm_wrap.c -I/usr/include/python2.7
	gcc -o $@ -shared denorm_wrap.o -lpython2.7

_callgrind.so: callgrind.i
	swig -python $<
	gcc -fPIC -c callgrind_wrap.c -I/usr/include/python2.7 -I/usr/include/valgrind
	gcc -o _callgrind.so -shared callgrind_wrap.o -lpython2.7

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
all: mix emfit

tractor/mix.py tractor/mix_wrap.c: tractor/mix.i
	swig -python -I. $<

tractor/_mix.so: tractor/mix_wrap.c setup-mix.py
	python setup-mix.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_mix.so $@

mix: tractor/_mix.so tractor/mix.py
.PHONY: mix


tractor/emfit.py tractor/emfit_wrap.c: tractor/emfit.i
	swig -python -I. $<

tractor/_emfit.so: tractor/emfit_wrap.c setup-emfit.py
	python setup-emfit.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_emfit.so $@

emfit: tractor/_emfit.so tractor/emfit.py
.PHONY: emfit


include Makefile.sphinx
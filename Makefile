all: mix

mix_wrap.c: mix.i
	swig -python -I. $<

_mix.so: mix_wrap.c setup-mix.py
	python setup-mix.py build --force --build-base build --build-platlib build/lib
	cp build/lib/_mix.so $@

mix: _mix.so
.PHONY: mix



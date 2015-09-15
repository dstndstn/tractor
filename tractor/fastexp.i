%module(package="tractor") fastexp

%inline %{

#include "fastapprox.h"

float my_fastexp(double x) {
    return fastexp(x);
}



/// from ngmix
#include "fastexp-table.h"

union fmath_di {
    double d;
    uint64_t i;
};

double ngmix_exp(double x) {
    union fmath_di di;
    di.d = x * a + b;
    uint64_t iax = dtbl[di.i & sbit_masked];
    double t = (di.d - b) * ra - x;
    uint64_t u = ((di.i + adj) >> sbit) << 52;
    double y = (C3 - t) * (t * t) * C2 - t + C1;
    di.i = u | iax;
    return y * di.d;
}



 %}

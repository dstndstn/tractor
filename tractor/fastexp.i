%module(package="tractor") fastexp

%inline %{

#include "fastapprox.h"

float my_fastexp(double x) {
    return fastexp(x);
}

 %}

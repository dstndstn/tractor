/*

   Fast exponential function for doubles.  Code adapted from
   here

        https://github.com/herumi/fmath

    On modern compilers, a factor of 5 faster with accuracy
    good to about 1.55e-15


    To build the tests:
        python build.py

    If the code is slow or inaccurate, try remaking the
    lookup table.  Run
        python build.py --make-dtbl
        ./make-dtbl > fmath-dtbl.c
        python build.py
    and rerun the tests.
*/
#ifndef _FMATH_HEADER_GUARD
#define _FMATH_HEADER_GUARD

#include <stdint.h>

    union fmath_di {
        double d;
        uint64_t i;
    };

    // holds definition of the table and C1,C2,C3, a, ra
#include "fastexp-table.h"

    static inline double expd(double x)
    {

        union fmath_di di;

        di.d = x * a + b;
        uint64_t iax = dtbl[di.i & sbit_masked];

        double t = (di.d - b) * ra - x;
        uint64_t u = ((di.i + adj) >> sbit) << 52;
        double y = (C3 - t) * (t * t) * C2 - t + C1;

        di.i = u | iax;
        return y * di.d;

    }


#endif

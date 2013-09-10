%module(package="tractor") tsnnls

%include <typemaps.i>

%{
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
%}

%init %{
    // numpy
    import_array();
%}


%inline %{
#if 0
 } // fool emacs indenter
#endif

#include "libtsnnls/tsnnls.h"

static PyObject* tsnnls_lsqr(PyObject* np_colinds,
                             PyObject* np_sprows,
                             PyObject* np_spvals,
                             PyObject* np_b,
                             int Nrows, int Nvals) {
    taucs_ccs_matrix* A;
    double* b;
    double tol;
    double* x;
    double rnorm;
    npy_intp Ncols;
    PyObject* np_x = NULL;

    //printf("size of int: %i\n", sizeof(int));
    Ncols = PyArray_SIZE(np_colinds) - 1;

    /*
     {
        int* cind = PyArray_DATA(np_colinds);
        int* rows = PyArray_DATA(np_sprows);
        double* vals = PyArray_DATA(np_spvals);
        int i, j;
        for (i=0; i<Ncols; i++) {
            printf("Column %i: %i to %i\n", i, cind[i], cind[i+1]);
            for (j=cind[i]; j<cind[i+1]; j++) {
                printf("  row %i, val %g\n", rows[j], vals[j]);
            }
        }
     }
     */

    A = calloc(1, sizeof(taucs_ccs_matrix));
    A->n = Ncols;
    A->m = Nrows;
    A->flags = TAUCS_DOUBLE;
    A->colptr = PyArray_DATA(np_colinds);
    A->rowind = PyArray_DATA(np_sprows);
    A->values.d = PyArray_DATA(np_spvals);

    b = PyArray_DATA(np_b);

    tol = -1.; // always take an LSQR step

    /*
     {
        int i;
        printf("A matrix:\n");
        taucs_print_ccs_matrix(A);
        printf("\n");

        printf("b: [ ");
        for (i=0; i<Nrows; i++) {
            printf("%g ", b[i]);
        }
        printf("]\n");
    }
     */

    //x = t_snnls_fallback(A, b, &rnorm, tol, 1);
    x = t_snnls(A, b, &rnorm, tol, 1);
    free(A);
    if (x) {
        npy_intp dims = Ncols;
        np_x = PyArray_SimpleNewFromData(1, &dims, PyArray_DOUBLE, x);
        printf("rnorm = %g\n", rnorm);
        /*
        int i;
        printf("x = [ ");
        for (i=0; i<Ncols; i++) {
            printf("%g ", x[i]);
        }
        printf("]\n");
         */
        /*
         {
         int i;
         x = t_lsqr(A, b);
         printf("TSNNLS LSQR:\n");
         }*/
    } else {
        PyErr_SetString(PyExc_RuntimeError,
                        "t_snnls failed: maybe matrix not pos def.");
    }
    return np_x;
}

%}


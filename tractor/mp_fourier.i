%module(package="tractor") mp_fourier

%{
#define _GNU_SOURCE 1
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

static PyObject* mixture_profile_fourier_transform(
    PyObject* np_amps,
    PyObject* np_means,
    PyObject* np_vars,
    PyObject* np_w,
    PyObject* np_v
    ) {

    npy_intp K, NW,NV;
    const npy_intp D = 2;
    npy_intp i,j,k;
    double* amps, *means, *vars, *ww, *vv;
    PyObject* np_F;
    double* f;
    npy_intp dims[2];

    if (!PyArray_Check(np_amps) || !PyArray_Check(np_means) ||
        !PyArray_Check(np_vars) || !PyArray_Check(np_w) ||
        !PyArray_Check(np_v)) {
        PyErr_SetString(PyExc_ValueError, "Expected numpy arrays");
        return NULL;
    }

    /*
     PyObject* arr;
     PyArray_Descr* dtype = PyArray_DescrFromType(NPY_DOUBLE);
     int req = NPY_C_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED
     | NPY_ELEMENTSTRIDES;
    Py_INCREF(dtype);
    arr = PyArray_FromAny(np_amps, dtype, 1, 1, req, NULL);
    K = PyArray_DIM(arr, 0);
    amps = PyArray_DATA(arr);
     */

    if ((PyArray_TYPE(np_amps) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_means ) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_vars) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_w)    != NPY_DOUBLE) ||
        (PyArray_TYPE(np_v)    != NPY_DOUBLE)) {
        PyErr_SetString(PyExc_ValueError, "Expected numpy double arrays");
        return NULL;
    }

    if (PyArray_NDIM(np_amps) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'amps' to be 1-d");
        return NULL;
    }
    K = PyArray_DIM(np_amps, 0);
    if (PyArray_NDIM(np_means) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected 'means' to be 2-d");
        return NULL;
    }
    if ((PyArray_DIM(np_means, 0) != K) ||
        (PyArray_DIM(np_means, 1) != D)) {
        PyErr_SetString(PyExc_ValueError, "Expected 'means' to be K x D");
        return NULL;
    }
    if (PyArray_NDIM(np_vars) != 3) {
        PyErr_SetString(PyExc_ValueError, "Expected 'vars' to be 3-d");
        return NULL;
    }
    if ((PyArray_DIM(np_vars, 0) != K) ||
        (PyArray_DIM(np_vars, 1) != D) ||
        (PyArray_DIM(np_vars, 2) != D)) {
        PyErr_SetString(PyExc_ValueError, "Expected 'vars' to be K x D x D");
        return NULL;
    }

    if (PyArray_NDIM(np_w) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'w' to be 1-d");
        return NULL;
    }
    if (PyArray_NDIM(np_v) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'v' to be 1-d");
        return NULL;
    }

    NW = PyArray_DIM(np_w, 0);
    NV = PyArray_DIM(np_v, 0);

    dims[0] = NV;
    dims[1] = NW;
    np_F = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    f = PyArray_DATA(np_F);
    amps = PyArray_DATA(np_amps);
    means = PyArray_DATA(np_means);
    vars = PyArray_DATA(np_vars);
    ww = PyArray_DATA(np_w);
    vv = PyArray_DATA(np_v);

    memset(f, 0, 2*NW*NV*sizeof(double));

    for (k=0; k<K; k++) {
        if ((means[k*D] != means[0]) ||
            (means[k*D+1] != means[1])) {
            PyErr_SetString(PyExc_ValueError, "Assume all means are equal");
            return NULL;
        }
    }

    for (k=0; k<K; k++) {
        double* V = vars + k*D*D;
        double det;
        double a,b,d, amp;
        double* ff = f;
        //printf("k=%i: mean %g, %g\n", (int)k, means[k*D], means[k*D+1]);
        det = V[0]*V[3] - V[1]*V[2];
        a = 0.5 *  V[3]/det;
        b = 0.5 * -V[1]/det;
        d = 0.5 *  V[0]/det;
        det = a*d - b*b;
        amp = amps[k];
        /*
         double prefactor = - M_PI * M_PI / det;
         for (i=0; i<NW; i++)
         wfactor[i] = prefactor * d * ww[i] * ww[i];
         for (i=0; i<NV; i++)
         vfactor[i] = prefactor * a * vv[i] * vv[i];
         prefactor *= 2. * b;
         */
        double mu0 = means[k*D];
        double mu1 = means[k*D+1];
        for (i=0; i<NV; i++) {
            for (j=0; j<NW; j++) {
                //double s = amp * exp(vfactor[i] + wfactor[j]
                //- prefactor * vv[i] * ww[j]);
                double s = amp * exp(-M_PI*M_PI/det *
                                     (a * vv[i]*vv[i] +
                                      d * ww[j]*ww[j] -
                                      2*b*vv[i]*ww[j]));
                (*ff) += s;
                ff++;
                ff++;
                /*
                 //double real,imag;
                 //sincos(angle, &imag, &real);
                 //(*ff) += s * real;
                 (*ff) += s * cos(angle);
                 ff++;
                 //(*ff) += s * imag;
                 (*ff) += s * sin(angle);
                 ff++;
                 */
                //if (j<5 && i<5)
                //printf("%g ", s);
                //printf("%g ", s*cos(angle));
            }
            //if (i<5)
            //printf("\n");
        }
    }

    double* ff = f;
    double mu0 = means[0];
    double mu1 = means[1];
    for (i=0; i<NV; i++) {
        for (j=0; j<NW; j++) {
            double angle = -2. * M_PI * (mu0 * ww[j] + mu1 * vv[i]);
            ff[1] = ff[0] * sin(angle);
            ff[0] *= cos(angle);
            ff++;
            ff++;
        }
    }
    return np_F;
}

    %}

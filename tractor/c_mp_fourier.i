%module(package="tractor") c_mp_fourier

%{
#define SWIG_FILE_WITH_INIT
#define _GNU_SOURCE 1
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
%}

%include "numpy.i"

%init %{
    // numpy
    import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {
    (double *amps, int amps_len),
    (double *v, int v_len),
    (double *w, int w_len)
};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
    (double *means, int means_dim1, int means_dim2)
};
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {
    (double *vars, int vars_dim1, int vars_dim2, int vars_dim3)
};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double *out, int out_dim1, int out_dim2)
};

%inline %{

static void mixture_profile_fourier_transform(double *amps, int amps_len,
                                              double *means, int means_dim1, int means_dim2,
                                              double *vars, int vars_dim1, int vars_dim2, int vars_dim3,
                                              double *v, int v_len,
                                              double *w, int w_len,
                                              double *out, int out_dim1, int out_dim2)
{
    const double twopisquare = -2. * M_PI * M_PI;

    int K = amps_len;
    int NV = v_len;
    int NW = w_len;
    int i, j, k;

    double *s = (double*)malloc(sizeof(double) * NV * NW);
    memset(s, 0, sizeof(double) * NV * NW);

    for (j = 0; j < NW; j++) {
        double w_j = w[j];
        double w_j_sqr = w_j * w_j;
        for (i = 0; i < NV; i++) {
            int index = NV * j + i;
            double v_i = v[i];
            double v_i_sqr = v_i * v_i;
            for (k = 0; k < K; k++) {
                int offset = k * 4;
                double a = vars[offset];
                double b = vars[offset + 1];
                double d = vars[offset + 3];

                s[index] += amps[k] * exp(twopisquare * (a *  v_i_sqr + 2. * b * v_i * w_j + d * w_j_sqr));
            }
            out[index] = s[index];
        }
    }
    free(s);
    return;
}

%}

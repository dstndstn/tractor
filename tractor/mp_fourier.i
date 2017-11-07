%module(package="tractor") mp_fourier

%{
#define SWIG_FILE_WITH_INIT
#define _GNU_SOURCE 1
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>

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


%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double *work, int work_dim1, int work_dim2)
};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double *img, int img_dim1, int img_dim2)
};
%apply (double* IN_ARRAY1, int DIM1) {
    (double *filtx, int filtx_dim)
};
%apply (double* IN_ARRAY1, int DIM1) {
    (double *filty, int filty_dim)
};


%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (float *work, int work_dim1, int work_dim2)
};
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (float *img, int img_dim1, int img_dim2)
};


%inline %{

// #ifdef __INTEL_COMPILER
// #define RESTRICT restrict
// #else
// #define RESTRICT __restrict
// #endif

// #ifndef __INTEL_COMPILER
// #define restrict __restrict
// #endif

#if 0
 } // fool emacs indenter
#endif

static void correlate7(double* restrict img, int img_dim1, int img_dim2,
                       double* restrict filtx, int filtx_dim,
                       double* restrict filty, int filty_dim,
                       double* restrict work, int work_dim1, int work_dim2) {
    // Output goes back into "img"!

#ifdef __INTEL_COMPILER
    __assume_aligned(img, 64);
    __assume_aligned(work, 64);
#endif

    assert(filtx_dim == 7);
    assert(filty_dim == 7);

    assert(work_dim1 >= img_dim1);
    assert(work_dim2 >= img_dim2);

    double filter[7];
    int i, j, k;
    int W = img_dim2;
    int H = img_dim1;

    assert(W > 8);
    assert(H > 8);

    memcpy(filter, filtx, 7 * sizeof(double));

    // first run the filtx over image rows
    for (j=0; j<H; j++) {
        // special handling of left edge
        int offset = j*W;
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[3-i+k] * img[offset + k];
            work[i*H + j] = sum;
        }
        // middle section
        for (i=0; i<=(W-7); i++) {
            double sum = 0.0;
            for (k=0; k<7; k++)
                sum += filter[k] * img[offset + i+k];
            work[(i+3)*H + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[k] * img[offset + W-(4+i)+k];
            work[(W-1-i)*H + j] = sum;
        }
    }

    memcpy(filter, filty, 7 * sizeof(double));

    int workH = W;
    int workW = H;

    // Output goes back into "img"!

    // Now run filty over rows of the 'work' array
    for (j=0; j<workH; j++) {
        // special handling of left edge
        int offset = j * workW;
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[3-i+k] * work[offset + k];
            img[i*W + j] = sum;
        }
        // middle section
        for (i=0; i<=(workW-7); i++) {
            double sum = 0.0;
            for (k=0; k<7; k++)
                sum += filter[k] * work[offset + i+k];
            img[(i+3)*W + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[k] * work[offset +  workW-(4+i)+k];
            img[(workW-1-i)*W + j] = sum;
        }
    }
}


static void correlate7f(float*  restrict img, int img_dim1, int img_dim2,
                        double* restrict filtx, int filtx_dim,
                        double* restrict filty, int filty_dim,
                        float*  restrict work, int work_dim1, int work_dim2) {
    // Output goes back into "img"!

#ifdef __INTEL_COMPILER
    __assume_aligned(img, 64);
    __assume_aligned(work, 64);
#endif

    assert(filtx_dim == 7);
    assert(filty_dim == 7);

    assert(work_dim1 >= img_dim1);
    assert(work_dim2 >= img_dim2);

    float filter[7];
    int i, j, k;
    int W = img_dim2;
    int H = img_dim1;

    assert(W > 8);
    assert(H > 8);

    //memcpy(filter, filtx, 7 * sizeof(double));
    for (i=0; i<7; i++)
        filter[i] = filtx[i];

    // first run the filtx over image rows
    for (j=0; j<H; j++) {
        // special handling of left edge
        int offset = j*W;
        for (i=0; i<3; i++) {
            float sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[3-i+k] * img[offset + k];
            work[i*H + j] = sum;
        }
        // middle section
        for (i=0; i<=(W-7); i++) {
            float sum = 0.0;
            for (k=0; k<7; k++)
                sum += filter[k] * img[offset + i+k];
            work[(i+3)*H + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            float sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[k] * img[offset + W-(4+i)+k];
            work[(W-1-i)*H + j] = sum;
        }
    }

    for (i=0; i<7; i++)
        filter[i] = filty[i];

    int workH = W;
    int workW = H;

    // Output goes back into "img"!

    // Now run filty over rows of the 'work' array
    for (j=0; j<workH; j++) {
        // special handling of left edge
        int offset = j * workW;
        for (i=0; i<3; i++) {
            float sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[3-i+k] * work[offset + k];
            img[i*W + j] = sum;
        }
        // middle section
        for (i=0; i<=(workW-7); i++) {
            float sum = 0.0;
            for (k=0; k<7; k++)
                sum += filter[k] * work[offset + i+k];
            img[(i+3)*W + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            float sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[k] * work[offset +  workW-(4+i)+k];
            img[(workW-1-i)*W + j] = sum;
        }
    }
}


static void gaussian_fourier_transform_zero_mean(
    double * restrict amps, int amps_len,
    double * restrict vars, int vars_dim1, int vars_dim2, int vars_dim3,
    double * restrict v, int v_len,
    double * restrict w, int w_len,
    double * restrict out, int out_dim1, int out_dim2)
{
    const double negtwopisquare = -2. * M_PI * M_PI;

    int K = amps_len;
    int NV = v_len;
    int NW = w_len;
    int i, j, k;

    assert(vars_dim1 == K);
    assert(vars_dim2 == 2);
    assert(vars_dim3 == 2);
    assert(out_dim1 == NV);
    assert(out_dim2 == NW);

    for (j = 0; j < NW; j++) {
        double w_j = w[j];
        double w_j_sqr = w_j * w_j;
        for (i = 0; i < NV; i++) {
            int index = NV * j + i;
            double v_i = v[i];
            double v_i_sqr = v_i * v_i;
            double sum = 0.0;
            for (k = 0; k < K; k++) {
                int offset = k * 4;
                double a = vars[offset];
                double b = vars[offset + 1];
                double d = vars[offset + 3];

                sum += amps[k] * exp(negtwopisquare *
                                     (a *  v_i_sqr + 2. * b * v_i * w_j + d * w_j_sqr));
            }
            out[index] = sum;
        }
    }
    return;
}


%}

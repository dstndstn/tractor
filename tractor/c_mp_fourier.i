%module(package="tractor") c_mp_fourier

%{
#define SWIG_FILE_WITH_INIT
#define _GNU_SOURCE 1
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

    //#include <ipp.h>
    //#include <ippcv.h>
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
%apply (double* INPLACE_ARRAY1, int DIM1) {
    (double *filtx, int filtx_dim)
};
%apply (double* INPLACE_ARRAY1, int DIM1) {
    (double *filty, int filty_dim)
};


%inline %{

#if 0
 }
#endif

static void correlate(double* img, int img_dim1, int img_dim2,
                      double* filtx, int filtx_dim,
                      double* filty, int filty_dim,
                      double* work, int work_dim1, int work_dim2,
                      double* out, int out_dim1, int out_dim2) {

    //printf("img %i x %i, filter x %i, filter y %i\n", img_dim1, img_dim2, filtx_dim, filty_dim);

    __assume_aligned(img, 64);
    __assume_aligned(work, 64);
    __assume_aligned(out, 64);

    assert(filtx_dim <= 8);
    assert(filty_dim <= 8);

    assert(img_dim1 == out_dim1);
    assert(img_dim2 == out_dim2);
    assert(work_dim1 >= img_dim1);
    assert(work_dim2 >= img_dim2);

    double filter[8];
    int i, j, k;
    int W = img_dim2;
    int H = img_dim1;

    assert(W > 8);
    assert(H > 8);

    /*
      for (i=0; i<filtx_dim; i++)
      filter[i] = filtx[i];
      for (; i<8; i++)
      filter[i] = 0.0;
    */
    // Copy 'filtx' into the end of 'filter'
    for (i=0; i<filtx_dim; i++)
        filter[7-i] = filtx[7-i];
    for (i=0; i<(8 - filtx_dim); i++)
        filter[i] = 0;

    // first run the filtx over image rows
    for (j=0; j<H; j++) {
        // special handling of left edge
        double* img_row = img + j*W;
        for (i=0; i<4; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[4-i+k] * img_row[k];
            work[i*H + j] = sum;
        }
        // middle section
        for (i=4; i<=(W-4); i++) {
            double sum = 0.0;
            for (k=0; k<8; k++)
                sum += filter[k] * img_row[i-4+k];
            work[i*H + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<5+i; k++)
                sum += filter[k] * img_row[W-(5+i)+k];
            work[(W-1-i)*H + j] = sum;
        }
    }
    /*
    for (j=0; j<H; j++) {
        for (i=0; i<W; i++) {
            out[j*W + i] = work[i*H + j];
        }
    }

    for (j=0; j<H; j++) {
        for (i=0; i<W; i++) {
            work[i*H + j] = img[j*W + i];
        }
    }
    */

    /*
    for (i=0; i<filty_dim; i++)
        filter[i] = filty[i];
    for (; i<8; i++)
        filter[i] = 0.0;
    */

    // Copy 'filty' into the end of 'filter'
    for (i=0; i<filty_dim; i++)
        filter[7-i] = filty[7-i];
    for (i=0; i<(8 - filty_dim); i++)
        filter[i] = 0;

    int workH = W;
    int workW = H;

    // Now run filty over rows of the 'work' array
    for (j=0; j<workH; j++) {
        // special handling of left edge
        double* work_row = work + j * workW;
        for (i=0; i<4; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[4-i+k] * work_row[k];
            out[i*W + j] = sum;
        }
        // middle section
        for (i=4; i<=(workW-4); i++) {
            double sum = 0.0;
            for (k=0; k<8; k++)
                sum += filter[k] * work_row[i-4+k];
            out[i*W + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<5+i; k++)
                sum += filter[k] * work_row[workW-(5+i)+k];
            out[(workW-1-i)*W + j] = sum;
        }
    }

}






static void correlate7(double* img, int img_dim1, int img_dim2,
                       double* filtx, int filtx_dim,
                       double* filty, int filty_dim,
                       double* work, int work_dim1, int work_dim2,
                       double* out, int out_dim1, int out_dim2) {

    __assume_aligned(img, 64);
    __assume_aligned(work, 64);
    __assume_aligned(out, 64);

    assert(filtx_dim == 7);
    assert(filty_dim == 7);

    assert(img_dim1 == out_dim1);
    assert(img_dim2 == out_dim2);
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
        double* img_row = img + j*W;
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[3-i+k] * img_row[k];
            work[i*H + j] = sum;
        }
        // middle section
        for (i=3; i<=(W-4); i++) {
            double sum = 0.0;
            for (k=0; k<7; k++)
                sum += filter[k] * img_row[i-3+k];
            work[i*H + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[k] * img_row[W-(4+i)+k];
            work[(W-1-i)*H + j] = sum;
        }
    }

    /*
    for (j=0; j<H; j++) {
        for (i=0; i<W; i++) {
            out[j*W + i] = work[i*H + j];
        }
    }
    */
    /*
    for (j=0; j<H; j++) {
        for (i=0; i<W; i++) {
            work[i*H + j] = img[j*W + i];
        }
    }
    */

    memcpy(filter, filty, 7 * sizeof(double));

    int workH = W;
    int workW = H;

    // Now run filty over rows of the 'work' array
    for (j=0; j<workH; j++) {
        // special handling of left edge
        double* work_row = work + j * workW;
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[3-i+k] * work_row[k];
            out[i*W + j] = sum;
        }
        // middle section
        for (i=3; i<=(workW-4); i++) {
            double sum = 0.0;
            for (k=0; k<7; k++)
                sum += filter[k] * work_row[i-3+k];
            out[i*W + j] = sum;
        }
        // special handling of right edge
        // i=0 is the rightmost pixel
        for (i=0; i<3; i++) {
            double sum = 0.0;
            for (k=0; k<4+i; k++)
                sum += filter[k] * work_row[workW-(4+i)+k];
            out[(workW-1-i)*W + j] = sum;
        }
    }
}








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

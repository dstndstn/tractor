%module(package="tractor") mix

%include <typemaps.i>

%{
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
#include <sys/param.h>

int n_exp = 0;
int n_expf = 0;

static double eval_g(double I[3], double dx, double dy) {
    double dsq = (I[0] * dx * dx +
                  I[1] * dx * dy +
                  I[2] * dy * dy);
    if (dsq < -100)
        // ~ 1e-44
        return 0.0;
    n_exp++;
    return exp(dsq);
}


static double eval_all(int K, double* scales, double* I, double* means,
                       double x, double y) {
    double r = 0;
    int k;
    for (k=0; k<K; k++) {
        double dx,dy;
        dx = x - means[2*k+0];
        dy = y - means[2*k+1];
        r += scales[k] * eval_g(I + 3*k, dx, dy);
    }
    return r;
}

static double eval_all_dxy(int K, double* scales, double* I, double* means,
                           double x, double y, double* xderiv, double* yderiv,
                           double* maxD) {
    double r = 0;
    int k;
    if (xderiv)
        *xderiv = 0;
    if (yderiv)
        *yderiv = 0;

    for (k=0; k<K; k++) {
        double dx,dy;
        double G;
        double* Ik = I + 3*k;
        double dsq;
        dx = x - means[2*k+0];
        dy = y - means[2*k+1];
        dsq = (Ik[0] * dx * dx +
               Ik[1] * dx * dy +
               Ik[2] * dy * dy);
        // "maxD" is slightly (ok totally) misnamed: it includes the
        // -0.5 factor * mahalanobis distance so is actually a *minimum*.
        if (dsq < maxD[k])
            continue;
        n_exp++;
        G = scales[k] * exp(dsq);
        r += G;
        // The negative sign here is because we want the derivatives
        // with respect to the means, not x,y.
        if (xderiv)
            *xderiv += -G * (2. * Ik[0] * dx + Ik[1] * dy);
        if (yderiv)
            *yderiv += -G * (2. * Ik[2] * dy + Ik[1] * dx);
    }
    return r;
}


static double eval_all_dxy_f(int K, float* scales, float* I, float* means,
                             float x, float y, float* xderiv, float* yderiv,
                             float* maxD) {
    float r = 0;
    int k;
    if (xderiv)
        *xderiv = 0;
    if (yderiv)
        *yderiv = 0;

    for (k=0; k<K; k++) {
        float dx,dy;
        float G;
        float* Ik;
        float dsq;
        if (scales[k] == 0)
            continue;
        Ik = I + 3*k;
        dx = x - means[2*k+0];
        dy = y - means[2*k+1];
        dsq = (Ik[0] * dx * dx +
               Ik[1] * dx * dy +
               Ik[2] * dy * dy);
        // "maxD" is slightly (ok totally) misnamed: it includes the
        // -0.5 factor * mahalanobis distance so is actually a *minimum*.
        if (dsq < maxD[k])
            continue;
        n_expf++;
        G = scales[k] * expf(dsq);
        r += G;
        // The negative sign here is because we want the derivatives
        // with respect to the means, not x,y.
        if (xderiv)
            *xderiv += -G * (2. * Ik[0] * dx + Ik[1] * dy);
        if (yderiv)
            *yderiv += -G * (2. * Ik[2] * dy + Ik[1] * dx);
    }
    return r;
}



#define ERR(x, ...) printf(x, ## __VA_ARGS__)
// PyErr_SetString(PyExc_ValueError, x, __VA_ARGS__)

static int get_np(PyObject* ob_amp,
                  PyObject* ob_mean,
                  PyObject* ob_var,
                  PyObject* ob_result,
                  PyObject* ob_xderiv,
                  PyObject* ob_yderiv,
                  PyObject* ob_mask,
                  int NX, int NY,
                  int* K,
                  PyObject **np_amp,
                  PyObject **np_mean,
                  PyObject **np_var,
                  PyObject **np_result,
                  PyObject **np_xderiv,
                  PyObject **np_yderiv,
                  PyObject **np_mask,
                  PyArray_Descr* dtype) {
    PyArray_Descr* btype = NULL;
    int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
    int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
    const int D = 2;
    if (!dtype)
        dtype = PyArray_DescrFromType(PyArray_DOUBLE);

    Py_INCREF(dtype);
    Py_INCREF(dtype);
    Py_INCREF(dtype);
    Py_INCREF(dtype);
    *np_amp = PyArray_FromAny(ob_amp, dtype, 1, 1, req, NULL);
    *np_mean = PyArray_FromAny(ob_mean, dtype, 2, 2, req, NULL);
    *np_var = PyArray_FromAny(ob_var, dtype, 3, 3, req, NULL);
    *np_result = PyArray_FromAny(ob_result, dtype, 2, 2, reqout, NULL);
    if (ob_xderiv != Py_None) {
        Py_INCREF(dtype);
        *np_xderiv = PyArray_FromAny(ob_xderiv, dtype, 2, 2, reqout, NULL);
    }
    if (ob_yderiv != Py_None) {
        Py_INCREF(dtype);
        *np_yderiv = PyArray_FromAny(ob_yderiv, dtype, 2, 2, reqout, NULL);
    }
    if (ob_mask != Py_None) {
        btype = PyArray_DescrFromType(PyArray_BOOL);
        Py_INCREF(btype);
        *np_mask = PyArray_FromAny(ob_mask, btype, 2, 2, req, NULL);
        Py_CLEAR(btype);
    }
    Py_DECREF(dtype);

    if (!*np_amp || !*np_mean || !*np_var || !*np_result ||
        ((ob_xderiv != Py_None) && !*np_xderiv) ||
        ((ob_yderiv != Py_None) && !*np_yderiv) ||
        ((ob_mask   != Py_None) && !*np_mask)) {
        if (!*np_amp) {
            ERR("amp wasn't the type expected");
            Py_DECREF(dtype);
        }
        if (!*np_mean) {
            ERR("mean wasn't the type expected");
            Py_DECREF(dtype);
        }
        if (!*np_var) {
            ERR("var wasn't the type expected");
            Py_DECREF(dtype);
        }
        if (!*np_result) {
            ERR("result wasn't the type expected");
            Py_DECREF(dtype);
        }
        if ((ob_xderiv != Py_None) && !*np_xderiv) {
            ERR("xderiv wasn't the type expected");
            Py_DECREF(dtype);
        }
        if ((ob_yderiv != Py_None) && !*np_yderiv) {
            ERR("yderiv wasn't the type expected");
            Py_DECREF(dtype);
        }
        if ((ob_mask != Py_None) && !*np_mask) {
            ERR("mask wasn't the type expected");
            Py_DECREF(btype);
        }
        return 1;
    }
    *K = (int)PyArray_DIM(*np_amp, 0);
    if ((PyArray_DIM(*np_mean, 0) != *K) ||
        (PyArray_DIM(*np_mean, 1) != D)) {
        ERR("np_mean must be K x D");
        return 1;
    }
    if ((PyArray_DIM(*np_var, 0) != *K) ||
        (PyArray_DIM(*np_var, 1) != D) ||
        (PyArray_DIM(*np_var, 2) != D)) {
        ERR("np_var must be K x D x D");
        return 1;
    }
    if ((PyArray_DIM(*np_result, 0) != NY) ||
        (PyArray_DIM(*np_result, 1) != NX)) {
        ERR("np_result must be size NY x NX (%i x %i), got %i x %i",
            NY, NX, (int)PyArray_DIM(*np_result, 0),
            (int)PyArray_DIM(*np_result, 1));
        return 1;
    }
    if (np_xderiv && *np_xderiv) {
        if ((PyArray_DIM(*np_xderiv, 0) != NY) ||
            (PyArray_DIM(*np_xderiv, 1) != NX)) {
            ERR("np_xderiv must be size NY x NX (%i x %i), got %i x %i",
                NY, NX, (int)PyArray_DIM(*np_xderiv, 0),
                (int)PyArray_DIM(*np_xderiv, 1));
            return 1;
        }
    }
    if (np_yderiv && *np_yderiv) {
        if ((PyArray_DIM(*np_yderiv, 0) != NY) ||
            (PyArray_DIM(*np_yderiv, 1) != NX)) {
            ERR("np_yderiv must be size NY x NX (%i x %i), got %i x %i",
                NY, NX, (int)PyArray_DIM(*np_yderiv, 0),
                (int)PyArray_DIM(*np_yderiv, 1));
            return 1;
        }
    }
    if (np_mask && *np_mask) {
        if ((PyArray_DIM(*np_mask, 0) != NY) ||
            (PyArray_DIM(*np_mask, 1) != NX)) {
            ERR("np_mask must be size NY x NX (%i x %i), got %i x %i",
                NY, NX, (int)PyArray_DIM(*np_mask, 0),
                (int)PyArray_DIM(*np_mask, 1));
            return 1;
        }
    }
    return 0;
}


    %}

%init %{
    // numpy
    import_array();
    %}


%apply int *OUTPUT { int* p_sx0, int* p_sx1, int* p_sy0, int* p_sy1 };

%inline %{

#if 0
 } // fool silly text editor indenters...
#endif

static int c_gauss_2d(PyObject* ob_pos, PyObject* ob_amp,
                      PyObject* ob_mean, PyObject* ob_var,
                      PyObject* ob_result) {
    int i, N, d, K, k;
    const int D = 2;
    double *pos, *amp, *mean, *var, *result;
    PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
    int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
    int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
    double tpd;
    PyObject *np_pos=NULL, *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
    double *scale=NULL, *ivar=NULL;
    int rtn = -1;

    tpd = pow(2.*M_PI, D);

    Py_INCREF(dtype);
    Py_INCREF(dtype);
    Py_INCREF(dtype);
    Py_INCREF(dtype);
    Py_INCREF(dtype);
    np_pos = PyArray_FromAny(ob_pos, dtype, 2, 2, req, NULL);
    np_amp = PyArray_FromAny(ob_amp, dtype, 1, 1, req, NULL);
    np_mean = PyArray_FromAny(ob_mean, dtype, 2, 2, req, NULL);
    np_var = PyArray_FromAny(ob_var, dtype, 3, 3, req, NULL);
    np_result = PyArray_FromAny(ob_result, dtype, 1, 1, reqout, NULL);
    Py_CLEAR(dtype);

    if (!(np_pos && np_amp && np_mean && np_var && np_result)) {
        if (!np_pos) {
            ERR("pos wasn't the type expected");
            Py_DECREF(dtype);
        }
        if (!np_amp) {
            ERR("amp wasn't the type expected");
            Py_DECREF(dtype);
        }
        if (!np_mean) {
            ERR("mean wasn't the type expected");
            Py_DECREF(dtype);
        }
        if (!np_var) {
            ERR("var wasn't the type expected");
            Py_DECREF(dtype);
        }
        if (!np_result) {
            ERR("result wasn't the type expected");
            Py_DECREF(dtype);
        }
        goto bailout;
    }

    N = (int)PyArray_DIM(np_pos, 0);
    d = (int)PyArray_DIM(np_pos, 1);
    if (d != D) {
        ERR("must be 2-D");
        goto bailout;
    }
    K = (int)PyArray_DIM(np_amp, 0);
    //printf("K=%i\n", K);
    if ((PyArray_DIM(np_mean, 0) != K) ||
        (PyArray_DIM(np_mean, 1) != D)) {
        ERR("np_mean must be K x D");
        goto bailout;
    }
    if ((PyArray_DIM(np_var, 0) != K) ||
        (PyArray_DIM(np_var, 1) != D) ||
        (PyArray_DIM(np_var, 2) != D)) {
        ERR("np_var must be K x D x D");
        goto bailout;
    }
    if (PyArray_DIM(np_result, 0) != N) {
        ERR("np_result must be size N");
        goto bailout;
    }

    pos = PyArray_DATA(np_pos);
    amp = PyArray_DATA(np_amp);
    mean = PyArray_DATA(np_mean);
    var = PyArray_DATA(np_var);
    result = PyArray_DATA(np_result);

    scale = malloc(K * sizeof(double));
    ivar = malloc(K * D * D * sizeof(double));

    for (k=0; k<K; k++) {
        double* V = var + k*D*D;
        double* I = ivar + k*D*D;
        double det;
        det = V[0]*V[3] - V[1]*V[2];
        I[0] =  V[3] / det;
        I[1] = -V[1] / det;
        I[2] = -V[2] / det;
        I[3] =  V[0] / det;
        scale[k] = amp[k] / sqrt(tpd * det);
    }
    
    for (i=0; i<N; i++) {
        for (k=0; k<K; k++) {
            double dsq;
            double dx,dy;
            dx = pos[i*D+0] - mean[k*D+0];
            dy = pos[i*D+1] - mean[k*D+1];
            dsq = ivar[k*D*D + 0] * dx * dx
                + ivar[k*D*D + 1] * dx * dy
                + ivar[k*D*D + 2] * dx * dy
                + ivar[k*D*D + 3] * dy * dy;
            if (dsq >= 100)
                continue;
            result[i] += scale[k] * exp(-0.5 * dsq);
        }
    }
    rtn = 0;

bailout:
    free(scale);
    free(ivar);
    Py_XDECREF(np_pos);
    Py_XDECREF(np_amp);
    Py_XDECREF(np_mean);
    Py_XDECREF(np_var);
    Py_XDECREF(np_result);
    return rtn;
}




static int c_gauss_2d_grid(int x0, int x1, int y0, int y1, double fx, double fy,
                           PyObject* ob_amp, PyObject* ob_mean, PyObject* ob_var,
                           PyObject* ob_result) {
    int i, K, k;
    const int D = 2;
    double *amp, *mean, *var, *result;
    double tpd;
    PyObject *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
    int rtn = -1;
    int NX = x1 - x0;
    int NY = y1 - y0;

    tpd = pow(2.*M_PI, D);

    if (get_np(ob_amp, ob_mean, ob_var, ob_result, Py_None, Py_None, Py_None,
               NX, NY, &K, &np_amp, &np_mean, &np_var, &np_result,
               NULL, NULL, NULL, NULL))
        goto bailout;

    amp    = PyArray_DATA(np_amp);
    mean   = PyArray_DATA(np_mean);
    var    = PyArray_DATA(np_var);
    result = PyArray_DATA(np_result);

    {
        double scale[K];
        double ivar[K*3];
        int x1 = x0 + NX;
        int y1 = y0 + NY;
        int ix,iy;

        for (k=0; k<K; k++) {
            double* V = var + k*D*D;
            double* I = ivar + k*3;
            double det;
            det = V[0]*V[3] - V[1]*V[2];
            I[0] =  V[3] / det;
            I[1] = -(V[1]+V[2]) / det;
            I[2] =  V[0] / det;
            scale[k] = amp[k] / sqrt(tpd * det);
        }
    
        i = 0;
        for (iy=y0; iy<y1; iy++) {
            for (ix=x0; ix<x1; ix++) {
                for (k=0; k<K; k++) {
                    double dsq;
                    double dx,dy;
                    dx = ix - fx - mean[k*D+0];
                    dy = iy - fy - mean[k*D+1];
                    dsq = ivar[k*3 + 0] * dx * dx
                        + ivar[k*3 + 1] * dx * dy
                        + ivar[k*3 + 2] * dy * dy;
                    if (dsq >= 100)
                        continue;
                    result[i] += scale[k] * exp(-0.5 * dsq);
                }
                i++;
            }
        }
        rtn = 0;
    }

bailout:
    Py_XDECREF(np_amp);
    Py_XDECREF(np_mean);
    Py_XDECREF(np_var);
    Py_XDECREF(np_result);
    return rtn;
}

static int c_gauss_2d_approx(int x0, int x1, int y0, int y1,
                             double fx, double fy,
                             double minval,
                             PyObject* ob_amp,
                             PyObject* ob_mean,
                             PyObject* ob_var,
                             PyObject* ob_result) {
    double *amp, *mean, *var, *result;
    const int D=2;
    int K, k;
    int rtn = -1;
    PyObject *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
    double tpd;
    int W,H;
    W = x1 - x0;
    H = y1 - y0;
    tpd = pow(2.*M_PI, D);

    if (get_np(ob_amp, ob_mean, ob_var, ob_result, Py_None, Py_None, Py_None, W, H,
               &K, &np_amp, &np_mean, &np_var, &np_result, NULL, NULL, NULL, NULL))
        goto bailout;

    amp = PyArray_DATA(np_amp);
    mean = PyArray_DATA(np_mean);
    var = PyArray_DATA(np_var);
    result = PyArray_DATA(np_result);

    for (k=0; k<K; k++) {
        // We symmetrize the covariance matrix,
        // so V,I just have three elements: x**2, xy, y**2.
        // We also scale the the I to make the Gaussian evaluation easier
        int dyabs;
        double V[3];
        double I[3];
        double det;
        double isc;
        double scale;
        double mx,my;
        double mv;
        //int xc;
        int yc;
        V[0] = var[k*D*D + 0];
        V[1] = (var[k*D*D + 1] + var[k*D*D + 2])*0.5;
        V[2] = var[k*D*D + 3];
        det = V[0]*V[2] - V[1]*V[1];
        // we fold the -0.5 in the Gaussian exponent term in here...
        isc = -0.5 / det;
        I[0] =  V[2] * isc;
        // we also fold in the 2*dx*dy term here
        I[1] = -V[1] * isc * 2.0;
        I[2] =  V[0] * isc;
        scale = amp[k] / sqrt(tpd * det);
        mx = mean[k*D+0] + fx;
        my = mean[k*D+1] + fy;
        mv = minval * fabs(amp[k]);
        //printf("minval %g: amp %g, allowing mv %g\n", minval, amp[k], mv);
        //printf("minval %g, amp %g, scale %g, mv %g\n", minval, amp[k], scale, mv);
        //xc = MAX(x0, MIN(x1-1, (int)lround(mx)));
        yc = MAX(y0, MIN(y1-1, (int)lround(my)));
        //printf("mx,my (%.1f, %.1f)   xc,yc (%i,%i)\n", mx,my,xc,yc);
        for (dyabs=0; dyabs < MAX(y1-yc, 1+yc-y0); dyabs++) {
            int dysign;
            int ngood = 0;
            for (dysign=-1; dysign<=1; dysign+=2) {
                int dy;
                double g, v;
                int dir;
                int xm;
                int x, y;
                double* rrow;
                // only do the dy=0 row once
                if ((dyabs == 0) && (dysign == 1))
                    continue;
                dy = dyabs * dysign;
                y = yc + dy;
                if ((y < y0) || (y >= y1))
                    continue;
                // mean of conditional distribution of dx given dy
                xm = (int)lround(V[1] / V[2] * (y - my) + mx);
                xm = MAX(x0, MIN(x1-1, xm));
                // eval at dx=0
                // eval at dx = +- 1, ...
                // stop altogether if neither are accepted
                x = xm;
                g = eval_g(I, x - mx, y - my);
                //printf("g = %g vs mv %g\n", g, mv);
                rrow = result + (y - y0)*W - x0;
                v = scale * g;
                rrow[x] += v;
                if (v > mv)
                    ngood++;
                for (dir=0; dir<2; dir++) {
                    for (x = xm + (dir ? 1 : -1); (dir ? x < x1 : x >= x0); dir ? x++ : x--) {
                        g = eval_g(I, x - mx, y - my);
                        //printf("dx %i, g = %g vs mv %g\n", dx, g, mv);
                        v = scale * g;
                        rrow[x] += v;
                        if (v > mv)
                            ngood++;
                        else
                            break;
                    }
                }
            }
            // If this whole row (+ and -) was all < mv, we're done
            if (ngood == 0)
                break;
        }
    }
    rtn = 0;
bailout:
    Py_XDECREF(np_amp);
    Py_XDECREF(np_mean);
    Py_XDECREF(np_var);
    Py_XDECREF(np_result);
    return rtn;
}



static int c_gauss_2d_approx2(int x0, int x1, int y0, int y1,
                              // (fx,fy): center position
                              // which offsets "means"
                              double fx, double fy,
                              double minval,
                              PyObject* ob_amp,
                              PyObject* ob_mean,
                              PyObject* ob_var,
                              PyObject* ob_result) {
    double *amp, *mean, *var, *result;
    const int D=2;
    int K, k;
    int rtn = -1;
    PyObject *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
    double tpd;
    int W,H;
    double *II = NULL;
    double *VV = NULL;
    double *scales = NULL;
    double mx,my;
    int xc,yc;
    double maxpix;
    uint8_t *doT=NULL, *doB=NULL, *doL=NULL, *doR=NULL;
    uint8_t *nextT=NULL, *nextB=NULL, *nextL=NULL, *nextR=NULL;
    int R;

    W = x1 - x0;
    H = y1 - y0;
    tpd = pow(2.*M_PI, D);

    if (get_np(ob_amp, ob_mean, ob_var, ob_result, Py_None, Py_None, Py_None, W, H,
               &K, &np_amp, &np_mean, &np_var, &np_result, NULL, NULL, NULL, NULL))
        goto bailout;

    amp = PyArray_DATA(np_amp);
    mean = PyArray_DATA(np_mean);
    var = PyArray_DATA(np_var);
    result = PyArray_DATA(np_result);

    II = malloc(sizeof(double) * 3 * K);
    VV = malloc(sizeof(double) * 3 * K);
    scales = malloc(sizeof(double) * K);

    // We symmetrize the covariance matrix,
    // so V,I just have three elements for each K: x**2, xy, y**2.
    for (k=0; k<K; k++) {
        // We also scale the I to make the Gaussian evaluation easier
        double det;
        double isc;
        double scale;
        double* V = VV + 3*k;
        double* I = II + 3*k;
        V[0] =  var[k*D*D + 0];
        V[1] = (var[k*D*D + 1] + var[k*D*D + 2])*0.5;
        V[2] =  var[k*D*D + 3];
        det = V[0]*V[2] - V[1]*V[1];
        // we fold the -0.5 in the Gaussian exponent term in here...
        isc = -0.5 / det;
        I[0] =  V[2] * isc;
        // we also fold in the 2*dx*dy term here
        I[1] = -V[1] * isc * 2.0;
        I[2] =  V[0] * isc;
        if (det <= 0.) {
            // FIXME -- Abort?
            scales[k] = 0.;
        } else {
            scale = amp[k] / sqrt(tpd * det);
            scales[k] = scale;
        }
    }

    // Find (likely) max pixel.  This looks for the max pixel within
    // the box bounds for each component separately.  This isn't
    // correct, since we should look at the max of the SUM of
    // components...
    maxpix = 0.;
    xc = x0;
    yc = y0;
    for (k=0; k<K; k++) {
        double val;
        int ix,iy;
        mx = mean[k*D+0] + fx;
        my = mean[k*D+1] + fy;
        // inside postage stamp?
        ix = (int)lround(mx);
        iy = (int)lround(my);
        if ((ix >= x0) && (ix < x1) && (iy >= y0) && (iy < y1)) {
            val = scales[k];
        } else {
            double maxd = -HUGE_VAL;
            double maxx = x0;
            double maxy = y0;
            double yy, xx, dd;
            double dx, dy;
            double* I = II + 3*k;
            // outside postage stamp.  Min Mahalanobis distance along
            // four corners of the box.
            //
            // The I array is the *negative* inverse-covariance
            // so we want to *maximize*:
            //    dd = I[0]dx**2 + I[1]dx dy + I[2] dy**2
            //
            // x = x0
            // dx = x0 - mx,  dy = y - my
            // dd = I[0]dx**2 + I[1]dx dy + I[2] dy**2
            // d(dd)/dy = I[1]dx + 2 I[2] dy = 0
            // dy = -I[1]*dx / (2*I[2])
            // If within bounds, OR min of y0 or y1.
            // x = x0
            xx = x0;
            dx = xx - mx;
            dy = -I[1] * dx / (2. * I[2]);
            yy = dy + my;
            if (yy < y0) {
                yy = y0;
                dy = yy - my;
            } else if (yy > (y1-1)) {
                yy = y1-1;
                dy = yy - my;
            }
            dd = I[0]*dx*dx + I[1]*dx*dy + I[2]*dy*dy;
            if (dd > maxd) {
                maxd = dd;
                maxx = xx;
                maxy = yy;
            }
            // x = right edge
            xx = x1-1;
            dx = xx - mx;
            dy = -I[1] * dx / (2. * I[2]);
            yy = dy + my;
            if (yy < y0) {
                yy = y0;
                dy = yy - my;
            } else if (yy > (y1-1)) {
                yy = y1-1;
                dy = yy - my;
            }
            dd = I[0]*dx*dx + I[1]*dx*dy + I[2]*dy*dy;
            if (dd > maxd) {
                maxd = dd;
                maxx = xx;
                maxy = yy;
            }
            // y = bottom edge
            yy = y0;
            dy = yy - my;
            dx = -I[1] * dy / (2. * I[0]);
            xx = dx + mx;
            if (xx < x0) {
                xx = x0;
                dx = xx - mx;
            } else if (xx > (x1-1)) {
                xx = x1-1;
                dx = xx - mx;
            }
            dd = I[0]*dx*dx + I[1]*dx*dy + I[2]*dy*dy;
            if (dd > maxd) {
                maxd = dd;
                maxx = xx;
                maxy = yy;
            }
            // y = top edge
            yy = y1 - 1;
            dy = yy - my;
            dx = -I[1] * dy / (2. * I[0]);
            xx = dx + mx;
            if (xx < x0) {
                xx = x0;
                dx = xx - mx;
            } else if (xx > (x1-1)) {
                xx = x1-1;
                dx = xx - mx;
            }
            dd = I[0]*dx*dx + I[1]*dx*dy + I[2]*dy*dy;
            if (dd > maxd) {
                maxd = dd;
                maxx = xx;
                maxy = yy;
            }
            val = scales[k] * exp(maxd);
            ix = (int)lround(maxx);
            iy = (int)lround(maxy);
        }

        /*
         printf("component %i: max val %g at (%i,%i) (vs x [%i,%i), y [%i,%i)\n",
         k, val, ix, iy, x0, x1, y0, y1);
         */

        if (val > maxpix) {
            maxpix = val;
            xc = ix;
            yc = iy;
        }
        assert(xc >= x0);
        assert(xc <  x1);
        assert(yc >= y0);
        assert(yc <  y1);
    }

    // Starting from the central pixel (xc,yc), evaluate expanding
    // rings of pixels.  We mark pixels we want to evaluate in four
    // arrays, for the top, bottom, left, and right of the ring.  For
    // any pixel that evaluates above minval, we mark all its
    // neighbors for evaluation next time.
    // The "top" and "bottom" arrays take priority over the L,R.
    // That is, we only mark pixel +R,+R in the TOP array, not the Left.
    doT   = calloc(W, 1);
    doB   = calloc(W, 1);
    doL   = calloc(H, 1);
    doR   = calloc(H, 1);
    nextT = calloc(W, 1);
    nextB = calloc(W, 1);
    nextL = calloc(H, 1);
    nextR = calloc(H, 1);

#define SET(arr, i, lo, hi)                     \
    { if ((i >= lo) && (i < hi)) { arr[i - lo] = 1; } }

    // Mark the eight neighbors around the central pixel as needing to be done.
    if (yc > y0) {
        SET(nextB, xc+1, x0,x1);
        SET(nextB, xc,   x0,x1);
        SET(nextB, xc-1, x0,x1);
    }
    if (yc < (y1-1)) {
        SET(nextT, xc,   x0,x1);
        SET(nextT, xc+1, x0,x1);
        SET(nextT, xc-1, x0,x1);
    }
    if (xc > x0) {
        SET(nextL, yc,   y0,y1);
    }
    if (xc < (x1-1)) {
        SET(nextR, yc,   y0,y1);
    }
    result[(yc - y0)*W + (xc - x0)] = eval_all(K, scales, II, mean, xc-fx, yc-fy);

    for (R=1;; R++) {
        int any = 0;
        int xx, yy;
        int i;
        double* rrow;
        memcpy(doT, nextT, W);
        memcpy(doB, nextB, W);
        memcpy(doL, nextL, H);
        memcpy(doR, nextR, H);
        memset(nextT, 0, W);
        memset(nextB, 0, W);
        memset(nextL, 0, H);
        memset(nextR, 0, H);

        // Beautiful ASCII art...
        /*
         printf("R = %i, xc,yc = (%i,%i)\n", R, xc, yc);
         for (i=MAX(xc-x0-R, 0); i<=MIN(xc-x0+R, W-1); i++) {
         printf("%c", (doT[i] ? '*' : '-'));
         }
         printf("\n");
         for (i=MIN(yc-y0+R-1, H-1); i>=MAX(yc-y0-R+1, 0); i--) {
         printf("%c", (doL[i] ? '*' : '|'));
         int j;
         for (j=MAX(xc-x0-R, 0)+1; j<=MIN(xc-x0+R, W-1)-1; j++)
         printf(" ");
         printf("%c", (doR[i] ? '*' : '|'));
         printf("\n");
         }
         for (i=MAX(xc-x0-R, 0); i<=MIN(xc-x0+R, W-1); i++) {
         printf("%c", (doB[i] ? '*' : '-'));
         }
         printf("\n");
         printf("\n");

         for (i=0; i<H; i++) {
         int j;
         for (j=0; j<W; j++) {
         if (i == (yc+R-y0)) {
         printf("%c", (doT[j] ? '*' : '-'));
         } else if (i == (yc-R-y0)) {
         printf("%c", (doB[j] ? '*' : '-'));
         } else if (j == (xc-R-x0)) {
         printf("%c", (doL[i] ? '*' : '-'));
         } else if (j == (xc+R-x0)) {
         printf("%c", (doR[i] ? '*' : '-'));
         } else {
         printf(".");
         }
         }
         printf("\n");
         }
         printf("\n");
         */

        // top
        yy = yc + R;
        if (yy < y1) {
            rrow = result + (yy - y0)*W;
            for (i=0; i<W; i++) {
                double r;
                if (!doT[i])
                    continue;
                any = 1;
                xx = x0 + i;
                r = eval_all(K, scales, II, mean, xx-fx, yy-fy);
                //result[(yy - y0)*W + (xx - x0)] = r;
                rrow[i] = r;
                //printf("top[xx=%i] = %g\n", xx, r);
                if (r < minval)
                    continue;
                // leftmost pixel of Top, and not at the left edge...
                if ((xx == (xc - R)) && (xx > x0)) {
                    //printf("setting L\n");
                    SET(nextL, yy  , y0,y1);
                    SET(nextL, yy-1, y0,y1);
                }
                // rightmost pixel of Top, and not at the right edge...
                if ((xx == (xc + R)) && (xx < (x1-1))) {
                    //printf("setting R\n");
                    SET(nextR, yy  , y0,y1);
                    SET(nextR, yy-1, y0,y1);
                }
                // not the top edge...
                if (yy < (y1-1)) {
                    //printf("setting T\n");
                    SET(nextT, xx-1, x0,x1);
                    SET(nextT, xx  , x0,x1);
                    SET(nextT, xx+1, x0,x1);
                }
            }
        }
        // bottom
        yy = yc - R;
        if (yy >= y0) {
            rrow = result + (yy - y0)*W;
            for (i=0; i<W; i++) {
                double r;
                if (!doB[i])
                    continue;
                any = 1;
                xx = x0 + i;
                r = eval_all(K, scales, II, mean, xx-fx, yy-fy);
                //result[(yy - y0)*W + (xx - x0)] = r;
                rrow[i] = r;
                //printf("bottom[xx=%i] = %g\n", xx, r);
                if (r < minval)
                    continue;
                // leftmost pixel in Bottom, and not left edge?
                if ((xx == (xc - R)) && (xx > x0)) {
                    //printf("setting L\n");
                    SET(nextL, yy  , y0,y1);
                    SET(nextL, yy+1, y0,y1);
                }
                // rightmost pixel in Bottom, and not right edge?
                if ((xx == (xc + R)) && (xx < (x1-1))) {
                    //printf("setting R\n");
                    SET(nextR, yy  , y0,y1);
                    SET(nextR, yy+1, y0,y1);
                }
                // not the bottom edge?
                if (yy > y0) {
                    //printf("setting B\n");
                    SET(nextB, xx-1, x0,x1);
                    SET(nextB, xx  , x0,x1);
                    SET(nextB, xx+1, x0,x1);
                }
            }
        }

        // left
        xx = xc - R;
        if (xx >= x0) {
            for (i=0; i<H; i++) {
                double r;
                if (!doL[i])
                    continue;
                any = 1;
                yy = y0 + i;
                r = eval_all(K, scales, II, mean, xx-fx, yy-fy);
                result[(yy - y0)*W + (xx - x0)] = r;
                //printf("left[yy=%i] = %g\n", xx, r);
                if (r < minval)
                    continue;
                // not the left edge?
                if (xx > x0) {
                    //printf("setting L\n");
                    SET(nextL, yy-1, y0,y1);
                    SET(nextL, yy  , y0,y1);
                    SET(nextL, yy+1, y0,y1);
                }
            }
        }
        // right
        xx = xc + R;
        if (xx < x1) {
            for (i=0; i<H; i++) {
                double r;
                if (!doR[i])
                    continue;
                any = 1;
                yy = y0 + i;
                r = eval_all(K, scales, II, mean, xx-fx, yy-fy);
                result[(yy - y0)*W + (xx - x0)] = r;
                //printf("right[yy=%i] = %g\n", yy, r);
                if (r < minval)
                    continue;
                // not the right edge?
                if (xx < (x1-1)) {
                    //printf("setting R\n");
                    SET(nextR, yy-1, y0,y1);
                    SET(nextR, yy  , y0,y1);
                    SET(nextR, yy+1, y0,y1);
                }
            }
        }
        if (!any)
            break;
    }
    rtn = 0;

#undef SET

bailout:
    free(doT);
    free(doB);
    free(doL);
    free(doR);
    free(nextT);
    free(nextB);
    free(nextL);
    free(nextR);

    free(II);
    free(VV);
    free(scales);

    Py_XDECREF(np_amp);
    Py_XDECREF(np_mean);
    Py_XDECREF(np_var);
    Py_XDECREF(np_result);
    return rtn;
}


static int c_gauss_2d_approx3(int x0, int x1, int y0, int y1,
                              // (fx,fy): center position
                              // which offsets "means"
                              double fx, double fy,
                              double minval,
                              PyObject* ob_amp,
                              PyObject* ob_mean,
                              PyObject* ob_var,
                              PyObject* ob_result,
                              PyObject* ob_xderiv,
                              PyObject* ob_yderiv,
                              PyObject* ob_mask,
                              int xc, int yc,
                              int minradius,
                              int* p_sx0, int* p_sx1, int* p_sy0, int* p_sy1
                              );

#include "approx3.c"


static int c_gauss_2d_masked(int x0, int y0, int W, int H,
                             // (fx,fy): center position
                             // which offsets "means"
                             double fx, double fy,
                             PyObject* ob_amp,
                             PyObject* ob_mean,
                             PyObject* ob_var,
                             PyObject* ob_result,
                             PyObject* ob_xderiv,
                             PyObject* ob_yderiv,
                             PyObject* ob_mask);

#include "gauss_masked.c"


%}


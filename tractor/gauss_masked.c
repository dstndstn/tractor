static int c_gauss_2d_masked(int x0, int y0, int W, int H,
                             // (fx,fy): center position
                             // which offsets "means"
                             double fxd, double fyd,
                             PyObject* ob_amp,
                             PyObject* ob_mean,
                             PyObject* ob_var,
                             PyObject* ob_result,
                             PyObject* ob_xderiv,
                             PyObject* ob_yderiv,
                             PyObject* ob_mask) {

    // ob_mask: numpy array, boolean: which pixels to evaluate.
    //
    // ob_xderiv: if not NULL, result array for x derivative
    // ob_yderiv: if not NULL, result array for y derivative

    int nexpf0 = n_expf;

    float *amp, *mean, *var, *result;
    float *xderiv=NULL, *yderiv=NULL;
    float fx = (float)fxd;
    float fy = (float)fyd;
    uint8_t* mask=NULL;
    const int D=2;
    int K, k;
    PyObject *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
    PyObject *np_xderiv=NULL, *np_yderiv=NULL, *np_mask=NULL;
    float tpd;
    float *pxd = NULL, *pyd = NULL;
    int rtn;
    PyArray_Descr* ftype = PyArray_DescrFromType(PyArray_FLOAT32);

    tpd = pow(2.*M_PI, D);

    if (get_np(ob_amp, ob_mean, ob_var, ob_result, ob_xderiv, ob_yderiv,
               ob_mask, W, H,
               &K, &np_amp, &np_mean, &np_var, &np_result, &np_xderiv, &np_yderiv,
               &np_mask, ftype)) {
        printf("get_np failed\n");
        goto bailout;
    }

    rtn = 0;
    amp    = PyArray_DATA(np_amp);
    mean   = PyArray_DATA(np_mean);
    var    = PyArray_DATA(np_var);
    result = PyArray_DATA(np_result);
    if (np_xderiv)
        xderiv = PyArray_DATA(np_xderiv);
    if (np_yderiv)
        yderiv = PyArray_DATA(np_yderiv);
    if (np_mask)
        mask   = PyArray_DATA(np_mask);

    {
        float II[3*K];
        float VV[3*K];
        float scales[K];
        float maxD[K];
        float ampsum = 0.;
        int allzero = 1;

        for (k=0; k<K; k++)
            ampsum += amp[k];
            
        // We symmetrize the covariance matrix,
        // so V,I just have three elements for each K: x**2, xy, y**2.
        for (k=0; k<K; k++) {
            // We also scale the I to make the Gaussian evaluation easier
            float det;
            float isc;
            float* V = VV + 3*k;
            float* I = II + 3*k;
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
            scales[k] = amp[k] / sqrt(tpd * det);
            maxD[k] = -100.;

            if (!(isfinite(V[0]) && isfinite(V[1]) && isfinite(V[2]) &&
                  isfinite(I[0]) && isfinite(I[1]) && isfinite(I[2]) &&
                  isfinite(scales[k]))) {
                printf("Warning: infinite variance or scale.  Zeroing.\n");
                // large variance can cause this... set scale = 0.
                scales[k] = 0.;
                V[0] = V[1] = V[2] = 1.;
                I[0] = I[1] = I[2] = 1.;
                det = 1.;
                isc = 1.;
            }

            if (scales[k] != 0)
                allzero = 0;
        }

        if (allzero)
            goto bailout;

        int dx, dy;
        for (dy=0; dy<H; dy++) {
            int y = y0 + dy;
            int i0 = dy * W;
            for (dx=0; dx<W; dx++) {
                if (!mask[i0 + dx])
                    continue;
                int x = x0 + dx;
                if (xderiv)
                    pxd = xderiv + i0 + dx;
                if (yderiv)
                    pyd = yderiv + i0 + dx;
                float v = eval_all_dxy_f(K, scales, II, mean, x-fx, y-fy, pxd, pyd, maxD);
                result[i0 + dx] = v;
            }
        }
    }
 bailout:
    Py_XDECREF(np_amp);
    Py_XDECREF(np_mean);
    Py_XDECREF(np_var);
    Py_XDECREF(np_result);
    Py_XDECREF(np_xderiv);
    Py_XDECREF(np_yderiv);
    Py_XDECREF(np_mask);

    //printf("N expf calls: %i\n", n_expf - nexpf0);
    return rtn;
}

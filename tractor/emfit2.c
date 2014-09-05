static int em_fit_2d_reg2(PyObject* np_img, int x0, int y0,
                          PyObject* np_amp,
                          PyObject* np_mean,
                          PyObject* np_var,
                          double alpha,
                          int steps) {
    npy_intp i, N, K, k;
    npy_intp ix, iy;
    npy_intp NX, NY;
    const npy_intp D = 2;
    double* scale = NULL, *ivar = NULL;
    int step;
    double tpd;
    int result;

    PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
    int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
    int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;

    double* imgcursor;

    double* amp;
    double* mean;
    double* var;
    double* img;

    tpd = pow(2.*M_PI, D);

    Py_INCREF(dtype);
    np_img = PyArray_FromAny(np_img, dtype, 2, 2, req, NULL);
    if (!np_img) {
        ERR("img wasn't the type expected");
        Py_DECREF(dtype);
        return -1;
    }
    Py_INCREF(dtype);
    np_amp = PyArray_FromAny(np_amp, dtype, 1, 1, reqout, NULL);
    if (!np_amp) {
        ERR("amp wasn't the type expected");
        Py_DECREF(np_img);
        Py_DECREF(dtype);
        return -1;
    }
    Py_INCREF(dtype);
    np_mean = PyArray_FromAny(np_mean, dtype, 2, 2, reqout, NULL);
    if (!np_mean) {
        ERR("mean wasn't the type expected");
        Py_DECREF(np_img);
        Py_DECREF(np_amp);
        Py_DECREF(dtype);
        return -1;
    }
    Py_INCREF(dtype);
    np_var = PyArray_FromAny(np_var, dtype, 3, 3, reqout, NULL);
    if (!np_var) {
        ERR("var wasn't the type expected");
        Py_DECREF(np_img);
        Py_DECREF(np_amp);
        Py_DECREF(np_mean);
        Py_DECREF(dtype);
        return -1;
    }

    K = PyArray_DIM(np_amp, 0);
    // printf("K=%i\n", K);
    if ((PyArray_DIM(np_mean, 0) != K) ||
        (PyArray_DIM(np_mean, 1) != D)) {
        ERR("np_mean must be K x D");
        return -1;
    }
    if ((PyArray_DIM(np_var, 0) != K) ||
        (PyArray_DIM(np_var, 1) != D) ||
        (PyArray_DIM(np_var, 2) != D)) {
        ERR("np_var must be K x D x D");
        return -1;
    }
    NY = PyArray_DIM(np_img, 0);
    NX = PyArray_DIM(np_img, 1);

    amp  = PyArray_DATA(np_amp);
    mean = PyArray_DATA(np_mean);
    var  = PyArray_DATA(np_var);
    img  = PyArray_DATA(np_img);

    N = NX*NY;
    scale = malloc(K * sizeof(double));
    ivar = malloc(K * D * D * sizeof(double));
    assert(scale && ivar);

    for (step=0; step<steps; step++) {
        double x,y;
        double new_mean[K*D];
        double Zi[K];

        /*              {
         int d;
         printf("step=%i: ", step);
         printf("w=[");
         for (k=0; k<K; k++)
         printf("%g ", amp[k]);
         printf("], mu=[");
         for (k=0; k<K; k++) {
         printf("[");
         for (d=0; d<D; d++)
         printf("%s%g", (d?" ":""), mean[k*D+d]);
         printf("], ");
         }
         printf("], var=[");
         for (k=0; k<K; k++) {
         printf("[ ");
         for (d=0; d<D*D; d++)
         printf("%s%g", (d?" ":""), var[k*D*D+d]);
         printf("], ");
         }
         printf("]\n");
         }
         */
        for (k=0; k<K; k++) {
            // ASSUME ordering
            double* V = var + k*D*D;
            double* I = ivar + k*D*D;
            double det;
            det = V[0]*V[3] - V[1]*V[2];
            if (det <= 0.0) {
                printf("det = %g\n", det);
                ERR("Got non-positive determinant\n");
                result = -1;
                goto cleanup;
            }
            I[0] =  V[3] / det;
            I[1] = -V[1] / det;
            I[2] = -V[2] / det;
            I[3] =  V[0] / det;
            scale[k] = amp[k] / sqrt(tpd * det);
        }

        imgcursor = img;
        imgcursor--;

        // We pre-compute amp -> scale, used during the loop; amp
        // array is not needed, so we use it as accumulator for new
        // amp.
        memset(amp, 0, K*sizeof(double));
        memset(new_mean, 0, K*D*sizeof(double));

        // We pre-compute ivar, which is used during the loop; var
        // (old var) is not needed during the loop, so we use that
        // array to accumulate new_var.
        for (k=0; k<K; k++) {
            var[k*D*D + 0] = alpha;
            var[k*D*D + 1] = 0;
            var[k*D*D + 2] = 0;
            var[k*D*D + 3] = alpha;
        }

        for (iy=0; iy<NY; iy++) {
            for (ix=0; ix<NX; ix++) {
                double zi;
                double zsum = 0;
                // next pixel
                imgcursor++;
                x = x0 + ix;
                y = y0 + iy;
                // E step
                for (k=0; k<K; k++) {
                    double dsq;
                    double dx,dy;
                    dx = x - mean[k*D+0];
                    dy = y - mean[k*D+1];
                    dsq = ivar[k*D*D + 0] * dx * dx
                        + ivar[k*D*D + 1] * dx * dy
                        + ivar[k*D*D + 2] * dx * dy
                        + ivar[k*D*D + 3] * dy * dy;
                    if (dsq >= 100) {
                        Zi[k] = 0.;
                        continue;
                    }
                    zi = scale[k] * exp(-0.5 * dsq);
                    zsum += zi;
                    Zi[k] = zi;
                }
                if (zsum == 0)
                    continue;

                // M step
                for (k=0; k<K; k++) {
                    double dx, dy;
                    double wi = *imgcursor * Zi[k] / zsum;
                    amp[k] += wi;
                    x = x0 + ix;
                    y = y0 + iy;
                    new_mean[k*D + 0] += wi * x;
                    new_mean[k*D + 1] += wi * y;
                    dx = x - mean[k*D+0];
                    dy = y - mean[k*D+1];
                    var[k*D*D + 0] += wi * dx*dx;
                    var[k*D*D + 1] += wi * dx*dy;
                    var[k*D*D + 2] += wi * dy*dx;
                    var[k*D*D + 3] += wi * dy*dy;
                }
            }
        }
        for (k=0; k<K; k++) {
            mean[k*D + 0] = new_mean[k*D + 0] / amp[k];
            mean[k*D + 1] = new_mean[k*D + 1] / amp[k];
            for (i=0; i<D*D; i++)
                var[k*D*D + i] /= (amp[k] + alpha);
        }
    }

    result = 0;
        
 cleanup:
    free(scale);
    free(ivar);

    Py_DECREF(np_img);
    Py_DECREF(np_amp);
    Py_DECREF(np_mean);
    Py_DECREF(np_var);
    Py_DECREF(dtype);

    return result;
}

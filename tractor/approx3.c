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
                              int minradius) {

    //
    // ob_mask: numpy array, shape (y1-y0, x1-x0), boolean: which
    // pixels to evaluate.
    //
    // ob_xderiv: if not NULL, result array for x derivative
    // ob_yderiv: if not NULL, result array for y derivative
    //
    // xc, yc: "center" pixel from which to begin evaluation.  If
    // outside x0,x1,y0,y1, the largest boundary value will be chosen
    // as the start point.
    //
    // minradius: minimum number of pixels to evaluate, regardless of minval

    double *amp, *mean, *var, *result;
    double *xderiv=NULL, *yderiv=NULL;
    uint8_t* mask=NULL;
    const int D=2;
    int K, k;
    int rtn = -1;
    PyObject *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
    PyObject *np_xderiv=NULL, *np_yderiv=NULL, *np_mask=NULL;
    double tpd;
    int W,H;
    double *II = NULL;
    double *VV = NULL;
    double *scales = NULL;
    uint8_t *doT=NULL, *doB=NULL, *doL=NULL, *doR=NULL;
    uint8_t *nextT=NULL, *nextB=NULL, *nextL=NULL, *nextR=NULL;
    int R;
    double *pxd = NULL, *pyd = NULL;
    int off;

    W = x1 - x0;
    H = y1 - y0;
    tpd = pow(2.*M_PI, D);

    if (get_np(ob_amp, ob_mean, ob_var, ob_result, ob_xderiv, ob_yderiv,
               ob_mask, W, H,
               &K, &np_amp, &np_mean, &np_var, &np_result, &np_xderiv, &np_yderiv,
               &np_mask)) {
        printf("get_np failed\n");
        goto bailout;
    }

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

    II = malloc(sizeof(double) * 3 * K);
    VV = malloc(sizeof(double) * 3 * K);
    scales = malloc(sizeof(double) * K);

    // We symmetrize the covariance matrix,
    // so V,I just have three elements for each K: x**2, xy, y**2.
    for (k=0; k<K; k++) {
        // We also scale the I to make the Gaussian evaluation easier
        double det;
        double isc;
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
        scales[k] = amp[k] / sqrt(tpd * det);
    }

    // is the given starting pixel xc,yc outside the box the be evaluated?
    if ((xc < x0) || (yc < y0) || (xc >= x1) || (yc >= y1)) {
        // Find max pixel on the boundary and use that as xc,yc
        int newxc, newyc;
        int ix,iy;
        double maxpix = 0.;
        newxc = (x0+x1)/2;
        newyc = (y0+y1)/2;
        // We search along the edge(s) nearest the given xc,yc.
        if ((xc < x0) || (xc >= x1)) {
            if (xc < x0) {
                ix = x0;
            } else {
                ix = x1-1;
            }
            for (iy=y0; iy<y1; iy++) {
                double v = eval_all(K, scales, II, mean, ix-fx, iy-fy);
                if (v > maxpix) {
                    maxpix = v;
                    newxc = ix;
                    newyc = iy;
                }
            }
        }
        if ((yc < y0) || (yc >= y1)) {
            if (yc < y0) {
                iy = y0;
            } else {
                iy = y1-1;
            }
            for (ix=x0; ix<x1; ix++) {
                double v = eval_all(K, scales, II, mean, ix-fx, iy-fy);
                if (v > maxpix) {
                    maxpix = v;
                    newxc = ix;
                    newyc = iy;
                }
            }
        }
        xc = newxc;
        yc = newyc;
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

    off = (yc - y0)*W + (xc - x0);
    if (xderiv)
        pxd = xderiv + off;
    if (yderiv)
        pyd = yderiv + off;
    result[off] = eval_all_dxy(K, scales, II, mean, xc-fx, yc-fy, pxd, pyd);

    //printf("xc,yc %i,%i, vs x [%i,%i], y [%i,%i] -> res %g\n",
    //xc, yc, x0, x1, y0, y1, result[off]);

    for (R=1;; R++) {
        int any = 0;
        int xx, yy;
        int i;
        double* rrow;
        uint8_t* tmparr;

        // Swap "do" and "next" arrays (do <= next)
        tmparr = doT;
        doT = nextT;
        nextT = tmparr;
        tmparr = doB;
        doB = nextB;
        nextB = tmparr;
        tmparr = doL;
        doL = nextL;
        nextL = tmparr;
        tmparr = doR;
        doR = nextR;
        nextR = tmparr;
        memset(nextT, 0, W);
        memset(nextB, 0, W);
        memset(nextL, 0, H);
        memset(nextR, 0, H);

        // Beautiful ASCII art...
        if (0) {
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
        }

        // top
        yy = yc + R;
        if (yy < y1) {
            off = (yy - y0)*W;
            rrow = result + off;
            if (xderiv)
                pxd = xderiv + off;
            if (yderiv)
                pyd = yderiv + off;
            for (i=0; i<W; i++) {
                double r;
                if (!doT[i])
                    continue;
                any = 1;
                xx = x0 + i;
                r = eval_all_dxy(K, scales, II, mean, xx-fx, yy-fy,
                                 (pxd ? pxd+i : NULL), (pyd ? pyd+i : NULL));

                //result[(yy - y0)*W + (xx - x0)] = r;
                rrow[i] = r;
                //printf("top[xx=%i] = %g\n", xx, r);

                // If we're inside the minradius, mark the next pixels as game
                printf("r=%g vs minval %g; R=%i vs minradius %i\n", r, minval, R, minradius);
                if ((R > minradius) && (r < minval))
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
            off = (yy - y0)*W;
            rrow = result + off;
            if (xderiv)
                pxd = xderiv + off;
            if (yderiv)
                pyd = yderiv + off;
            for (i=0; i<W; i++) {
                double r;
                if (!doB[i])
                    continue;
                any = 1;
                xx = x0 + i;
                r = eval_all_dxy(K, scales, II, mean, xx-fx, yy-fy,
                                 (pxd?pxd+i:NULL), (pyd?pyd+i:NULL));
                //result[(yy - y0)*W + (xx - x0)] = r;
                rrow[i] = r;
                //printf("bottom[xx=%i] = %g\n", xx, r);
                printf("r=%g vs minval %g; R=%i vs minradius %i\n", r, minval, R, minradius);
                if ((R > minradius) && (r < minval))
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
                off = (yy - y0)*W + (xx - x0);
                r = eval_all_dxy(K, scales, II, mean, xx-fx, yy-fy,
                                 (xderiv ? xderiv+off : NULL),
                                 (yderiv ? yderiv+off : NULL));
                result[off] = r;
                //printf("left[yy=%i] = %g\n", xx, r);
                printf("r=%g vs minval %g; R=%i vs minradius %i\n", r, minval, R, minradius);
                if ((R > minradius) && (r < minval))
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
                off = (yy - y0)*W + (xx - x0);
                r = eval_all_dxy(K, scales, II, mean, xx-fx, yy-fy,
                                 (xderiv ? xderiv+off : NULL),
                                 (yderiv ? yderiv+off : NULL));
                result[off] = r;
                //printf("right[yy=%i] = %g\n", yy, r);
                printf("r=%g vs minval %g; R=%i vs minradius %i\n", r, minval, R, minradius);
                if ((R > minradius) && (r < minval))
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
    Py_XDECREF(np_xderiv);
    Py_XDECREF(np_yderiv);
    Py_XDECREF(np_mask);

    return rtn;
    }

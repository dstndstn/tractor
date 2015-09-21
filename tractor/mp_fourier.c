const static double amps6[] = {
    1.99485977e-04,   2.61612679e-03,   1.89726655e-02,
    1.00186544e-01,   3.68534484e-01,   5.09490694e-01 };


#define ALIGNED(x) x

// #define ALIGNED(x) __builtin_assume_aligned(x, 32)


static inline void
mp_fourier_core(int NW, int NV, int K,
                double mu0, double mu1,
                const double* restrict vv_in,
                const double* restrict ww_in,
                const double* restrict amps_in,
                const double* restrict vars_in,
                double* restrict f_in) {

    int i, j, k;

    double*       restrict ff   = ALIGNED(f_in   );
    const double* restrict vv   = ALIGNED(vv_in  );
    const double* restrict ww   = ALIGNED(ww_in  );
    const double* restrict amps = ALIGNED(amps_in);
    const double* restrict vars = ALIGNED(vars_in);

    const double twopisquare = -2. * M_PI * M_PI;
    for (j=0; j<NW; j++) {
        for (i=0; i<NV; i++) {
            double s = 0;
            const double* restrict V = ALIGNED(vars);
            for (k=0; k<K; k++) {
                double a, b, d;
                a = *V;
                V++;
                b = *V;
                V++;
                // skip c
                V++;
                d = *V;
                V++;

                s += amps[k] * exp(twopisquare * (a *  vv[i]*vv[i] +
                                                  2.*b*vv[i]*ww[j] +
                                                  d *  ww[j]*ww[j]));
            }
            double angle = -2. * M_PI * (mu0 * vv[i] + mu1 * ww[j]);
            *ff = s * cos(angle);
            ff++;
            *ff = s * sin(angle);
            ff++;
        }
    }

}

                              


static inline void
mp_fourier_core_vw(int NW, int NV, int K,
                   double mu0, double mu1,
                   double v0, double dv,
                   double w0, double dw,
                   const double* restrict amps_in,
                   const double* restrict vars_in,
                   double* restrict f_in) {

    int i, j, k;

    double*       restrict ff   = ALIGNED(f_in   );
    const double* restrict amps = ALIGNED(amps_in);
    const double* restrict vars = ALIGNED(vars_in);

    const double twopisquare = -2. * M_PI * M_PI;
    double w = w0;
    for (j=0; j<NW; j++) {
        w += dw;
        double v = v0;
        for (i=0; i<NV; i++) {
            double s = 0;
            v += dv;
            const double* restrict V = ALIGNED(vars);
            for (k=0; k<K; k++) {
                double a, b, d;
                a = *V;
                V++;
                b = *V;
                V++;
                // skip c
                V++;
                d = *V;
                V++;

                s += amps[k] * exp(twopisquare * (a *  v*v +
                                                  2.*b*v*w +
                                                  d *  w*w));
            }
            double angle = -2. * M_PI * (mu0 * v + mu1 * w);
            *ff = s * cos(angle);
            ff++;
            *ff = s * sin(angle);
            ff++;
        }
    }

}


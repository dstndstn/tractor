#include <stdio.h>
#include <stdint.h>
#include <sys/param.h>

#include <vector>

#include "ceres-tractor.h"

ForcedPhotCostFunction::ForcedPhotCostFunction(Patch data,
                                               std::vector<Patch> sources) :
    _data(data), _sources(sources) {

    set_num_residuals(data.npix());
    std::vector<int16_t>* bs = mutable_parameter_block_sizes();
    for (size_t i=0; i<sources.size(); i++) {
        bs->push_back(1);
    }
    /*
     printf("ForcedPhotCostFunction: npix %i, nsources %i\n",
     num_residuals(), (int)parameter_block_sizes().size());
     */
}

ForcedPhotCostFunction::~ForcedPhotCostFunction() {}

bool ForcedPhotCostFunction::Evaluate(double const* const* parameters,
                                      double* residuals,
                                      double** jacobians) const {
    const std::vector<int16_t> bs = parameter_block_sizes();

    /*
     printf("ForcedPhotCostFunction::Evaluate\n");
     printf("Parameter blocks:\n");
     for (size_t i=0; i<bs.size(); i++) {
     printf("  %i: [", (int)i);
     for (int j=0; j<bs[i]; j++) {
     printf(" %g,", parameters[i][j]);
     }
     printf(" ]\n");
     }
     */

    double* mod;
    if (_data._mod0) {
        mod = (double*)malloc(_data.npix() * sizeof(double));
        memcpy(mod, _data._mod0, _data.npix() * sizeof(double));
    } else {
        mod = (double*)calloc(_data.npix(), sizeof(double));
    }

    for (size_t i=0; i<bs.size(); i++) {
        assert(bs[i] == 1);
        int j = 0;
        double flux = parameters[i][j];
        // add source*flux to mod
        Patch source = _sources[i];

        int xlo = MAX(source._x0, _data._x0);
        int xhi = MIN(source._x0 + source._w, _data._x0 + _data._w);
        int ylo = MAX(source._y0, _data._y0);
        int yhi = MIN(source._y0 + source._h, _data._y0 + _data._h);

        /*
         printf("Adding source %i: x [%i, %i), y [%i, %i)\n",
         (int)i, xlo, xhi, ylo, yhi);
         */

        int nx = xhi - xlo;
        for (int y=ylo; y<yhi; y++) {
            double* orow =         mod + ((y -  _data._y0) *  _data._w) +
                (xlo -  _data._x0);
            double* irow = source._img + ((y - source._y0) * source._w) +
                (xlo - source._x0);
            for (int x=0; x<nx; x++, orow++, irow++) {
                (*orow) += (*irow) * flux;
            }
        }
    }

    for (int i=0; i<_data.npix(); i++) {
        residuals[i] = (_data._img[i] - mod[i]) * _data._ierr[i];
    }

    free(mod);

    /*
     printf("Returning residual:\n");
     for (int y=0; y<_data._h; y++) {
     printf("row %i: [ ", y);
     for (int x=0; x<_data._w; x++) {
     printf("%6.1f ", residuals[y * _data._w + x]);
     }
     printf(" ]\n");
     }
     */

    for (size_t i=0; i<bs.size(); i++) {
        if (!jacobians || !jacobians[i])
            continue;
        for (int k=0; k<_data.npix(); k++)
            jacobians[i][k] = 0.;
        Patch source = _sources[i];
        int xlo = MAX(source._x0, _data._x0);
        int xhi = MIN(source._x0 + source._w, _data._x0 + _data._w);
        int ylo = MAX(source._y0, _data._y0);
        int yhi = MIN(source._y0 + source._h, _data._y0 + _data._h);
        int nx = xhi - xlo;
        for (int y=ylo; y<yhi; y++) {
            double* orow = jacobians[i] + ((y -  _data._y0) *  _data._w) +
                (xlo -  _data._x0);
            double* irow = source._img  + ((y - source._y0) * source._w) +
                (xlo - source._x0);
            double* erow =  _data._ierr + ((y -  _data._y0) *  _data._w) +
                (xlo -  _data._x0);
            for (int x=0; x<nx; x++, orow++, irow++, erow++) {
                (*orow) = -1.0 * (*irow) * (*erow);
            }
        }

        /*
         printf("Returning Jacobian:\n");
         for (int y=0; y<_data._h; y++) {
         printf("row %i: [ ", y);
         for (int x=0; x<_data._w; x++) {
         printf("%6.1f ", jacobians[i][y * _data._w + x]);
         }
         printf(" ]\n");
         }
         */

    }

    return true;
}


#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdint.h>
#include <sys/param.h>

#include <vector>

#include "ceres-tractor.h"

template<typename T>
ForcedPhotCostFunction<T>::ForcedPhotCostFunction(Patch<T> data,
                                                  std::vector<Patch<T> > sources,
                                                  int nonneg) :
    _data(data), _sources(sources), _nonneg(nonneg) {

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

template<typename T>
ForcedPhotCostFunction<T>::~ForcedPhotCostFunction() {}

template<typename T>
bool ForcedPhotCostFunction<T>::Evaluate(double const* const* parameters,
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

    //double maxJ = 0.;

    T* mod;
    if (_data._mod0) {
        mod = (T*)malloc(_data.npix() * sizeof(T));
        memcpy(mod, _data._mod0, _data.npix() * sizeof(T));
    } else {
        mod = (T*)calloc(_data.npix(), sizeof(T));
    }

    for (size_t i=0; i<bs.size(); i++) {
        assert(bs[i] == 1);
        int j = 0;

        double flux;
        if (_nonneg) {
            flux = exp(parameters[i][j]);
        } else {
            flux = parameters[i][j];
        }
        Patch<T> source = _sources[i];

        int xlo = MAX(source._x0, _data._x0);
        int xhi = MIN(source._x0 + source._w, _data._x0 + _data._w);
        int ylo = MAX(source._y0, _data._y0);
        int yhi = MIN(source._y0 + source._h, _data._y0 + _data._h);

        /*
         printf("Adding source %i: x [%i, %i), y [%i, %i)\n",
         (int)i, xlo, xhi, ylo, yhi);
         */

        // Compute model & jacobians
        if (jacobians && jacobians[i]) {
            for (int k=0; k<_data.npix(); k++)
                jacobians[i][k] = 0.;
        }
        int nx = xhi - xlo;
        for (int y=ylo; y<yhi; y++) {
            T* modrow  =         mod + ((y -  _data._y0) *  _data._w) +
                (xlo -  _data._x0);
            T* umodrow = source._img + ((y - source._y0) * source._w) +
                (xlo - source._x0);

            if (!jacobians || !jacobians[i]) {
                // Model: add source*flux to mod
                for (int x=0; x<nx; x++, modrow++, umodrow++) {
                    (*modrow) += (*umodrow) * flux;
                }
            } else {
                // Jacobians: d(residual)/d(param)
                //    = d( (data - model) * ierr ) /d(param)
                //    = d( -model * ierr ) / d(param)
                //    = -ierr * d( flux * umod ) / d(param)
                //    = -ierr * umod * d(flux) / d(param)
                double* jrow = jacobians[i] + ((y -  _data._y0) *  _data._w) +
                    (xlo -  _data._x0);
                T*      erow =  _data._ierr + ((y -  _data._y0) *  _data._w) +
                    (xlo -  _data._x0);

                if (_nonneg) {
                    // flux = exp(param)
                    // d(flux)/d(param) = d(exp(param))/d(param)
                    //                  = exp(param)
                    //                  = flux
                    for (int x=0; x<nx; x++, modrow++, umodrow++, jrow++, erow++) {
                        double m = (*umodrow) * flux;
                        (*modrow) += m;
                        (*jrow) = -1.0 * m * (*erow);
                        //maxJ = MAX(maxJ, fabs(*jrow));
                    }
                } else {
                    for (int x=0; x<nx; x++, modrow++, umodrow++, jrow++, erow++) {
                        (*modrow) += (*umodrow) * flux;
                        (*jrow) = -1.0 * (*umodrow) * (*erow);
                        //maxJ = MAX(maxJ, fabs(*jrow));
                    }
                }
            }
        }
    }

    //if (jacobians)
    //printf("Max jacobian: %g\n", maxJ);

    T* dptr = _data._img;
    T* mptr = mod;
    T* eptr = _data._ierr;
    double* rptr = residuals;
    for (int i=0; i<_data.npix(); i++, dptr++, mptr++, eptr++, rptr++) {
        (*rptr) = ((*dptr) - (*mptr)) * (*eptr);
        //residuals[i] = (_data._img[i] - mod[i]) * _data._ierr[i];
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
    return true;
}



template class Patch<float>;
template class Patch<double>;
template class ForcedPhotCostFunction<float>;
template class ForcedPhotCostFunction<double>;




ImageCostFunction::ImageCostFunction(PyObject* tractor,
                                     int imagei, int nparams) :
    _tractor(tractor), _imagei(imagei), _image(NULL),
    _npix(0), _nparams(nparams) {

    //PyObject* tractorGetImage = PyObject_GetAttrString(tractor, "getImage");
    //assert(tractorGetImage);
    //assert(PyCallable_Check(tractorGetImage);

    PyObject* ret;

    _image = PyObject_CallMethod(_tractor, (char*)"getImage",
                                 (char*)"i", _imagei);
    //PyInt_FromLong(_imagei), NULL);

    ret = PyObject_CallMethod(_image, (char*)"numberOfPixels", NULL);
    _npix = PyInt_AsLong(ret);
    Py_DECREF(ret);
    //NULL);

    printf("Image %i: number of pixels %i\n", _imagei, _npix);

    set_num_residuals(_npix);
    std::vector<int16_t>* bs = mutable_parameter_block_sizes();
    bs->push_back(_nparams);
    /*
     for (int i=0; i<_nparams; i++) {
     bs->push_back(1);
     }
     */
}

ImageCostFunction::~ImageCostFunction() {
    Py_XDECREF(_image);
}

bool ImageCostFunction::Evaluate(double const* const* parameters,
                                 double* residuals,
                                 double** jacobians) const {

    const std::vector<int16_t> bs = parameter_block_sizes();
    printf("ImageCostFunction::Evaluate\n");
    printf("Parameter blocks:\n");
    for (size_t i=0; i<bs.size(); i++) {
        printf("  %i: [", (int)i);
        for (int j=0; j<bs[i]; j++) {
            printf(" %g,", parameters[i][j]);
        }
        printf(" ]\n");
    }

    npy_intp dims[1];
    dims[0] = _nparams;

    PyObject* np_params = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
                                                    (void*)(parameters[0]));
    PyObject* np_chi;
    printf("Calling getChiImage(%i)\n", _imagei);
    np_chi = PyObject_CallMethod(_tractor, (char*)"getChiImage", (char*)"i",
                                 _imagei);
    Py_DECREF(np_params);
    if (!np_chi) {
        printf("getChiImage() failed\n");
        return false;
    }
    if (PyArray_TYPE(np_chi) != NPY_DOUBLE) {
        printf("expected getChiImage() to return double\n");
        Py_DECREF(np_chi);
        return false;
    }
    printf("Got chi image of size: %i\n", (int)PyArray_Size(np_chi));

    double* chi = (double*)PyArray_DATA(np_chi);
    // FIXME -- ASSUME contiguous C-style...

    memcpy(residuals, chi, sizeof(double) * _npix);
    
    Py_DECREF(np_chi);

    return true;
}

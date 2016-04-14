#include <Python.h>

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#import_array
#define PY_ARRAY_UNIQUE_SYMBOL tractorceres_ARRAY_API
#define NO_IMPORT_ARRAY
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
    std::vector<int32_t>* bs = mutable_parameter_block_sizes();
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
    const std::vector<int32_t> bs = parameter_block_sizes();

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
                                     int imagei, int nparams,
                                     PyObject* np_params) :
    _tractor(tractor), _imagei(imagei), _image(NULL),
    _npix(0), _nparams(nparams), _W(0), _H(0), _np_params(np_params) {

    _image = PyObject_CallMethod(_tractor, (char*)"getImage",
                                 (char*)"i", _imagei);
    PyObject* ret;
    ret = PyObject_CallMethod(_image, (char*)"getWidth", NULL);
    _W = PyInt_AsLong(ret);
    Py_DECREF(ret);
    ret = PyObject_CallMethod(_image, (char*)"getHeight", NULL);
    _H = PyInt_AsLong(ret);
    Py_DECREF(ret);
    _npix = _W * _H;

    //printf("Image %i: %i x %i -> number of pixels %i\n", _imagei, _W, _H,_npix);

    set_num_residuals(_npix);
    std::vector<int32_t>* bs = mutable_parameter_block_sizes();
    for (int i=0; i<_nparams; i++) {
        bs->push_back(1);
    }
}

ImageCostFunction::~ImageCostFunction() {
    Py_XDECREF(_image);
}

bool ImageCostFunction::Evaluate(double const* const* parameters,
                                 double* residuals,
                                 double** jacobians) const {
    bool result = _Evaluate(parameters, residuals, jacobians);
    //printf("ImageCostFunction::Evaluate: returning %i\n", (int)result);
    return result;
}

bool ImageCostFunction::_Evaluate(double const* const* parameters,
                                  double* residuals,
                                  double** jacobians) const {
    const std::vector<int32_t> bs = parameter_block_sizes();

    //printf("ImageCostFunction::Evaluate\n");
    /*
     printf("Parameter blocks: (%i)\n", (int)(bs.size()));
     for (size_t i=0; i<bs.size(); i++) {
     printf("  %i: n=%i, [", (int)i, (int)bs[i]);
     for (int j=0; j<bs[i]; j++) {
     printf(" %g,", parameters[i][j]);
     }
     printf(" ]\n");
     }
     */
    // Copy from "parameters" into "_np_params"
    int e0 = 0;
    double* pdata = (double*)PyArray_DATA(_np_params);
    for (size_t i=0; i<parameter_block_sizes().size(); i++) {
        int n = parameter_block_sizes()[i];
        memcpy(pdata + e0, parameters[i], n * sizeof(double));
        e0 += n;
    }
    assert(e0 == _nparams);

    /*{
     double* pdata = (double*)PyArray_DATA(_np_params);
     printf("np_params = [ ");
     for (int i=0; i<_nparams; i++) {
     printf("%g ", pdata[i]);
     }
     printf("]\n");
     }
     */

    // Call _tractor.setParams(_np_params)
    PyObject* setparams = PyString_FromString((char*)"setParams");
    PyObject* ret = PyObject_CallMethodObjArgs(_tractor, setparams,
                                               _np_params, NULL);
    Py_DECREF(setparams);
    if (!ret) {
        printf("failed to setParams()\n");
        return false;
    }

    // Get residuals (chi image)
    PyObject* np_chi;
    //printf("Calling getChiImage(%i)\n", _imagei);
    np_chi = PyObject_CallMethod(_tractor, (char*)"getChiImage",
                                 (char*)"i", _imagei);
    if (!np_chi) {
        printf("getChiImage() failed\n");
        return false;
    }

    if (PyArray_TYPE(np_chi) == NPY_DOUBLE) {
        double* chi = (double*)PyArray_DATA(np_chi);
        // FIXME -- ASSUME contiguous C-style...
        memcpy(residuals, chi, sizeof(double) * _npix);
        Py_DECREF(np_chi);
    } else if (PyArray_TYPE(np_chi) == NPY_FLOAT) {
        float* chi = (float*)PyArray_DATA(np_chi);
        // FIXME -- ASSUME contiguous C-style...
        int i;
        for (i=0; i<_npix; i++)
            residuals[i] = chi[i];
        Py_DECREF(np_chi);
    } else {
        printf("expected getChiImage() to return double or float\n");
        Py_DECREF(np_chi);
        return false;
    }
    //printf("Got chi image of size: %i\n", (int)PyArray_Size(np_chi));


    // Get Jacobian (derivatives)
    if (!jacobians) {
        //printf("Jacobians not requested\n");
        return true;
    }

    // FIXME -- we don't include priors here!

    // NOTE -- _getOneImageDerivs() returns dCHI / dParam, not the usual
    // dModel / dParam!
    PyObject* allderivs = PyObject_CallMethod(
        _tractor, (char*)"_getOneImageDerivs", (char*)"i", _imagei);
    if (!allderivs) {
        printf("_getOneImageDerivs() returned NULL\n");
        return false;
    }
    if (!PyList_Check(allderivs)) {
        printf("Expecting allderivs to be a list\n");
        Py_XDECREF(allderivs);
        return false;
    }
    int n = (int)PyList_Size(allderivs);
    //printf("Got %i derivatives\n", n);
    for (int i=0; i<n; i++) {
        PyObject* deriv = PyList_GetItem(allderivs, i);
        if (!PyTuple_Check(deriv)) {
            printf("Expected allderivs element %i to be a tuple\n", i);
            Py_DECREF(allderivs);
            return false;
        }
        int j = PyInt_AsLong(PyTuple_GetItem(deriv, 0));

        if (!jacobians[j])
            continue;

        int x0 = PyInt_AsLong(PyTuple_GetItem(deriv, 1));
        int y0 = PyInt_AsLong(PyTuple_GetItem(deriv, 2));
        PyObject* np_deriv = PyTuple_GetItem(deriv, 3);
        if (!PyArray_Check(np_deriv)) {
            printf("Expected third element of allderivs element %i to be an array\n", i);
            Py_DECREF(allderivs);
            return false;
        }

        if (PyArray_NDIM(np_deriv) != 2) {
            printf("Expected 2-d derivative image\n");
            Py_DECREF(allderivs);
            return false;
        }
        int dH = PyArray_DIM(np_deriv, 0);
        int dW = PyArray_DIM(np_deriv, 1);
        double* deriv_data = (double*)PyArray_DATA(np_deriv);

        //printf("jacobian %i: x0,y0 %i,%i, W,H %i,%i\n", j, x0, y0, dW, dH);

        double* J = jacobians[j];

        // Pad with zeros
        if (y0)
            memset(J, 0, y0*_W*sizeof(double));
        if (y0 + dH < _H)
            memset(J + (y0+dH)*_W, 0, (_H-(y0+dH))*_W*sizeof(double));

        for (int k=0; k<dH; k++) {
            double* row0 = J + (y0+k)*_W;
            if (x0)
                memset(row0, 0, x0*sizeof(double));
            row0 += x0;
            // Copy
            //for (int m=0; m<dW; m++)
            //row0[m] = deriv_data[k*dW + m];
            memcpy(row0, deriv_data + k*dW, dW * sizeof(double));
            if (x0 + dW < _W)
                memset(row0 + dW, 0, (_W - (x0+dW)) * sizeof(double));
        }
    }
    Py_DECREF(allderivs);

    return true;
}

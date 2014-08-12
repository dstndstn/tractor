#include "ceres/ceres.h"

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

template <typename T>
class Patch {
 public:

 Patch(int x0, int y0, int w, int h, T* img,
       T* mod0=NULL, T* ierr=NULL) :
    _x0(x0), _y0(y0), _w(w), _h(h), _img(img), _ierr(ierr),
        _mod0(mod0) {}

    int npix() const {
        return _w * _h;
    }
    
    int _x0;
    int _y0;
    int _w;
    int _h;
    T* _img;
    T* _ierr;
    T* _mod0;
};

template <typename T>
class ForcedPhotCostFunction : public CostFunction {
 public:
    virtual ~ForcedPhotCostFunction();

    ForcedPhotCostFunction(Patch<T> data,
                           std::vector<Patch<T> > sources,
                           int nonneg
                           );

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
 protected:
    Patch<T> _data;
    std::vector<Patch<T> > _sources;

    int _nonneg;
};








class ImageCostFunction : public CostFunction {
 public:
    virtual ~ImageCostFunction();

    ImageCostFunction(PyObject* tractor, int imagei, int nparams, PyObject* np_params);

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    int nPix() const { return _npix; }

 protected:

    bool _Evaluate(double const* const* parameters,
                   double* residuals,
                   double** jacobians) const;


    PyObject* _tractor;
    int _imagei;
    PyObject* _image;
    int _npix;
    int _nparams;
    int _W;
    int _H;
    PyObject* _np_params;
};

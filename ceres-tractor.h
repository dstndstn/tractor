#include "ceres/ceres.h"

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

class Patch {
 public:

 Patch(int x0, int y0, int w, int h, double* img, double* ierr=NULL) :
    _x0(x0), _y0(y0), _w(w), _h(h), _img(img), _ierr(ierr) {}

    int npix() const {
        return _w * _h;
    }
    
    int _x0;
    int _y0;
    int _w;
    int _h;
    double* _img;
    double* _ierr;
};

class ForcedPhotCostFunction : public CostFunction {
 public:
    virtual ~ForcedPhotCostFunction();

    ForcedPhotCostFunction(Patch data,
                           std::vector<Patch> sources
                           );

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
 protected:
    Patch _data;
    std::vector<Patch> _sources;
};


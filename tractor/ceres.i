%module(package="tractor") ceres

%include <typemaps.i>

%{
#define PY_ARRAY_UNIQUE_SYMBOL tractorceres_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <Python.h>
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#define PyNumber_Int   PyNumber_Long
#endif

#include <math.h>
#include <assert.h>
#include <sys/param.h>
#include "glog/logging.h"
%}

%init %{
    // numpy
    import_array();

    google::InitGoogleLogging("ceres.i");
%}

%{
#include "ceres/ceres.h"
#include "ceres/version.h"
#include "ceres/normal_prior.h"
#include <vector>
#include <Eigen/Core>

#if 0
 } // fool emacs indenter
#endif

#include "ceres-tractor.h"

template <typename T>
static PyObject* real_ceres_forced_phot(PyObject* blocks,
                                        PyObject* np_fluxes,
                                        int npy_type,
                                        int nonneg,
                                        int verbose) {
    // Note, if you change this function signature, you also need
    // to change the template instantiations below!
    /*
     blocks: [ (data, sources), (data, sources), ... ]
       data: (x0, y0, np_img, np_mod0, np_inverr)
       sources: [ (index, x0, y0, np_img), ... ]
     */
    npy_intp Nblocks;
    assert(PyList_Check(blocks));
    Nblocks = PyList_Size(blocks);
    //printf("N blocks: %i\n", (int)Nblocks);
    assert(PyArray_Check(np_fluxes));
    assert(PyArray_TYPE((PyArrayObject*)np_fluxes) == NPY_DOUBLE);
    int Nfluxes = (int)PyArray_Size(np_fluxes);
    double* realfluxes = (double*)PyArray_DATA((PyArrayObject*)np_fluxes);
    T* mod0data;
    Problem problem;
    int totaldatapix = 0;
    int totalderivpix = 0;
    int totalsources = 0;

    if (nonneg) {
        // params = log(flux)
        for (int j=0; j<Nfluxes; j++) {
            realfluxes[j] = log(MAX(realfluxes[j], 1e-6));
        }
    }

    for (int i=0; i<Nblocks; i++) {
        PyObject* block;
        PyObject* srclist;
        PyObject* obj;
        int x0, y0;
        PyArrayObject *img, *mod0, *ierr;
        int w, h;
        int Nsources;

        // block = (data, [source, source, ...])
        block = PyList_GET_ITEM(blocks, i);
        assert(PyTuple_Check(block));
        assert(PyTuple_Size(block) == 2);

        // data = (x0, y0, image, mod0, inverror)
        obj = PyTuple_GET_ITEM(block, 0);
        assert(PyTuple_Check(obj));
        assert(PyTuple_Size(obj) == 5);
        x0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 0));
        y0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 1));
        img = (PyArrayObject*)PyTuple_GET_ITEM(obj, 2);
        assert(PyArray_Check(img));
        h = PyArray_DIM(img, 0);
        w = PyArray_DIM(img, 1);
        mod0 = (PyArrayObject*)PyTuple_GET_ITEM(obj, 3);
        if ((PyObject*)mod0 == Py_None) {
            mod0data = NULL;
        } else {
            assert(PyArray_Check(mod0));
            assert(PyArray_DIM(mod0, 0) == h);
            assert(PyArray_DIM(mod0, 1) == w);
            assert(PyArray_TYPE(mod0) == npy_type);
            mod0data = (T*)PyArray_DATA(mod0);
        }
        ierr = (PyArrayObject*)PyTuple_GET_ITEM(obj, 4);
        assert(PyArray_Check(ierr));
        assert(PyArray_DIM(ierr, 0) == h);
        assert(PyArray_DIM(ierr, 1) == w);
        assert(PyArray_TYPE(img) == npy_type);
        assert(PyArray_TYPE(ierr) == npy_type);
        Patch<T> data(x0, y0, w, h, (T*)PyArray_DATA(img), mod0data,
                   (T*)PyArray_DATA(ierr));

        std::vector<Patch<T> > srcs;
        std::vector<double*> fluxes;
        
        srclist = PyTuple_GET_ITEM(block, 1);
        assert(PyList_Check(srclist));
        Nsources = PyList_Size(srclist);
        int nderivpix = 0;
        for (int j=0; j<Nsources; j++) {
            int index, uh, uw;
            PyArrayObject* uimg;
            obj = PyList_GET_ITEM(srclist, j);
            assert(PyTuple_Check(obj));
            assert(PyTuple_Size(obj) == 4);
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 0)));
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 1)));
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 2)));
            index = PyInt_AsLong(PyTuple_GET_ITEM(obj, 0));
            assert(index >= 0);
            assert(index < Nfluxes);
            x0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 1));
            y0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 2));
            uimg = (PyArrayObject*)PyTuple_GET_ITEM(obj, 3);
            assert(PyArray_Check(uimg));
            uh = PyArray_DIM(uimg, 0);
            uw = PyArray_DIM(uimg, 1);
            nderivpix += (uw*uh);
            /*
             printf("param index %i: x0,y0 %i,%i, size %ix%i\n",
             index, x0, y0, w, h);
             */
            assert(PyArray_TYPE(uimg) == npy_type);
            srcs.push_back(Patch<T>(x0, y0, uw, uh, (T*)PyArray_DATA(uimg)));
            fluxes.push_back(realfluxes + index);
        }

        CostFunction* cost = new ForcedPhotCostFunction<T>(data, srcs, nonneg);
        problem.AddResidualBlock(cost, NULL, fluxes);
        //printf("added residual block with %i pixels and %i sources
        //(%i pix derivatives)\n", (w*h), Nsources, nderivpix);

        totaldatapix += (w*h);
        totalderivpix += nderivpix;
        totalsources += Nsources;
    }
    if (verbose)
        printf("Ceres: %i blocks, total %i pixels, %i sources-in-blocks, %i sources, %i deriv elements\n",
               (int)Nblocks, totaldatapix, totalsources, Nfluxes, totalderivpix);
    
    // Run the solver!
    Solver::Options options;
    if (verbose)
        options.minimizer_progress_to_stdout = true;
    //options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::SPARSE_SCHUR;

    options.jacobi_scaling = false;
    //options.jacobi_scaling = true;

    // .minimizer_type = TRUST_REGION / LINE_SEARCH
    // .linear_solver_type = SPARSE_NORMAL_CHOLESKY / DENSE_QR
    // / DENSE_SCHUR / SPARSE_SCHUR
    // .trust_region_strategy_type = LEVENBERG_MARQUARDT / DOGLEG
    // .dogleg_type = TRADITIONAL_DOGLEG / SUBSPACE_DOGLEG 

    // linear subspaces
    // .use_inner_iterations = true;
    // .minimizer_type = TRUST_REGION / LINE_SEARCH
    // .line_search_direction_type = LBFGS / STEEPEST_DESCENT / NONLINEAR_CONJUGATE_GRADIENT / BFGS
    // .line_search_type = WOLFE / ARMIJO
    // .nonlinear_conjugate_gradient_type = FLETCHER_REEVES / POLAK_RIBIRERE / HESTENES_STIEFEL
    // .max_lbfs_rank = 20
    // .use_approximate_eigenvalue_bfgs_scaling
    // .line_search_interpolation_type = CUBIC / ...
    // .min_line_search_step_size
    // .line_search_sufficient_function_decrease
    // .max_line_search_step_contraction
    // .min_line_search_step_contraction
    // .max_num_line_search_step_size_iterations
    // .max_num_line_search_direction_restarts
    // .line_search_sufficient_curvature_decrease
    // .max_line_search_step_expansion
    // .use_nonmonotonic_steps
    // .max_consecutive_nonmonotonic_steps
    // .max_num_iterations
    // .max_solver_time_in_seconds
    // .num_threads
    // .initial_trust_region_radius
    // .max_trust_region_radius
    // .min_trust_region_radius
    // .min_relative_decrease
    // .min_lm_diagonal
    // .max_lm_diagonal
    // .max_num_consecutive_invalid_steps
    // .function_tolerance = 1e-6
    // .gradient_tolerance
    // .parameter_tolerance = 1e-8
    // .preconditioner_type
    // .dense_linear_algebra_library_type
    // .sparse_linear_algebra_library_type
    // .num_linear_solver_threads
    // .linear_solver_ordering
    // .use_post_ordering
    // .min_linear_solver_iterations
    // .max_linear_solver_iterations
    // .eta
    //
    // Jacobian is scaled by the norm of its columns before being passed to the linear solver. This improves the numerical conditioning of the normal equations.
    // .jacobi_scaling = true
    //
    // .inner_itearation_tolerance
    // .inner_iteration_ordering
    // .logging_type
    // .minimizer_progress_to_stdout
    // .numeric_derivative_relative_step_size
    
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    if (verbose)
        printf("%s\n", summary.BriefReport().c_str());
    //std::cout << summary.FullReport() << "\n";

    if (nonneg) {
        for (int j=0; j<Nfluxes; j++) {
            realfluxes[j] = exp(realfluxes[j]);
        }
    }


    // CERES 1.9.0
    const char* errstring = summary.message.c_str();
    // CERES 1.8.0
    //const char* errstring = summary.error.c_str();

    return Py_BuildValue("{sisssdsdsdsssssisisi}",
                         "termination", int(summary.termination_type),
                         "error", errstring,
                         "initial_cost", summary.initial_cost,
                         "final_cost", summary.final_cost,
                         "fixed_cost", summary.fixed_cost,
                         "brief_report", summary.BriefReport().c_str(),
                         "full_report", summary.FullReport().c_str(),
                         "steps_successful", summary.num_successful_steps,
                         "steps_unsuccessful", summary.num_unsuccessful_steps,
                         "steps_inner", summary.num_inner_iteration_steps);
                         
}

template PyObject* real_ceres_forced_phot<float>(PyObject*, PyObject*, int, int, int);
template PyObject* real_ceres_forced_phot<double>(PyObject*, PyObject*, int, int, int);


%}

%inline %{

#if 0
 } // fool emacs indenter
#endif
    

static PyObject* ceres_forced_phot(PyObject* blocks,
                                   PyObject* np_fluxes,
                                   int nonneg,
                                   int verbose) {
	assert(PyList_Check(blocks));
    assert(PyList_Size(blocks) > 0);
    PyObject* block;
    block = PyList_GET_ITEM(blocks, 0);
    assert(PyTuple_Check(block));
    assert(PyTuple_Size(block) == 2);
    PyObject* obj;
    // data
    obj = PyTuple_GET_ITEM(block, 0);
    assert(PyTuple_Check(obj));
    assert(PyTuple_Size(obj) == 5);
    PyArrayObject* img;
    img = (PyArrayObject*)PyTuple_GET_ITEM(obj, 2);
    assert(PyArray_Check(img));

    if (PyArray_TYPE(img) == NPY_FLOAT) {
        if (verbose)
            printf("Calling float version\n");
        return real_ceres_forced_phot<float>(blocks, np_fluxes, NPY_FLOAT,
                                             nonneg, verbose);
    } else if (PyArray_TYPE(img) == NPY_DOUBLE) {
        if (verbose)
            printf("Calling double version\n");
        return real_ceres_forced_phot<double>(blocks, np_fluxes, NPY_DOUBLE,
                                              nonneg, verbose);
    }
    printf("Unknown PyArray type %i\n", PyArray_TYPE(img));

    Py_RETURN_NONE;
}


// Generic optimization


class NumericDiffImageCost {
public:
    NumericDiffImageCost(ImageCostFunction* im) : _im(im) {}
    
    ~NumericDiffImageCost() {
        delete _im;
    }

    bool operator()(double const* const* parameters, double* residuals) const {
        return _im->Evaluate(parameters, residuals, NULL);
    }

protected:
    ImageCostFunction* _im;
};

class DlnpCallback : public ceres::IterationCallback {
public:
    DlnpCallback(double dlnp) : _dlnp(dlnp) {}
    virtual ~DlnpCallback() {}

    virtual ceres::CallbackReturnType operator()
    (const ceres::IterationSummary& summary) {
        //printf("Cost change: %g\n", summary.cost_change);
        printf("Callback: step size %g, line search evals %i,%i,%i, linear solver iters %i\n",
               summary.step_size, summary.line_search_function_evaluations,
               summary.line_search_gradient_evaluations,
               summary.line_search_iterations,
               summary.linear_solver_iterations);
        if (summary.cost_change > 0 && summary.cost_change < _dlnp) {
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        }
        return ceres::SOLVER_CONTINUE;
    }
protected:
    const double _dlnp;
};


static PyObject* ceres_opt(PyObject* tractor,
                           int nims,
                           PyObject* py_params,
                           PyObject* py_variance,
                           int scale_columns,
                           int numeric,
                           float numeric_stepsize,
                           float dlnp,
                           int max_iterations,
                           PyObject* gaussian_priors,
                           PyObject* lubounds,
                           int print_progress) {
    /*
     np_params: numpy array, type double, length number of params.
     np_variance: ditto

     Methods called on the "tractor" object include:

     img = tractor.getImage(i)
        img.getWidth()
        img.getHeight()
     tractor.setParams(np_params)   # with the np_params obj passed in
     tractor.getChiImage(i)
     tractor._getOneImageDerivs(i)
     
     if priors:
     tractor.getGaussianPriors()

     */
    Problem problem;
    int i;
    double* params;
    int nparams;
    int get_variance;
    int variance_ok = 0;
    DlnpCallback cb(dlnp);
    PyArrayObject *np_params, *np_variance=NULL;

    assert(PyArray_Check(py_params));
    np_params = (PyArrayObject*)py_params;
    assert(PyArray_TYPE(np_params) == NPY_DOUBLE);
    if (!(PyArray_Check(np_params) &&
          (PyArray_TYPE(np_params) == NPY_DOUBLE))) {
        printf("ceres_opt: wrong type for params variable\n");
        return NULL;
    }
    nparams = (int)PyArray_SIZE(np_params);
    params = (double*)PyArray_DATA(np_params);

    get_variance = (py_variance != Py_None);

    //printf("ceres_opt, nims %i, nparams %i, get_variance %i\n",
    //       nims, nparams, get_variance);

    std::vector<double*> allparams;
    // Single-param blocks
    for (i=0; i<nparams; i++)
        allparams.push_back(params + i);

    for (i=0; i<nims; i++) {
        ImageCostFunction* icf = new ImageCostFunction
            (tractor, i, nparams, np_params);
        CostFunction* cost = NULL;
        if (numeric) {
#if CERES_VERSION_MINOR >= 12
            ceres::NumericDiffOptions numopts;
            numopts.relative_step_size = numeric_stepsize;
            
            ceres::DynamicNumericDiffCostFunction<NumericDiffImageCost>* dyncost = 
                new ceres::DynamicNumericDiffCostFunction<NumericDiffImageCost>
                (new NumericDiffImageCost(icf), ceres::TAKE_OWNERSHIP,
                 numopts);
#else
            ceres::DynamicNumericDiffCostFunction<NumericDiffImageCost>* dyncost = 
                new ceres::DynamicNumericDiffCostFunction<NumericDiffImageCost>
                (new NumericDiffImageCost(icf), ceres::TAKE_OWNERSHIP,
                 numeric_stepsize);
#endif
            for (i=0; i<nparams; i++)
                dyncost->AddParameterBlock(1);
            dyncost->SetNumResiduals(icf->nPix());
            cost = dyncost;

        } else {
            cost = icf;
        }
        problem.AddResidualBlock(cost, NULL, allparams);
        //problem.AddResidualBlock(cost, NULL, params);
    }

    if (gaussian_priors != Py_None) {
        if (!PyList_Check(gaussian_priors)) {
            printf("Expected gaussian_priors to be a list\n");
            return NULL;
        }
        size_t nterms = PyList_Size(gaussian_priors);
        for (size_t i=0; i<nterms; i++) {
            PyObject* tup = PyList_GetItem(gaussian_priors, i);
            if (!PySequence_Check(tup)) {
                printf("Expected gaussian_priors to contain iterables; element %i is not\n", (int)i);
                return NULL;
            }
            if (PySequence_Size(tup) != 3) {
                printf("Expected gaussian_priors to contain length-3 iterables; element %i is not\n", (int)i);
                return NULL;
            }
            PyObject* pyi = PySequence_GetItem(tup, 0);
            PyObject* pym = PySequence_GetItem(tup, 1);
            PyObject* pys = PySequence_GetItem(tup, 2);

            PyObject* ii = PyNumber_Int(pyi);
            if (!ii) {
                printf("Expected gaussian_priors element %i, index 0, to be an integer\n", (int)i);
                return NULL;
            }
            int index = PyInt_AsLong(ii);

            PyObject* fm = PyNumber_Float(pym);
            PyObject* fs = PyNumber_Float(pys);
            if (!fm) {
                printf("Expected gaussian_priors element %i, index 1, to be a float\n", (int)i);
                return NULL;
            }
            if (!fs) {
                printf("Expected gaussian_priors element %i, index 2, to be a float\n", (int)i);
            }
            double mean  = PyFloat_AsDouble(fm);
            double sigma = PyFloat_AsDouble(fs);

            Py_DECREF(ii);
            Py_DECREF(fm);
            Py_DECREF(fs);
            Py_DECREF(pyi);
            Py_DECREF(pym);
            Py_DECREF(pys);

            Eigen::MatrixXd A(1,1);
            A(0,0) = 1. / sigma;
            Eigen::VectorXd mu(1);
            mu[0] = mean;
            //printf("Adding Gaussian prior on parameter %i (current value "
            //"%f): mean %f, sigma %f\n", index, params[index], mean, sigma);
            //printf("A size: %i, %i.  Mu size: %i\n", 
            //A.rows(), A.cols(), mu.size());
            CostFunction* prior = new ceres::NormalPrior(A, mu);
            problem.AddResidualBlock(prior, NULL, params + index);
        }
    }

    if (lubounds != Py_None) {
        if (!PyList_Check(lubounds)) {
            printf("Expected lubounds to be a list\n");
            return NULL;
        }
        size_t nterms = PyList_Size(lubounds);
        for (size_t i=0; i<nterms; i++) {
            PyObject* tup = PyList_GetItem(lubounds, i);
            if (!PySequence_Check(tup)) {
                printf("Expected lubounds to contain iterables; element %i is not\n", (int)i);
                return NULL;
            }
            if (PySequence_Size(tup) != 3) {
                printf("Expected lubounds to contain length-3 iterables; element %i is not\n", (int)i);
                return NULL;
            }
            // (int index, float bound, bool lower)
            PyObject* pyi = PySequence_GetItem(tup, 0);
            PyObject* pyb = PySequence_GetItem(tup, 1);
            PyObject* pyl = PySequence_GetItem(tup, 2);

            if (!PyInt_Check(pyi)) {
                printf("Expected lubounds element %i, index 0, to be an integer\n", (int)i);
                return NULL;
            }
            int index = PyInt_AsLong(pyi);
            Py_DECREF(pyi);

            if (!PyFloat_Check(pyb)) {
                printf("Expected lubounds element %i, index 1, to be a float\n", (int)i);
                return NULL;
            }
            double bound = PyFloat_AsDouble(pyb);
            Py_DECREF(pyb);
            
            if (!PyBool_Check(pyl)) {
                printf("Expected lubounds element %i, index 2, to be a bool\n", (int)i);
                return NULL;
            }
            int islower = (pyl == Py_True);
            Py_DECREF(pyl);

            printf("Bound on element %i, bound %g, %s bound\n",
                   index, bound, (islower ? "lower" : "upper"));

            if (islower)
                problem.SetParameterLowerBound(params + index, 0, bound);
            else
                problem.SetParameterUpperBound(params + index, 0, bound);

        }
    }

    // Run the solver!
    Solver::Options options;
    options.minimizer_progress_to_stdout = print_progress;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.jacobi_scaling = scale_columns;
    // not in ceres-solver 1.12.0?
    //options.numeric_derivative_relative_step_size = numeric_stepsize;
    if (max_iterations) {
        options.max_num_iterations = max_iterations;
    }
    if (dlnp > 0) {
        options.function_tolerance = 1e-16;
        options.callbacks.push_back(&cb);
        //printf("Added Dlnp callback: now %i callbacks\n",(int)options.callbacks.size());
    }

    Solver::Summary summary;
    Solve(options, &problem, &summary);
    //printf("%s\n", summary.BriefReport().c_str());

    printf("%s\n", summary.FullReport().c_str());

    if (get_variance && (summary.termination_type == ceres::CONVERGENCE)) {
        if (!PyArray_Check(py_variance)) {
            printf("ceres_opt: variance must be a numpy array\n");
            return NULL;
        }
        np_variance = (PyArrayObject*)py_variance;
        if (PyArray_TYPE(np_variance) != NPY_DOUBLE) {
            printf("ceres_opt: wrong type for variance variable\n");
            return NULL;
        }
        if (PyArray_SIZE(np_variance) != PyArray_SIZE(np_params)) {
            printf("ceres_opt: wrong size for variance variable\n");
            return NULL;
        }
        double* cov_out = (double*)PyArray_DATA(np_variance);
        for (i=0; i<nparams; i++)
            cov_out[i] = 0.0;

        ceres::Covariance::Options options;

        options.algorithm_type = ceres::DENSE_SVD;
        options.null_space_rank = -1;
        //options.algorithm_type = SPARSE_QR;
        //options.algorithm_type = SPARSE_CHOLESKY;

        ceres::Covariance covariance(options);

        std::vector<std::pair<const double*, const double*> > covar_blocks;
        for (i=0; i<nparams; i++)
            covar_blocks.push_back(std::make_pair(params+i, params+i));
        if (!covariance.Compute(covar_blocks, &problem)) {
            printf("ceres_opt: failed to compute variance\n");
            // ?
            return NULL;
        } else {
            variance_ok = 1;
        }
        for (i=0; i<nparams; i++)
            covariance.GetCovarianceBlock(params+i, params+i, cov_out+i);
    }

    const char* errstring = summary.message.c_str();

    return Py_BuildValue
        ("{sisssdsdsdsssssisisisi}",
         "termination", int(summary.termination_type),
         "error", errstring,
         "initial_cost", summary.initial_cost,
         "final_cost", summary.final_cost,
         "fixed_cost", summary.fixed_cost,
         "brief_report", summary.BriefReport().c_str(),
         "full_report", summary.FullReport().c_str(),
         "steps_successful", summary.num_successful_steps,
         "steps_unsuccessful", summary.num_unsuccessful_steps,
         "steps_inner", summary.num_inner_iteration_steps,
         "variance_ok", variance_ok);
}










%}



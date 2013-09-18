%module(package="tractor") ceres

%include <typemaps.i>

%{
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
#include "glog/logging.h"
%}

%init %{
    // numpy
    import_array();

    google::InitGoogleLogging("ceres.i");
%}

%{
#include "ceres/ceres.h"
#include <vector>

#if 0
 } // fool emacs indenter
#endif

#include "ceres-tractor.h"

template <typename T>
static int real_ceres_forced_phot(PyObject* blocks,
                                  PyObject* np_fluxes,
                                  int npy_type) {
    /*
     blocks: [ (data, sources), (data, sources), ... ]
       data: (x0, y0, np_img, np_mod0, np_inverr)
       sources: [ (index, x0, y0, np_img), ... ]
     */
    npy_intp Nblocks;

	assert(PyList_Check(blocks));
    Nblocks = PyList_Size(blocks);
    printf("N blocks: %i\n", (int)Nblocks);

    assert(PyArray_Check(np_fluxes));
    assert(PyArray_TYPE(np_fluxes) == NPY_DOUBLE);
    //assert(PyArray_ITEMSIZE(np_fluxes) == sizeof(double));
    double* realfluxes = (double*)PyArray_DATA(np_fluxes);
    T* mod0data;
    Problem problem;
    int totaldatapix = 0;
    int totalderivpix = 0;
    int totalsources = 0;
    //int maxsource = 0;

    for (int i=0; i<Nblocks; i++) {
        PyObject* block;
        PyObject* srclist;
        PyObject* obj;
        int x0, y0;
        PyObject *img, *mod0, *ierr;
        int w, h;
        int Nsources;

        block = PyList_GET_ITEM(blocks, i);
        assert(PyTuple_Check(block));
        assert(PyTuple_Size(block) == 2);

        // data
        obj = PyTuple_GET_ITEM(block, 0);
        assert(PyTuple_Check(obj));
        assert(PyTuple_Size(obj) == 5);
        x0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 0));
        y0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 1));
        img   = PyTuple_GET_ITEM(obj, 2);
        assert(PyArray_Check(img));
        h = PyArray_DIM(img, 0);
        w = PyArray_DIM(img, 1);
        mod0  = PyTuple_GET_ITEM(obj, 3);
        if (mod0 == Py_None) {
            mod0data = NULL;
        } else {
            assert(PyArray_Check(mod0));
            assert(PyArray_DIM(mod0, 0) == h);
            assert(PyArray_DIM(mod0, 1) == w);
            assert(PyArray_TYPE(mod0) == npy_type);
            mod0data = (T*)PyArray_DATA(mod0);
        }
        ierr  = PyTuple_GET_ITEM(obj, 4);
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
            PyObject* uimg;
            obj = PyList_GET_ITEM(srclist, j);
            assert(PyTuple_Check(obj));
            assert(PyTuple_Size(obj) == 4);
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 0)));
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 1)));
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 2)));
            index = PyInt_AsLong(PyTuple_GET_ITEM(obj, 0));
            //maxsource = MAX(maxsource, index);
            x0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 1));
            y0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 2));
            uimg = PyTuple_GET_ITEM(obj, 3);
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
        CostFunction* cost = new ForcedPhotCostFunction<T>(data, srcs);
        problem.AddResidualBlock(cost, NULL, fluxes);
        printf("added residual block with %i pixels and %i sources (%i pix derivatives)\n", (w*h), Nsources, nderivpix);

        totaldatapix += (w*h);
        totalderivpix += nderivpix;
        totalsources += Nsources;
    }
    int nfluxes = (int)PyArray_SIZE(np_fluxes);
    printf("Ceres: %i blocks, total %i pixels, %i sources-in-blocks, %i sources, %i deriv elements\n",
           (int)Nblocks, totaldatapix, totalsources, nfluxes, totalderivpix);
    
    // Run the solver!
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    //options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.jacobi_scaling = false;

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

    printf("%s\n", summary.BriefReport().c_str());
    //std::cout << summary.BriefReport() << "\n";
    //std::cout << summary.FullReport() << "\n";

    return 0;
}

template int real_ceres_forced_phot<float>(PyObject*, PyObject*, int);
template int real_ceres_forced_phot<double>(PyObject*, PyObject*, int);


%}

%inline %{

#if 0
 } // fool emacs indenter
#endif
    

static int ceres_forced_phot(PyObject* blocks,
                             PyObject* np_fluxes) {
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
    PyObject* img;
    img   = PyTuple_GET_ITEM(obj, 2);
    assert(PyArray_Check(img));

    if (PyArray_TYPE(img) == NPY_FLOAT) {
        printf("Calling float version\n");
        return real_ceres_forced_phot<float>(blocks, np_fluxes, NPY_FLOAT);
    } else if (PyArray_TYPE(img) == NPY_DOUBLE) {
        printf("Calling double version\n");
        return real_ceres_forced_phot<double>(blocks, np_fluxes, NPY_DOUBLE);
    }
    printf("Unknown PyArray type %i\n", PyArray_TYPE(img));
    return -1;
}


%}



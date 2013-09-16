%module(package="tractor") ceres

%include <typemaps.i>

%{
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
%}

%init %{
    // numpy
    import_array();
%}

%{
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <vector>
    %}

%inline %{
#if 0
 } // fool emacs indenter
#endif

#include "ceres-tractor.h"

static int ceres_forced_phot(PyObject* blocks,
                             PyObject* np_fluxes) {
    /*
     blocks: [ (data, sources), (data, sources), ... ]
       data: (x0, y0, np_img, np_inverr)
       sources: [ (index, x0, y0, np_img), ... ]
     */

    /// FIXME -- Need to be able to specify that a source touches multiple
    /// blocks!!!

    npy_intp Nblocks;

	assert(PyList_Check(blocks));
    Nblocks = PyList_Size(blocks);
    printf("N blocks: %i\n", (int)Nblocks);

    google::InitGoogleLogging("ceres.i");

    assert(PyArray_Check(np_fluxes));
    double* realfluxes = (double*)PyArray_DATA(np_fluxes);

    Problem problem;
    
    for (int i=0; i<Nblocks; i++) {
        PyObject* block;
        PyObject* srclist;
        PyObject* obj;
        int x0, y0;
        PyObject *img, *ierr;
        int w, h;
        int Nsources;

        block = PyList_GET_ITEM(blocks, i);
        assert(PyTuple_Check(block));
        assert(PyTuple_Size(block) == 2);

        obj = PyTuple_GET_ITEM(block, 0);
        assert(PyTuple_Check(obj));
        assert(PyTuple_Size(obj) == 4);
        x0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 0));
        y0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 1));
        img = PyTuple_GET_ITEM(obj, 2);
        ierr  = PyTuple_GET_ITEM(obj, 3);
        assert(PyArray_Check(img));
        assert(PyArray_Check(ierr));
        h = PyArray_DIM(img, 0);
        w = PyArray_DIM(img, 1);
        Patch data(x0, y0, w, h, (double*)PyArray_DATA(img), (double*)PyArray_DATA(ierr));

        std::vector<Patch> srcs;
        std::vector<double*> fluxes;
        
        srclist = PyTuple_GET_ITEM(block, 1);
        assert(PyList_Check(srclist));
        Nsources = PyList_Size(srclist);
        for (int j=0; j<Nsources; j++) {
            int index;
            obj = PyList_GET_ITEM(srclist, j);
            assert(PyTuple_Check(obj));
            assert(PyTuple_Size(obj) == 4);
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 0)));
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 1)));
            assert(PyInt_Check(PyTuple_GET_ITEM(obj, 2)));
            index = PyInt_AsLong(PyTuple_GET_ITEM(obj, 0));
            x0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 1));
            y0 = PyInt_AsLong(PyTuple_GET_ITEM(obj, 2));
            //assert(x0 >= 0);
            //assert(y0 >= 0);
            img = PyTuple_GET_ITEM(obj, 3);
            assert(PyArray_Check(img));
            h = PyArray_DIM(img, 0);
            w = PyArray_DIM(img, 1);
            printf("param index %i: x0,y0 %i,%i, size %ix%i\n",
                   index, x0, y0, w, h);
            srcs.push_back(Patch(x0, y0, w, h, (double*)PyArray_DATA(img)));

            fluxes.push_back(realfluxes + index);
        }
        CostFunction* cost = new ForcedPhotCostFunction(data, srcs);
        problem.AddResidualBlock(cost, NULL, fluxes);
    }
    
    // Run the solver!
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.jacobi_scaling = false;

    // .linear_solver_type = SPARSE_NORMAL_CHOLESKY / DENSE_QR
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



%}


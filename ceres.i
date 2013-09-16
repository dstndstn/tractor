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
       data: (x0, y0, np_img, np_invvar)
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
        PyObject *img, *iv;
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
        iv  = PyTuple_GET_ITEM(obj, 3);
        assert(PyArray_Check(img));
        assert(PyArray_Check(iv));
        h = PyArray_DIM(img, 0);
        w = PyArray_DIM(img, 1);
        Patch data(x0, y0, w, h, (double*)PyArray_DATA(img), (double*)PyArray_DATA(iv));

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
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    return 0;
}



%}


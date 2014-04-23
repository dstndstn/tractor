%module denorm

%{
#include <math.h>
#include <numpy/arrayobject.h>
%}

%init %{
	// numpy
	import_array();
	%}

%inline %{

int denorm_to_zero(PyObject* np_arrobj) {
	// Sample code yoinked from
	// http://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#simple-iteration-example

	NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp *strideptr, *innersizeptr;

	int denorm_count = 0;
	PyArrayObject* np_arr;

	int isfloat;
	//PyArrayDescr* descr;

	if (!PyArray_Check(np_arrobj)) {
		printf("denorm: Object not an np array\n");
		return -1;
	}
	np_arr = (PyArrayObject*)np_arrobj;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(np_arr) == 0) {
        return 0;
    }

	// warning: eye-bleeding code ahead.

	if (PyArray_DESCR(np_arr) == PyArray_DescrFromType(NPY_FLOAT)) {
		isfloat = 1;
	} else if (PyArray_DESCR(np_arr) == PyArray_DescrFromType(NPY_DOUBLE)) {
		isfloat = 0;
	} else {
		printf("Unknown type\n");
		return -1;
	}

	iter = NpyIter_New(np_arr, 
					   NPY_ITER_READWRITE |
					   NPY_ITER_EXTERNAL_LOOP | 
					   NPY_ITER_NBO |
					   NPY_ITER_ALIGNED,
					   NPY_KEEPORDER, NPY_NO_CASTING,
					   NULL);
					   //PyArray_DESCR(np_arr));
					   //PyArray_DescrFromType(NPY_FLOAT));
	if (iter == NULL) {
		printf("denorm: failed to get iter\n");
		return -1;
	}
	iternext = NpyIter_GetIterNext(iter, NULL);
	if (iternext == NULL) {
		NpyIter_Deallocate(iter);
		printf("denorm: failed to getIterNext\n");
		return -1;
	}
	/* The location of the data pointer which the iterator may update */
	dataptr = NpyIter_GetDataPtrArray(iter);
	/* The location of the stride which the iterator may update */
	strideptr = NpyIter_GetInnerStrideArray(iter);
	/* The location of the inner loop size which the iterator may update */
	innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
	/* The iteration loop */
	do {
		/* Get the inner loop data/stride/count values */
		char* data = *dataptr;
		npy_intp stride = *strideptr;
		npy_intp count = *innersizeptr;

		if (isfloat) {
			while (count--) {
				float* f = (float*)data;
				//if (!isnormal(*f)) {
				// zero isn't normal!
				if (fpclassify(*f) == FP_SUBNORMAL) {
					printf("Found %g not normal\n", *f);
					denorm_count++;
					*f = 0.0;
				}
				data += stride;
			}
		} else {
			while (count--) {
				double* f = (double*)data;
				if (fpclassify(*f) == FP_SUBNORMAL) {
					printf("Found %g not normal\n", *f);
					denorm_count++;
					*f = 0.0;
				}
				data += stride;
			}
		}
		/* Increment the iterator to the next inner loop */
	} while(iternext(iter));
	NpyIter_Deallocate(iter);
	return denorm_count;
}

 %}

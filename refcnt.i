%module refcnt

%{
#include "python2.7/object.h"
	%}

%inline %{
	int refcnt(PyObject* obj) {
		if (!obj) {
			printf("refcnt: NULL object\n");
			return -1;
		}
		return Py_REFCNT(obj);
	}
	%}


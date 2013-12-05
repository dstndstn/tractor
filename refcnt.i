%module refcnt

%{
#include "object.h"
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


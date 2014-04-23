%module callgrind
%{
#include "callgrind.h"
	%}

%inline %{

void callgrind_start_instrumentation() {
	CALLGRIND_START_INSTRUMENTATION;
}

void callgrind_stop_instrumentation() {
	CALLGRIND_STOP_INSTRUMENTATION;
}

 %}


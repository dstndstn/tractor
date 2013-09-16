#include <stdio.h>
#include <stdint.h>

#include <vector>

#include "ceres-tractor.h"

ForcedPhotCostFunction::ForcedPhotCostFunction(Patch data,
                                               std::vector<Patch> sources) :
    _data(data), _sources(sources) {

    set_num_residuals(data.npix());
    std::vector<int16_t>* bs = mutable_parameter_block_sizes();
    for (size_t i=0; i<sources.size(); i++) {
        bs->push_back(1);
    }
    printf("ForcedPhotCostFunction: npix %i, nsources %i\n",
           num_residuals(), (int)parameter_block_sizes().size());
}

ForcedPhotCostFunction::~ForcedPhotCostFunction() {}

bool ForcedPhotCostFunction::Evaluate(double const* const* parameters,
                                      double* residuals,
                                      double** jacobians) const {
    printf("ForcedPhotCostFunction::Evaluate\n");
    const std::vector<int16_t> bs = parameter_block_sizes();
    printf("Parameter blocks:\n");
    for (size_t i=0; i<bs.size(); i++) {
        printf("  %i: [", (int)i);
        for (int j=0; j<bs[i]; j++) {
            printf(" %g,", parameters[i][j]);
        }
        printf(" ]\n");
    }
    return false;
}


'''
This file is part of the Tractor project.
Copyright 2014, Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`sersic.py`
===========

General Sersic galaxy model.
'''
from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d

from tractor import mixture_profiles as mp
from tractor.engine import *
from tractor.utils import *
from tractor.cache import *
from tractor.galaxy import *

class SersicMixture(object):
    singleton = None

    @staticmethod
    def getProfile(sindex):
        if SersicMixture.singleton is None:
            SersicMixture.singleton = SersicMixture()
        return SersicMixture.singleton._getProfile(sindex)

    def __init__(self):
        # GalSim: supports n=0.3 to 6.2.

        # A set of ranges [ser_lo, ser_hi], plus a list of fit parameters
        # that have a constant number of components.
        fits = [
        # 3 components
        (0.29, 0.42, [
            # amp prior std=2
            # (with amp prior std=1, the two positive components become degenerate)
            (0.29 , [ 29.7333, 10.0627, -32.1111,  ], [ 0.400918, 0.223734, 0.29063,  ]),
            (0.30 , [ 28.1273, 8.97218, -29.3501,  ], [ 0.410641, 0.227158, 0.29584,  ]),
            (0.31 , [ 26.5807, 7.94616, -26.7126,  ], [ 0.420884, 0.230796, 0.301292,  ]),
            (0.32 , [ 25.0955, 6.98467, -24.2006,  ], [ 0.431669, 0.234678, 0.307003,  ]),
            (0.34 , [ 22.3149, 5.25392, -19.5576,  ], [ 0.454958, 0.243336, 0.319275,  ]),
            (0.36 , [ 19.7907, 3.76919, -15.4161,  ], [ 0.480685, 0.253583, 0.332854,  ]),
            (0.38 , [ 17.5149, 2.50552, -11.7434,  ], [ 0.509045, 0.266156, 0.348058,  ]),
            # Now we start kicking in a prior on the amplitude to force the second component to zero.
            (0.39 , [ 16.5252, 1.55052, -9.73179,  ], [ 0.524296, 0.26729, 0.359764,  ]),
            (0.40 , [ 15.5942, 0.836228, -8.01953,  ], [ 0.540194, 0.267179, 0.372381,  ]),
            (0.41 , [ 14.7227, 0.344169, -6.58915,  ], [ 0.556667, 0.265585, 0.385658,  ]),
            # from the 2-component fit
            (0.42 , [ 14.0586, 0.0, -5.51399,  ], [ 0.572984, 0.265585, 0.403086,  ]),
            ]),
        # 0.42 to 0.6: 2 components
        #(0.42, 0.6, [
        (0.42, 0.51, [
            #(0.40 , [ 17.438, -9.0197,  ], [ 0.538194, 0.410385,  ]),
            (0.41 , [ 15.7428, -7.26198,  ], [ 0.55406, 0.408598,  ]),
            (0.42 , [ 14.0586, -5.51399,  ], [ 0.572984, 0.403086,  ]),
            (0.43 , [ 12.5707, -3.96071,  ], [ 0.594559, 0.393181,  ]),
            (0.44 , [ 11.4452, -2.76895,  ], [ 0.616783, 0.380574,  ]),
            (0.45 , [ 10.6628, -1.9203,  ], [ 0.637898, 0.367583,  ]),
            (0.46 , [ 10.1197, -1.31122,  ], [ 0.657378, 0.355276,  ]),
            (0.47 , [ 9.7311, -0.857421,  ], [ 0.675295, 0.343829,  ]),
            (0.48 , [ 9.44434, -0.506166,  ], [ 0.691845, 0.333043,  ]),
            (0.49 , [ 9.22735, -0.225454,  ], [ 0.70721, 0.322121,  ]),            
            (0.50 , [ 9.06467, 0.,], [ 0.721342, 0.316688, ]),
            (0.51 , [ 8.92798, 0.198966,  ], [ 0.735017, 0.311146,  ]),
            (0.52 , [ 8.82471, 0.363539,  ], [ 0.747636, 0.301751,  ]),
            (0.54 , [ 8.67782, 0.630599,  ], [ 0.770737, 0.286705,  ]),
            (0.56 , [ 8.58632, 0.839043,  ], [ 0.791347, 0.273568,  ]),
            (0.58 , [ 8.5322, 1.00693,  ], [ 0.80978, 0.261654,  ]),
            (0.60 , [ 8.50421, 1.14553,  ], [ 0.826286, 0.250709,  ]),
            ]),

        (0.51, 0.57, [
        # third variance component from fit for 0.52 with prior of 0.51
        (0.51 , [ 8.92797, 0.198981, 0. ], [ 0.735018, 0.311154, 0.0214237  ]),
        (0.515 , [ 8.73228, 0.424472, 0.00330697,  ], [ 0.746698, 0.366195, 0.0414387,  ]),
        (0.52 , [ 8.46752, 0.714782, 0.0107365,  ], [ 0.760192, 0.406511, 0.0692738,  ]),
        (0.53 , [ 7.85715, 1.3651, 0.0360094,  ], [ 0.789684, 0.456885, 0.10255,  ]),
        (0.54 , [ 7.76883, 1.50907, 0.043058,  ], [ 0.805917, 0.437708, 0.0952802,  ]),
        (0.55 , [ 7.65445, 1.67716, 0.0516891,  ], [ 0.822863, 0.427311, 0.0912697,  ]),
        (0.56 , [ 7.54335, 1.84057, 0.0613179,  ], [ 0.839849, 0.420414, 0.088685,  ]),
        (0.57 , [ 7.44456, 1.99074, 0.0714226,  ], [ 0.856566, 0.414979, 0.0866439,  ]),
        ]),
            
        # 0.6+ to 0.75+: 4 components
        #(0.57, 0.76, [
        #(0.54, 0.76, [

        # (0.55 , [ 8.48032, 0.887229, 0.004641, 2.17472e-15,  ], [ 0.789271, 0.30969, 0.0206317, 2.57958e+09,  ]),
        # (0.56 , [ 8.22566, 1.19593, 0.0150014, 1.00065e-11,  ], [ 0.811016, 0.335293, 0.0387941, 2.55998e+09,  ]),
        # (0.57 , [ 7.82365, 1.63757, 0.0411003, 0.000194472,  ], [ 0.840165, 0.373703, 0.0649509, 0.00298538,  ]),
        # (0.58 , [ 7.3678, 2.11332, 0.0851715, 0.00116142,  ], [ 0.872321, 0.410417, 0.0912836, 0.00811122,  ]),
        # (0.59 , [ 6.89805, 2.57825, 0.150862, 0.00403253,  ], [ 0.906138, 0.444427, 0.118216, 0.0157737,  ]),
        # (0.60 , [ 6.53494, 2.92485, 0.224343, 0.00923437,  ], [ 0.936531, 0.468137, 0.140115, 0.0236033,  ]),

        ## # From 2-component fit
        ## (0.54 , [ 8.67782, 0.630599,  0., 0.], [ 0.770737, 0.286705, 0.00830397, 0.00298538, ]),
        ## #(0.54 , [ 8.64119, 0.667547, 0.0, 0.0 ], [ 0.772674, 0.295691, 0.00830397, 2.63229e+09,  ]),
        ## (0.55 , [ 8.48032, 0.887229, 0.004641, 0.0  ], [ 0.789271, 0.30969, 0.0206317, 0.00298538, ]),
        ## (0.56 , [ 8.22566, 1.19593, 0.0150014, 0.0  ], [ 0.811016, 0.335293, 0.0387941, 0.00298538, ]),
        ## (0.57 , [ 7.82365, 1.63757, 0.0411003, 0.0  ], [ 0.840165, 0.373703, 0.0649509, 0.00298538, ]),
        ## (0.58 , [ 7.3678, 2.11332, 0.0851715, 0.00116142,  ], [ 0.872321, 0.410417, 0.0912836, 0.00811122,  ]),
        ## (0.59 , [ 6.89805, 2.57825, 0.150862, 0.00403253,  ], [ 0.906138, 0.444427, 0.118216, 0.0157737,  ]),
        ## (0.60 , [ 6.53494, 2.92485, 0.224343, 0.00923437,  ], [ 0.936531, 0.468137, 0.140115, 0.0236033,  ]),

        # (0.54, 0.56, [
        # # From 2-component fit
        # (0.54 , [ 8.67782, 0.630599,  0. ], [ 0.770737, 0.286705, 0.00830397 ]),
        # #(0.54 , [ 8.64119, 0.667547, 0.0, 0.0 ], [ 0.772674, 0.295691, 0.00830397, 2.63229e+09,  ]),
        # (0.55 , [ 8.48032, 0.887229, 0.004641 ], [ 0.789271, 0.30969, 0.0206317 ]),
        # (0.56 , [ 8.22566, 1.19593, 0.0150014 ], [ 0.811016, 0.335293, 0.0387941 ]),
        # (0.57 , [ 7.82365, 1.63757, 0.0411003 ], [ 0.840165, 0.373703, 0.0649509 ]),
        # ]),
        #(0.57, 0.76, [
        (0.57, 0.62, [
        #(0.56 , [ 7.55834, 1.82665, 0.0601492, 0. ], [ 0.839285, 0.418841, 0.0877974, 0.00415219 ]),
        #    (0.57 , [ 7.34568, 2.07926, 0.0820918, 0.000402361,  ], [ 0.860494, 0.425241, 0.0951868, 0.00415219,  ]),

        # from sersicsC
        (0.57 , [ 7.44456, 1.99074, 0.0714226, 0. ], [ 0.856566, 0.414979, 0.0866439, 0.0015899, ]),
        (0.575 , [ 7.2992, 2.15041, 0.0880113, 0.000440884,  ], [ 0.868967, 0.422919, 0.0942549, 0.00412029,  ]),
        (0.58 , [ 7.08949, 2.36274, 0.115856, 0.00162587,  ], [ 0.884274, 0.43718, 0.107525, 0.00953343,  ]),
        (0.6 , [ 6.39791, 3.02772, 0.256114, 0.0121952,  ], [ 0.942439, 0.480795, 0.152127, 0.0277215,  ]),
        (0.62 , [ 6.29153, 3.20888, 0.298822, 0.0148133,  ], [ 0.978226, 0.474438, 0.14661, 0.0262404,  ]),
        # Keep this to improve the spline at the endpoint
        (0.63 , [ 6.24609, 3.29044, 0.320663, 0.016228,  ], [ 0.995781, 0.471937, 0.144342, 0.0256225,  ]),
        #(0.65 , [ 6.17248, 3.43504, 0.364076, 0.0192057,  ], [ 1.03, 0.467533, 0.140271, 0.0245156,  ]),
        #(0.7 , [ 6.07511, 3.70769, 0.466212, 0.0270964,  ], [ 1.10965, 0.457509, 0.131157, 0.0220772,  ]),
        #(0.75 , [ 6.06413, 3.89387, 0.556591, 0.0352166,  ], [ 1.18078, 0.447338, 0.122734, 0.0199081,  ]),
        #(0.76 , [ 6.06909, 3.92395, 0.573272, 0.0368418,  ], [ 1.19401, 0.445243, 0.121115, 0.0195012,  ]),
            ]),

        (0.62, 0.71, [
        # sersicsB
        # 0.62 is from above, with variance of last component from here.
        (0.62 , [ 6.29153, 3.20888, 0.298822, 0.0148133, 0. ], [ 0.978226, 0.474438, 0.14661, 0.0262404, 0.0019734, ]),
        #(0.62 , [ 6.21467, 3.25855, 0.322441, 0.0185052, 0.000209265,  ], [ 0.981889, 0.482007, 0.155074, 0.0312349, 0.0019734,  ]),
        (0.63 , [ 6.09398, 3.38171, 0.372264, 0.0254702, 0.000682271,  ], [ 1.00341, 0.487314, 0.161969, 0.0366079, 0.00408524,  ]),
        (0.64 , [ 5.99342, 3.48297, 0.42144, 0.0340188, 0.0015078,  ], [ 1.02426, 0.491834, 0.168392, 0.0422743, 0.00631834,  ]),
        (0.65 , [ 5.92916, 3.55924, 0.460143, 0.0412147, 0.00229882,  ], [ 1.04346, 0.493458, 0.171137, 0.0454779, 0.00760041,  ]),
        (0.7 , [ 5.81571, 3.81324, 0.586725, 0.0589626, 0.00359706,  ], [ 1.12784, 0.487001, 0.162894, 0.0426197, 0.00699607,  ]),
        #(0.71 , [ 5.80384, 3.85246, 0.611158, 0.0627668, 0.0038732,  ], [ 1.14378, 0.485858, 0.161486, 0.0420891, 0.00686317,  ]),
        (0.75 , [ 5.78581, 3.98447, 0.700589, 0.0772783, 0.00495831,  ], [ 1.20435, 0.480685, 0.155308, 0.0396603, 0.00628884,  ]),
        #(0.8 , [ 5.80548, 4.10613, 0.802727, 0.0961115, 0.00646245,  ], [ 1.27352, 0.474151, 0.148343, 0.0369821, 0.00567392,  ]),
            ]),

        # 0.75 to 1.5+: 6 components
        (0.71, 1.5, [
        # sersicsA
        # fit for 0.71 is from above, with last variance from here
        (0.71 , [ 5.80384, 3.85246, 0.611158, 0.0627668, 0.0038732, 0. ], [ 1.14378, 0.485858, 0.161486, 0.0420891, 0.00686317, 0.000125639, ]),
        (0.72 , [ 5.78449, 3.89063, 0.64027, 0.0693474, 0.00482271, 4.7711e-05,  ], [ 1.16019, 0.486018, 0.161901, 0.0432836, 0.00770569, 0.000360731,  ]),
        (0.73 , [ 5.76622, 3.92465, 0.670321, 0.0771626, 0.00617073, 0.000149421,  ], [ 1.17644, 0.4865, 0.162878, 0.0450234, 0.00889761, 0.000772094,  ]),
        (0.74 , [ 5.73834, 3.95798, 0.706138, 0.0881532, 0.00842185, 0.000385661,  ], [ 1.19343, 0.488511, 0.165609, 0.048145, 0.0108527, 0.00140575,  ]),
        (0.75 , [ 5.74038, 3.98136, 0.726531, 0.0942424, 0.0102561, 0.000673631,  ], [ 1.20789, 0.487188, 0.1649, 0.0492449, 0.0122331, 0.00192144,  ]),
        (0.8 , [ 5.75213, 4.09637, 0.834928, 0.119308, 0.0137876, 0.000939471,  ], [ 1.27836, 0.482225, 0.159263, 0.0470525, 0.0114908, 0.00174917,  ]),
        (0.85 , [ 5.79382, 4.18044, 0.93266, 0.145078, 0.0176549, 0.00124088,  ], [ 1.34231, 0.477026, 0.153978, 0.0449401, 0.0107664, 0.00158922,  ]),
        (0.9 , [ 5.85237, 4.24456, 1.02239, 0.171488, 0.0218791, 0.00158456,  ], [ 1.40053, 0.471837, 0.149149, 0.0429814, 0.0101053, 0.00144948,  ]),
        (0.95 , [ 5.9154, 4.29436, 1.10933, 0.200379, 0.0267765, 0.00199386,  ], [ 1.4543, 0.467524, 0.145402, 0.0414866, 0.00958089, 0.0013368,  ]),
        (1 , [ 5.99798, 4.33895, 1.17948, 0.223347, 0.0308197, 0.00235229,  ], [ 1.50163, 0.460978, 0.139974, 0.039152, 0.00885129, 0.00120215,  ]),
        #(0.75 , [ 5.73017, 3.98402, 0.732308, 0.0959582, 0.0104227, 0.000681892,  ], [ 1.20875, 0.488435, 0.166055, 0.0497645, 0.0123354, 0.0019356,  ]),
        #(0.80 , [ 5.75014, 4.09764, 0.836108, 0.119013, 0.0136534, 0.00092986,  ], [ 1.27856, 0.482428, 0.159252, 0.0468932, 0.0114183, 0.00173846,  ]),
        #(0.85 , [ 5.79392, 4.18038, 0.932621, 0.145083, 0.0176554, 0.00124148,  ], [ 1.3423, 0.477017, 0.153977, 0.0449414, 0.0107673, 0.00158981,  ]),
        #(0.90 , [ 5.85235, 4.24457, 1.0224, 0.171489, 0.0218797, 0.00158463,  ], [ 1.40054, 0.47184, 0.149151, 0.0429818, 0.0101055, 0.00144954,  ]),
        #(0.95 , [ 5.92083, 4.29582, 1.10511, 0.198049, 0.0263517, 0.00196063,  ], [ 1.45361, 0.466645, 0.144623, 0.0411061, 0.00947885, 0.00132211,  ]),
        #(1.00 , [ 5.99709, 4.33914, 1.1799, 0.223597, 0.030886, 0.00235662,  ], [ 1.50178, 0.461093, 0.140049, 0.0391944, 0.00886401, 0.00120353,  ]),
        (1.10 , [ 6.14939, 4.40829, 1.32037, 0.276163, 0.0406625, 0.00324514,  ], [ 1.58704, 0.451176, 0.132327, 0.0359017, 0.00780277, 0.00100656,  ]),
        (1.20 , [ 6.30415, 4.46728, 1.4427, 0.325106, 0.0503068, 0.00416872,  ], [ 1.65845, 0.440928, 0.124826, 0.0327071, 0.00682354, 0.00083812,  ]),
        (1.30 , [ 6.45448, 4.52149, 1.55158, 0.37086, 0.0598431, 0.00512976,  ], [ 1.71861, 0.430659, 0.117736, 0.0297692, 0.00596945, 0.00070013,  ]),
        (1.40 , [ 6.59957, 4.57433, 1.64865, 0.412699, 0.0689351, 0.00608707,  ], [ 1.76917, 0.420192, 0.110904, 0.0270357, 0.00521398, 0.000585129,  ]),
        (1.50 , [ 6.7389, 4.62721, 1.73564, 0.450737, 0.0774992, 0.00702291,  ], [ 1.81152, 0.409534, 0.104336, 0.0245173, 0.00455116, 0.000489465,  ]),
        (1.55 , [ 6.80691, 4.6537, 1.7755, 0.468334, 0.0815596, 0.00747883,  ], [ 1.82984, 0.40408, 0.101139, 0.023336, 0.00425152, 0.000447901,  ]),
        ]),
        # 1.5 to 3+: 7 components
        (1.5, 3.0, [
        (1.50 , [ 6.7389, 4.62721, 1.73564, 0.450737, 0.0774992, 0.00702291, 0. ], [ 1.81152, 0.409534, 0.104336, 0.0245173, 0.00455116, 0.000489465, 1.26919e-07 ]),
        # sersics3
        #(1.5 , [ 6.73869, 4.62729, 1.73568, 0.450758, 0.077575, 0.00704877, 1.45795e-06,  ], [ 1.81159, 0.409561, 0.104347, 0.0245277, 0.00455809, 0.00049163, 1.26919e-07,  ]),
        (1.6 , [ 6.84022, 4.6525, 1.83264, 0.513258, 0.0983529, 0.0108713, 0.000253333,  ], [ 1.85423, 0.405045, 0.102384, 0.0243061, 0.0047071, 0.000586314, 2.35149e-05,  ]),
        (1.7 , [ 6.98843, 4.72222, 1.89212, 0.528331, 0.0984345, 0.0100329, 9.44416e-05,  ], [ 1.87886, 0.390304, 0.0937353, 0.0208931, 0.00372688, 0.000401139, 7.24892e-06,  ]),
        (1.8 , [ 7.00162, 4.67387, 2.01228, 0.655506, 0.156872, 0.025264, 0.00206884,  ], [ 1.93056, 0.401045, 0.101788, 0.0252612, 0.00545105, 0.000891309, 8.09711e-05,  ]),
        (1.9 , [ 7.13743, 4.75313, 2.06704, 0.661156, 0.150748, 0.0219627, 0.00135914,  ], [ 1.94667, 0.386024, 0.092685, 0.0213977, 0.00420332, 0.000594403, 4.27877e-05,  ]),
        (2.0 , [ 7.18732, 4.74718, 2.15135, 0.742103, 0.189478, 0.0333073, 0.00317256,  ], [ 1.98023, 0.38757, 0.0943672, 0.0225633, 0.00474161, 0.000771134, 7.0628e-05,  ]),
        (2.50 , [ 7.65824, 4.99436, 2.39904, 0.865839, 0.230773, 0.0424127, 0.0042325,  ], [ 2.02971, 0.34077, 0.0718309, 0.0148612, 0.00271532, 0.000383605, 3.00222e-05,  ]),
        (3.00 , [ 8.14102, 5.24059, 2.53707, 0.922238, 0.249248, 0.0466651, 0.00474693,  ], [ 1.99426, 0.286884, 0.052207, 0.00942273, 0.00151295, 0.000187568, 1.26448e-05,  ]),
        (3.10 , [ 8.24161, 5.28628, 2.55339, 0.926919, 0.250588, 0.0469803, 0.0047839,  ], [ 1.97757, 0.275642, 0.0487155, 0.00856339, 0.00134121, 0.00016213, 1.06116e-05,  ]),
        #(3.20 , [ 8.34394, 5.32945, 2.56627, 0.929876, 0.25136, 0.0471554, 0.00480307,  ], [ 1.95759, 0.264343, 0.0453888, 0.00777446, 0.00118816, 0.000140071, 8.90155e-06,  ]),
        ]),
        # 3 to 6: 8 components
        #(3., 6.3, [
        (3.0, 6.3, [

        (3.00 , [ 8.14102, 5.24059, 2.53707, 0.922238, 0.249248, 0.0466651, 0.00474693, 0. ], [ 1.99426, 0.286884, 0.052207, 0.00942273, 0.00151295, 0.000187568, 1.26448e-05, 1.25508e-07 ]),
        #(3.0 , [ 8.10521, 5.2249, 2.55636, 0.944806, 0.26151, 0.0508683, 0.0056095, 4.33514e-05,  ], [ 2.00803, 0.292206, 0.053997, 0.00992823, 0.00163513, 0.000211379, 1.57459e-05, 1.25508e-07,  ]),
        (3.1 , [ 8.07544, 5.2095, 2.63977, 1.03299, 0.310611, 0.0687529, 0.00971923, 0.00049079,  ], [ 2.04225, 0.300424, 0.0570212, 0.0109142, 0.0019162, 0.00027784, 2.71048e-05, 1.13468e-06,  ]),
        (3.2 , [ 8.03033, 5.17514, 2.72194, 1.13299, 0.371686, 0.093523, 0.0165082, 0.00155907,  ], [ 2.08117, 0.311506, 0.0612546, 0.0123072, 0.00231779, 0.000376581, 4.59253e-05, 2.92292e-06,  ]),
        (3.3 , [ 8.10278, 5.21812, 2.74999, 1.14796, 0.378683, 0.0961179, 0.0172133, 0.0016746,  ], [ 2.07426, 0.303209, 0.0583139, 0.0114924, 0.00212925, 0.000341377, 4.12787e-05, 2.60765e-06,  ]),
        (3.4 , [ 8.1839, 5.2655, 2.77223, 1.15587, 0.381374, 0.0969334, 0.0174001, 0.00169875,  ], [ 2.06218, 0.293548, 0.0550206, 0.0105945, 0.00192158, 0.000301843, 3.57481e-05, 2.20559e-06,  ]),
        #(3.5 , [ 8.26594, 5.31105, 2.79169, 1.16265, 0.383854, 0.0977587, 0.0176062, 0.0017298,  ], [ 2.04785, 0.283872, 0.0518883, 0.00977241, 0.00173718, 0.000267693, 3.11075e-05, 1.8819e-06,  ]),
        (3.50 , [ 8.2659, 5.31123, 2.79165, 1.16263, 0.383863, 0.0977581, 0.0176055, 0.00172958,  ], [ 2.04792, 0.283875, 0.051887, 0.00977245, 0.0017372, 0.000267685, 3.11052e-05, 1.88165e-06,  ]),
        (4.00 , [ 8.65642, 5.50645, 2.87114, 1.19672, 0.400482, 0.104839, 0.0198321, 0.0021867,  ], [ 1.95979, 0.240012, 0.0393036, 0.00675003, 0.00110949, 0.000160129, 1.77656e-05, 1.074e-06,  ]),
        (4.50 , [ 8.99621, 5.65283, 2.9379, 1.23994, 0.427649, 0.117971, 0.0244073, 0.0032999,  ], [ 1.86001, 0.205048, 0.0308801, 0.00497934, 0.000782798, 0.000110491, 1.24644e-05, 8.32241e-07,  ]),
        (5.00 , [ 9.29528, 5.76673, 2.99459, 1.28529, 0.459118, 0.134222, 0.0305692, 0.00507613,  ], [ 1.75283, 0.176342, 0.0248332, 0.003824, 0.000585179, 8.23549e-05, 9.65102e-06, 7.17946e-07,  ]),
        (5.50 , [ 9.55843, 5.85539, 3.04284, 1.3303, 0.492811, 0.152703, 0.0381636, 0.00763483,  ], [ 1.64141, 0.15243, 0.0203223, 0.00302437, 0.000455735, 6.46352e-05, 7.92247e-06, 6.48304e-07,  ]),
        (6.00 , [ 9.78462, 5.92346, 3.08624, 1.37584, 0.528399, 0.17315, 0.0471669, 0.0111288,  ], [ 1.52991, 0.132595, 0.0169103, 0.00245378, 0.000366884, 5.27404e-05, 6.76674e-06, 6.00242e-07,  ]),
        (6.10 , [ 9.82588, 5.93499, 3.09429, 1.38485, 0.535646, 0.177436, 0.0491334, 0.0119536,  ], [ 1.50771, 0.129027, 0.0163265, 0.0023592, 0.000352456, 5.08287e-05, 6.58083e-06, 5.92307e-07,  ]),
        (6.20 , [ 9.86538, 5.94567, 3.10234, 1.39406, 0.54302, 0.181814, 0.0511623, 0.0128242,  ], [ 1.48564, 0.125603, 0.0157758, 0.00227077, 0.000339019, 4.9052e-05, 6.40777e-06, 5.84841e-07,  ]),
        (6.30 , [ 9.90411, 5.95575, 3.10989, 1.40294, 0.550347, 0.186235, 0.0532466, 0.013741,  ], [ 1.46346, 0.122261, 0.015246, 0.00218663, 0.00032635, 4.73849e-05, 6.24559e-06, 5.77773e-07,  ]),
            ]),
            ]

        self.orig_fits = fits
        
        self.fits = []
        for lo, hi, grid in fits:
            (s,a,v) = grid[0]
            K = len(a)
            # spline degree
            spline_k = 3
            if len(grid) <= 3:
                spline_k = 1

            if True:
                amp_funcs = [InterpolatedUnivariateSpline(
                    [ser for ser,amps,varr in grid],
                    [amps[i] for ser,amps,varr in grid], k=spline_k)
                    for i in range(K)]
                logvar_funcs = [InterpolatedUnivariateSpline(
                    [ser for ser,amps,varr in grid],
                    [np.log(varr[i]) for ser,amps,varr in grid], k=spline_k)
                    for i in range(K)]
            else:
                amp_funcs = [interp1d(
                    [ser for ser,amps,varr in grid],
                    [amps[i] for ser,amps,varr in grid], assume_sorted=True)
                    for i in range(K)]
                logvar_funcs = [interp1d(
                    [ser for ser,amps,varr in grid],
                    [np.log(varr[i]) for ser,amps,varr in grid], assume_sorted=True)
                    for i in range(K)]

            # amp_funcs = [UnivariateSpline(
            #     [ser for ser,amps,varr in grid],
            #     [amps[i] for ser,amps,varr in grid], k=spline_k, s=1.)
            #     for i in range(K)]
            # logvar_funcs = [UnivariateSpline(
            #     [ser for ser,amps,varr in grid],
            #     [np.log(varr[i]) for ser,amps,varr in grid], k=spline_k, s=1.)
            #     for i in range(K)]

            self.fits.append((lo, hi, amp_funcs, logvar_funcs))
        (lo,hi,a,v) = self.fits[0]
        self.lowest = lo
        (lo,hi,a,v) = self.fits[-1]
        self.highest = hi

    def _getProfile(self, sindex):
        matches = []
        # clamp
        if sindex <= self.lowest:
            matches.append(self.fits[0])
            #(lo,hi,a,v) = self.fits[0]
            #amp_funcs = a
            #logvar_funcs = v
            sindex = self.lowest
        elif sindex >= self.highest:
            matches.append(self.fits[-1])
            #(lo,hi,a,v) = self.fits[-1]
            #amp_funcs = a
            #logvar_funcs = v
            sindex = self.highest
        else:
            for f in self.fits:
                lo,hi,a,v = f
                if sindex >= lo and sindex < hi:
                    matches.append(f)
                    #print('Sersic index', sindex, '-> range', lo, hi)
                    #amp_funcs = a
                    #logvar_funcs = v
                    #break

        if len(matches) == 2:
            # Two ranges overlap.  Ramp between them.
            # Assume self.fits is ordered in increasing Sersic index
            m0,m1 = matches
            lo0,hi0,a0,v0 = m0
            lo1,hi1,a1,v1 = m1
            assert(lo0 < lo1)
            assert(lo1 < hi0) # overlap is in here
            ramp_lo = lo1
            ramp_hi = hi0
            assert(ramp_lo < ramp_hi)
            assert(ramp_lo <= sindex)
            assert(sindex < ramp_hi)
            ramp_frac = (sindex - ramp_lo) / (ramp_hi - ramp_lo)
            #print('Sersic index', sindex, ': ramping between ranges', (lo0,hi0), 'and', (lo1,hi1), '; ramp', (ramp_lo, ramp_hi), 'fraction', ramp_frac)
            amps0 = np.array([f(sindex) for f in a0])
            amps0 /= amps0.sum()
            amps1 = np.array([f(sindex) for f in a1])
            amps1 /= amps1.sum()
            amps = np.append((1.-ramp_frac)*amps0, ramp_frac*amps1)
            varr = np.exp(np.array([f(sindex) for f in v0 + v1]))
            #print('amps', amps, 'sum', amps.sum())
        else:
            assert(len(matches) == 1)
            lo,hi,amp_funcs,logvar_funcs = matches[0]
            amps = np.array([f(sindex) for f in amp_funcs])
            amps /= amps.sum()
            varr = np.exp(np.array([f(sindex) for f in logvar_funcs]))
            #print('amps', amps, 'sum', amps.sum())

        # print('varr', varr)
        return mp.MixtureOfGaussians(amps, np.zeros((len(amps), 2)), varr)

class SersicIndex(ScalarParam):
    stepsize = 0.01

    def __init__(self, val=0):
        super(SersicIndex, self).__init__(val=val)
        # Bounds
        self.lower = 0.29
        self.upper = 6.3
        self.maxstep = 0.1

    def isLegal(self):
        return self.val <= self.upper and self.val >= self.lower

    def getLogPrior(self):
        if not self.isLegal():
            return -np.inf
        return 0.

class SersicGalaxy(HoggGalaxy):
    nre = 8.

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, shape=2, sersicindex=3)

    def __init__(self, pos, brightness, shape, sersicindex, **kwargs):
        # super(SersicMultiParams.__init__(self, pos, brightness, shape, sersicindex)
        #self.name = self.getName()
        self.nre = SersicGalaxy.nre
        super(SersicGalaxy, self).__init__(pos, brightness, shape, sersicindex)
        #**kwargs)
        #self.sersicindex = sersicindex

    def __str__(self):
        return (super(SersicGalaxy, self).__str__() +
                ', Sersic index %.3f' % self.sersicindex.val)

    def getName(self):
        return 'SersicGalaxy'

    def getProfile(self):
        return SersicMixture.getProfile(self.sersicindex.val)

    def copy(self):
        return SersicGalaxy(self.pos.copy(), self.brightness.copy(),
                            self.shape.copy(), self.sersicindex.copy())

    def _getUnitFluxDeps(self, img, px, py):
        return hash(('unitpatch', self.getName(), px, py,
                     img.getWcs().hashkey(),
                     img.getPsf().hashkey(),
                     self.shape.hashkey(),
                     self.sersicindex.hashkey()))

    def getParamDerivatives(self, img, modelMask=None):
        # superclass produces derivatives wrt pos, brightness, and shape.
        derivs = super(SersicGalaxy, self).getParamDerivatives(
            img, modelMask=modelMask)

        pos0 = self.getPosition()
        (px0, py0) = img.getWcs().positionToPixel(pos0, self)
        patch0 = self.getUnitFluxModelPatch(img, px=px0, py=py0, modelMask=modelMask)
        if patch0 is None:
            derivs.append(None)
            return derivs
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)

        # derivatives wrt Sersic index
        steps = self.sersicindex.getStepSizes()
        if not self.isParamFrozen('sersicindex'):
            inames = self.sersicindex.getParamNames()
            oldvals = self.sersicindex.getParams()
            ups = self.sersicindex.getUpperBounds()
            for i,step in enumerate(steps):
                # Assume step is positive, and check whether stepping
                # would exceed the upper bound.
                newval = oldvals[i] + step
                if newval > ups[i]:
                    step *= -1.
                    newval = oldvals[i] + step

                oldval = self.sersicindex.setParam(i, newval)
                patchx = self.getUnitFluxModelPatch(
                    img, px=px0, py=py0, modelMask=modelMask)
                self.sersicindex.setParam(i, oldval)
                if patchx is None:
                    print('patchx is None:')
                    print('  ', self)
                    print('  stepping galaxy sersicindex',
                          self.sersicindex.getParamNames()[i])
                    print('  stepped', step)
                    print('  to', self.sersicindex.getParams()[i])
                    derivs.append(None)

                dx = (patchx - patch0) * (counts / step)
                dx.setName('d(%s)/d(%s)' % (self.dname, inames[i]))
                derivs.append(dx)
        return derivs


if __name__ == '__main__':
    from basics import *
    from ellipses import *
    from astrometry.util.plotutils import PlotSequence
    import pylab as plt

    # mix = SersicMixture()
    # plt.clf()
    # for (n, amps, vars) in mix.fits:
    #     plt.loglog(vars, amps, 'b.-')
    #     plt.text(vars[0], amps[0], '%.2f' % n, ha='right', va='top')
    # plt.xlabel('Variance')
    # plt.ylabel('Amplitude')
    # plt.savefig('serfits.png')

    import galsim
    from tractor.psf import *
    
    H,W = 100,100

    xx,yy = np.meshgrid(np.arange(25),np.arange(25))
    psf_sigma = 2.
    psf_img = np.exp(-0.5 * ((xx-12)**2 + (yy-12)**2) / psf_sigma**2)
    psf_img /= np.sum(psf_img)
    gausspsf = GaussianMixturePSF(1., 0., 0., psf_sigma**2, psf_sigma**2, 0.)
    pixpsf = PixelizedPSF(psf_img)
    hpsf = HybridPixelizedPSF(pixpsf, gauss=gausspsf)    

    tim = Image(data=np.zeros((H,W)), inverr=np.ones((H,W)), psf=hpsf)
    SersicIndex.stepsize = 0.001
    re = 15.0
    cx = 50
    cy = 50
    rr = (np.arange(W)-cx)/re
    #serstep = 0.01
    serstep = SersicIndex.stepsize
    gal = SersicGalaxy(PixPos(50., 50.), Flux(1.), EllipseE(re, 0.0, 0.0), SersicIndex(1.0))
    gal.freezeAllBut('sersicindex')
    tr = Tractor([tim], [gal])

    ss = SersicIndex.stepsize

    #sersics = np.linspace(2.999, 6.0, 3)
    #sersics = [2.999]
    #sersics = [0.395, 0.399, 0.400]
    #sersics = [0.3, 0.35, 0.4-ss/2., 0.4, 0.4+ss/2., 0.405, 0.41, 0.5-ss/2, 0.5,]
    #sersics = [0.55-ss, 0.55, 0.55+ss, 0.555, 0.56-ss, 0.56]
    #sersics = [0.599, 0.6, 0.605, 0.609, 0.61,
    #           0.69, 0.699, 0.7, 0.705, 0.709, 0.71,
    #          ]

    #sersics = [
    ##0.4,
    ##0.55, 0.58, 0.59, 0.5999, 0.6,
    ##0.3, 0.399, 0.4,
    ##0.549, 0.55,
    ##0.599, 0.6,
    ##0.7, 0.749, 0.75, 0.76,
    ##0.799, 0.8,
    ##0.799, 0.800,
    ##1.499, 1.500,
    ##1.5, 2.0, 2.5, 2.55, 2.6, 2.69, 2.8, 3.0,
    ##2.5, 2.9,
    ##2.999, 3.000,
    #3.0, 3.1, 3.2,
    #4., 5., 6., 6.19
    #]

    if False:
        ps = PlotSequence('ser', format='%03i')

        np.random.seed(13)
        
        #import logging
        #import sys
        #logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stdout)
        
        re = 15.0
        W,H = 100,100

        #sersics = np.linspace(0.35, 5.5, 21)
        sersics = [6.]

        true_ser = []
        fit_ser = []
        fit_dser = []
        fit_allparams = []
        fit_alld = []

        noise_sigma = 1.

        img = Image(data=np.zeros((H,W)), inverr=np.ones((H,W), np.float32) / noise_sigma,
                    psf=GaussianMixturePSF(1., 0., 0., 4., 4., 0.))
        tim = img

        from tractor.constrained_optimizer import ConstrainedOptimizer

        for i,si in enumerate(sersics):
            pixel_scale = 1.0
            gs_psf = galsim.Gaussian(flux=1., sigma=2.)
            gs_gal = galsim.Sersic(n=si, half_light_radius=re)
            gs_gal = gs_gal.shear(galsim.Shear(g1=0.1, g2=0.0))
            gs_gal = gs_gal.shift(0.5, 0.5)
            gs_final = galsim.Convolve([gs_gal, gs_psf])
            gs_image = gs_final.drawImage(scale=pixel_scale)#, method='sb')
            gs_image = gs_image.array
            iy,ix = np.unravel_index(np.argmax(gs_image), gs_image.shape)
            gs_image = gs_image[iy-H//2:, ix-W//2:][:H,:W]

            
            flux = 2000.
            # + 3.*(float(si) / max(1, len(sersics)-1))
            #flux *= 50.

            for j in range(1):
                tim.setImage(flux * gs_image + np.random.normal(size=(H,W), scale=noise_sigma))
                print('Max:', tim.getImage().max(), 'Sum:', tim.getImage().sum())
                gal = SersicGalaxy(PixPos(50.3, 49.7), Flux(100.),
                                   EllipseE(re*0.5, 0.0, 0.0), SersicIndex(2.5))
                #gal = DevGalaxy(PixPos(50., 50.), Flux(1.), EllipseE(re, 0.0, 0.0))
                tr = Tractor([tim], [gal], optimizer=ConstrainedOptimizer())
                tr.freezeParam('images')
                #tr.printThawedParams()
                try:
                    tr.optimize_loop(dchisq=0.1, shared_params=False)
                    v = tr.optimize(variance=True, just_variance=True, shared_params=False)
                except:
                    print('Failed', si)
                    import traceback
                    traceback.print_exc()
                    continue
                dser = np.sqrt(v[-1])

                print('Fit galaxy:', gal)
                
                plt.clf()
                plt.subplot(1,2,1)
                sig1 = noise_sigma
                ima = dict(vmin=-2.*sig1, vmax=5.*sig1, interpolation='nearest', origin='lower')
                plt.imshow(tim.getImage(), **ima)
                #plt.colorbar()
                plt.subplot(1,2,2)
                plt.imshow(tr.getModelImage(0), **ima)
                ps.savefig()
                continue
            
                #print('ser', si, 'trial', j, '->', gal.sersicindex.getValue(), '+-', dser)
                if dser == 0:
                    ### HACK FIXME
                    continue
                true_ser.append(si)
                fit_ser.append(gal.sersicindex.getValue())
                fit_dser.append(dser)
                fit_allparams.append(gal.getParams())
                fit_alld.append(np.sqrt(v))
        import sys
        sys.exit(0)

    if True:
        #sersics = np.logspace(np.log10(0.3001), np.log10(6.19), 1000)
        #sersics = np.logspace(np.log10(0.3001), np.log10(0.6), 500)
        #sersics = np.linspace(0.35, 0.5, 101)
        slo,shi = 0.3, 6.2
        #sersics = np.linspace(slo, shi, 101)
        sersics = np.logspace(np.log10(slo), np.log10(shi), 1000).clip(slo,shi)

        plt.figure(figsize=(12,5))
        plt.clf()
        plt.subplot(1,2,1)
        for s in sersics:
            mix = SersicMixture.getProfile(s)
            plt.plot(s+np.zeros_like(mix.amp), mix.amp, 'k.', ms=1)
        #plt.xscale('log')
        plt.yscale('symlog', linthreshy=1e-3)
        plt.xlabel('Sersic index')
        plt.ylabel('Mixture amplitudes')
        plt.subplot(1,2,2)
        for s in sersics:
            mix = SersicMixture.getProfile(s)
            plt.plot(s+np.zeros_like(mix.amp), mix.var[:,0,0], 'k.', ms=1)
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Sersic index')
        plt.ylabel('Mixture variances')
        plt.savefig('mix.png')

        plt.subplot(1,2,1)
        plt.xscale('log')
        plt.subplot(1,2,2)
        plt.xscale('log')
        plt.savefig('mix2.png')

    #sersics = np.logspace(np.log10(0.3001), np.log10(6.19), 200)
    #sersics = np.linspace(0.35, 0.5, 25)
    #sersics = np.linspace(slo, shi, 33)
    #sersics = np.linspace(slo, shi, 101)
    sersics = np.logspace(np.log10(slo), np.log10(shi), 101).clip(slo,shi)
    
    ps = PlotSequence('ser', format='%03i')

    plt.figure(figsize=(12,4))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.85)
    
    pixel_scale = 1.0
    draw_args = dict(scale=pixel_scale)

    #gs_psf = galsim.Gaussian(flux=1., sigma=psf_sigma)
    pim = galsim.Image(psf_img)
    gs_psf = galsim.InterpolatedImage(pim, scale=pixel_scale)
    draw_args.update(method='no_pixel')
    
    for i,si in enumerate(sersics):
        gs_gal = galsim.Sersic(si, half_light_radius=re)
        gs_gal = gs_gal.shift(0.5, 0.5)
        gs_final = galsim.Convolve([gs_gal, gs_psf])
        gs_image = gs_final.drawImage(**draw_args)
        gs_image = gs_image.array
        iy,ix = np.unravel_index(np.argmax(gs_image), gs_image.shape)
        gs_image = gs_image[iy-H//2:, ix-W//2:][:H,:W]

        step = serstep
        if si >= 6.2:
            step *= -1.
        
        gs_gal = galsim.Sersic(si + step, half_light_radius=re)
        gs_gal = gs_gal.shift(0.5, 0.5)
        gs_final2 = galsim.Convolve([gs_gal, gs_psf])
        gs_image2 = gs_final2.drawImage(**draw_args)
        gs_image2 = gs_image2.array
        iy,ix = np.unravel_index(np.argmax(gs_image2), gs_image2.shape)
        gs_image2 = gs_image2[iy-cy:, ix-cx:][:H,:W]

        # # SB method
        # gs_image3 = gs_final.drawImage(scale=pixel_scale, method='sb')
        # gs_image3 = gs_image3.array
        # iy,ix = np.unravel_index(np.argmax(gs_image3), gs_image3.shape)
        # gs_image3 = gs_image3[iy-H//2:, ix-W//2:][:H,:W]
        # 
        # gs_image4 = gs_final2.drawImage(scale=pixel_scale, method='sb')
        # gs_image4 = gs_image4.array
        # iy,ix = np.unravel_index(np.argmax(gs_image4), gs_image4.shape)
        # gs_image4 = gs_image4[iy-H//2:, ix-W//2:][:H,:W]
        
        ds = (gs_image2 - gs_image) / step
        # galsim derivative estimate
        #ds = (gs_image4 - gs_image3) / (step)

        # galsim model estimate
        #gsim = gs_image3
        gsim = gs_image
        gs = gsim[H//2,:]

        gal.sersicindex.setValue(si)
        derivs = gal.getParamDerivatives(tim)
        assert(len(derivs) == 1)
        deriv = derivs[0]
        dd = np.zeros((H,W))
        deriv.addTo(dd)

        mod = tr.getModelImage(0)

        iy,ix = np.unravel_index(np.argmax(mod), mod.shape)
        assert(iy == H//2)
        assert(ix == W//2)

        print('GS sum:', gsim.sum(), 'Tractor sum:', mod.sum(), 'Ratio (T/GS):', mod.sum()/gsim.sum())
        gs *= mod.sum()/gsim.sum()

        # SCALE deriv by same factor
        ds *= mod.sum()/gsim.sum()
        
        N = gal.getProfile().K
        
        plt.clf()

        plt.subplot(1,3,1)

        scale = 1000.
        #plt.plot(scale * gs_image[H//2,:], label='galsim')
        plt.plot(scale * gs, label='galsim')
        plt.plot(scale * mod[H//2,:], label='tractor')
        plt.legend()
        plt.title('Model')
        #plt.title('Model: %.4f' % si)
        #ps.savefig()

        #mx = max(scale * gs_image[H//2,:]) * 0.01
        mx = max(scale * gs) * 0.01
        
        plt.subplot(1,3,2)

        #plt.plot(scale * (mod[H//2,:] - gs_image[H//2,:]))
        plt.plot(scale * (mod[H//2,:] - gs))
        #plt.ylim(-mx, mx)
        plt.axhline(0., color='k', alpha=0.1)
        plt.axhline(mx*0.1, color='k', alpha=0.1)
        plt.axhline(mx, color='k', alpha=0.1)
        plt.axhline(mx*-0.1, color='k', alpha=0.1)
        plt.title('Difference')
        
        #plt.clf()
        plt.subplot(1,3,3)
        # plt.subplot(1,4,1)
        # plt.imshow(tr.getModelImage(0), interpolation='nearest', origin='lower')
        # 
        # mn = min(np.min(dd), np.min(ds))
        # mx = max(np.max(dd), np.max(ds))
        # da = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        # plt.subplot(1,4,2)
        # plt.imshow(dd, **da)
        # 
        # plt.subplot(1,4,3)
        # plt.imshow(ds, **da)
        # 
        # plt.subplot(1,4,4)
        d1 = dd[:,H//2]
        d2 = ds[:,H//2]
        plt.plot(d2, label='galsim')
        yl,yh = plt.ylim()
        plt.plot(d1, label='tractor')
        plt.ylim(yl*1.5, yh*1.5)
        plt.axhline(0., color='k', alpha=0.1)
        plt.legend()
        #plt.suptitle('%0.4f'%si)
        #plt.show()
        plt.title('Deriv')
        #plt.title('Deriv: %.4f' % si)
        plt.suptitle('Sersic index: %.4f (N=%i)' % (si,N))
        ps.savefig()

    import sys
    sys.exit(0)

    for ser in [0.699, 0.7, 0.701, 0.705, 0.709, 0.71]:
    
        s = SersicGalaxy(PixPos(100., 100.),
                         Flux(1000.),
                         EllipseE(5., -0.5, 0.),
                         SersicIndex(ser))
        print()
        print(s)
        s.getProfile()
        #print(s.getProfile())

    sys.exit(0)
        
    s.sersicindex.setValue(4.0)
    print(s.getProfile())

    d = DevGalaxy(s.pos, s.brightness, s.shape)
    print(d)
    print(d.getProfile())

    # Extrapolation!
    # s.sersicindex.setValue(0.5)
    # print s.getProfile()

    ps = PlotSequence('ser')

    # example PSF (from WISE W1 fit)
    w = np.array([0.77953706,  0.16022146,  0.06024237])
    mu = np.array([[-0.01826623, -0.01823262],
                   [-0.21878855, -0.0432496],
                   [-0.83365747, -0.13039277]])
    sigma = np.array([[[7.72925584e-01,   5.23305564e-02],
                       [5.23305564e-02,   8.89078473e-01]],
                      [[9.84585869e+00,   7.79378820e-01],
                       [7.79378820e-01,   8.84764455e+00]],
                      [[2.02664489e+02,  -8.16667434e-01],
                       [-8.16667434e-01,   1.87881670e+02]]])

    psf = GaussianMixturePSF(w, mu, sigma)

    data = np.zeros((200, 200))
    invvar = np.zeros_like(data)
    tim = Image(data=data, invvar=invvar, psf=psf)

    tractor = Tractor([tim], [s])

    nn = np.linspace(0.5, 5.5, 12)
    cols = int(np.ceil(np.sqrt(len(nn))))
    rows = int(np.ceil(len(nn) / float(cols)))

    xslices = []
    disable_galaxy_cache()

    plt.clf()
    for i, n in enumerate(nn):
        s.sersicindex.setValue(n)
        print(s.getParams())
        #print(s.getProfile())

        mod = tractor.getModelImage(0)

        plt.subplot(rows, cols, i + 1)
        #plt.imshow(np.log10(np.maximum(1e-16, mod)), interpolation='nearest',
        plt.imshow(mod, interpolation='nearest', origin='lower')
        plt.axis([50,150,50,150])
        plt.title('index %.2f' % n)

        xslices.append(mod[100,:])
    ps.savefig()

    plt.clf()
    for x in xslices:
        plt.plot(x, '-')
    ps.savefig()

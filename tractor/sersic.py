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
        from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
        # GalSim: supports n=0.3 to 6.2.

        # A set of ranges [ser_lo, ser_hi], plus a list of fit parameters
        # that have a constant number of components.
        fits = [
        # 3 components
        (0.29, 0.42, [
        # amp prior std=2
        # (with amp prior std=1, the two positive components become degenerate)
        # sersicsG
        (0.29 , [ 29.7334, 10.0626, -32.1111,  ], [ 0.400917, 0.223733, 0.29063,  ]),
        (0.3 , [ 28.1272, 8.97219, -29.3501,  ], [ 0.410641, 0.227158, 0.29584,  ]),
        (0.31 , [ 26.5806, 7.94611, -26.7125,  ], [ 0.420884, 0.230796, 0.301292,  ]),
        (0.32 , [ 25.0955, 6.98474, -24.2006,  ], [ 0.431669, 0.234678, 0.307003,  ]),
        (0.33 , [ 23.6732, 6.08762, -21.8156,  ], [ 0.44302, 0.238842, 0.31299,  ]),
        (0.34 , [ 22.3148, 5.25393, -19.5576,  ], [ 0.454958, 0.243336, 0.319275,  ]),
        (0.35 , [ 21.0211, 4.4825, -17.4262,  ], [ 0.467504, 0.248227, 0.325884,  ]),
        (0.36 , [ 19.7908, 3.76912, -15.416,  ], [ 0.480684, 0.253582, 0.332854,  ]),
        (0.37 , [ 18.6228, 3.11202, -13.5245,  ], [ 0.494522, 0.259516, 0.340225,  ]),
        (0.38 , [ 17.5146, 2.50564, -11.7432,  ], [ 0.509046, 0.266156, 0.348055,  ]),
        #(0.39 , [ 16.4614, 1.94395, -10.0617,  ], [ 0.524299, 0.273676, 0.35641,  ]),
        #(0.4 , [ 15.4473, 1.43828, -8.47514,  ], [ 0.540383, 0.28262, 0.365116,  ]),
        #(0.41 , [ 14.4933, 0.921142, -6.93744,  ], [ 0.557234, 0.292441, 0.37502,  ]),
        #(0.42 , [ 13.5581, 0.433215, -5.44774,  ], [ 0.575113, 0.304653, 0.385431,  ]),

        # sersicsFG
        # kick in a prior on the amplitude to force the second component to zero.
        (0.39 , [ 16.5252, 1.55052, -9.73179,  ], [ 0.524296, 0.26729, 0.359764,  ]),
        (0.4 , [ 15.5942, 0.836228, -8.01953,  ], [ 0.540194, 0.267179, 0.372381,  ]),
        (0.41 , [ 14.7227, 0.344169, -6.58915,  ], [ 0.556667, 0.265585, 0.385658,  ]),
        #(0.42 , [ 14.0601, -0.0213796, -5.49404,  ], [ 0.572973, 0.403082, 0.403103,  ]),
        # from sersicsF
        (0.42 , [ 14.0586, 0., -5.51399,  ], [ 0.572984, 0.403086, 0.403086 ]),
        ]),

        (0.42, 0.51, [
        # sersicsF
        #(0.4 , [ 17.4381, -9.01977,  ], [ 0.538194, 0.410386,  ]),
        (0.41 , [ 15.7428, -7.26198,  ], [ 0.55406, 0.408598,  ]),
        (0.42 , [ 14.0586, -5.51399,  ], [ 0.572984, 0.403086,  ]),
        (0.43 , [ 12.5707, -3.96071,  ], [ 0.594559, 0.393181,  ]),
        (0.44 , [ 11.4452, -2.76895,  ], [ 0.616783, 0.380574,  ]),
        (0.45 , [ 10.6629, -1.92036,  ], [ 0.637897, 0.367586,  ]),
        (0.46 , [ 10.1197, -1.31131,  ], [ 0.657376, 0.355282,  ]),
        (0.47 , [ 9.7311, -0.857421,  ], [ 0.675295, 0.343829,  ]),
        (0.48 , [ 9.44434, -0.506166,  ], [ 0.691845, 0.333043,  ]),
        (0.49 , [ 9.22735, -0.225454,  ], [ 0.70721, 0.322121,  ]),

        # sersicsE, with second amp 0 and variance interp. between 0.49 and 0.51
        (0.5 , [ 9.06467, 0. ], [ 0.721342,  0.316586 ]),

        # sersicsD
        (0.51 , [ 8.92798, 0.198966,  ], [ 0.735017, 0.311146,  ]),
        (0.52 , [ 8.82471, 0.363539,  ], [ 0.747636, 0.301751,  ]),
        ]),
        # (0.53 , [ 8.7428, 0.505938,  ], [ 0.759522, 0.293898,  ]),
        # (0.54 , [ 8.67782, 0.6306,  ], [ 0.770737, 0.286704,  ]),
        # (0.55 , [ 8.6265, 0.740798,  ], [ 0.781331, 0.279957,  ]),
        # (0.56 , [ 8.58636, 0.839008,  ], [ 0.791344, 0.273563,  ]),
        # (0.57 , [ 8.55541, 0.927229,  ], [ 0.800818, 0.267477,  ]),
        # (0.58 , [ 8.53217, 1.00695,  ], [ 0.809782, 0.261658,  ]),
        # (0.59 , [ 8.51548, 1.07934,  ], [ 0.818259, 0.25607,  ]),
        # (0.6 , [ 8.50421, 1.14553,  ], [ 0.826286, 0.250709,  ]),
        # (0.61 , [ 8.49772, 1.20617,  ], [ 0.833872, 0.245539,  ]),        

        # sersicsC2
        (0.51, 0.57, [
        # third variance from 0.515
        #(0.51 , [ 8.92797, 0.198981, -3.33278e-11,  ], [ 0.735018, 0.311154, 1.64344e+10,  ]),
        (0.51 , [ 8.92797, 0.198981,  0.  ], [ 0.735018, 0.311154, 0.0414387,  ]),
        (0.515 , [ 8.73228, 0.424472, 0.00330697,  ], [ 0.746698, 0.366195, 0.0414387,  ]),
        (0.52 , [ 8.46752, 0.714782, 0.0107365,  ], [ 0.760192, 0.406511, 0.0692738,  ]),
        (0.53 , [ 7.85715, 1.3651, 0.0360094,  ], [ 0.789684, 0.456885, 0.10255,  ]),
        (0.54 , [ 7.76883, 1.50907, 0.043058,  ], [ 0.805917, 0.437708, 0.0952802,  ]),
        (0.55 , [ 7.65445, 1.67716, 0.0516891,  ], [ 0.822863, 0.427311, 0.0912697,  ]),
        (0.56 , [ 7.54335, 1.84057, 0.0613179,  ], [ 0.839849, 0.420414, 0.088685,  ]),
        (0.57 , [ 7.44456, 1.99074, 0.0714226,  ], [ 0.856566, 0.414979, 0.0866439,  ]),
        ]),

        # sersicsC
        (0.57, 0.62, [
        # first three components from 3-comp 0.57, above
        (0.57 , [ 7.41719, 2.01535, 0.0742881, 0.  ], [ 0.856566, 0.414979, 0.0866439, 0.0015899,  ]),
        #(0.57 , [ 7.41719, 2.01535, 0.0742881, 9.69107e-05,  ], [ 0.857665, 0.417844, 0.0889867, 0.0015899,  ]),
        (0.575 , [ 7.2992, 2.1504, 0.0880109, 0.000440864,  ], [ 0.868967, 0.422919, 0.0942545, 0.00412021,  ]),
        (0.58 , [ 7.0897, 2.36255, 0.115839, 0.00162557,  ], [ 0.884264, 0.437162, 0.107518, 0.00953237,  ]),
        (0.6 , [ 6.39792, 3.02771, 0.256107, 0.0121941,  ], [ 0.942439, 0.480793, 0.152123, 0.0277202,  ]),
        (0.62 , [ 6.29146, 3.20893, 0.298839, 0.0148154,  ], [ 0.97823, 0.474444, 0.146615, 0.0262425,  ]),
        (0.63 , [ 6.24609, 3.29044, 0.320663, 0.0162281,  ], [ 0.995781, 0.471937, 0.144342, 0.0256226,  ]),        
        ]),
        
        # sersicsB
        (0.62, 0.71, [
        # 0.62 is from above, with variance of last component from here.
        (0.62 , [ 6.29146, 3.20893, 0.298839, 0.0148154, 0. ], [ 0.97823, 0.474444, 0.146615, 0.0262425, 0.00197332, ]),
        #(0.62 , [ 6.21467, 3.25855, 0.32244, 0.0185049, 0.000209253,  ], [ 0.981889, 0.482007, 0.155073, 0.0312345, 0.00197332,  ]),
        (0.63 , [ 6.09398, 3.38172, 0.372265, 0.0254703, 0.00068227,  ], [ 1.00341, 0.487314, 0.161969, 0.0366079, 0.00408525,  ]),
        (0.64 , [ 5.99343, 3.48298, 0.421426, 0.0340169, 0.00150778,  ], [ 1.02426, 0.491832, 0.168388, 0.0422731, 0.0063184,  ]),
        (0.65 , [ 5.92929, 3.55914, 0.460104, 0.0412235, 0.00229928,  ], [ 1.04345, 0.493446, 0.171136, 0.0454849, 0.00760083,  ]),
        (0.7 , [ 5.80983, 3.81707, 0.588617, 0.0591976, 0.00360959,  ], [ 1.12831, 0.487538, 0.163206, 0.0427118, 0.00700954,  ]),
        (0.71 , [ 5.80384, 3.85246, 0.611158, 0.0627668, 0.0038732,  ], [ 1.14378, 0.485858, 0.161486, 0.0420891, 0.00686317,  ]),
        (0.75 , [ 5.78581, 3.98447, 0.700589, 0.0772783, 0.00495831,  ], [ 1.20435, 0.480685, 0.155308, 0.0396603, 0.00628884,  ]),
        #(0.8 , [ 5.80548, 4.10613, 0.802727, 0.0961115, 0.00646245,  ], [ 1.27352, 0.474151, 0.148343, 0.0369821, 0.00567392,  ]),
        ]),

        (0.71, 1.5, [
        # sersicsA
        # fit for 0.71 is from above, with last variance from here
        (0.71 , [ 5.80384, 3.85246, 0.611158, 0.0627668, 0.0038732, 0. ], [ 1.14378, 0.485858, 0.161486, 0.0420891, 0.00686317, 0.000125639, ]),
        #(0.71 , [ 5.79976, 3.85582, 0.611454, 0.0631766, 0.00401389, 9.72068e-06,  ], [ 1.1442, 0.486145, 0.161699, 0.0424391, 0.00709661, 0.000125639,  ]),
        (0.72 , [ 5.78449, 3.89063, 0.64027, 0.0693474, 0.00482271, 4.7711e-05,  ], [ 1.16019, 0.486018, 0.161901, 0.0432836, 0.00770569, 0.000360731,  ]),
        (0.73 , [ 5.76622, 3.92465, 0.670321, 0.0771626, 0.00617073, 0.000149421,  ], [ 1.17644, 0.4865, 0.162878, 0.0450234, 0.00889761, 0.000772094,  ]),
        (0.74 , [ 5.73834, 3.95798, 0.706138, 0.0881532, 0.00842185, 0.000385661,  ], [ 1.19343, 0.488511, 0.165609, 0.048145, 0.0108527, 0.00140575,  ]),
        (0.75 , [ 5.74038, 3.98136, 0.726531, 0.0942424, 0.0102561, 0.000673631,  ], [ 1.20789, 0.487188, 0.1649, 0.0492449, 0.0122331, 0.00192144,  ]),
        (0.8 , [ 5.75213, 4.09637, 0.834928, 0.119308, 0.0137876, 0.000939471,  ], [ 1.27836, 0.482225, 0.159263, 0.0470525, 0.0114908, 0.00174917,  ]),
        (0.85 , [ 5.79382, 4.18044, 0.93266, 0.145078, 0.0176549, 0.00124088,  ], [ 1.34231, 0.477026, 0.153978, 0.0449401, 0.0107664, 0.00158922,  ]),
        (0.9 , [ 5.85237, 4.24456, 1.02239, 0.171488, 0.0218791, 0.00158456,  ], [ 1.40053, 0.471837, 0.149149, 0.0429814, 0.0101053, 0.00144948,  ]),
        (0.95 , [ 5.9154, 4.29436, 1.10933, 0.200379, 0.0267765, 0.00199386,  ], [ 1.4543, 0.467524, 0.145402, 0.0414866, 0.00958089, 0.0013368,  ]),
        (1. , [ 5.99798, 4.33895, 1.17948, 0.223347, 0.0308197, 0.00235229,  ], [ 1.50163, 0.460978, 0.139974, 0.039152, 0.00885129, 0.00120215,  ]),
        # sersics15
        #(1 , [ 5.99471, 4.33873, 1.18173, 0.224466, 0.0310354, 0.00236898,  ], [ 1.50212, 0.461467, 0.14033, 0.0393117, 0.00889359, 0.00120778,  ]),
        (1.1 , [ 5.83586, 4.57038, 1.46335, 0.317796, 0.0476365, 0.00386133,  ], [ 1.67005, 0.487971, 0.14593, 0.0400137, 0.00876429, 0.001151,  ]),
        (1.2 , [ 5.73216, 4.75622, 1.71514, 0.410113, 0.0656856, 0.00563594,  ], [ 1.83935, 0.5119, 0.148904, 0.0396002, 0.00839362, 0.0010718,  ]),
        (1.3 , [ 5.66968, 4.91188, 1.93948, 0.499674, 0.0846701, 0.00766234,  ], [ 2.00901, 0.53331, 0.150175, 0.0386185, 0.00792685, 0.000989366,  ]),
        (1.4 , [ 5.64047, 5.05007, 2.13639, 0.5838, 0.104037, 0.00990716,  ], [ 2.17789, 0.551721, 0.149915, 0.0372631, 0.00742588, 0.000910892,  ]),
        (1.5 , [ 5.64124, 5.1737, 2.30778, 0.662012, 0.12352, 0.0123547,  ], [ 2.34334, 0.566734, 0.148472, 0.035715, 0.00693027, 0.000839978,  ]),
        #(1.51 , [ 5.64275, 5.18569, 2.32359, 0.669356, 0.12542, 0.0126034,  ], [ 2.35968, 0.568033, 0.148253, 0.0355445, 0.00687945, 0.000833015,  ]),
        (1.55 , [ 5.65049, 5.2323, 2.38519, 0.698456, 0.133083, 0.0136284,  ], [ 2.42472, 0.573034, 0.14735, 0.0348749, 0.00668341, 0.000806695,  ]),        
        ]),

        (1.5, 3.0, [
        # sersics3 -- 1.5 is from above, with last component from here.
        (1.5 , [ 5.64124, 5.1737, 2.30778, 0.662012, 0.12352, 0.0123547, 0. ], [ 2.34334, 0.566734, 0.148472, 0.035715, 0.00693027, 0.000839978, 1.27404e-07 ]),
        #(1.5 , [ 5.64168, 5.17397, 2.30736, 0.66175, 0.12347, 0.0123548, 6.38226e-07,  ], [ 2.34321, 0.566637, 0.148426, 0.0357024, 0.00692878, 0.000840288, 1.27404e-07,  ]),
        (1.6 , [ 5.61537, 5.2611, 2.48815, 0.765994, 0.15635, 0.018094, 0.00024499,  ], [ 2.52291, 0.589568, 0.151672, 0.0363008, 0.00717518, 0.000947623, 4.35091e-05,  ]),
        (1.7 , [ 5.69233, 5.38532, 2.59845, 0.81004, 0.165966, 0.018716, 6.74404e-05,  ], [ 2.66711, 0.591014, 0.144734, 0.032994, 0.00619227, 0.00076251, 1.64929e-05,  ]),
        (1.8 , [ 5.4949, 5.33007, 2.85122, 1.0466, 0.272322, 0.0468646, 0.00363625,  ], [ 2.93426, 0.659773, 0.170579, 0.0427035, 0.0094335, 0.001642, 0.000200853,  ]),
        (1.9 , [ 5.65598, 5.49575, 2.90782, 1.0314, 0.252496, 0.0382267, 0.00159733,  ], [ 3.04492, 0.640408, 0.153817, 0.0354658, 0.00708065, 0.0010568, 9.93631e-05,  ]),

        (2.0 , [ 5.41041, 5.36575, 3.14515, 1.31485, 0.406678, 0.0902769, 0.0114168,  ], [ 3.37063, 0.732207, 0.188988, 0.0485424, 0.0114482, 0.00228187, 0.000330023,  ]),
        (2.1 , [ 5.43605, 5.43024, 3.25475, 1.39479, 0.444065, 0.102018, 0.0133629,  ], [ 3.56316, 0.75047, 0.188957, 0.0475758, 0.0110707, 0.00219774, 0.000322841,  ]),
        (2.3 , [ 5.50695, 5.5516, 3.44571, 1.54199, 0.517969, 0.126929, 0.017716,  ], [ 3.94332, 0.782323, 0.188001, 0.0456873, 0.0103958, 0.00205463, 0.00031153,  ]),
        (2.5 , [ 5.59652, 5.66277, 3.6078, 1.67472, 0.59009, 0.153297, 0.0226131,  ], [ 4.31446, 0.808878, 0.186254, 0.0438849, 0.00980639, 0.0019366, 0.000303158,  ]),
        (2.7 , [ 5.69708, 5.76403, 3.74798, 1.79608, 0.66079, 0.180995, 0.0280086,  ], [ 4.67575, 0.831388, 0.184166, 0.0422426, 0.00930319, 0.00184046, 0.00029719,  ]),
        (3.0 , [ 5.85922, 5.89892, 3.92624, 1.96084, 0.764378, 0.224729, 0.0369456,  ], [ 5.19717, 0.85919, 0.180844, 0.0401082, 0.00869111, 0.00173002, 0.00029187,  ]),
        (3.1 , [ 5.91409, 5.9398, 3.97918, 2.01211, 0.798292, 0.239767, 0.040112,  ], [ 5.36623, 0.867383, 0.179783, 0.0394844, 0.00851896, 0.0017002, 0.000290821,  ]),
        #(3.2 , [ 5.96931, 5.97853, 4.02915, 2.06163, 0.831951, 0.255044, 0.0433694,  ], [ 5.53217, 0.875044, 0.178751, 0.038903, 0.00836214, 0.00167365, 0.000290112,  ]),
        ]),

        (3.0, 6.3, [
        # sersics4
        # 3.0 is from above, last component from here.
        (3.0 , [ 5.85922, 5.89892, 3.92624, 1.96084, 0.764378, 0.224729, 0.0369456, 0. ], [ 5.19717, 0.85919, 0.180844, 0.0401082, 0.00869111, 0.00173002, 0.00029187, 1.2738e-07, ]),
        #(3 , [ 5.85846, 5.89849, 3.92653, 1.96133, 0.764712, 0.224883, 0.0369991, 1.68316e-06,  ], [ 5.19808, 0.859469, 0.180928, 0.0401323, 0.00869748, 0.00173158, 0.000292223, 1.2738e-07,  ]),
        (3.1 , [ 5.89261, 5.92745, 3.98572, 2.02531, 0.808698, 0.245297, 0.0422136, 0.000112894,  ], [ 5.39377, 0.875448, 0.182264, 0.0402242, 0.00872866, 0.0017551, 0.000303572, 1.19031e-05,  ]),
        (3.2 , [ 5.78724, 5.86899, 4.07897, 2.17279, 0.923106, 0.306152, 0.0641956, 0.00213177,  ], [ 5.78016, 0.947204, 0.200892, 0.0455095, 0.0102528, 0.00217963, 0.000413628, 5.93462e-05,  ]),
        (3.3 , [ 5.47036, 5.62078, 4.1782, 2.44848, 1.17503, 0.461173, 0.135215, 0.0191445,  ], [ 6.50002, 1.13024, 0.257658, 0.0632557, 0.0156834, 0.00377045, 0.000851159, 0.000178873,  ]),
        (3.4 , [ 5.4215, 5.571, 4.22317, 2.551, 1.27341, 0.527162, 0.168619, 0.0282894,  ], [ 6.86724, 1.19689, 0.275578, 0.0686685, 0.0173948, 0.00431013, 0.00100836, 0.000212777,  ]),
        (3.5 , [ 5.44946, 5.59631, 4.2688, 2.60459, 1.31816, 0.554882, 0.180674, 0.0307001,  ], [ 7.12481, 1.22187, 0.278094, 0.0687613, 0.0173456, 0.00429558, 0.00100893, 0.000214998,  ]),
        
        # sersics6
        #(2.70 , [ 5.23741, 5.35015, 3.81887, 2.12106, 0.948976, 0.344098, 0.095497, 0.0145339,  ], [ 5.1295, 1.01952, 0.259567, 0.0695757, 0.0184349, 0.00463351, 0.00105641, 0.000204265,  ]),
        #(3.00 , [ 5.30489, 5.44742, 4.01317, 2.32324, 1.0956, 0.423048, 0.125619, 0.0200763,  ], [ 5.86908, 1.10014, 0.267615, 0.0693436, 0.0179822, 0.00448395, 0.001033, 0.000207863,  ]),
        #(3.50 , [ 5.44891, 5.59587, 4.26884, 2.60502, 1.31856, 0.555054, 0.180707, 0.0307064,  ], [ 7.12575, 1.22218, 0.278196, 0.0687918, 0.0173527, 0.00429675, 0.00100908, 0.000215022,  ]),
        (4.00 , [ 5.60921, 5.72377, 4.46477, 2.83675, 1.51974, 0.686093, 0.240192, 0.0426452,  ], [ 8.40275, 1.33355, 0.287321, 0.0685146, 0.0169499, 0.00418865, 0.00100242, 0.000223784,  ]),
        (4.50 , [ 6.00513, 5.81635, 4.51757, 2.92305, 1.62101, 0.766762, 0.282114, 0.0516328,  ], [ 8.54868, 1.27631, 0.26422, 0.0613012, 0.014927, 0.00367536, 0.000889046, 0.000204293,  ]),
        (5.00 , [ 6.34001, 5.893, 4.56595, 3.00318, 1.71658, 0.844897, 0.323944, 0.0607105,  ], [ 8.64798, 1.22663, 0.245659, 0.0557539, 0.0134131, 0.00329657, 0.000805513, 0.000189682,  ]),
        (5.50 , [ 6.62811, 5.95913, 4.61021, 3.07655, 1.80515, 0.919469, 0.365005, 0.0697116,  ], [ 8.71157, 1.18229, 0.230188, 0.0513001, 0.0122278, 0.0030045, 0.000741307, 0.000178303,  ]),
        (6.00 , [ 6.87598, 6.01584, 4.65195, 3.14545, 1.88867, 0.990882, 0.404981, 0.0785137,  ], [ 8.75587, 1.14398, 0.217394, 0.0477077, 0.0112856, 0.00277359, 0.000690429, 0.000169168,  ]),
        
        # sersics7
        (6.10 , [ 6.91899, 6.02455, 4.66025, 3.16021, 1.90634, 1.00582, 0.413331, 0.0803644,  ], [ 8.76912, 1.13837, 0.21546, 0.0471597, 0.01114, 0.00273729, 0.000682265, 0.000167666,  ]),
        (6.20 , [ 6.96467, 6.03488, 4.66744, 3.17248, 1.92163, 1.01935, 0.421289, 0.0821744,  ], [ 8.77167, 1.13077, 0.213104, 0.0465242, 0.0109783, 0.00269904, 0.000674078, 0.000166197,  ]),
        (6.30 , [ 7.00941, 6.04646, 4.67528, 3.1842, 1.93614, 1.03217, 0.428476, 0.0837091,  ], [ 8.77363, 1.12301, 0.210661, 0.0458606, 0.0108084, 0.00265766, 0.000664904, 0.000164538,  ]),
        ]),

        ]
        self.orig_fits = fits

        # Core flux omitted by softening.
        # specifically, these values are  1. - soft/true
        self.cores = [
            (1.00, 0.000000),
            (1.25, 0.000070),
            (1.50, 0.000202),
            (1.75, 0.000425),
            (2.00, 0.000769),
            (2.25, 0.001262),
            (2.50, 0.001928),
            (2.75, 0.002788),
            (3.00, 0.003853),
            (3.25, 0.005130),
            (3.50, 0.006622),
            (3.75, 0.008325),
            (4.00, 0.010233),
            (4.25, 0.011748),
            (4.50, 0.013358),
            (4.75, 0.015057),
            (5.00, 0.016839),
            (5.25, 0.018698),
            (5.50, 0.020628),
            (5.75, 0.022625),
            (6.00, 0.024683),
            (6.25, 0.026796),
            (6.50, 0.028961),
        ]
        self.core_func = InterpolatedUnivariateSpline([s for s,c in self.cores],
                                                      [c for s,c in self.cores], k=3)

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
        else:
            assert(len(matches) == 1)
            lo,hi,amp_funcs,logvar_funcs = matches[0]
            amps = np.array([f(sindex) for f in amp_funcs])
            amps /= amps.sum()
            varr = np.exp(np.array([f(sindex) for f in logvar_funcs]))

        # Core
        if sindex > 1.:
            core = self.core_func(sindex)
            amps *= (1. - core) / amps.sum()
            amps = np.append(amps, core)
            varr = np.append(varr, 0.)

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

    def __repr__(self):
        return super(SersicGalaxy, self).__repr__().replace(
            ')', ', sersicindex=%.3f)' % self.sersicindex.val)

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

    def getParamDerivatives(self, img, modelMask=None, **kwargs):
        # superclass produces derivatives wrt pos, brightness, and shape.
        derivs = super(SersicGalaxy, self).getParamDerivatives(
            img, modelMask=modelMask, **kwargs)

        pos0 = self.getPosition()
        (px0, py0) = img.getWcs().positionToPixel(pos0, self)
        patch0 = self.getUnitFluxModelPatch(img, px=px0, py=py0,
                                            modelMask=modelMask, **kwargs)
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
                    img, px=px0, py=py0, modelMask=modelMask, **kwargs)
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

    def getDerivativeShearedProfiles(self, img, px, py):
        # superclass produces derivatives wrt shape
        derivs = super().getDerivativeShearedProfiles(img, px, py)
        # Returns a list of sheared profiles that will be needed to compute
        # derivatives for this source; this is assumed in addition to the
        # sheared profile at the current parameter settings.
        if self.isParamThawed('sersicindex'):
            steps = self.sersicindex.getStepSizes()
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
                pro = self._getShearedProfile(img, px, py)
                self.sersicindex.setParam(i, oldval)
                derivs.append(('sersicindex.'+inames[i], pro, step))
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
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        p = SersicMixture.getProfile(1.)

        for lo,hi,comps in SersicMixture.singleton.orig_fits:
            for v,amps,vars in comps:
                if v >= lo and v < hi:
                    amps /= np.sum(amps)
                    ax1.plot(v + np.zeros_like(amps), amps, 'o', mec='r', mfc='none', ms=5)
                    ax2.plot(v + np.zeros_like(vars), vars, 'o', mec='r', mfc='none', ms=5)

        from tractor.mixture_profiles import dev_amp,dev_var,exp_amp,exp_var
        ax1.plot([4.]*len(dev_amp), dev_amp, 'o', mec='b', mfc='none', ms=5)
        ax2.plot([4.]*len(dev_var), dev_var, 'o', mec='b', mfc='none', ms=5)
        ax1.plot([1.]*len(exp_amp), exp_amp, 'o', mec='b', mfc='none', ms=5)
        ax2.plot([1.]*len(exp_var), exp_var, 'o', mec='b', mfc='none', ms=5)
                    
        for s in sersics:
            mix = SersicMixture.getProfile(s)
            ax1.plot(s+np.zeros_like(mix.amp), mix.amp, 'k.', ms=1)
            ax2.plot(s+np.zeros_like(mix.amp), mix.var[:,0,0], 'k.', ms=1)

        #plt.xscale('log')
        ax1.set_yscale('symlog', linthreshy=1e-3)
        ax1.set_xlabel('Sersic index')
        ax1.set_ylabel('Mixture amplitudes')
        #plt.xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Sersic index')
        ax2.set_ylabel('Mixture variances')
        plt.savefig('mix.png')

        xt = [0.3, 0.5, 0.7, 1, 2, 3, 4, 5, 6]
        #plt.subplot(1,2,1)
        #plt.xscale('log')
        #plt.xticks(xt)
        #plt.subplot(1,2,2)
        #plt.xscale('log')
        #plt.xticks(xt)
        ax1.set_xscale('log')
        ax1.set_xticks(xt)
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_xscale('log')
        ax2.set_xticks(xt)
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.savefig('mix2.png')
        print('Wrote mix2.png')
        #sys.exit(0)
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

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

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline

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
        # 0.3 to 0.4+: 3 components
        (0.3, 0.42, [
            # amp prior std=2
            # (with amp prior std=1 -- the two positive components become degenerate)
            (0.30 , [ 28.2624, 8.86967, -29.3828,  ], [ 0.410382, 0.226964, 0.296305,  ]),
            (0.31 , [ 26.5302, 7.98289, -26.6988,  ], [ 0.420989, 0.230867, 0.301094,  ]),
            (0.32 , [ 25.164, 6.91623, -24.2007,  ], [ 0.431515, 0.234492, 0.30735,  ]),
            (0.33 , [ 23.7372, 6.12669, -21.9187,  ], [ 0.442811, 0.23908, 0.313087,  ]),
            (0.34 , [ 22.2576, 5.28883, -19.5351,  ], [ 0.455131, 0.243431, 0.318959,  ]),
            (0.35 , [ 21.0483, 4.47032, -17.4412,  ], [ 0.46741, 0.248201, 0.326043,  ]),
            (0.36 , [ 19.7558, 3.89, -15.502,  ], [ 0.480764, 0.254386, 0.332208,  ]),
            (0.37 , [ 18.6861, 2.97227, -13.448,  ], [ 0.494319, 0.258367, 0.341254,  ]),
            (0.38 , [ 17.6259, 2.53168, -11.8807,  ], [ 0.508462, 0.266862, 0.348707,  ]),
            (0.39 , [ 16.5312, 2.10138, -10.2891,  ], [ 0.523787, 0.276232, 0.356039,  ]),
            (0.40 , [ 15.4034, 1.44981, -8.44278,  ], [ 0.540655, 0.282509, 0.36455,  ]),
            (0.41 , [ 14.4519, 0.998542, -6.97335,  ], [ 0.557467, 0.294148, 0.373739, ]),
            (0.42 , [ 13.7357, 1.49789, -6.69082,  ], [ 0.573105, 0.334529, 0.380582,  ]),
            ]),
        # 0.4 to 0.6: 2 components
        (0.4, 0.6, [
            (0.40 , [ 17.4548, -9.03653,  ], [ 0.538118, 0.410483,  ]),
            (0.41 , [ 15.7532, -7.27249,  ], [ 0.553999, 0.408682,  ]),
            (0.42 , [ 14.0443, -5.49963,  ], [ 0.573098, 0.402914,  ]),
            (0.43 , [ 12.5725, -3.96256,  ], [ 0.594539, 0.393216,  ]),
            (0.44 , [ 11.4282, -2.75186,  ], [ 0.617041, 0.380054,  ]),
            (0.45 , [ 10.6572, -1.91458,  ], [ 0.638009, 0.367307,  ]),
            (0.46 , [ 10.1211, -1.31265,  ], [ 0.657343, 0.355383,  ]),
            (0.47 , [ 9.72809, -0.854359,  ], [ 0.675384, 0.343458,  ]),
            (0.48 , [ 9.44568, -0.507529,  ], [ 0.691799, 0.333334,  ]),
            (0.49 , [ 9.22775, -0.225867,  ], [ 0.707194, 0.322326,  ]),
            (0.50 , [ 9.06467, 0.,], [ 0.721342, 0.316688, ]),
            (0.51 , [ 8.92798, 0.198969,  ], [ 0.735018, 0.311148,  ]),
            (0.52 , [ 8.82442, 0.363841,  ], [ 0.747652, 0.301851,  ]),
            (0.53 , [ 8.7428, 0.505937,  ], [ 0.759522, 0.293898,  ]),
            (0.54 , [ 8.67782, 0.630592,  ], [ 0.770736, 0.286703,  ]),
            (0.55 , [ 8.62648, 0.740827,  ], [ 0.781334, 0.279961,  ]),
            (0.56 , [ 8.58635, 0.83901,  ], [ 0.791344, 0.273563,  ]),
            (0.57 , [ 8.55541, 0.927238,  ], [ 0.800819, 0.267478,  ]),
            (0.58 , [ 8.53217, 1.00695,  ], [ 0.809781, 0.261658,  ]),
            (0.59 , [ 8.51543, 1.0794,  ], [ 0.818263, 0.256077,  ]),
            (0.60 , [ 8.50421, 1.14553,  ], [ 0.826285, 0.250709,  ]),
            ]),
        # 0.6+ to 0.75+: 4 components
        (0.57, 0.76, [
            (0.57 , [ 6.4488, 2.83314, 0.218854, 0.00998291,  ], [ 0.892224, 0.508341, 0.173774, 0.0330768,  ]),
            (0.58 , [ 6.49729, 2.84447, 0.220223, 0.0101379,  ], [ 0.906864, 0.492428, 0.161543, 0.0301401,  ]),
            (0.60 , [ 6.39787, 3.02775, 0.256118, 0.0121954,  ], [ 0.942441, 0.480798, 0.152128, 0.027722,  ]),
            (0.65 , [ 6.17247, 3.43505, 0.364078, 0.0192057,  ], [ 1.03, 0.467534, 0.140271, 0.0245156,  ]),
            (0.70 , [ 6.07509, 3.70771, 0.466221, 0.0270969,  ], [ 1.10965, 0.457512, 0.131159, 0.0220774,  ]),
            (0.75 , [ 6.06411, 3.89388, 0.556595, 0.035217,  ], [ 1.18078, 0.447339, 0.122735, 0.0199082,  ]),
            (0.76 , [ 6.06908, 3.92404, 0.573212, 0.0368273,  ], [ 1.19402, 0.445236, 0.121101, 0.0194962,  ]),
            ]),
        # 0.75 to 1.5+: 6 components
        (0.75, 1.55, [
            (0.75 , [ 5.73017, 3.98402, 0.732308, 0.0959582, 0.0104227, 0.000681892,  ], [ 1.20875, 0.488435, 0.166055, 0.0497645, 0.0123354, 0.0019356,  ]),
            (0.80 , [ 5.75014, 4.09764, 0.836108, 0.119013, 0.0136534, 0.00092986,  ], [ 1.27856, 0.482428, 0.159252, 0.0468932, 0.0114183, 0.00173846,  ]),
            (0.85 , [ 5.79392, 4.18038, 0.932621, 0.145083, 0.0176554, 0.00124148,  ], [ 1.3423, 0.477017, 0.153977, 0.0449414, 0.0107673, 0.00158981,  ]),
            (0.90 , [ 5.85235, 4.24457, 1.0224, 0.171489, 0.0218797, 0.00158463,  ], [ 1.40054, 0.47184, 0.149151, 0.0429818, 0.0101055, 0.00144954,  ]),
            (0.95 , [ 5.92083, 4.29582, 1.10511, 0.198049, 0.0263517, 0.00196063,  ], [ 1.45361, 0.466645, 0.144623, 0.0411061, 0.00947885, 0.00132211,  ]),
            (1.00 , [ 5.99709, 4.33914, 1.1799, 0.223597, 0.030886, 0.00235662,  ], [ 1.50178, 0.461093, 0.140049, 0.0391944, 0.00886401, 0.00120353,  ]),
            (1.10 , [ 6.14939, 4.40829, 1.32037, 0.276163, 0.0406625, 0.00324514,  ], [ 1.58704, 0.451176, 0.132327, 0.0359017, 0.00780277, 0.00100656,  ]),
            (1.20 , [ 6.30415, 4.46728, 1.4427, 0.325106, 0.0503068, 0.00416872,  ], [ 1.65845, 0.440928, 0.124826, 0.0327071, 0.00682354, 0.00083812,  ]),
            (1.30 , [ 6.45448, 4.52149, 1.55158, 0.37086, 0.0598431, 0.00512976,  ], [ 1.71861, 0.430659, 0.117736, 0.0297692, 0.00596945, 0.00070013,  ]),
            (1.40 , [ 6.59957, 4.57433, 1.64865, 0.412699, 0.0689351, 0.00608707,  ], [ 1.76917, 0.420192, 0.110904, 0.0270357, 0.00521398, 0.000585129,  ]),
            (1.50 , [ 6.7389, 4.62721, 1.73564, 0.450737, 0.0774992, 0.00702291,  ], [ 1.81152, 0.409534, 0.104336, 0.0245173, 0.00455116, 0.000489465,  ]),
            (1.55 , [ 6.80691, 4.6537, 1.7755, 0.468334, 0.0815596, 0.00747883,  ], [ 1.82984, 0.40408, 0.101139, 0.023336, 0.00425152, 0.000447901,  ]),
        ]),
        # 1.5 to 3+: 7 components
        #(1.5, 3.2, [
        (1.5, 2.9, [
            (1.50 , [ 6.65995, 4.53932, 1.76681, 0.531461, 0.122459, 0.0197282, 0.00172856,  ], [ 1.8282, 0.426099, 0.118189, 0.0326576, 0.00798516, 0.00151833, 0.000165901,  ]),
            (2.00 , [ 7.18733, 4.74714, 2.1514, 0.742118, 0.189465, 0.0333001, 0.0031723,  ], [ 1.98023, 0.387572, 0.0943684, 0.0225626, 0.00474101, 0.000771006, 7.06243e-05,  ]),
            (2.50 , [ 7.65824, 4.99436, 2.39904, 0.865839, 0.230773, 0.0424127, 0.0042325,  ], [ 2.02971, 0.34077, 0.0718309, 0.0148612, 0.00271532, 0.000383605, 3.00222e-05,  ]),
            (3.00 , [ 8.14102, 5.24059, 2.53707, 0.922238, 0.249248, 0.0466651, 0.00474693,  ], [ 1.99426, 0.286884, 0.052207, 0.00942273, 0.00151295, 0.000187568, 1.26448e-05,  ]),
            (3.10 , [ 8.24161, 5.28628, 2.55339, 0.926919, 0.250588, 0.0469803, 0.0047839,  ], [ 1.97757, 0.275642, 0.0487155, 0.00856339, 0.00134121, 0.00016213, 1.06116e-05,  ]),
            (3.20 , [ 8.34394, 5.32945, 2.56627, 0.929876, 0.25136, 0.0471554, 0.00480307,  ], [ 1.95759, 0.264343, 0.0453888, 0.00777446, 0.00118816, 0.000140071, 8.90155e-06,  ]),
        ]),
        # 3 to 6: 8 components
        #(3., 6.3, [
        (2.7, 6.3, [
            (2.70 , [ 7.6403, 4.92511, 2.55191, 1.05911, 0.346176, 0.0867453, 0.0153226, 0.00147992,  ], [ 2.09543, 0.357192, 0.0802787, 0.0183499, 0.00390731, 0.000717937, 0.000100154, 7.53241e-06,  ]),
            (3.00 , [ 7.86558, 5.07114, 2.66611, 1.11404, 0.366817, 0.0927074, 0.0165192, 0.00160337,  ], [ 2.09647, 0.33148, 0.0689552, 0.0146165, 0.00289797, 0.000496432, 6.43457e-05, 4.42298e-06,  ]),
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

            amp_funcs = [InterpolatedUnivariateSpline(
                [ser for ser,amps,varr in grid],
                [amps[i] for ser,amps,varr in grid], k=spline_k)
                for i in range(K)]
            logvar_funcs = [InterpolatedUnivariateSpline(
                [ser for ser,amps,varr in grid],
                [np.log(varr[i]) for ser,amps,varr in grid], k=spline_k)
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
        self.lower = 0.3
        self.upper = 6.0


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
        isteps = self.sersicindex.getStepSizes()
        if not self.isParamFrozen('sersicindex'):
            inames = self.sersicindex.getParamNames()
            oldvals = self.sersicindex.getParams()
            for i, istep in enumerate(isteps):
                oldval = self.sersicindex.setParam(i, oldvals[i] + istep)
                patchx = self.getUnitFluxModelPatch(
                    img, px=px0, py=py0, modelMask=modelMask)
                self.sersicindex.setParam(i, oldval)
                if patchx is None:
                    print('patchx is None:')
                    print('  ', self)
                    print('  stepping galaxy sersicindex',
                          self.sersicindex.getParamNames()[i])
                    print('  stepped', isteps[i])
                    print('  to', self.sersicindex.getParams()[i])
                    derivs.append(None)

                dx = (patchx - patch0) * (counts / istep)
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

    H,W = 100,100
    tim = Image(data=np.zeros((H,W)), inverr=np.ones((H,W)),
                psf=GaussianMixturePSF(1., 0., 0., 2., 2., 0.))
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

    sersics = np.logspace(np.log10(0.3001), np.log10(6.19), 200)
    print('Sersic indices:', sersics)
    
    ps = PlotSequence('ser', format='%03i')

    plt.figure(figsize=(8,4))
    
    pixel_scale = 1.0
    gs_psf = galsim.Gaussian(flux=1., sigma=np.sqrt(2.))
    
    for i,si in enumerate(sersics):
        gs_gal = galsim.Sersic(si, half_light_radius=re)
        gs_gal = gs_gal.shift(0.5, 0.5)
        gs_final = galsim.Convolve([gs_gal, gs_psf])
        gs_image = gs_final.drawImage(scale=pixel_scale)
        gs_image = gs_image.array
        iy,ix = np.unravel_index(np.argmax(gs_image), gs_image.shape)
        gs_image = gs_image[iy-cy:, ix-cx:][:H,:W]

        gs_gal = galsim.Sersic(si+serstep, half_light_radius=re)
        gs_gal = gs_gal.shift(0.5, 0.5)
        gs_final = galsim.Convolve([gs_gal, gs_psf])
        gs_image2 = gs_final.drawImage(scale=pixel_scale)
        gs_image2 = gs_image2.array
        iy,ix = np.unravel_index(np.argmax(gs_image2), gs_image2.shape)
        gs_image2 = gs_image2[iy-cy:, ix-cx:][:H,:W]

        ds = (gs_image2 - gs_image) / (serstep)
    
        gal.sersicindex.setValue(si)
        derivs = gal.getParamDerivatives(tim)
        assert(len(derivs) == 1)
        deriv = derivs[0]
        dd = np.zeros((H,W))
        deriv.addTo(dd)

        mod = tr.getModelImage(0)

        N = gal.getProfile().K
        
        plt.clf()

        plt.subplot(1,2,1)

        plt.plot(gs_image[cy,:], label='galsim')
        plt.plot(mod[cy,:], label='tractor')
        plt.legend()
        plt.title('Model')
        #plt.title('Model: %.4f' % si)
        #ps.savefig()
        
        #plt.clf()
        plt.subplot(1,2,2)
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
        d1 = dd[:,50]
        d2 = ds[:,50]
        plt.plot(d2, label='galsim')
        Y = plt.ylim()
        plt.plot(d1, label='tractor')
        plt.ylim(*Y)
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

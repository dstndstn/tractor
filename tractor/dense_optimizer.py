from __future__ import print_function
import numpy as np
from astrometry.util.ttime import Time
from tractor.engine import logverb, isverbose, logmsg
from tractor.optimize import Optimizer
from tractor.lsqr_optimizer import LsqrOptimizer
from tractor.constrained_optimizer import ConstrainedOptimizer

from numpy.linalg import lstsq

# from astrometry.util.plotutils import PlotSequence
# ps = PlotSequence('opt')
# import pylab as plt

        
class ConstrainedDenseOptimizer(ConstrainedOptimizer):

    # TODO: implement:
    #   _optimized_forcedphot_core

    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None, variance=False,
                           shared_params=True,
                           get_A_matrix=False):

        # I don't want to deal with this right now!
        assert(shared_params == False)
        assert(scales_only == False)
        assert(damp == 0.)
        assert(variance == False)
        #assert(priors == False)

        # Returns: numpy array containing update direction.
        # If *variance* is True, return    (update,variance)
        # If *get_A_matrix* is True, returns the sparse matrix of derivatives.
        # If *scale_only* is True, return column scalings
        # In cases of an empty matrix, returns the list []
        #
        # allderivs: [
        #    (param0:)  [  (deriv, img), (deriv, img), ... ],
        #    (param1:)  [],
        #    (param2:)  [  (deriv, img), ],
        # ]
        # The "img"s may repeat
        # "deriv" are Patch objects.

        # Each position in the "allderivs" array corresponds to a
        # model parameter that we are optimizing

        # We want to minimize:
        #   || chi + (d(chi)/d(params)) * dparams ||^2
        # So  b = chi
        #     A = -d(chi)/d(params)
        #     x = dparams
        #
        # chi = (data - model) / std = (data - model) * inverr
        # derivs = d(model)/d(param)
        # A matrix = -d(chi)/d(param)
        #          = + (derivs) * inverr

        # Parameters to optimize go in the columns of matrix A
        # Pixels go in the rows.

        #print('Getting update direction:')
        #tractor.printThawedParams()
        
        # Keep track of row offsets for each image.
        imgoffs = {}
        Npixels = 0
        for param in allderivs:
            for deriv, img in param:
                if img in imgoffs:
                    continue
                npix = img.numberOfPixels()
                imgoffs[img] = (Npixels, npix)
                Npixels += npix
        Ncols = len(allderivs)

        Npriors = 0
        if priors:
            ''' getLogPriorDerivatives()
            Returns a "chi-like" approximation to the log-prior at the
            current parameter values.

            This will go into the least-squares fitting (each term in the
            prior acts like an extra "pixel" in the fit).

            Returns (rowA, colA, valA, pb, mub), where:
            rowA, colA, valA: describe a sparse matrix pA
            pA: has shape N x numberOfParams
            pb: has shape N
            mub: has shape N
            rowA: list of iterables of ints
            colA: list of iterables of ints
            valA: list of iterables of floats
            pb:   list of iterables of floats
            mub:  list of iterables of floats

            where "N" is the number of "pseudo-pixels" or Gaussian terms.
            "pA" will be appended to the least-squares "A" matrix, and
            "pb" will be appended to the least-squares "b" vector, and the
            least-squares problem is minimizing
            || A * (delta-params) - b ||^2

            We also require *mub*, which is like "pb" but not shifted
            relative to the current parameter values; ie, it's the mean of
            the Gaussian.
            '''
            priorVals = tractor.getLogPriorDerivatives()
            #print('Priors:', priorVals)
            if priorVals is not None:
                rA, cA, vA, pb, mub = priorVals
                Npriors = max(Npriors, max([1+max(r) for r in rA]))
                #print('Priors: rows', rA)
                #print('Cols:', cA)
                #print('Number of prior "pixels":', Npriors)
                #assert(False)
                # sprows.extend([ri + Nrows for ri in rA])
                # spcols.extend(cA)
                # spvals.extend([vi / colscales[ci] for vi, ci in zip(vA, cA)])
                # oldnrows = Nrows
                # nr = listmax(rA, -1) + 1
                # Nrows += nr
                # logverb('Nrows was %i, added %i rows of priors => %i' %
                #         (oldnrows, nr, Nrows))
                # # if len(cA) == 0:
                # #     Ncols = 0
                # # else:
                # #     Ncols = 1 + max(cA)
                # 
                # b = np.zeros(Nrows)
                # b[oldnrows:] = np.hstack(pb)

        Nrows = Npixels + Npriors
        #logmsg('Allocating', Nrows, 'x', Ncols, 'matrix for update direction')
        A = np.zeros((Nrows, Ncols), np.float32)
        # 'B' holds the chi values
        B = np.zeros(Nrows, np.float32)

        colscales = np.ones(Ncols)
        for col,param in enumerate(allderivs):
            scale2 = 0.
            for (deriv, img) in param:
                inverrs = img.getInvError()
                H,W = img.shape
                row0,npix = imgoffs[img]
                rows = slice(row0, row0+npix)

                # FIXME -- shortcut if deriv bounds == img bounds?
                dimg = np.zeros((H,W), np.float32)
                deriv.addTo(dimg)
                dimg *= inverrs
                dimg = dimg.flat
                if scale_columns:
                    # accumulate L2 norm
                    scale2 += np.dot(dimg, dimg)
                A[rows, col] = dimg
            if scale_columns:
                scale = np.sqrt(scale2)
                if scale > 0:
                    colscales[col] = scale
                    A[:,col] /= scale

        #print('Colscales:', colscales)
        if Npriors > 0:
            rA, cA, vA, pb, mub = priorVals
            #print('Priors: pb', pb, 'mub', mub)
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                if scale_columns:
                    A[Npixels + ri, ci] = vi / colscales[ci]
                else:
                    A[Npixels + ri, ci] = vi
                B[Npixels + ri] += bi
            del priorVals

        chimap = {}
        if chiImages is not None:
            for img, chi in zip(tractor.getImages(), chiImages):
                chimap[img] = chi
        # iterating this way avoids setting the elements more than once
        for img, (row0,npix) in imgoffs.items():
            chi = chimap.get(img, None)
            if chi is None:
                chi = tractor.getChiImage(img=img)
            chi = chi.flat
            #npix = len(chi)
            B[row0: row0 + npix] = chi
        del chi

        # X, resids, rank, singular_vals
        X,_,_,_ = lstsq(A, B, rcond=None)

        if False:
            Aold = super(ConstrainedDenseOptimizer, self).getUpdateDirection(
                tractor, allderivs, damp=damp, priors=priors,
                scale_columns=scale_columns,
                get_A_matrix=True, shared_params=False)
            Aold = Aold.toarray()
    
            print('Sparse A matrix:', Aold.shape)
            if Aold.shape == A.shape:
                print('Average relative difference in matrix elements:',
                      100. * np.mean(np.abs((A - Aold) / np.maximum(1e-6, Aold))), 'percent')
    
            plt.clf()
            plt.plot(Aold.ravel(), A.ravel(), 'b.')
            plt.xlabel('Old matrix element')
            plt.ylabel('New matrix element')
            plt.title('scale columns: %s, priors: %i' % (scale_columns, Npriors))
            ps.savefig()
    
            plt.clf()
            plt.plot(Aold.ravel(), A.ravel(), 'b.')
            plt.xscale('symlog', linthreshx=1e-6)
            plt.yscale('symlog', linthreshy=1e-6)
            plt.xlabel('Old matrix element')
            plt.ylabel('New matrix element')
            ps.savefig()

            # plt.clf()
            # plt.plot(Aold.ravel(), (A-Aold).ravel(), 'b.')
            # plt.xlabel('Old matrix element')
            # plt.ylabel('New-Old matrix element')
            # #plt.title('scale columns: %s, priors: %i' % (scale_columns, Npriors))
            # ps.savefig()
                
            plt.clf()
            #plt.plot(Aold.ravel(), A.ravel(), 'b.')
            reldif = ((A-Aold)/np.maximum(1e-6, Aold)).ravel()
            mx = reldif.max()
            plt.plot(Aold.ravel(), reldif, 'b.')
            plt.xlabel('Old matrix element')
            #plt.ylabel('New matrix element')
            plt.ylabel('Relative change in New matrix element')
            plt.ylim(-mx, mx)
            ps.savefig()
            
        del A
        del B

        if scale_columns:
            #X[colscales > 0] /= colscales[colscales > 0]
            X /= colscales

        if False:
            print('Returning:  ', X)
            Xold = super(ConstrainedDenseOptimizer, self).getUpdateDirection(
                tractor, allderivs, damp=damp, priors=priors,
                scale_columns=scale_columns, chiImages=chiImages, variance=variance,
                shared_params=False)
            print('COpt result:', Xold)
            print('Relative difference:', ', '.join(['%.1f' % d for d in 100.*((X - Xold) / np.maximum(1e-18, np.abs(Xold)))]), 'percent')

        return X



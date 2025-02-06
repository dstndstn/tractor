from __future__ import print_function
import numpy as np
from tractor.engine import logverb
from tractor.constrained_optimizer import ConstrainedOptimizer

from numpy.linalg import lstsq, LinAlgError

# from astrometry.util.plotutils import PlotSequence
# ps = PlotSequence('opt')
# import pylab as plt
        
class ConstrainedDenseOptimizer(ConstrainedOptimizer):

    # TODO: implement:
    #   _optimized_forcedphot_core

    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None,
                           variance=False,
                           shared_params=True,
                           get_A_matrix=False):

        #print('ConstrainedDenseOptimizer.getUpdateDirection: shared_params=', shared_params,
        #      'scales_only=', scales_only, 'damp', damp, 'variance', variance)
        if shared_params or scales_only or damp>0 or variance:
            raise RuntimeError('Not implemented')
        # I don't want to deal with this right now!
        assert(shared_params == False)
        assert(scales_only == False)
        assert(variance == False)
        assert(damp == 0.)

        # Returns: numpy array containing update direction.
        # If *variance* is True, return    (update,variance)
        # If *get_A_matrix* is True, returns the matrix of derivatives.
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

        Nrows = Npixels + Npriors
        if Nrows == 0:
            #print('ConstrainedDenseOptimizer.getUpdateDirection: Nrows = 0')
            return None
        #print('Allocating', Nrows, 'x', Ncols, 'matrix for update direction')
        A = np.zeros((Nrows, Ncols), np.float32)
        # 'B' holds the chi values
        B = np.zeros(Nrows, np.float32)

        colscales2 = np.ones(Ncols)
        for col,param in enumerate(allderivs):
            scale2 = 0.
            for (deriv, img) in param:
                if deriv.patch is None:
                    continue
                inverrs = img.getInvError()
                H,W = img.shape
                row0,npix = imgoffs[img]
                rows = slice(row0, row0+npix)

                # shortcut for deriv bounds == img bounds
                if deriv.x0 == 0 and deriv.y0 == 0 and deriv.patch.shape==(H,W):
                    dimg = (deriv.patch * inverrs).flat
                else:
                    dimg = np.zeros((H,W), np.float32)
                    deriv.addTo(dimg)
                    dimg *= inverrs
                    dimg = dimg.flat
                assert(np.all(np.isfinite(dimg)))
                #print('Derivative', col, 'in image', img, 'gave non-finite value!')
                #tractor.printThawedParams()
                if scale_columns:
                    # accumulate L2 norm
                    scale2 += np.dot(dimg, dimg)

                A[rows, col] = dimg
                del dimg
            if scale_columns:
                colscales2[col] = scale2
        print('colscales2:', colscales2)

        if Npriors > 0:
            rA, cA, vA, pb, mub = priorVals
            #print('Priors: pb', pb, 'mub', mub)
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                if scale_columns:
                    colscales2[ci] += np.dot(vi, vi)
                for rij,vij,bij in zip(ri, vi, bi):
                    A[Npixels + rij, ci] = vij
                    B[Npixels + rij] += bij
            del priorVals, rA, cA, vA, pb, mub

        if scale_columns:
            for col,scale2 in enumerate(colscales2):
                if scale2 > 0:
                    A[:,col] /= np.sqrt(scale2)
                else:
                    # Set to safe value...
                    colscales2[col] = 1.
            colscales = np.sqrt(colscales2)
            print('colscales:', colscales)

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

        try:
            X,_,_,_ = lstsq(A, B, rcond=None)
        except LinAlgError as e:
            print('Exception in lstsq:', e)
            from collections import Counter
            import traceback
            traceback.print_exc()
            if scale_columns:
                print('Column scales:', colscales)
            print('A finite:', Counter(np.isfinite(A.ravel())))
            print('B finite:', Counter(np.isfinite(B.ravel())))
            return None

        ## X, resids, rank, singular_vals
        #X,_,_,_ = lstsq(A, B, rcond=None)

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

        if not get_A_matrix:
            del A
            del B
        #del B

        if scale_columns:
            X /= colscales

        if False:
            print('Returning:  ', X)
            Xold = super(ConstrainedDenseOptimizer, self).getUpdateDirection(
                tractor, allderivs, damp=damp, priors=priors,
                scale_columns=scale_columns, chiImages=chiImages, variance=variance,
                shared_params=False)
            print('COpt result:', Xold)
            print('Relative difference:', ', '.join(['%.1f' % d for d in 100.*((X - Xold) / np.maximum(1e-18, np.abs(Xold)))]), 'percent')

        if not np.all(np.isfinite(X)):
            print('ConstrainedDenseOptimizer.getUpdateDirection: X not all finite!')
            print('X = ', X)
            return None

        if get_A_matrix:
            if scale_columns:
                A *= colscales[np.newaxis,:]
            return X,A,colscales,B

        return X



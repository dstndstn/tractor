from tractor.dense_optimizer import ConstrainedDenseOptimizer
import numpy as np
from numpy.linalg import lstsq, LinAlgError

class SmarterDenseOptimizer(ConstrainedDenseOptimizer):

    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None,
                           variance=False,
                           shared_params=True,
                           get_A_matrix=False):
        if shared_params or scales_only or damp>0 or variance:
            raise RuntimeError('Not implemented')
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


        # Keep track of active pixels in each image
        img_bounds = {}
        # which parameters actually have derivatives?
        # FIXME -- careful with how this interacts with priors!
        live_params = set()

        for iparam,derivs in enumerate(allderivs):
            if len(derivs) == 0:
                continue
            live_params.add(iparam)

            for deriv, img in derivs:
                print('deriv', deriv)
                dx0,dx1,dy0,dy1 = deriv.extent

                if img in img_bounds:
                    x0,x1,y0,y1 = img_bounds[img]
                    img_bounds[img] = min(dx0, x0), max(dx1, x1), min(dy0, y0), max(dy1, y1)
                else:
                    img_bounds[img] = dx0, dx1, dy0, dy1
        Ncols = len(live_params)

        # Where in the A & B arrays will the image pixels start?

        img_offsets = {}
        Npixels = 0
        for iparam,derivs in enumerate(allderivs):
            for deriv, img in derivs:
                if img in img_offsets:
                    continue
                x0,x1,y0,y1 = img_bounds[img]
                img_offsets[img] = Npixels
                Npixels += (x1-x0) * (y1-y0)

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
                print('Priors.  live_params:', live_params, ', cA:', cA)
                live_params.update(cA)
                print('live_params after:', live_params)
                Ncols = len(live_params)
                print('new Ncols:', Ncols)

        print('DenseOptimizer.getUpdateDirection : N params (cols) %i, N pix %i, N priors %i' %
              (Ncols, Npixels, Npriors))

        Nrows = Npixels + Npriors
        if Nrows == 0:
            #print('ConstrainedDenseOptimizer.getUpdateDirection: Nrows = 0')
            return None
        #print('Allocating', Nrows, 'x', Ncols, 'matrix for update direction')
        A = np.zeros((Nrows, Ncols), np.float32)
        # 'B' holds the chi values
        B = np.zeros(Nrows, np.float32)

        live_params = list(live_params)
        live_params.sort()

        column_map = dict([(c,i) for i,c in enumerate(live_params)])
        print('Column map:', column_map)

        colscales2 = np.ones(Ncols)
        for iparam,derivs in enumerate(allderivs):
            if len(derivs) == 0:
                continue
            col = column_map[iparam]
            scale2 = 0.
            for deriv, img in derivs:
                if deriv.patch is None:
                    continue
                inverrs = img.getInvError()

                dx0,dx1,dy0,dy1 = deriv.extent
                x0,x1,y0,y1 = img_bounds[img]
                print('deriv extent', dx0,dx1, dy0,dy1)
                print('image bounds', x0,x1,y0,y1)
                if x0 == dx0 and x1 == dx1 and y0 == dy0 and y1 == dy1:
                    rowstart = img_offsets[img]
                    print('row start:', rowstart)
                    print('col', col)
                    w = x1-x0
                    h = y1-y0
                    apix = (deriv.patch * inverrs[y0:y1, x0:x1])
                    A[rowstart:rowstart+w*h, col] = apix.flat
                    if scale_columns:
                        # accumulate L2 norm
                        scale2 += np.sum(apix**2)
                    del apix
                else:
                    assert(False)

            if scale_columns:
                colscales2[col] = scale2
        print('colscales2:', colscales2)

        if Npriors > 0:
            rA, cA, vA, pb, mub = priorVals
            #print('Priors: pb', pb, 'mub', mub)
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                print('prior: r,c', ri, ci, 'v', vi, 'b', bi)
                col = column_map[ci]
                if scale_columns:
                    colscales2[col] += np.sum(vi**2)
                for rij,vij,bij in zip(ri, vi, bi):
                    A[Npixels + rij, col] = vij
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

        for img,rowstart in img_offsets.items():
            chi = chimap.get(img, None)
            if chi is None:
                ### FIXME.... set tractor's modelMask???
                chi = tractor.getChiImage(img=img)
            x0,x1,y0,y1 = img_bounds[img]
            chi = chi[y0:y1, x0:x1]
            B[rowstart: rowstart + w*h] = chi.flat
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

        if not get_A_matrix:
            del A
            del B
        #del B

        if scale_columns:
            X /= colscales

        if not np.all(np.isfinite(X)):
            print('ConstrainedDenseOptimizer.getUpdateDirection: X not all finite!')
            print('X = ', X)
            return None

        # Expand x back out (undo the column_map)
        print('Expanding X: column_map =', column_map)
        print('X:', X)
        X_full = np.zeros(1+max(live_params))
        for c,i in column_map.items():
            X_full[c] = X[i]
        X = X_full
        print('-> X', X)

        if get_A_matrix:
            if scale_columns:
                A *= colscales[np.newaxis,:]
            # HACK
            # expand the colscales array too!
            c_full = np.zeros(len(X))
            for c,i in column_map.items():
                c_full[c] = colscales[i]
            colscales = c_full

            # Expand A matrix
            r,c = A.shape
            A_full = np.zeros((r, len(X)), np.float32)
            for c,i in column_map.items():
                A_full[:,c] = A[:,i]
            A = A_full

            return X,A,colscales,B

        return X

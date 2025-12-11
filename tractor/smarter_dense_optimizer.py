from tractor.constrained_optimizer import ConstrainedOptimizer
import numpy as np
from numpy.linalg import lstsq, LinAlgError
from tractor.utils import savetxt_cpu_append

class SmarterDenseOptimizer(ConstrainedOptimizer):

    def getPriorsHessianAndGradient(self, tr):
        priorVals = tr.getLogPriorDerivatives()
        if priorVals is None:
            return None

        # Get the total number of parameters from the tractor object
        Ntotal = tr.numberOfParams()
        rA, cA, vA, pb, _ = priorVals

        # Initialize the full Hessian and gradient matrices for the priors
        ATA_prior = np.zeros((Ntotal, Ntotal), np.float32)
        ATB_prior = np.zeros(Ntotal, np.float32)

        # The columns of the prior matrices correspond to the indices of the parameters.
        # We can directly use cA as indices into the full-sized matrices.
        for ri, ci, vi, bi in zip(rA, cA, vA, pb):
            # ci is the column index for the parameter with a prior
            col = ci
            for rij, vij, bij in zip(ri, vi, bi):
                # The prior's Hessian term is A.T @ A. In this case, for a single row,
                # this is just vij^2, and it goes in the (col, col) element of the matrix.
                ATA_prior[col, col] += vij * vij

                # The prior's gradient term is A.T @ B. For a single row,
                # this is vij * bij. It goes in the `col` element of the vector.
                ATB_prior[col] += vij * bij

        return ATA_prior, ATB_prior

    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True,
                           scales_only=False,
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

        # which parameters actually have derivatives (or priors)?
        Ncols_total = len(allderivs)
        cols_live = np.zeros(Ncols_total, bool)

        for iparam,derivs in enumerate(allderivs):
            if len(derivs) == 0:
                continue
            cols_live[iparam] = True
            for deriv, img in derivs:
                h,w = img.shape
                # FIXME - use_gpu arg!
                use_gpu = False
                if not deriv.clipTo(w, h, use_gpu=use_gpu):
                    print('No overlap between derivative and image.')
                    print('deriv extents:', deriv.extent, 'image size %i x %i' % (w,h))
                    continue
                dx0,dx1,dy0,dy1 = deriv.extent
                assert(type(dx0) is int)
                assert(type(dx1) is int)
                assert(type(dy0) is int)
                assert(type(dy1) is int)
                if img in img_bounds:
                    x0,x1,y0,y1 = img_bounds[img]
                    img_bounds[img] = min(dx0, x0), max(dx1, x1), min(dy0, y0), max(dy1, y1)
                else:
                    img_bounds[img] = dx0, dx1, dy0, dy1

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
            rowA: list of iterables of ints
            colA: list of ints
            valA: list of iterables of floats
            pb:   list of iterables of floats

            where "N" is the number of "pseudo-pixels" or Gaussian terms.
            "pA" will be appended to the least-squares "A" matrix, and
            "pb" will be appended to the least-squares "b" vector, and the
            least-squares problem is minimizing
            || A * (delta-params) - b ||^2
            '''
            priorVals = tractor.getLogPriorDerivatives()
            if priorVals is not None:
                rA, cA, vA, pb, _ = priorVals
                Npriors = max(Npriors, max([1+max(r) for r in rA]))
                cols_live[cA] = True

        # Where in the A & B arrays will the image pixels start?
        img_offsets = {}
        Npixels = 0
        for iparam,derivs in enumerate(allderivs):
            for deriv, img in derivs:
                if img in img_offsets:
                    continue
                x0,x1,y0,y1 = img_bounds[img]
                img_offsets[img] = Npixels
                # pixel coords can end up as int16; cast to int
                # to avoid overflow!
                Npixels += (int(x1)-int(x0)) * (int(y1)-int(y0))

        Ncols_live = np.sum(cols_live)

        Nrows = Npixels + Npriors
        if Nrows == 0:
            return None

        A = np.zeros((Nrows, Ncols_live), np.float32)
        # 'B' holds the chi values
        B = np.zeros(Nrows, np.float32)

        # Original -> live mapping
        inv_column_map = np.flatnonzero(cols_live)
        column_map = np.empty(Ncols_total, int)
        column_map[:] = -1
        column_map[inv_column_map] = np.arange(len(inv_column_map))
        #print('Column map:', column_map)
        #print('Inv column map:', inv_column_map)

        colscales = np.zeros(Ncols_live, np.float32)
        for iparam,derivs in enumerate(allderivs):
            if len(derivs) == 0:
                continue
            col = column_map[iparam]
            assert(col >= 0)
            scale = 0.
            for deriv, img in derivs:
                if deriv.patch is None:
                    continue
                inverrs = img.getInvError()

                dx0,dx1,dy0,dy1 = deriv.extent
                x0,x1,y0,y1 = img_bounds[img]

                # The derivative covers the whole area for this image.
                if x0 == dx0 and x1 == dx1 and y0 == dy0 and y1 == dy1:
                    rowstart = img_offsets[img]
                    w = x1-x0
                    h = y1-y0
                    apix = (deriv.patch * inverrs[y0:y1, x0:x1])
                    A[rowstart: rowstart+(h*w), col] = apix.flat
                    if scale_columns:
                        # accumulate L2 norm
                        scale += np.sum(apix**2)
                    del apix

                else:
                    # There are multiple modelMasks for this image
                    # (eg from multiple sources), so need to pad it out
                    print('multiple modelMasks (sources) for this image?')
                    print('\tderiv extent', dx0,dx1, dy0,dy1)
                    print('\timage bounds', x0,x1,y0,y1, img)
                    return None

            if scale_columns:
                colscales[col] = scale

        if Npriors > 0:
            rA, cA, vA, pb, _ = priorVals
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                col = column_map[ci]
                assert(col >= 0)
                if scale_columns:
                    # (note, np.dot works when vi is a list)
                    colscales[col] += np.dot(vi, vi)
                for rij,vij,bij in zip(ri, vi, bi):
                    A[Npixels + rij, col] = vij
                    B[Npixels + rij] += bij
            del priorVals, rA, cA, vA, pb, _

        if scale_columns:
            colscales = np.sqrt(colscales)
            for col,scale in enumerate(colscales):
                if scale > 0:
                    A[:,col] /= scale
                else:
                    # Set to safe value...
                    colscales[col] = 1.

        chimap = {}
        if chiImages is not None:
            for img, chi in zip(tractor.getImages(), chiImages):
                chimap[img] = chi

        for img,rowstart in img_offsets.items():
            chi = chimap.get(img, None)
            if chi is None:
                ### FIXME.... set tractor's modelMask???  Get only the pixel area we need?
                chi = tractor.getChiImage(img=img)
            x0,x1,y0,y1 = img_bounds[img]
            chi = chi[y0:y1, x0:x1]
            w = x1-x0
            h = y1-y0
            B[rowstart: rowstart + w*h] = chi.flat
            del chi

        try:
            #print('Smarter: cond', np.linalg.cond(A))
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

        if scale_columns:
            X /= colscales

        if not np.all(np.isfinite(X)):
            print('ConstrainedDenseOptimizer.getUpdateDirection: X not all finite!')
            print('X = ', X)
            return None

        # Expand x back out (undo the column_map)
        X_full = np.zeros(Ncols_total, np.float32)
        X_full[inv_column_map] = X

        if get_A_matrix:
            if scale_columns:
                A *= colscales[np.newaxis,:]
            # HACK
            # expand the colscales array too!
            c_full = np.zeros(Ncols_total, np.float32)
            c_full[inv_column_map] = colscales

            # Expand A matrix
            r,c = A.shape
            assert(r == Nrows)
            A_full = np.zeros((Nrows, Ncols_total), np.float32)
            A_full[:,inv_column_map] = A

            return X_full, A_full, c_full, B

        return X_full

from __future__ import print_function
import numpy as np
import cupy as cp
from cupy.sparse import csr_matrix as cpxr_matrix
#from cupy.sparse.linalg import lsqr as cplsqr
from cupy.sparse.linalg import lsmr as cplsmr # ADD THIS
from astrometry.util.ttime import Time
from tractor.engine import logverb, isverbose, logmsg
from tractor.optimize import Optimizer
from tractor.utils import listmax
from tractor.lsqr_optimizer import LsqrOptimizer
import time

glt = np.zeros(10)
gc = np.zeros(3, dtype=np.int32)

def listmax_gpu(X, default=0):
    # X is expected to be a list of CuPy arrays
    mx_vals = []
    for x in X:
        if len(x) > 0:
            mx_vals.append(cp.max(x))
    
    if len(mx_vals) == 0:
        return default
    # If mx_vals contains CuPy scalars, cp.max works
    return cp.max(cp.array(mx_vals)).item() # .item() to get Python scalar if needed

class GPULsqrOptimizer(LsqrOptimizer):
    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None, variance=False,
                           shared_params=True,
                           get_A_matrix=False):
        print ("GPULsqrOptimizer!")
        #print ("TRACTOR", type(tractor), tractor.getParams, tractor.setParams)
        t = time.time()
        gc[0] += 1

        if shared_params:
            # Assuming tractor.getParams() now returns CuPy arrays
            p0 = tractor.getParamsGPU()
            tractor.setParamsGPU(cp.arange(len(p0))) # Using cp.arange directly
            p1 = tractor.getParamsGPU()
            tractor.setParamsGPU(p0) # Restore original params
            
            # np.unique replaced by cp.unique, returns CuPy arrays
            U, I = cp.unique(p1, return_inverse=True)
            logverb(len(p0), 'params;', len(U), 'unique')
            paramindexmap = I # I is already a CuPy array
            
        # Build the sparse matrix of derivatives:
        sprows = [] # Will hold CuPy arrays of row indices
        spcols = [] # Will hold CuPy arrays of column indices (or just ints if single col)
        spvals = [] # Will hold CuPy arrays of values

        # Keep track of row offsets for each image.
        imgoffs = {}
        nextrow = 0
        for param in allderivs:
            for deriv, img in param:
                if img in imgoffs:
                    continue
                imgoffs[img] = nextrow
                nextrow += img.numberOfPixels()
        Nrows = nextrow
        del nextrow
        Ncols = len(allderivs)
        if variance:
            var = cp.zeros(Ncols, cp.float32)

        colscales = cp.ones(Ncols)
        for col, param in enumerate(allderivs):
            RR = []
            VV = []
            WW = []
            for (deriv, img) in param:
                gc[1] += 1
                tx = time.time()
                # Assuming img.getInvError() returns CuPy array
                inverrs = img.getInvErrorGPU() 
                (H, W) = img.shape
                row0 = imgoffs[img]
                deriv.clipTo(W, H)
                # Assuming deriv.getPixelIndices() returns CuPy array
                pix = deriv.getPixelIndicesGPU(img) 
                glt[6] += time.time()-tx
                tx = time.time()

                if len(pix) == 0:
                    logverb('Col %i: this param does not influence this image!' % col)
                    continue

                assert(cp.all(pix < img.numberOfPixels()))

                # Assuming deriv.getImage() returns CuPy array
                dimg = deriv.getImageGPU()
                nz = cp.flatnonzero(dimg)
                glt[7] += time.time()-tx
                tx = time.time()

                if len(nz) == 0:
                    logverb('Col %i: all derivs are zero')
                    continue
                
                rows = row0 + pix[nz]
                vals = dimg.ravel()[nz] # Changed from dimg.flat[nz]
                inverrs_patch = inverrs[deriv.getSlice(img)]
                w = inverrs_patch.ravel()[nz] # Fixed here
                #w = inverrs[deriv.getSlice(img)].flat[nz]
                
                assert(vals.shape == w.shape)
                
                RR.append(rows)
                VV.append(vals)
                WW.append(w)
                glt[8] += time.time()-tx

            if len(VV) == 0:
                continue
            
            tx = time.time()
            rows = cp.hstack(RR)
            VV = cp.hstack(VV)
            WW = cp.hstack(WW)
            vals = VV * WW

            mx = cp.max(cp.abs(vals))
            if mx == 0:
                logmsg('mx == 0:', len(cp.flatnonzero(VV)), 'of', len(VV), 'non-zero derivatives,',
                       len(cp.flatnonzero(WW)), 'of', len(WW), 'non-zero weights;',
                       len(cp.flatnonzero(vals)), 'non-zero products')
                continue
            
            FACTOR = 1.e-10
            I = (cp.abs(vals) > (FACTOR * mx))
            rows = rows[I]
            vals = vals[I]
            
            scale = cp.sqrt(cp.dot(vals, vals))
            colscales[col] = scale

            if scales_only:
                continue

            if variance:
                var[col] = scale**2

            sprows.append(rows)
            spcols.append(col) # col is a Python int here
            if scale_columns:
                if scale == 0.:
                    spvals.append(vals)
                else:
                    spvals.append(vals / scale)
            else:
                spvals.append(vals)
            glt[9] += time.time()-tx

        glt[0] += time.time()-t
        if scales_only:
            glt[4] += time.time()-t
            print ("GLTimes:",glt)
            return colscales.get()

        b = None
        if priors:
            # Assuming tractor.getLogPriorDerivatives() now returns Python lists/NumPy arrays
            X_prior = tractor.getLogPriorDerivatives()
            if X_prior is not None:
                # Unpack lists/NumPy arrays from the CPU
                rA, cA, vA, pb, mub = X_prior 

                # sprows gets a list of CuPy arrays. The addition Nrows to a NumPy array is fine
                # as the result is still a NumPy array. These are then stored as lists within sprows.
                # If rA contains lists of ints, np.array() converts them.
                # rA elements are typically 1D NumPy arrays, so addition is element-wise.
                # Then append these NumPy arrays to the list sprows.
                sprows.extend([np.array(rij) + Nrows for rij in rA])
                
                # cA is a List
                spcols.extend(cA)

                if scale_columns:
                    # Perform division using NumPy arrays and Python lists
                    # colscales is a CuPy array, but its elements can be accessed on CPU for zip.
                    # Or better, convert colscales to NumPy for this small op if many prior terms.
                    # For a few terms, accessing elements of colscales (which is on GPU) via indexing
                    # will implicitly transfer small scalars, which is generally acceptable.
                    # Let's explicitly bring colscales to CPU if we're doing list comprehension with it.
                    colscales_cpu = colscales.get()
                    scaled_vA = [v_val / colscales_cpu[c_idx] for v_val, c_idx in zip(vA, cA)]
                    spvals.extend(scaled_vA) # Append Python list to spvals list
                else:
                    spvals.extend(vA) # Append Python list to spvals list

                oldnrows = Nrows
                # Call the NumPy-aware listmax
                nr = listmax(rA if rA else [], -1) + 1 
                Nrows += nr
                logverb('Nrows was %i, added %i rows of priors => %i' %
                                (oldnrows, nr, Nrows))

                # Now, when building 'b', we convert prior values to CuPy
                b = cp.zeros(Nrows, dtype=(np.array(pb)).dtype) # Use dtype from pb
                b[oldnrows:] = cp.asarray(np.hstack(pb)) # Ensure pb is hstacked as NumPy before CuPy conversion

        glt[1] += time.time()-t

        if len(spcols) == 0:
            logverb("len(spcols) == 0")
            glt[4] += time.time()-t
            print ("GLTimes:",glt)
            return []

        # Convert spcols list of Python ints to a CuPy array
        spcols_cp = cp.array(spcols, dtype=cp.int32) 
        # nrowspercol can be derived on GPU if sprows elements are CuPy arrays
        nrowspercol = cp.array([len(x) for x in sprows])

        if shared_params:
            spcols_cp = paramindexmap[spcols_cp]
            Ncols = cp.max(spcols_cp) + 1
            logverb('Set Ncols=', Ncols)

        if b is None:
            b = cp.zeros(Nrows)

        # Build chi vector 'b'
        # Assuming tractor.getChiImage() returns CuPy arrays
        # And if chiImages is provided, it contains CuPy arrays
        if chiImages is not None:
            chimap = {img: chi for img, chi in zip(tractor.getImagesGPU(), chiImages)}
        else:
            chimap = {}
        glt[2] += time.time()-t

        for img, row0 in imgoffs.items():
            gc[2] += 1
            tx = time.time()
            chi = chimap.get(img, None)
            if chi is None:
                # Assuming tractor.getChiImage() returns CuPy array
                chi = tractor.getChiImageGPU(img=img)
            chi_ravelled = chi.ravel()
            NP = len(chi_ravelled)
            assert(cp.all(b[row0: row0 + NP] == 0))
            assert(cp.all(cp.isfinite(chi_ravelled)))
            b[row0: row0 + NP] = chi_ravelled
            glt[5] += time.time()-tx

        spvals_cp = cp.hstack(spvals)
        if not cp.all(cp.isfinite(spvals_cp)):
            print('Warning: infinite derivatives; bailing out')
            return None
        assert(cp.all(cp.isfinite(spvals_cp)))

        sprows_cp = cp.hstack(sprows)
        assert(len(sprows_cp) == len(spvals_cp))
        
        # For LSQR, expand 'spcols' to be the same length as 'sprows'.
        # This part requires iterating over CPU values (`spcols_cp.get()` and `nrowspercol.get()`)
        # to correctly fill `cc`.
        cc = cp.empty(len(sprows_cp), dtype=spcols_cp.dtype)
        i = 0
        # Iterate over CPU versions of spcols_cp and nrowspercol
        for c, n in zip(spcols_cp.get(), nrowspercol.get()):
            cc[i: i + n] = c
            i += n
        spcols_cp = cc
        assert(i == len(sprows_cp))
        assert(len(sprows_cp) == len(spcols_cp))

        logverb('  Number of sparse matrix elements:', len(sprows_cp))
        urows = cp.unique(sprows_cp)
        ucols = cp.unique(spcols_cp)
        logverb('  Unique rows (pixels):', len(urows))
        logverb('  Unique columns (params):', len(ucols))
        if len(urows) == 0 or len(ucols) == 0:
            return []
        logverb('  Max row:', urows[-1].item()) # .item() for logging
        logverb('  Max column:', ucols[-1].item()) # .item() for logging
        logverb('  Sparsity factor (possible elements / filled elements):',
                float(len(urows) * len(ucols)) / float(len(sprows_cp)))

        glt[3] += time.time()-t


        # Build sparse matrix A
        A = cpxr_matrix((spvals_cp, (sprows_cp, spcols_cp)), shape=(Nrows, Ncols))

        # Remove the complex damping augmentation logic, as lsmr handles it internally
        # A_lsqr = A # These will now just be A and b directly
        # b_lsqr = b

        cplsmr_opts = dict(damp=damp) # Now damp is a valid argument for lsmr!
        # You can add other parameters here if you need them later, e.g.,
        # cplsmr_opts['atol'] = 1e-6
        # cplsmr_opts['btol'] = 1e-6
        # cplsmr_opts['conlim'] = 1e8

        logverb('LSMR: %i cols (%i unique), %i elements' % # Changed log message
                (Ncols, len(ucols), len(spvals_cp) - 1))

        bail = False
        try:
            # CHANGE THIS LINE: Call cplsmr instead of cplsqr
            (X, istop, niters, r1norm, arnorm, anorm, acond, xnorm) = cplsmr(A, b, **cplsmr_opts)
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f'CuPy CUDA Runtime Error caught: {e}. Returning zero.')
            bail = True
        except Exception as e:
            print(f'General error caught: {e}. Returning zero.')
            bail = True

        del A
        del b
        # No need to delete A_augmented or b_augmented since we're using lsmr

        if bail:
            glt[4] += time.time()-t
            print ("GLTimes:",glt)
            if shared_params:
                return cp.zeros(len(paramindexmap)).get()
            return cp.zeros(len(allderivs)).get()

        logverb('scaled  X=', X)

        if shared_params:
            X = X[paramindexmap]

        if scale_columns:
            positive_colscales_mask = colscales > 0
            X[positive_colscales_mask] /= colscales[positive_colscales_mask]
        logverb('  X=', X)

        glt[4] += time.time()-t
        print ("GLTimes:",glt, gc)

        if variance:
            if shared_params:
                var = var[paramindexmap]
            # Ensure variance is not zero to avoid division by zero
            var_reciprocal = cp.zeros_like(var)
            non_zero_mask = var > 0
            var_reciprocal[non_zero_mask] = 1. / var[non_zero_mask]
            # lsmr doesn't return lsqrvar directly like scipy's lsqr
            # If you need this, it needs to be calculated after the solution X
            # based on (A'A + damp^2*I)^{-1}. This is typically the diagonal of the inverse
            # of the normal equations matrix. It's not trivial to get from LSMR directly.
            # You might need to compute it explicitly if calc_var is True and you need this.
            # For now, I'll return None for lsqrvar to match what lsmr might imply.
            #return X, var_reciprocal
            return X.get(), var_reciprocal.get() # lsmr doesn't return calc_var directly
                                                 # You might need to re-think what var_reciprocal means here
                                                 # if it was intended to be derived from lsqrvar.

        return X.get()
        #return X

        """
        # Build sparse matrix A
        A = cpxr_matrix((spvals_cp, (sprows_cp, spcols_cp)), shape=(Nrows, Ncols))

        # Handle damping by augmenting the matrix and vector
        if damp > 0:
            # Create a CuPy identity matrix for damping
            # It should be a sparse identity matrix
            damp_identity = cpxr_matrix(cp.eye(Ncols)) # cp.eye creates dense, then convert to sparse
            damp_identity = cpxr_matrix(damp_identity) # Ensure it's a sparse CuPy matrix

            # Augment A
            # cp.sparse.vstack stacks sparse matrices vertically
            A_augmented = cp.sparse.vstack([A, damp * damp_identity])

            # Augment b
            # cp.zeros(Ncols) creates a CuPy array of zeros of length Ncols
            b_augmented = cp.hstack([b, cp.zeros(Ncols)])

            # Use the augmented system for LSQR
            A_lsqr = A_augmented
            b_lsqr = b_augmented
        else:
            A_lsqr = A
            b_lsqr = b

        # cplsqr_opts = dict(damp=damp) # This line should be removed
        cplsqr_opts = dict() # No specific keyword arguments for CuPy lsqr

        logverb('LSQR: %i cols (%i unique), %i elements' %
                (Ncols, len(ucols), len(spvals_cp) - 1))

        bail = False
        try:
            (X, istop, niters, r1norm, r2norm, anorm, acond,
             arnorm, xnorm, lsqrvar) = cplsqr(A_lsqr, b_lsqr, **cplsqr_opts) # Use A_lsqr, b_lsqr
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f'CuPy CUDA Runtime Error caught: {e}. Returning zero.')
            bail = True
        except Exception as e:
            print(f'General error caught: {e}. Returning zero.')
            bail = True

        del A # Delete the original A
        if damp > 0: # Delete augmented matrices if created
            del A_augmented
            del b_augmented
        del b # Delete the original b

        if bail:
            if shared_params:
                return cp.zeros(len(paramindexmap)).get()
            return cp.zeros(len(allderivs)).get()

        logverb('scaled  X=', X)
        
        if shared_params:
            X = X[paramindexmap]

        if scale_columns:
            positive_colscales_mask = colscales > 0
            X[positive_colscales_mask] /= colscales[positive_colscales_mask]
        logverb('  X=', X)

        glt[4] += time.time()-t
        print ("GLTimes:",glt)

        if variance:
            if shared_params:
                var = var[paramindexmap]
            # Ensure variance is not zero to avoid division by zero
            var_reciprocal = cp.zeros_like(var)
            non_zero_mask = var > 0
            var_reciprocal[non_zero_mask] = 1. / var[non_zero_mask]
            return X.get(), var_reciprocal.get()

        return X.get()
        """

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

gsub = np.zeros(10)
g2 = np.zeros(8)
g3 = np.zeros(8)
gsub2 = np.zeros(6)
gsub3 = np.zeros(8)

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
    _gpumode = 4

    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None, variance=False,
                           shared_params=True,
                           get_A_matrix=False):
        if self._gpumode == 4:
            print ("GPUMODE 4")
            X1 = self.getUpdateDirection_orig(tractor, allderivs, damp, priors,
                                        scale_columns, scales_only,
                                        chiImages, variance,
                                        shared_params,
                                        get_A_matrix)
            print ("GLTimes: NEW ", gsub[0], "OLD ",gsub[5])
            return X1
        try:
            X1 = self.getUpdateDirection_new(tractor, allderivs, damp, priors,
                                        scale_columns, scales_only,
                                        chiImages, variance,
                                        shared_params,
                                        get_A_matrix)

        except Exception as ex:
            print ("GPULsqrOptimizer Exception: "+str(ex)+"; Running previous version instead.")
            X1 = self.getUpdateDirection_orig(tractor, allderivs, damp, priors,
                                        scale_columns, scales_only,
                                        chiImages, variance,
                                        shared_params,
                                        get_A_matrix)
        #print (f'{gsub=}\n{g2=}\n{gsub2=}\n{g3=}\n{gsub3=}')
        print ("GLTimes: NEW ", gsub[0], "OLD ",gsub[5])
        return X1

    def getUpdateDirection_new(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None, variance=False,
                           shared_params=True,
                           get_A_matrix=False):
        print ("GPULsqrOptimizerBatch!")
        #print ("TRACTOR", type(tractor), tractor.getParams, tractor.setParams)
        t = time.time()

        if shared_params:
            # Assuming tractor.getParams() now returns CuPy arrays
            p0 = tractor.getParams()
            tractor.setParams(np.arange(len(p0))) # Using cp.arange directly
            p1 = tractor.getParams()
            tractor.setParams(p0) # Restore original params

            # np.unique replaced by cp.unique, returns CuPy arrays
            U, I = np.unique(p1, return_inverse=True)
            logverb(len(p0), 'params;', len(U), 'unique')
            paramindexmap = I # I is already a CuPy array

        # Build the sparse matrix of derivatives:
        sprows = [] # Will hold CuPy arrays of row indices
        spcols = [] # Will hold CuPy arrays of column indices (or just ints if single col)
        spvals = [] # Will hold CuPy arrays of values

        # Keep track of row offsets for each image.
        imgoffs = {}
        nextrow = 0
        skyVal = None
        for param in allderivs:
            for deriv, img in param:
                if skyVal is None:
                    skyVal = deriv.getImage()[0,0]
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

        t1 = time.time()
        active_tims = list(imgoffs.keys())
        if len(active_tims) == 0:
            print ("GPULsqrOptimizer> Empty list of tims")
            return []
        # PRE-COMPUTE: These are our "Reference Tensors"
        ie3d = tractor.get_3d_stack_InvErrors(active_tims)
        Nsky = len(active_tims)
        # Get dimensions from the padded ie3d
        depth, max_h, max_w = ie3d.shape
        # Initialize the row map on the GPU
        row_map3d = cp.zeros((depth, max_h, max_w), dtype=cp.int32)

        for i, tim in enumerate(active_tims):
            h, w = tim.shape
            row0 = imgoffs[tim]
            # Create a 2D grid of row indices for this specific image
            # row0, row0+1, row0+2 ... row0 + (h*w - 1)
            img_rows = cp.arange(row0, row0 + (h * w), dtype=cp.int32).reshape(h, w)
            # Inject it into the 3D stack
            row_map3d[i, :h, :w] = img_rows

        # Create a global mask of where InvErr is strictly positive
        # This captures both image boundaries and bad-pixel masks
        global_nonzero_mask = ie3d > 0

        # Compress ie3d and row_map3d into 1D 'Super-Vectors'
        # This removes 99% of the zeros before the loops start
        ie_vector = ie3d[global_nonzero_mask]
        master_rows = row_map3d[global_nonzero_mask]

        # Pre-calculate offsets in the compressed vector for each image
        compressed_offsets = []
        curr = 0
        for i in range(len(active_tims)):
            # count non-zero pixels in this specific image slice
            n_valid = int(cp.sum(global_nonzero_mask[i]))
            compressed_offsets.append((curr, curr + n_valid))
            curr += n_valid

        g2[0] += time.time()-t1
        t1 = time.time()
        # 2. SKY FITTING
        for col in range(Nsky):
            start, end = compressed_offsets[col]

            # These are already non-zero!
            vals = ie_vector[start:end] * skyVal
            rows = master_rows[start:end]

            # Scale is now a simple norm of a much smaller vector
            scale = cp.sqrt(cp.sum(vals**2))
            colscales[col] = scale

            if not scales_only and scale > 0:
                spvals.append(vals / scale if scale_columns else vals)
                sprows.append(rows)
                spcols.append(cp.full(vals.size, col, dtype=cp.int32))
        g2[1] += time.time()-t1

        # 3. POINT SOURCE PHASE (Remaining Columns)
        FACTOR = 1.e-10
        for col in range(Nsky, len(allderivs)):
            col_vals = []
            col_rows = []

            for deriv, img in allderivs[col]:
                # 1. CPU-side extraction (Fast)
                t2 = time.time()
                d_img = deriv.patch
                ie_patch = img.getInvError(use_gpu=False)[deriv.getSlice(img)]

                # 2. CPU Math
                v_flat = (d_img * ie_patch).ravel()
                gsub2[0] += time.time()-t2
                t2 = time.time()

                # 3. Filtering (Masking on CPU is faster for < 10k pixels)
                mx = np.abs(v_flat).max()
                if mx == 0: continue

                m = (np.abs(v_flat) > (FACTOR * mx))

                # 4. Row Mapping (Original Logic)
                row0 = imgoffs[img]
                # pix is the 1D offset of every pixel in the patch
                pix = deriv.getPixelIndices(img).ravel()
                gsub2[1] += time.time()-t2
                t2 = time.time()

                # Apply the same mask 'm' to both values and pixel indices
                col_vals.append(v_flat[m])
                col_rows.append(row0 + pix[m])
                gsub2[2] += time.time()-t2

            if not col_vals:
                colscales[col] = 0.0
                continue

            # 5. Move to GPU in bulk
            # Moving one concatenated array is MUCH faster than many small ones
            t2 = time.time()
            vals_cpu = np.concatenate(col_vals)
            rows_cpu = np.concatenate(col_rows)
            gsub2[3] += time.time()-t2

            t2 = time.time()
            vals = cp.asarray(vals_cpu)
            rows = cp.asarray(rows_cpu)
            gsub2[4] += time.time()-t2

            #Scaling
            scale = cp.sqrt(cp.sum(vals**2))
            colscales[col] = scale

            if scales_only or scale == 0:
                continue

            # 6. Append
            spvals.append(vals / scale if scale_columns else vals)
            sprows.append(rows)
            spcols.append(cp.full(vals.size, col, dtype=cp.int32))

        g2[3] += time.time()-t1

        if scales_only:
            return colscales.get()

        t1 = time.time()
        im3d = tractor.get_3d_stack_Data(active_tims)
        gsub[2] += time.time()-t1

        t1 = time.time()
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
                sprows.extend([cp.array(rij) + Nrows for rij in rA])

                # cA is a List
                spcols.append(cp.asarray(cA, dtype=cp.int32))

                if scale_columns:
                    # Perform division using NumPy arrays and Python lists
                    # colscales is a CuPy array, but its elements can be accessed on CPU for zip.
                    # Or better, convert colscales to NumPy for this small op if many prior terms.
                    # For a few terms, accessing elements of colscales (which is on GPU) via indexing
                    # will implicitly transfer small scalars, which is generally acceptable.
                    # Let's explicitly bring colscales to CPU if we're doing list comprehension with it.
                    colscales_cpu = colscales.get()
                    scaled_vA = [v_val / colscales_cpu[c_idx] for v_val, c_idx in zip(vA, cA)]
                    spvals.extend(cp.asarray(scaled_vA)) #Convert list to cp array
                else:
                    spvals.extend(cp.asarray([vA])) #Convert list to cp array

                oldnrows = Nrows
                # Call the NumPy-aware listmax
                nr = listmax(rA if rA else [], -1) + 1
                Nrows += nr
                logverb('Nrows was %i, added %i rows of priors => %i' %
                                (oldnrows, nr, Nrows))

                # Now, when building 'b', we convert prior values to CuPy
                b = cp.zeros(Nrows, dtype=(np.array(pb)).dtype) # Use dtype from pb
                b[oldnrows:] = cp.asarray(np.hstack(pb)) # Ensure pb is hstacked as NumPy before CuPy conversion

        if len(spcols) == 0:
            logverb("len(spcols) == 0")
            return []

        if b is None:
            b = cp.zeros(Nrows)
            #print ("NROWS",Nrows, Ncols)

        # Build chi vector 'b'
        # Assuming tractor.getChiImage() returns CuPy arrays
        # And if chiImages is provided, it contains CuPy arrays
        gsub[1] += time.time()-t1
        if chiImages is not None:
            #TODO: update so that getImages is not called
            chimap = {img: chi for img, chi in zip(tractor.getImagesGPU(), chiImages)}
        else:
            chimap = {}

        t1 = time.time()
        chi3d = tractor.get_3d_stack_Chi(active_tims, ie=ie3d, im=im3d)
        assert(cp.all(cp.isfinite(chi3d)))
        g2[4] += time.time()-t1
        t1 = time.time()
        i = 0
        #Loop over chi3d
        for img, row0 in imgoffs.items():
            (h, w) = active_tims[i].shape
            NP = h*w
            b[row0: row0 + NP] =  chi3d[i][:h,:w].ravel()
            i += 1
        g2[7] += time.time()-t1
        t1 = time.time()
        spvals_cp = cp.concatenate(spvals)
        sprows_cp = cp.concatenate(sprows)
        spcols_cp = cp.concatenate(spcols)
        g2[5] += time.time()-t1
        del sprows
        del spvals
        del spcols
        if not cp.all(cp.isfinite(spvals_cp)):
            print('Warning: infinite derivatives; bailing out')
            return None
        assert(len(sprows_cp) == len(spcols_cp))

        if isverbose:
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

        t1 = time.time()
        # Build sparse matrix A
        A = cpxr_matrix((spvals_cp, (sprows_cp, spcols_cp)), shape=(Nrows, Ncols))
        g2[6] += time.time()-t1

        # Remove the complex damping augmentation logic, as lsmr handles it internally
        # A_lsqr = A # These will now just be A and b directly
        # b_lsqr = b

        t1 = time.time()
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
        gsub[3] += time.time()-t1
        # No need to delete A_augmented or b_augmented since we're using lsmr
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

        gsub[0] += time.time()-t
        #print (f'{gsub=}\n{g2=}\n{gsub2=}\n{g3=}\n{gsub3=}')
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


    def getUpdateDirection_orig(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None, variance=False,
                           shared_params=True,
                           get_A_matrix=False):
        print ("GPULsqrOptimizer!")
        #print ("TRACTOR", type(tractor), tractor.getParams, tractor.setParams)
        t = time.time()

        if shared_params:
            # Assuming tractor.getParams() now returns CuPy arrays
            p0 = tractor.getParams()
            tractor.setParams(np.arange(len(p0))) # Using cp.arange directly
            p1 = tractor.getParams()
            tractor.setParams(p0) # Restore original params
            
            # np.unique replaced by cp.unique, returns CuPy arrays
            U, I = np.unique(p1, return_inverse=True)
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

        t1 = time.time()
        colscales = cp.ones(Ncols)
        for col, param in enumerate(allderivs):
            RR = []
            VV = []
            WW = []
            t2 = time.time()
            #print (f'{col=} {param=}')
            for deriv, img in param:
                #print (f'{deriv=} {img=}')
                # Use CPU for smaller arrays sizes, copy to GPU after accumulation 
                inverrs = img.getInvError(use_gpu=False) 
                (H, W) = img.shape
                row0 = imgoffs[img]
                deriv.clipTo(W, H)
                # Assuming deriv.getPixelIndices() returns numpy array
                pix = deriv.getPixelIndices(img)

                if len(pix) == 0:
                    logverb('Col %i: this param does not influence this image!' % col)
                    continue

                #assert(cp.all(pix < img.numberOfPixels()))
                #assert(np.all(pix < img.numberOfPixels()))
                #assert(pix.max() < img.numberOfPixels())
                #Pix is in order, only need to check last index
                assert(pix[-1] < img.numberOfPixels())

                # Assuming deriv.getImage() returns numPy array
                dimg = deriv.getImage()
                #nz = cp.flatnonzero(dimg)
                #dimg = deriv.getImage()
                #gsub[4] += time.time()-tx
                #dgpu = deriv.getImage(use_gpu=True) 
                #gsub[3] += time.time()-tx
                #nz = cp.flatnonzero(dgpu)
                #nz = dgpu.ravel() != 0
                #gsub[4] += time.time()-tx
                #nz = np.flatnonzero(dimg)
                nz = dimg.ravel() != 0

                if not np.any(nz):
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

            if len(param) == 1:
                gsub3[0] += time.time()-t2
                gsub3[6] += 1
            else:
                gsub3[5] += time.time()-t2
                gsub3[7] += 1
            gsub3[1] += time.time()-t2
            t2 = time.time()
            if len(VV) == 0:
                continue

            rows = np.hstack(RR)
            VV = np.hstack(VV)
            WW = np.hstack(WW)

            #Copy to GPU
            rows = cp.asarray(rows)
            VV = cp.asarray(VV)
            WW = cp.asarray(WW)
            vals = VV * WW
            gsub3[2] += time.time()-t2

            t2 = time.time()
            if len(vals) == 0:
                print ("Error: VALS LEN 0")   
                continue
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
            gsub3[3] += time.time()-t2

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

        g3[0] += time.time()-t1
        if scales_only:
            return colscales.get()

        t1 = time.time()
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
                sprows.extend([cp.array(rij) + Nrows for rij in rA])
                
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

        if len(spcols) == 0:
            logverb("len(spcols) == 0")
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
            #print ("NROWS",Nrows, Ncols)

        # Build chi vector 'b'
        # Assuming tractor.getChiImage() returns CuPy arrays
        # And if chiImages is provided, it contains CuPy arrays
        gsub[6] += time.time()-t1
        if chiImages is not None:
            #TODO: update so that getImages is not called
            chimap = {img: chi for img, chi in zip(tractor.getImagesGPU(), chiImages)}
        else:
            chimap = {}

        t1 = time.time()
        for img, row0 in imgoffs.items():
            t2 = time.time()
            chi = chimap.get(img, None)
            if chi is None:
                # Assuming tractor.getChiImage() returns CuPy array
                chi = tractor.getChiImageGPU(img=img)
            gsub3[4] += time.time()-t2
            #print (f'{i=} chi {chi.shape=} {type(chi)=} {chi.max()=} {chi.sum()=} {chi3d[i].shape=} {chi3d[i].max()=} {chi3d[i].sum()=}')
            #print (cp.where(chi == chi.max()), cp.where(chi3d[i] == chi3d[i].max()))
            chi_ravelled = chi.ravel()
            NP = len(chi_ravelled)
            assert(cp.all(b[row0: row0 + NP] == 0))
            assert(cp.all(cp.isfinite(chi_ravelled)))
            b[row0: row0 + NP] = chi_ravelled
        g3[4] += time.time()-t1
        t1 = time.time()

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


        g3[5] += time.time()-t1
        #print (f'{spvals_cp2.shape=} {sprows_cp2.shape=} {spcols_cp2.shape=} {Nrows=} {Ncols=}')
        #print (f'{spvals_cp.shape=} {sprows_cp.shape=} {spcols_cp.shape=}')
        #try:
        #    print ("AC ",cp.allclose(spvals_cp, spvals_cp2), cp.allclose(sprows_cp, sprows_cp2), cp.allclose(b, b2))
        #except Exception as ex:
        #    print ("AC Err", spvals_cp.max(), spvals_cp2.max())
        t1 = time.time()
        # Build sparse matrix A
        A = cpxr_matrix((spvals_cp, (sprows_cp, spcols_cp)), shape=(Nrows, Ncols))
        g3[6] += time.time()-t1
        #print (f'{A.shape=} {A2.shape=} {A.max()=} {A2.max()=}')

        # Remove the complex damping augmentation logic, as lsmr handles it internally
        # A_lsqr = A # These will now just be A and b directly
        # b_lsqr = b

        t1 = time.time()
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
        gsub[8] += time.time()-t1
        # No need to delete A_augmented or b_augmented since we're using lsmr
        #print (f'{X=} {X2=}', cp.allclose(X, X2))
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

        gsub[5] += time.time()-t
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

import sys
from tractor.dense_optimizer import ConstrainedDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF, PointSource
from tractor import mixture_profiles as mp
from tractor.psf import lanczos_shift_image
from tractor.sky import ConstantSky
from astrometry.util.miscutils import get_overlapping_region
import numpy as np
import scipy
import scipy.fft
import time
from tractor.batch_psf import BatchPixelizedPSF, lanczos_shift_image_batch_gpu
from tractor.batch_mixture_profiles import ImageDerivs, BatchImageParams, BatchMixtureOfGaussians, BatchGalaxyProfiles
from tractor.batch_galaxy import getShearedProfileGPU, getDerivativeShearedProfilesGPU
from tractor.patch import ModelMask
import cupy as cp
import time
import os, gc
from tractor.miscutils import get_splat_kernel, get_mog_eval_kernel
from tractor.brightness import NanoMaggies

tt = np.zeros(8)
tct = np.zeros(9, np.int32)
tt2 = np.zeros(1, np.int32)
tps = np.zeros(20)
cps = np.zeros(10)
gps = np.zeros(10)
tu = np.zeros(10)
tuc = np.zeros(5, np.int32)
t5 = np.zeros(2)

image_counter = 0
#from astrometry.util.plotutils import PlotSequence
#ps = PlotSequence('fourier')

class GPUFriendlyPointSourceFitter:
    def __init__(self, tractor, source, images):
        self.tractor = tractor
        self.source = source
        self.images = images
        self.cpucomp = None
        self.batch_psf = None
        self.tiny_alpha = 1e-8
        self.stepLimited = False
        self.current_bounds = None
        self.ie_batch = None

    def add_comp(self, R_cpu):
        self.cpucomp = R_cpu

    def get_batch_psf(self):
        psfs = [tim.getPsf() for tim in self.images]
        #for i, psf in enumerate(psfs):
        #    print ("Get Batch psf: ", i, psf.img.shape, psf.img.max())
        self.batch_psf = BatchPixelizedPSF(psfs, True)

    def compute_update_vec(self, allderivs):
        """
        Calculates the total Hessian (H) and Gradient (G) for the given derivatives
        across all images. Returns (H, G) as CuPy arrays.
        
        allderivs: list of lists [ (param0) [ (patch, img), ... ], (param1) [...] ]
        """
        t = time.time()
        num_params = len(allderivs)
        #print (f'{allderivs=}')
        #print (f'GPU {num_params=}')
        if num_params == 0:
            return None, None, 0

        # 1. Pivot: Group patches by Image to minimize GPU overhead
        # Structure: img -> {global_param_idx: PatchObject}
        img_map = {}
        for p_idx, patches in enumerate(allderivs):
            for patch, img in patches:
                if patch is None or patch.patch is None:
                    print (f'{p_idx=} patch is none')
                    continue
                if img not in img_map:
                    img_map[img] = {}
                img_map[img][p_idx] = patch

        # 2. CREATE THE ORDERED LIST OF ACTIVE IMAGES
        # This ensures the i-th slice of our batch matches the i-th image in our loop
        #psfs = []
        #active_tims = []
        #for i, (img, params) in enumerate(img_map.items()):
        #    active_tims.append(img)
        #    psfs.append(img.getPsf())
        active_tims = list(img_map.keys())

        ##2b update batch_psf
        """
        psfs = [tim.getPsf() for tim in active_tims]
        self.batch_psf = BatchPixelizedPSF(psfs, True)
        """

        """
        # 3. Determine the Global Super-Box across ALL images
        all_x0, all_x1, all_y0, all_y1 = [], [], [], []
        for patches in allderivs:
            for patch, _ in patches:
                if patch:
                    all_x0.append(patch.extent[0])
                    all_x1.append(patch.extent[1])
                    all_y0.append(patch.extent[2])
                    all_y1.append(patch.extent[3])

        if len(all_x0) == 0:
            return None, None, 0
        ux0, ux1 = min(all_x0), max(all_x1)
        uy0, uy1 = min(all_y0), max(all_y1)
        """

        # 3. Calculate the 'true' global union from patch extents
        # This replaces the logic that relies on the (sometimes too tight) modelMasks
        all_x0, all_x1, all_y0, all_y1 = [], [], [], []
        for img, params in img_map.items():
            for p_idx, patch in params.items():
                px0, px1, py0, py1 = patch.extent
                all_x0.append(px0); all_x1.append(px1)
                all_y0.append(py0); all_y1.append(py1)

        if len(all_x0) == 0:
            return None, None, 0
        ux0, ux1 = min(all_x0), max(all_x1)
        uy0, uy1 = min(all_y0), max(all_y1)
        ux0 = int(ux0)
        uy0 = int(uy0)
        ux1 = int(ux1)
        uy1 = int(uy1)
        #uH, uW = uy1 - uy0, ux1 - ux0
        u_height, u_width = uy1 - uy0, ux1 - ux0
        total_pixels = u_height * u_width
        num_imgs = len(img_map)
        num_params = len(allderivs)

        # 4. Pre-allocate Tensors on GPU
        # J: (N_images, N_params, Total_Pixels)
        # Chi: (N_images, Total_Pixels)
        #J_tensor = cp.zeros((num_imgs, num_params, total_pixels), dtype=cp.float64)
        J_tensor = cp.zeros((num_imgs, num_params, u_height, u_width), dtype=cp.float64)
        J_tensor2 = cp.zeros((num_imgs, num_params, u_height, u_width), dtype=cp.float64)
        #cJ_tensor = np.zeros((num_imgs, num_params, total_pixels), dtype=np.float64)
        cJ_tensor = np.zeros((num_imgs, num_params, u_height, u_width), dtype=np.float32)
        Chi_tensor = cp.zeros((num_imgs, total_pixels), dtype=cp.float64)
        #print ("ChiT", Chi_tensor.shape)
        print (f'{ux0=} {ux1=} {uy0=} {uy1=}')
        #print (f'{all_x0=} {all_x1=} {all_y0=} {all_y1=}')
        print (f'{num_imgs=} {num_params=} {total_pixels=} {u_width=} {u_height=}')
        tps[2] += time.time()-t

        t = time.time()

        masks = [self.tractor._getModelMaskFor(tim, self.source) for tim in active_tims]
        #masks = [self.tractor._getModelMaskByIdx(i, self.source) for i in range(len(self.images))]
        t1 = time.time()
        ie_batch = self.tractor.getBatchInvErrors(ux0=ux0, ux1=ux1, uy0=uy0, uy1=uy1, use_gpu=True, tims=active_tims)
        #Save params
        
        self.current_bounds = (ux0, ux1, uy0, uy1)
        if len(active_tims) == len(self.tractor.images):
            #If only a subset of images are "in frame" we can calculate this later
            self.ie_batch = ie_batch

        tps[5] += time.time()-t1 
        t1 = time.time()
        chi_batch = self.tractor.getBatchChiImages(ux0=ux0, ux1=ux1, uy0=uy0, uy1=uy1, use_gpu=True, batch_psf=self.batch_psf, tims=active_tims, ie_stack=ie_batch)
        tps[4] += time.time()-t1
        #model_batch = self.tractor.getBatchModelImage(ux0=ux0, ux1=ux1, uy0=uy0, uy1=uy1, use_gpu=True, batch_psf=self.batch_psf, tims=active_tims)
        #tps[6] += time.time()-t1
        #cp.savetxt('chi_batch.txt', chi_batch.ravel())
        #cp.savetxt('ie_batch.txt', ie_batch.ravel())
        #cp.savetxt('model_batch.txt', model_batch.ravel())
        #for i, tim in enumerate(active_tims):
        #    print (f'BATCH {i=} {tim.getImage().max()=} {tim.getPsf().img.max()=} {ie_batch[i].max()=} {model_batch.max()=}')
        #    print (f'MASKS {masks[i]}')

        print ("chi_batch", chi_batch.shape, ie_batch.shape)
        Chi_tensor = chi_batch.reshape((num_imgs, total_pixels))
        t = time.time()
        """
        for i, (img, params) in enumerate(img_map.items()):
            ie_full = img.getInvError(use_gpu=True)
            mod_full = cp.asarray(self.tractor.getModelImage(img=img))
            mask = self.tractor._getModelMaskFor(img, self.source)
            #print (f'FULL {i=} {img.getImage().max()=}  {tim.getPsf().img.max()=} {ie_full.max()=} {mod_full.max()=} {ie_full.shape=}')
            #print (cp.where(ie_full == ie_full.max()))
            #print (f'MASK2 {mask}')
        """
        for i, (img, params) in enumerate(img_map.items()):
            #print ("I", i, "IMG MAX", img.getImage().max(), img.getPsf().img.max())
            #if i >= 3:
            #    continue
            # Fetch and crop to global bounds
            # (Ensure getChiImage/getInvError return CuPy arrays to avoid transfers)
            """
            t1 = time.time()
            chi_full = self.tractor.getChiImage(img=img, use_gpu=True)
            ie_full = img.getInvError(use_gpu=True)
            mod_full = cp.asarray(self.tractor.getModelImage(img=img))
            tps[7] += time.time()-t1
            img_h, img_w = chi_full.shape
            #print ("CHI FULL", i, chi_full.shape)
            #print (f'{chi_batch[i].max()=} {chi_full.max()=}')
            #from tractor.batch_psf import plot_comparison
            #from tractor.batch_psf import plot_one
            #plot_one(model_batch[i].get(), "Model "+str(i))
            #plot_one(mod_full.get(), "Model F"+str(i))
            #plot_comparison(model_batch[i].get(), mod_full.get(), "Model "+str(i))

            # 2. Determine the intersection between this image and the Global Box
            # These are indices in the IMAGE coordinate system
            y_start = max(uy0, 0)
            y_end   = min(uy1, img_h)
            x_start = max(ux0, 0)
            x_end   = min(ux1, img_w)

            # 3. Determine where that intersection lands in the GLOBAL coordinate system
            # These are indices relative to the top-left of the (u_height, u_width) tensor
            d_y_start = y_start - uy0
            d_y_end   = y_end   - uy0
            d_x_start = x_start - ux0
            d_x_end   = x_end   - ux0

            for p_idx, patch in params.items():
                px0, px1, py0, py1 = patch.extent

            #print ("CHI FULL", i, chi_full.shape)
            #print ("SHAPES ", y_start, y_end, x_start, x_end, d_y_start, d_y_end, d_x_start, d_x_end)
            #print ("Ps:", py0, py1, px0, px1)
            chi_sub = chi_full[y_start:y_end, x_start:x_end]
            #print (f'{chi_batch[i].max()=} {chi_full.max()=} {chi_sub.max()=} {chi_sub.shape=}')
            #print ("WHERE", cp.where(chi_batch[i] == chi_batch[i].max()), cp.where(chi_full == chi_full.max()), cp.where(chi_sub == chi_sub.max()))
            
            # Map global bounds to image coordinates
            # Note: Handle images with different sizes/offsets via slicing
            # For simplicity, assuming images contain the global box:
            #chi_crop = chi_full[uy0:uy1, ux0:ux1].ravel()
            #ie_crop = ie_full[uy0:uy1, ux0:ux1].ravel()
            
            #print ("SHAPES", Chi_tensor.shape, Chi_tensor[i].shape, chi_crop.shape, ie_crop.shape, chi_full.shape, ie_full.shape)
            #ie_sub = ie_full[y_start:y_end, x_start:x_end]
            ie_sub = ie_full[py0:py1, px0:px1]
            #mod_sub = mod_full[y_start:y_end, x_start:x_end]
            mod_sub = mod_full[py0:py1, px0:px1]
            #print ("IE SUB", ie_sub.shape, ie_sub.max(), cp.where(ie_sub == ie_sub.max()))
            #if i < 3:
            #    from tractor.batch_psf import plot_comparison
            #    plot_comparison(model_batch[i].get(), mod_sub.get(), "Mod "+str(i))
            #    plot_comparison(ie_batch[i].get(), ie_sub.get(), "IE "+str(i))
            #    plot_comparison(chi_batch[i].get(), chi_sub.get(), "Chi "+str(i))

            #print (f'{uy0=} {uy1=} {ux0=} {ux1=}', chi_full[uy0:uy1, ux0:ux1].shape)
            #print ("IE", ie_batch[i].max(), ie_full.max(), ie_sub.max(), "MOD", model_batch[i].max(), mod_full.max(), mod_sub.max())
            #Chi_tensor[i] = chi_crop * ie_crop # Pre-weighted by inverse error

            # 4. Fill Chi_tensor (Pre-weighted by inverse error)
            # We reshape the row to 2D for easy slicing
            t1 = time.time()
            chi_row_2d = Chi_tensor[i].reshape(u_height, u_width)
            
            # Only update the region where the image actually exists
            # The rest of the row remains zeroed out
            #chi_row_2d[d_y_start:d_y_end, d_x_start:d_x_end] = chi_batch[i] 
            #chi_row_2d = chi_batch[i]
            chi_row_2d[d_y_start:d_y_end, d_x_start:d_x_end] = chi_batch[i][d_y_start:d_y_end, d_x_start:d_x_end]
            """
            iex = ie_batch[i]
            #tps[8] += time.time()-t1

            #print ("CHI ROW2d", chi_row_2d[d_y_start:d_y_end, d_x_start:d_x_end].shape, chi_row_2d.shape, chi_row_2d.max(), chi_batch[i].shape, chi_batch[i].max())
            #print (f'{chi_row_2d.sum()=} {chi_batch[i].sum()=} {chi_sub.sum()=}')
            #print ("IE", ie_batch[i].max(), ie_full.max(), "MOD", model_batch[i].max(), mod_full.max())
            #print ("ChiT", Chi_tensor[i].sum(), Chi_tensor[i].max(), Chi_tensor[i].shape, chi_row_2d[d_y_start:d_y_end, d_x_start:d_x_end].shape)
            #print ("START COORDS", y_start, y_end, x_start, x_end)
            #cp.savetxt('chi_row_2d_'+str(i)+'.txt', chi_row_2d)
            #cp.savetxt('ie_full_'+str(i)+'.txt', ie_full)
            #cp.savetxt('mod_full_'+str(i)+'.txt', mod_full)
            #cp.savetxt('chi_full_'+str(i)+'.txt', chi_full)
            #print ("MATCHES: ", cp.allclose(chi_batch[i], chi_row_2d, atol=1.e-5))
            #print ("CMATCHES: ", cp.allclose(chi_batch[i], chi_sub, atol=1.e-5))
            #print ("IE", ie_batch[i].shape, ie_full[y_start:y_end, x_start:x_end].shape, ie_sub.shape, ie_batch[i].max(), ie_full[y_start:y_end, x_start:x_end].max(), ie_sub.max())
            #if ie_full[y_start:y_end, x_start:x_end].shape == ie_batch[i].shape:
                #print ("MATCHES: ", cp.allclose(chi_batch[i], chi_row_2d, atol=1.e-5))
                #print ("IE MATCH", cp.allclose(ie_batch[i], ie_full[y_start:y_end, x_start:x_end], atol=1.e-5))
                #print ("MOD MATCH", cp.allclose(model_batch[i], mod_full[y_start:y_end, x_start:x_end], atol=1.e-5))

            for p_idx, patch in params.items():
                t1 = time.time()
                px0, px1, py0, py1 = patch.extent
                #dx0, dx1 = px0 - ux0, px1 - ux0
                #dy0, dy1 = py0 - uy0, py1 - uy0
                # Patch is already relative to image coordinates, 
                # but we need to place it relative to the Global Box
                dx0 = px0 - ux0
                dx1 = px1 - ux0
                dy0 = py0 - uy0
                dy1 = py1 - uy0
                
                # View into the specific image/parameter row as a 2D grid for insertion
                """
                J_view = J_tensor[i, p_idx].reshape((u_height, u_width))
                tps[17] += time.time()-t1
                t1 = time.time()
                """
                """
                t1 = time.time()
                J_tensor[i, p_idx, dy0:dy1, dx0:dx1] = patch.getPatch(use_gpu=True)
                gps[0] += time.time()-t1
                """

                #cJ_view = cJ_tensor[i, p_idx].reshape((u_height, u_width))
                #tps[14] += time.time()-t1
                #t1 = time.time()

                #d_cpu = patch.getPatch()
                #tps[15] += time.time()-t1
                #t1 = time.time()
                
                """
                # Weighted derivative: (deriv * inv_error)
                d_gpu = patch.getPatch(use_gpu=True)
                tps[18] += time.time()-t1
                t1 = time.time()

                #ie_full = img.getInvError(use_gpu=True)
                #print (f'{d_gpu.shape=} {J_view.shape=} {ie_batch.shape=} {iex[dy0:dy1, dx0:dx1].shape=}')
                #print (f'{dx0=} {dy0=} {dx1=} {dy1=} {px0=} {px1=} {py0=} {py1=}')
                """

                t1 = time.time()
                #cJ_view[dy0:dy1, dx0:dx1] = d_cpu * iec[dy0:dy1, dx0:dx1]
                #cJ_view[dy0:dy1, dx0:dx1] = d_cpu
                cJ_tensor[i, p_idx, dy0:dy1, dx0:dx1] = patch.getPatch()
                #cps[0] += time.time()-t1
                tps[3] += time.time()-t1


                """
                patch.patch_gpu = None
                t1 = time.time()
                J_tensor2[i, p_idx, dy0:dy1, dx0:dx1] = patch.getPatch(use_gpu=True)*iex[dy0:dy1, dx0:dx1]
                gps[6] += time.time()-t1
                """

                """
                #J_view[dy0:dy1, dx0:dx1] = d_gpu * ie_batch[i][dy0:dy1, dx0:dx1]
                J_view[dy0:dy1, dx0:dx1] = d_gpu * iex[dy0:dy1, dx0:dx1]
                #print ("JView", J_view.sum(), "IEX", iex.shape, iex[dy0:dy1, dx0:dx1].shape, iex.max(), iex[dy0:dy1, dx0:dx1].max())
                tps[19] += time.time()-t1
                #J_view[dy0:dy1, dx0:dx1] = d_gpu * ie_full[py0:py1, px0:px1]
                #J_view[dy0:dy1, dx0:dx1] = d_gpu * ie_full[py0:py1, px0:px1].astype(cp.float64)
                #print ("D GPU", d_gpu.shape, ie_full.shape)
                """

        tps[6] += time.time()-t
        t = time.time()
        # 4. The "Two-Dot" Solution (The Big Win)
        # Reshape J to (N_params, N_images * Total_Pixels)
        # This effectively flattens the image and pixel dimensions into one "measurement" space
        t = time.time()
        #cJ_tensor = cp.asarray(cJ_tensor)*ie_batch.reshape((num_imgs, 1, total_pixels))
        cJ_tensor = cp.asarray(cJ_tensor)
        cps[2] += time.time()-t
        t = time.time()
        cJ_tensor *= ie_batch[:, cp.newaxis, :,:]
        cps[3] += time.time()-t
        """
        t = time.time()
        J_tensor *= ie_batch[:, cp.newaxis, :,:]
        gps[3] += time.time()-t
        """
        t = time.time()
        #print ("ALLCLOSE", cp.allclose(J_tensor, cJ_tensor))
        #J_flat = J_tensor.transpose(1, 0, 2).reshape(num_params, -1)
        cJ_flat = cJ_tensor.transpose(1, 0, 2, 3).reshape(num_params, -1)
        cJ_flat = cp.ascontiguousarray(cJ_flat) # Force BLAS-friendly layout
        cps[9] += time.time()-t
        Chi_flat = Chi_tensor.ravel()
        #print ("J_flat", J_flat.shape, J_flat.sum(), J_flat.max())
        #print ("Chi_flat", Chi_flat.sum(), Chi_flat.max())
        #print ("Chi batch", chi_batch.sum(), chi_batch.max()) 

        # Gradient: G = J_flat @ Chi_flat
        # Result: (N_params,)
        G_full = cJ_flat.dot(Chi_flat)

        # Hessian: H = J_flat @ J_flat^T
        # Result: (N_params, N_params)
        H_full = cJ_flat.dot(cJ_flat.T)
        tps[10] += time.time()-t
        cps[4] += time.time()-t

        """
        t = time.time()
        J_flat = J_tensor.transpose(1, 0, 2, 3).reshape(num_params, -1)
        #J_flat = cp.ascontiguousarray(J_flat) # Force BLAS-friendly layout
        Chi_flat = Chi_tensor.ravel()
        print ("J_flat", J_flat.shape, J_flat.sum(), J_flat.max())
        print ("Chi_flat", Chi_flat.sum(), Chi_flat.max())
        print ("Chi batch", chi_batch.sum(), chi_batch.max())

        # Gradient: G = J_flat @ Chi_flat
        # Result: (N_params,)
        G_full = J_flat.dot(Chi_flat)

        # Hessian: H = J_flat @ J_flat^T
        # Result: (N_params, N_params)
        H_full = J_flat.dot(J_flat.T)
        gps[4] += time.time()-t


        t = time.time()
        J_flat2 = J_tensor2.transpose(1, 0, 2, 3).reshape(num_params, -1)
        #J_flat = cp.ascontiguousarray(J_flat) # Force BLAS-friendly layout
        Chi_flat = Chi_tensor.ravel()

        # Gradient: G = J_flat @ Chi_flat
        # Result: (N_params,)
        G_full = J_flat2.dot(Chi_flat)

        # Hessian: H = J_flat @ J_flat^T
        # Result: (N_params, N_params)
        H_full = J_flat2.dot(J_flat2.T)
        gps[7] += time.time()-t
        """

        return H_full, G_full, num_imgs


    def compute_update(self, allderivs):
        """
        Calculates the total Hessian (H) and Gradient (G) for the given derivatives
        across all images. Returns (H, G) as CuPy arrays.
        
        allderivs: list of lists [ (param0) [ (patch, img), ... ], (param1) [...] ]
        """
        t = time.time()
        num_params = len(allderivs)
        #print (f'{allderivs=}')
        #print (f'GPU {num_params=}')
        if num_params == 0:
            return None, None, 0

        # 1. Pivot: Group patches by Image to minimize GPU overhead
        # Structure: img -> {global_param_idx: PatchObject}
        img_map = {}
        for p_idx, patches in enumerate(allderivs):
            for patch, img in patches:
                if patch is None or patch.patch is None:
                    continue
                if img not in img_map:
                    img_map[img] = {}
                img_map[img][p_idx] = patch

        # 1. Determine the Global Super-Box across ALL images
        all_x0, all_x1, all_y0, all_y1 = [], [], [], []
        for patches in allderivs:
            for patch, _ in patches:
                if patch:
                    all_x0.append(patch.extent[0])
                    all_x1.append(patch.extent[1])
                    all_y0.append(patch.extent[2])
                    all_y1.append(patch.extent[3])

        if len(all_x0) == 0:
            return None, None, 0
        ux0, ux1 = int(min(all_x0)), int(max(all_x1))
        uy0, uy1 = int(min(all_y0)), int(max(all_y1))
        u_height, u_width = uy1 - uy0, ux1 - ux0
        total_pixels = u_height * u_width
        num_imgs = len(img_map)
        num_params = len(allderivs)

        # 2. Pre-allocate Tensors on GPU
        # J: (N_images, N_params, Total_Pixels)
        # Chi: (N_images, Total_Pixels)
        J_tensor = cp.zeros((num_imgs, num_params, total_pixels), dtype=cp.float64)
        Chi_tensor = cp.zeros((num_imgs, total_pixels), dtype=cp.float64)
        tps[11] += time.time()-t

        t = time.time()
        for i, (img, params) in enumerate(img_map.items()):
            #print ("OG I", i, "IMG MAX", img.getImage().max())
            #if i >= 3:
            #    continue
            # Fetch and crop to global bounds
            # (Ensure getChiImage/getInvError return CuPy arrays to avoid transfers)
            t1 = time.time()
            chi_full = self.tractor.getChiImage(img=img, use_gpu=True)
            ie_full = img.getInvError(use_gpu=True)
            mod_full = cp.asarray(self.tractor.getModelImage(img=img))
            tps[12] += time.time()-t1
            img_h, img_w = chi_full.shape

            # 2. Determine the intersection between this image and the Global Box
            # These are indices in the IMAGE coordinate system
            y_start = max(uy0, 0)
            y_end   = min(uy1, img_h)
            x_start = max(ux0, 0)
            x_end   = min(ux1, img_w)

            # 3. Determine where that intersection lands in the GLOBAL coordinate system
            # These are indices relative to the top-left of the (u_height, u_width) tensor
            d_y_start = y_start - uy0
            d_y_end   = y_end   - uy0
            d_x_start = x_start - ux0
            d_x_end   = x_end   - ux0
            
            # Map global bounds to image coordinates
            # Note: Handle images with different sizes/offsets via slicing
            # For simplicity, assuming images contain the global box:
            #chi_crop = chi_full[uy0:uy1, ux0:ux1].ravel()
            #ie_crop = ie_full[uy0:uy1, ux0:ux1].ravel()
            
            # 4. Fill Chi_tensor (Pre-weighted by inverse error)
            # We reshape the row to 2D for easy slicing
            t1 = time.time()
            chi_row_2d = Chi_tensor[i].reshape(u_height, u_width)
            
            # Only update the region where the image actually exists
            # The rest of the row remains zeroed out
            chi_row_2d[d_y_start:d_y_end, d_x_start:d_x_end] = (
                chi_full[y_start:y_end, x_start:x_end].astype(cp.float64)
            )
            #print (f"OLD {i=} {y_start=} {y_end=} {x_start=} {x_end=} {d_y_start=} {d_y_end=} {d_x_start=} {d_x_end=}")
            #print (f"IE {ie_full.max()=} {ie_full[y_start:y_end, x_start:x_end].max()=}")
            #print ("ChiT", Chi_tensor[i].sum(), Chi_tensor[i].max(), Chi_tensor[i].shape, chi_row_2d[d_y_start:d_y_end, d_x_start:d_x_end].shape)

            tps[13] += time.time()-t1

            for p_idx, patch in params.items():
                px0, px1, py0, py1 = patch.extent
                #dx0, dx1 = px0 - ux0, px1 - ux0
                #dy0, dy1 = py0 - uy0, py1 - uy0
                # Patch is already relative to image coordinates, 
                # but we need to place it relative to the Global Box
                dx0 = px0 - ux0
                dx1 = px1 - ux0
                dy0 = py0 - uy0
                dy1 = py1 - uy0
                
                # View into the specific image/parameter row as a 2D grid for insertion
                J_view = J_tensor[i, p_idx].reshape((u_height, u_width))
                
                # Weighted derivative: (deriv * inv_error)
                d_gpu = patch.getPatch(use_gpu=True)
                J_view[dy0:dy1, dx0:dx1] = d_gpu * ie_full[py0:py1, px0:px1]
                #print ("CPU IE Ps:",py0, py1, px0, px1, ie_full[py0:py1, px0:px1].max())
                #print ("JView", J_view.sum(), "IE", ie_full[py0:py1, px0:px1].shape, ie_full[py0:py1, px0:px1].max())

        tps[14] += time.time()-t
        t = time.time()
        # 4. The "Two-Dot" Solution (The Big Win)
        # Reshape J to (N_params, N_images * Total_Pixels)
        # This effectively flattens the image and pixel dimensions into one "measurement" space
        J_flat = J_tensor.transpose(1, 0, 2).reshape(num_params, -1)
        Chi_flat = Chi_tensor.ravel()
        print ("J_flat", J_flat.shape, J_flat.sum(), J_flat.max())
        print ("Chi_flat", Chi_flat.sum(), Chi_flat.max())

        # Gradient: G = J_flat @ Chi_flat
        # Result: (N_params,)
        G_full = J_flat.dot(Chi_flat)

        # Hessian: H = J_flat @ J_flat^T
        # Result: (N_params, N_params)
        H_full = J_flat.dot(J_flat.T)
        tps[15] += time.time()-t
        return H_full, G_full, num_imgs

    def compute_update_old(self, allderivs):
        """
        Calculates the total Hessian (H) and Gradient (G) for the given derivatives
        across all images. Returns (H, G) as CuPy arrays.
        
        allderivs: list of lists [ (param0) [ (patch, img), ... ], (param1) [...] ]
        """
        num_params = len(allderivs)
        #print (f'{allderivs=}')
        print (f'GPU {num_params=}')
        if num_params == 0:
            return None, None

        # 1. Pivot: Group patches by Image to minimize GPU overhead
        # Structure: img -> {global_param_idx: PatchObject}
        img_map = {}
        for p_idx, patches in enumerate(allderivs):
            for patch, img in patches:
                if patch is None or patch.patch is None:
                    continue
                if img not in img_map:
                    img_map[img] = {}
                img_map[img][p_idx] = patch

        # 2. Initialize H and G on GPU
        # We use float64 for the accumulation to match the precision of the CPU engine
        H_full = cp.zeros((num_params, num_params), dtype=cp.float64)
        G_full = cp.zeros(num_params, dtype=cp.float64)
        #A = []
        #B = []

        xicsum = 0
        icsum = 0

        # 3. Process image by image
        idx = 0
        ni = 0
        for img, params in img_map.items():
            t = time.time()
            H = cp.zeros((num_params, num_params), dtype=cp.float64)
            G = cp.zeros(num_params, dtype=cp.float64)
            R = None
            if self.cpucomp is not None:
                R = self.cpucomp[idx]
                cx,cx_icov,cA,ccolscales,cB,cAo = R
                print ("R", R)
                idx+=1
            #currA = [] 
            # Calculate the "Super-Bounding Box" for all patches in this image
            all_x0 = [p.extent[0] for p in params.values()]
            all_x1 = [p.extent[1] for p in params.values()]
            all_y0 = [p.extent[2] for p in params.values()]
            all_y1 = [p.extent[3] for p in params.values()]

            ux0, ux1 = min(all_x0), max(all_x1)
            uy0, uy1 = min(all_y0), max(all_y1)
            u_shape = (uy1 - uy0, ux1 - ux0)
            tps[2] += time.time()-t
            t = time.time()
            
            # Fetch residuals (chi) and inverse error for this image, cropped to the Super-Box
            # chi = (data - model) * inverr
            chi_full = self.tractor.getChiImage(img=img, use_gpu=True)
            ie_full = img.getInvError(use_gpu=True)
            tps[3] += time.time()-t
            t = time.time()
            
            # Transfer to GPU (float32 is sufficient for individual pixel math)
            chi_gpu = cp.array(chi_full[uy0:uy1, ux0:ux1].astype(np.float64))
            ie_gpu = cp.array(ie_full[uy0:uy1, ux0:ux1].astype(np.float64))
            tps[4] += time.time()-t
            t = time.time()
            #print ("chi", chi_gpu.shape, chi_gpu.max())
            #cp.savetxt('gchi.txt', chi_gpu)
            #B.append(chi_gpu.get())
            """
            if R is not None:
                try:
                    assert(np.allclose(cB, chi_gpu.get().ravel()))
                    print (f'{self.tractor.catalog[0]=} {img=} B match')
                except AssertionError as ex:
                    print (f"{self.tractor.catalog[0]=} {img=} B does not match!")
                    chi_gpu = cp.asarray(cB).rehsape(u_shape)
            """

            # Store padded Jacobians (J = deriv * inverr)
            js_gpu = {}
            for p_idx, patch in params.items():
                J_full = cp.zeros(u_shape, dtype=cp.float64)

                # Relative offsets within the Super-Box
                px0, px1, py0, py1 = patch.extent
                dx0, dx1 = px0 - ux0, px1 - ux0
                dy0, dy1 = py0 - uy0, py1 - uy0

                # Insert the derivative patch and weight by inverse error
                d_gpu = cp.array(patch.getPatch(use_gpu=True).astype(cp.float64))
                J_full[dy0:dy1, dx0:dx1] = d_gpu * ie_gpu[dy0:dy1, dx0:dx1]
                """
                if R is not None:
                    try:
                        assert(np.allclose(cA[:,p_idx], J_full.get().ravel()))
                        print (f'{self.tractor.catalog[0]=} {img=} A match')
                    except AssertionError as ex:
                        print (f'{self.tractor.catalog[0]=} {img=} {p_idx=} A does not match')
                        J_full = cp.asarray(cA[:,p_idx]).reshape(u_shape)
                """

                js_gpu[p_idx] = J_full

                # Accumulate Gradient: G = J^T * chi
                G[p_idx] += cp.sum((J_full * chi_gpu).astype(cp.float64))
                #print ("J", J_full.shape, J_full.max())
                #print ("G", G.shape, G.max())
                #cp.savetxt('gj_'+str(p_idx)+'.txt', J_full)
                #currA.append(J_full.get())
            tps[5] += time.time()-t
            t = time.time()

            #A.append(currA)
            # 4. Accumulate Hessian Cross-Terms (H = J^T * J)
            p_indices = sorted(js_gpu.keys())
            for i, p1 in enumerate(p_indices):
                J1 = js_gpu[p1].astype(cp.float64)
                for p2 in p_indices[i:]:
                    val = cp.sum(J1 * js_gpu[p2].astype(cp.float64))
                    H[p1, p2] += val
                    if p1 != p2:
                        H[p2, p1] += val
            
            # Clean up image-specific GPU memory
            del js_gpu, chi_gpu, ie_gpu
            #cp.savetxt("ggrad.txt", G)
            #cp.savetxt("ghess.txt", H)
            """
            gpux = cp.linalg.lstsq(H, G)[0]
            print (f'{idx=}')
            print ("SHAPES", cx_icov.shape, cx.shape, G.shape, H.shape, gpux.shape)
            print (f'{cx_icov=}')
            print (f'{G=}')
            print (f'{H=}')
            print (f'{gpux=} {self.tractor.catalog[0]=} {img=}')
            print (f'{cx=}')
            """
            H_full += H
            G_full += G
            ni += 1
            tps[6] += time.time()-t
            """
            dp = np.dot(cx_icov, cx)
            print (f'{dp=}')
            ddp = np.dot(cx_icov.astype(np.float64), cx.astype(np.float64))
            print (f'{ddp=}')

            xicsum = xicsum + np.dot(cx_icov, cx)
            icsum = icsum + cx_icov

            print (f'{H_full=}')
            print (f'{G_full=}')
            print (f'{xicsum=}')
            print (f'{icsum=}')
            if R is not None:
                try:
                    assert(np.allclose(cx_icov, H.get()))
                    print (f'{self.tractor.catalog[0]=} {img=} Hess match')
                except AssertionError as ex:
                    print (f'{self.tractor.catalog[0]=} {img=} Hess does not match')
                try:
                    assert(np.allclose(cx, gpux))
                    print (f'{self.tractor.catalog[0]=} {img=} X match')
                except AssertionError as ex:
                    print (f'{self.tractor.catalog[0]=} {img=} X does not match')
            """
            #print (f'{gpux=} {self.tractor.catalog[0]=}')
            #import sys
            #sys.exit(-1)

        """
        A = np.asarray(A)
        B = np.asarray(B)
        A = cp.asarray(A)
        B = cp.asarray(B)
        print ("A", A.shape)
        print ("B", B.shape)
        cp.savetxt("ga.txt", A.ravel())
        cp.savetxt("gb.txt", B.ravel())
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)
        #print ("Xicov", Xicov)
        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        colscales[colscales == 0] = 1.
        print ("COLSCALES", colscales.shape)
        cp.savetxt("gcols.txt", colscales.ravel())
        """
        #print ("J", J_full.shape, "Hess", H.shape, "Grad", G.shape)
        #cp.savetxt("gj.txt", J_full.ravel())
        #cp.savetxt("ggrad.txt", G.ravel())
        #cp.savetxt("ghess.txt", H.ravel())
        #import sys
        #sys.exit(-1)

        return H_full, G_full, ni

    def tryUpdates_gpu(self, X, alphas=None, check_step=None, use_less_mem=False):
        t_start = time.time()
        (ux0, ux1, uy0, uy1) = self.current_bounds
        if self.ie_batch is None:
            self.ie_batch = self.tractor.getBatchInvErrors(ux0=ux0, ux1=ux1, uy0=uy0, uy1=uy1, use_gpu=True)

        # 1. Prepare Alphas (inject 0.0 at the front)
        if alphas is None:
            alphas = np.append(0.0, np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.]))
        else:
            alphas = np.unique(np.append(0.0, alphas))

        # 2. Get Steps
        #s2 = self.getParameterSteps_gpu(self.tractor, X, alphas)
        #print (f'{s2=}')
        steps = self.getParameterSteps(X, alphas, firstAlphaIsZero=True)
        candidate_ps = [s['p'] for s in steps]
        #print (f'{alphas=} {steps=} {candidate_ps=}')
        #t1 = time.time()
        #-#lp_global_before = self.tractor.getLogProb()
        #tu[7] += time.time()-t1

        # 3. The Big Batch Evaluation
        # We pass the cached bounds and IE stack to avoid re-calculation
        t1 = time.time()
        logprobs = self.tractor.getLogProbBatch(
            candidate_ps,
            bounds=self.current_bounds,
            ie_stack=self.ie_batch,
            use_gpu=True
        )
        tu[8] += time.time()-t1

        # logprobs[0] is our baseline (alpha=0)
        # 3. Use the GPU's alpha=0 as the local baseline
        #lp_before = float(logprobs[0])
        lps_cpu = cp.asnumpy(logprobs)
        lp_before = lps_cpu[0]
        #print (f'{lps_cpu=}')
        #-#lp_local_baseline = lp_global_before-lps_cpu[0] 
        #-#lps_cpu += lp_local_baseline
        #-#lp_before = lp_global_before
        #-#print (f'{lp_global_before=} {lps_cpu=}')

        lp_best = lp_before
        alpha_best = 0.0
        p_best = candidate_ps[0]
        hit_limit_best = False

        # 4. Standard Line Search logic (Identify best alpha)
        for i, step in enumerate(steps[1:], start=1):
            lp = lps_cpu[i]
            
            # Tractor exit conditions: significantly worse or non-finite
            if lp < (lp_best - 1.0) or not np.isfinite(lp):
                break
                
            if lp > lp_best:
                lp_best = lp
                alpha_best = step['alpha']
                p_best = step['p']
                hit_limit_best = step['hit_limit']

        # 5. Finalize State
        self.tractor.setParams(p_best)
        self.last_step_hit_limit = hit_limit_best
        #print ("Pbest", p_best)
        #print ("LOGPROB", lp_best, lp_before)
        #print ("ALPHA", alpha_best, hit_limit_best, self.last_step_hit_limit)
        
        return lp_best - lp_before, alpha_best


    def getParameterSteps(self, X, alphas=None, firstAlphaIsZero=False):
        t = time.time()
        if alphas is None:
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        p0 = self.tractor.getParams()
        max_step_size = self.tractor.getMaxStep()
        lowers = self.tractor.getLowerBounds()
        uppers = self.tractor.getUpperBounds()
        #print (f'{p0=} {max_step_size=} {lowers=} {uppers=}')

        X_np = cp.asnumpy(X) # Ensure we work on CPU for the logic
        results = []

        for i, alpha in enumerate(alphas):
            do_break = False
            step_limit = False
            hit_limit = False

            # Apply max_step_size
            for i, mx in enumerate(max_step_size):
                if mx is None or abs(X_np[j]) == 0: continue
                step = alpha * abs(X_np[j])
                if step > mx:
                    step_limit = True
                    do_break = True
                    alpha = mx / abs(X_np[j])

            # Apply bounds and compute p
            p = []
            for p_start, s, l, u in zip(p0, X_np, lowers, uppers):
                px = p_start + alpha * s
                if l is not None and px < l:
                    px = l
                    hit_limit = True
                if u is not None and px > u:
                    px = u
                    hit_limit = True
                p.append(px)

            if alpha < self.tiny_alpha and alpha != 0:
                #if i > 0 or not firstAlphaIsZero or alpha != 0:
                break

            results.append({
                'alpha': alpha,
                'p': np.array(p),
                'step_limit': step_limit,
                'hit_limit': hit_limit
            })

            if do_break:
                break
        #print (f'{results=}')
        tu[9] += time.time()-t
        return results

    def getParameterSteps_gpu(self, tractor, step_direction, alphas=None):
        if alphas is None:
            # Default alphas: 1/1024 to 1, then sqrt(2) and 2
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        p0 = np.array(tractor.getParams())
        lowers = np.array([x if x is not None else -np.inf for x in tractor.getLowerBounds()])
        uppers = np.array([x if x is not None else  np.inf for x in tractor.getUpperBounds()])
        max_steps = np.array([x if x is not None else np.inf for x in tractor.getMaxStep()])

        step_direction = np.array(step_direction)
        results = []

        for alpha in alphas:
            # 1. Check Max Step Size
            # If alpha * |dir| > max_step, we cap alpha and then BREAK
            step_magnitudes = alpha * np.abs(step_direction)
            over_step = step_magnitudes > max_steps

            do_break = False
            step_limit = False
            if np.any(over_step):
                # Find the parameter that forces the smallest alpha
                alpha = np.min(max_steps[over_step] / np.abs(step_direction[over_step]))
                step_limit = True
                do_break = True
                self.stepLimited = True

            # 2. Check Bounds
            # px = p0 + alpha * s
            px_trial = p0 + alpha * step_direction
            hit_limit = False

            # Check lower bounds
            too_low = (lowers != -np.inf) & (px_trial < lowers)
            if np.any(too_low):
                alpha = np.min((lowers[too_low] - p0[too_low]) / step_direction[too_low])
                hit_limit = True

            # Check upper bounds
            too_high = (uppers != np.inf) & (px_trial > uppers)
            if np.any(too_high):
                alpha = np.min((uppers[too_high] - p0[too_high]) / step_direction[too_high])
                hit_limit = True

            if alpha < self.tiny_alpha:
                break

            # Final param calculation (clamped for safety)
            p_final = np.clip(p0 + alpha * step_direction, lowers, uppers)
            results.append((alpha, p_final.tolist(), step_limit, hit_limit))

            if do_break:
                break

        return results

'''
A mixin class for LsqrOptimizer that does the linear update direction step
by factorizing over images -- it solves the linear problem for each image
independently, and then combines those results (via their covariances) into
the overall result.
'''
class FactoredOptimizer(object):
    def __init__(self, *args, **kwargs):
        self.ps = None
        super().__init__(*args, **kwargs)

    def gsud_comp(self, tr, **kwargs):
        from tractor import Images
        img_opts = []
        imgs = tr.images
        mm = tr.modelMasks

        #Remove 'priors' from kwargs
        orig_priors = kwargs.pop('priors', True)
        max_size = 0
        for i,img in enumerate(imgs):
            tr.images = Images(img)
            if mm is not None:
                tr.modelMasks = [mm[i]]
            #Run with PRIORS = FALSE
            r = self.gsud1(tr, priors=False, max_size=max_size, **kwargs)
            if r is None:
                continue
            x,x_icov,A,colscales,B,Ao = r
            max_size = max(max_size, len(x))
            #print('FO: X', x, 'x_icov', x_icov)
            img_opts.append(r)
        tr.images = imgs
        tr.modelMasks = mm
        return img_opts

    def gsud1(self, tr, max_size=0, **kwargs):
        t = time.time()
        allderivs = tr.getDerivs()
        tt[7] += time.time()-t
        print ("GSUD ALLDERIVS", len(allderivs), tt[7])
        r = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, max_size=max_size, **kwargs)
        if r is None:
            return None
        x,A,colscales,B,Ao = r
        icov = np.matmul(A.T, A)
        return x, icov, A, colscales, B, Ao


    def getSingleImageUpdateDirection(self, tr, max_size=0, **kwargs):
        t = time.time()
        allderivs = tr.getDerivs()
        tt[7] += time.time()-t
        tps[0] += time.time()-t
        t = time.time()
        #print ("GSUD ALLDERIVS", len(allderivs), tt[7])
        r = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, max_size=max_size, **kwargs)
        tps[1] += time.time()-t
        if r is None:
            return None
        x,A,colscales,B,Ao = r
        #print ("X", x.shape, x)
        #print ("A", A.shape, A.max())
        #print ("B", B.shape, B.max())
        # print('SingeImageUpdateDirection: tr thawed params:')
        # tr.printThawedParams()
        # print('allderivs:', len(allderivs))

        if self.ps is not None:
            mod0 = tr.getModelImage(0)
            tim = tr.getImage(0)
            B = ((tim.getImage() - mod0) * tim.getInvError()).ravel()
            import pylab as plt
            plt.clf()
            ima = dict(interpolation='nearest', origin='lower')
            rr,cc = 3,4
            plt.subplot(rr,cc,1)
            plt.imshow(mod0, **ima)
            plt.title('mod0')
            sh = mod0.shape
            plt.subplot(rr,cc,2)
            mx = max(np.abs(B))
            imx = ima.copy()
            imx.update(vmin=-mx, vmax=+mx)
            plt.imshow(B.reshape(sh), **imx)
            plt.title('B')
            AX = np.dot(A, x)
            plt.subplot(rr,cc,3)
            plt.imshow(AX.reshape(sh), **imx)
            plt.title('A X')
            ah,aw = A.shape
            for i in range(min(aw, 8)):
                plt.subplot(rr,cc,5+i)
                plt.imshow(A[:,i].reshape(sh), **ima)
                if i == 0:
                    plt.title('dx')
                elif i == 1:
                    plt.title('dy')
                elif i == 2:
                    plt.title('dflux')
            self.ps.savefig()

        t = time.time()
        icov = np.matmul(A.T, A)
        tps[2] += time.time()-t
        del A
        return x, icov

    def getSingleImageUpdateDirections(self, tr, **kwargs):
        from tractor import Images
        img_opts = []
        imgs = tr.images
        mm = tr.modelMasks

        #Remove 'priors' from kwargs
        orig_priors = kwargs.pop('priors', True)
        max_size = 0
        for i,img in enumerate(imgs):
            tr.images = Images(img)
            if mm is not None:
                tr.modelMasks = [mm[i]]
            #Run with PRIORS = FALSE
            r = self.getSingleImageUpdateDirection(tr, priors=False, max_size=max_size, **kwargs)
            if r is None:
                continue
            x,x_icov = r
            max_size = max(max_size, len(x))
            #print('FO: X', x, 'x_icov', x_icov)
            img_opts.append((x,x_icov))
        tr.images = imgs
        tr.modelMasks = mm
        return img_opts

    def getLinearUpdateDirection(self, tr, **kwargs):

        # We can fit for image-based parameters (eg, sky level) and source-based parameters.
        # If image parameters are being fit, use the base code (eg in lsqr_optimizer.py)
        # to fit those, and prepend them to the results below.
        t = time.time()
        tct[1] += 1
        #tr.images = tr.images[:5]
        #if tct[1] > 2:
        #    import sys
        #    sys.exit(0)
        x_imgs = None
        image_thawed = tr.isParamThawed('images')
        if image_thawed:
            cat_frozen = tr.isParamFrozen('catalog')
            if not cat_frozen:
                tr.freezeParam('catalog')
            x_imgs = super().getLinearUpdateDirection(tr, **kwargs)
            if not cat_frozen:
                tr.thawParam('catalog')
            else:
                return x_imgs

        if image_thawed:
            tr.freezeParam('images')
        #print('getLinearUpdateDirection( kwargs=', kwargs, ')')
        #print ("Running Factored getSingleImageUpdateDirections")
        if len(tr.images) == 0:
            if x_imgs is not None:
                return x_imgs
            return None
        #Store original value of priors
        orig_priors = kwargs['priors']
        print (f'{orig_priors=}')
        img_opts = self.getSingleImageUpdateDirections(tr, **kwargs)
        tt[5] += time.time()-t
        t = time.time()
        if len(img_opts) == 0:
            if x_imgs is not None:
                return x_imgs
            return None
        # ~ inverse-covariance-weighted sum of img_opts...
        xicsum = 0
        icsum = 0
        if len(img_opts) == 1 and len(img_opts[0]) == 3:
            xicsum, icsum, _ = img_opts[0]
        else:
            for x,ic in img_opts:
                print(f'{x=} {ic=} {tr.catalog[0]=}')
                xicsum = xicsum + np.dot(ic, x)
                icsum = icsum + ic
        #C = np.linalg.inv(icsum)
        #x = np.dot(C, xicsum)
        print (f'{icsum=} {xicsum=}')
        if np.any(np.isnan(xicsum)):
            print (f"WARNING: np.dot failed with NAN {xicsum=}.  Running CPU version instead.")
            x = super().getLinearUpdateDirection(tr, **kwargs) 
            print ("SUPER x", x)
            return x 

        #Add the priors if needed.
        if orig_priors:
            priors_ATA, priors_ATB = self.getPriorsHessianAndGradient(tr)
            #print (f'{priors_ATA=}, {priors_ATB=}')
            # Add the raw priors to the sums
            #icsum += priors_ATA
            #xicsum += priors_ATB
            if priors_ATA.shape == icsum.shape:
                icsum += priors_ATA
                xicsum += priors_ATB
            elif np.all(priors_ATA == 0) and np.all(priors_ATB == 0):
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=} but priors are zero so ignorning.")
            else:
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=}; using CPU mode instead.")
                return super().getLinearUpdateDirection(tr, **kwargs)
        try:
            x,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
        except Exception as ex:
            print ("WARNING: lstsq failed "+str(ex)+"; using CPU version")
            return super().getLinearUpdateDirection(tr, **kwargs)
        tt[6] += time.time()-t
        #print (f'{tt=}')
        #print (f'{tps=}')
        print ("Final X: ",x, "Source", tr.catalog[0])
        #import sys
        #sys.exit(-1)
        if x_imgs is not None:
            x = np.append(x_imgs, x)
        #print (f'{icsum=} {xicsum=}')
        if image_thawed:
            tr.thawParam('images')
        return x

from tractor.smarter_dense_optimizer import SmarterDenseOptimizer

#class FactoredDenseOptimizer(FactoredOptimizer, ConstrainedDenseOptimizer):
class FactoredDenseOptimizer(FactoredOptimizer, SmarterDenseOptimizer):
    pass


class GPUFriendlyOptimizer(FactoredDenseOptimizer):
    _gpumode = 3

    def setGPUMode(self, gpumode):
        print ("Setting GPUMODE = ",gpumode)
        #0 = run CPU only
        #1 = run GPU only
        #2 = run VECTORIZED only
        #3 = run all
        #10 = run CPU only - Profile only
        #11 = run GPU only - Profile only
        #12 = run VECTORIZED only - Profile only
        #13 = run all - Profile only
        self._gpumode = gpumode

    def printTiming(self):
        print ("Times:",tt,tct)
        print ("PSTimes: ", tps)
        print ("CPSTimes ", cps)
        print ("GPSTimes:", gps)
        print ("TU Times:", tu, tuc)

    def printMemory(self, tag=""):
        import cupy as cp
        import datetime
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        tot_bytes = mempool.total_bytes()
        print (f'{tag=} {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=} at ',datetime.datetime.now())
        if free_mem/1.e+9 < 20:
            import gc
            print ("Freeing memory")
            gc.collect()
            print (f'{tag=} {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=} at ',datetime.datetime.now())
            print ("GPU free all blocks")
            mpool = cp.get_default_memory_pool()
            mpool.free_all_blocks()
            print (f'{tag=} {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=} at ',datetime.datetime.now())
        sys.stdout.flush()

    def getSingleImageUpdateDirections(self, tr, **kwargs):
        #import traceback
        #traceback.print_stack()
        print ("GPU getSingleImageUpdateDirections")
        print ("profile galaxy", isinstance(tr.catalog[0], ProfileGalaxy))
        print ("len", len(tr.catalog), "Nimages", len(tr.images))
        R_gpu = None
        R_cpu = None
        R_gpuv = None
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            if self._gpumode >= 10:
                print ("Skipping non-profile galaxy")
                return []
            if self._gpumode == 2 and isinstance(tr.catalog[0], PointSource):
                tct[5] += 1
                #print ("TCT5 PSCOUNT", tct[5])
                #im = tr.images
                #mm = tr.modelMasks
                #tr.images = tr.images[:1]
                #tr.modelMasks = tr.modelMasks[:1]
                t = time.time()
                # Initialize the GPU Fitter
                print ("Running new GPU point source fitter...")
                # Initialize and CACHE the fitter on self
                self.current_fitter = GPUFriendlyPointSourceFitter(tr, tr.catalog[0], tr.images)
                #R_cpu = super().gsud_comp(tr, **kwargs)
                #self.current_fitter.add_comp(R_cpu)
                tx1 = time.time()
                allderivs = tr.getDerivs()
                tt[7] += time.time()-tx1
                #print ("GPUPS ALLDERIVS", len(allderivs), tt[7])
                tps[0] += time.time()-t

                # Get the update
                # Solve A^T A x = A^T b
                """
                tx1 = time.time()
                ohess, ograd, ni = self.current_fitter.compute_update(allderivs)
                #ohess, ograd, ni = self.current_fitter.compute_update_old(allderivs) # Updated to use internal G
                t5[0] += time.time()-tx1
                ox,_,_,_ = np.linalg.lstsq(ohess.get(), ograd.get(), rcond=None)
                print (f'OLD {ohess=} {ograd=} {ni=} {ox=}')
                #self.current_fitter.get_batch_psf()
                hess, grad, x = ohess, ograd, ox
                """

                tx1 = time.time()
                hess, grad, ni = self.current_fitter.compute_update_vec(allderivs) # Updated to use internal G
                t5[1] += time.time()-tx1
                #x,_,_,_ = np.linalg.lstsq(hess.get(), grad.get(), rcond=None)

                #print (f'NEW {hess=} {grad=} {ni=} {x=}')
                #print (f'{t5=}')
                if ni == 0 or np.all(grad == 0):
                    print ("EMPTY SET")
                    return []
                """
                if np.allclose(ohess, hess, atol=1.e-5) and np.allclose(ograd, grad, atol=1.e-4):
                    print ("MATCH")
                else:
                    if np.allclose(ox, x, atol=1.e-5):
                        print ("MATCHX")
                    print ("NO MATCH")
                """

                #step = cp.linalg.solve(hess, grad)
                # Replace step = cp.linalg.solve(hess, grad)
                """
                res = cp.linalg.lstsq(hess, grad, rcond=1e-7)
                print (f'{res=}')
                step = res[0]
                """

                #if len(R_cpu) == 0:
                if ni == 0 or np.all(grad == 0):
                    print ("EMPTY SET")
                    return []

                """
                gopt = [(step.get(), hess.get())]
                gxicsum = 0
                gicsum = 0
                for gx, gic in gopt:
                    print(f'{gx=} {gic=} {tr.catalog[0]=}')
                    gxicsum = gxicsum + np.dot(gic, gx)
                    gicsum = gicsum + gic
                gxicsum = grad.get()
                gxf,_,_,_ = np.linalg.lstsq(gicsum, gxicsum, rcond=None)
                print ("gpu X_full: ",gxf)


                xicsum = 0
                icsum = 0
                for cx,cic,_,_,_,_ in R_cpu:
                    xicsum = xicsum + np.dot(cic, cx)
                    icsum = icsum + cic
                    print(f'{cx=} {cic=} {xicsum=} {icsum=} {tr.catalog[0]=}')
                print (f'{icsum=} {xicsum=}')
                cxf,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
                print ("cpu X full: ", cxf)

                if R_cpu is not None:
                    try:
                        assert(np.allclose(gicsum, icsum))
                        print (f'{tr.catalog[0]=} icsum full match')
                    except AssertionError as ex:
                        print (f'{tr.catalog[0]=} icsum full does not match {gicsum=} {icsum=} {hess=}')
                    try:
                        assert(np.allclose(gxicsum, xicsum))
                        print (f'{tr.catalog[0]=} xicsum full match')
                    except AssertionError as ex:
                        print (f'{tr.catalog[0]=} xicsum full does not match {gxicsum=} {xicsum=} {grad=} {step=}')
                    try:
                        assert(np.allclose(cxf, gxf))
                        print (f'{tr.catalog[0]=} X full match')
                    except AssertionError as ex:
                        print (f'{tr.catalog[0]=} X full does not match {gxf=} {cxf=}')

                """
                tt[0] += time.time()-t
                #tr.images = im
                #tr.modelMasks = mm

                #return [(step.get(), hess.get())]
                return [(grad.get(), hess.get(), 1)]
            elif isinstance(tr.catalog[0], PointSource):
                tct[6] += 1
            else:
                tct[4] += 1

            print ("Running CPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
            #p = self.ps
            #self.ps = None
            t = time.time()
            #print ("SRC", type(tr.catalog[0]))
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            #if len(R) != 0:
            #    fname = "bad_nonprofile.pickle"
            #    if not os.access(fname, os.F_OK):
            #        f = open(fname,'wb')
            #        import pickle
            #        pickle.dump(tr, f)
            #        f.close()
            tt[0] += time.time()-t
            tct[0] += 1
            #self.ps = p
            return R

        if self._gpumode == 1 or self._gpumode == 3 or self._gpumode == 11 or self._gpumode == 13:
            try:
                print('Running GPU code...')
                if not (tr.isParamFrozen('images') and (len(tr.catalog) == 1) and isinstance(tr.catalog[0], ProfileGalaxy)):
                    print ("Running GPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
                    tct[0] += 1
                t = time.time()
                R_gpu = self.gpuSingleImageUpdateDirections(tr, **kwargs)
                tt[1] += time.time()-t
                tct[1] += 1
                #print ("GPU time:",time.time()-t)
                if self._gpumode == 1 or self._gpumode == 11:
                    return R_gpu
            except AssertionError:
                import traceback
                print ("AssertionError in GPU code:")
                traceback.print_exc()
                print ("Running CPU version instead...")
                t = time.time()
                R_gpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 1 or self._gpumode == 11:
                    return R_gpu
            except:
                import traceback
                print('Exception in GPU code:')
                traceback.print_exc()
 
                src = tr.catalog[0]
                print('Source:', src)
                print(repr(src))
                print ("Running CPU version instead...")
                t = time.time()
                R_gpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 1 or self._gpumode == 11:
                    return R_gpu

        if self._gpumode == 2 or self._gpumode == 3 or self._gpumode == 12 or self._gpumode == 13 or self._gpumode == 4:
            import cupy as cp
            import datetime
            #Get free memory
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            #Estimate memory needed with new helper method based on padded image sizes and nd
            nimages = len(tr.images)
            imsize = tr.images[0].data.size
            nd = tr.numberOfParams()+2
            src = tr.catalog[0]
            masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
            if any(m is None for m in masks):
                print (f"WARNING: One or more modelMasks is None; running CPU version.")
                xout = super().getSingleImageUpdateDirections(tr, **kwargs)
                return xout

            assert(all([m is not None for m in masks]))
            # Pixel positions
            pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
                   for tim in tr.images]
            px, py = np.array(pxy).T
            if np.any(np.isnan(px)) or np.any(np.isnan(py)):
                print ("WARNING: px, py contain NANs.  Running CPU version instead.")
                xout = super().getSingleImageUpdateDirections(tr, **kwargs)
                return xout
            #New helper method predicts memory needed
            est_mem = self.predict_fft_memory(tr, masks, pxy, nd)
            free_mem /= 1.e+9
            # 3.2 factor: NGC 3585 example
            print (f'Estimated memory {est_mem} GB free memory {free_mem} for {nimages=} {imsize=} {nd=} at ', datetime.datetime.now(), "src = ", tr.catalog[0])

            if free_mem < est_mem:
                try:
                    import datetime
                    print(f"Warning: Estimated memory {est_mem} GB is greater than free memory {free_mem} GB; Running less-memory GPU mode instead!", datetime.datetime.now(), "src = ", tr.catalog[0])
                    #print("Warning: Estimated memory %.1f GB is greater than free memory %.1f GB; Running less-memory GPU mode instead!" % (est_mem / 1e9, free_mem / 1e9))
                    t = time.time()
                    R_gpu = self.gpuSingleImageUpdateDirectionsVectorized_less_mem(tr, **kwargs)
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    tt[4] += time.time()-t
                    return R_gpu
                except Exception as e:
                    print('Fallback to less-memory GPU version failed:', e)
                    import traceback
                    traceback.print_exc()
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()

                print('Running CPU version instead')
                R_cpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                return R_cpu

            try:
                #import datetime
                #print ('Running VECTORIZED GPU code...', datetime.datetime.now())
                t = time.time()
                R_gpuv = self.gpuSingleImageUpdateDirectionsVectorized(tr, **kwargs)
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                #print ("Finished VECTORIZED code")
                tt[2] += time.time()-t
                tct[2] += 1
                #print ("GPU Vectorized time:", time.time()-t)
                if self._gpumode == 2 or self._gpumode == 12 or self._gpumode == 4:
                    return R_gpuv
            except Exception as e:
                #New consolidated except
                import cupy as cp
                # 1. Capture the error string
                err_msg = str(e).lower()
                is_oom = "outofmemory" in err_msg or "out of memory" in err_msg

                # 2. Print immediately to stderr (bypasses buffering)
                import sys
                sys.stderr.write(f"\nCaught Exception in GPU code: {type(e).__name__}\n")
                if is_oom:
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    mempool = cp.get_default_memory_pool()
                    used_bytes = mempool.used_bytes()
                    tot_bytes = mempool.total_bytes()
                    sys.stderr.write(f"Confirmed OOM for source: {tr.catalog[0]}\n")
                    print (f'OOM Device {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=}')

                sys.stderr.write(f"\n----Traceback below----\n")
                import traceback
                traceback.print_exc()
                sys.stderr.flush()

                # 3. Emergency Memory Release
                try:
                    import cupy as cp
                    # Force drop image caches if they exist
                    for tim in tr.images:
                        if hasattr(tim, 'data_gpu'): tim.data_gpu = None
                        if hasattr(tim, 'inverr_gpu'): tim.inverr_gpu = None
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass

                # 4. Final Fallback
                print("Running CPU version instead... for source: "+str(tr.catalog[0]))
                sys.stdout.flush()
                t = time.time()
                R_gpuv = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 2 or self._gpumode == 12 or self._gpumode == 4:
                    return R_gpuv

        if self._gpumode == 0 or self._gpumode == 3 or self._gpumode == 10 or self._gpumode == 13:
            try:
                print('Running CPU code for comparison...')
                if not (tr.isParamFrozen('images') and (len(tr.catalog) == 1) and isinstance(tr.catalog[0], ProfileGalaxy)):
                    print ("Running CPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
                    tct[0] += 1
                    #print ("TCT0", tct[0])
                #else:
                #    return []
                t = time.time()
                R_cpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                tt[3] += time.time()-t
                tct[3] += 1
                #print ("TCT3", tct[3])
                #print ("CPU time", time.time()-t)
                if self._gpumode == 0 or self._gpumode == 10:
                    return R_cpu
            except Exception as ex:
                import traceback
                print('Exception in CPU code:')
                traceback.print_exc()

                src = tr.catalog[0]
                print('Source:', src)
                print(repr(src))
                raise(ex)
                #sys.exit(-1)

        xicacc_cpu = 0.
        icacc_cpu = 0.
        for x,ic in R_cpu:
            #print (f'{ic=} {x=}')
            xicacc_cpu = xicacc_cpu + np.dot(ic, x)
            icacc_cpu = icacc_cpu + ic
        #print (f'{icacc_cpu=} {xicacc_cpu=}')
        x_cpu,_,_,_ = np.linalg.lstsq(icacc_cpu, xicacc_cpu, rcond=None)
            
        xicacc_gpu = 0.
        icacc_gpu = 0.
        for x,ic in R_gpu:
            #print (f'{ic=} {x=}')
            xicacc_gpu = xicacc_gpu + np.dot(ic, x)
            icacc_gpu = icacc_gpu + ic
        #print (f'{icacc_gpu=} {xicacc_gpu=}')
        x_gpu,_,_,_ = np.linalg.lstsq(icacc_gpu, xicacc_gpu, rcond=None)

        xicacc_gpuv = 0.
        icacc_gpuv = 0.
        for x,ic in R_gpuv:
            xicacc_gpuv = xicacc_gpuv + np.dot(ic, x)
            icacc_gpuv = icacc_gpuv + ic
        x_gpuv,_,_,_ = np.linalg.lstsq(icacc_gpuv, xicacc_gpuv, rcond=None)

        print('CPU:', x_cpu)
        print('GPU:', x_gpu)
        print('GPU V:', x_gpuv)
        s = np.sum(x_cpu * x_gpu) / np.sqrt(np.sum(x_cpu**2) * np.sum(x_gpu**2))
        sv = np.sum(x_gpu * x_gpuv) / np.sqrt(np.sum(x_gpu**2) * np.sum(x_gpuv**2))
        print('Similarity CPU/GPU:', s)
        print('Similarity GPU/V:', sv)
        print ("Times:",tt,tct)
        """
        if s < 0.999 or s > 1.0001:
            src = tr.catalog[0]
            print('Source:', src)
            self.outputPickle("full.new.good", tr)
            print ("SIMILARITY BAD  source="+str(tr.catalog[0]))
            for i,((x_cpu,ic_cpu), (x_gpu,ic_gpu)) in enumerate(zip(R_cpu, R_gpu)):
                print('  CPU:', x_cpu)
                print('  GPU:', x_gpu)
                s = np.sum(x_cpu * x_gpu) / np.sqrt(np.sum(x_cpu**2) * np.sum(x_gpu**2))
                print('  similarity:', s)
                if s < 0.999 or s > 1.0001:
                  try:
                    print ("I", i, len(tr.images))
                    tim = tr.images[i]
                    print('Tim:', tim)

                    num = tt2[0]
                    fname = os.getenv('SCRATCH')+'/pickles/bad_x_new'+str(num)+'.pickle'
                    while os.access(fname, os.F_OK):
                        num += 100
                        fname = os.getenv('SCRATCH')+'/pickles/bad_x_new'+str(num)+'.pickle'
                    z = tr.images
                    tr.images = [tim]
                    self.outputPickle("x", tr, num=num)
                    tr.images = z
                    if tt2[0] > 40:
                        #Exit at 10 differences
                        sys.exit(-1)
                  except Exception as ex:
                    print ("WEIRD CRASH", str(ex))
            """
        return R_gpu

    def outputPickle(self, name, tr, num=None):
        if not os.access(os.getenv('SCRATCH')+'/pickles',os.F_OK):
            os.mkdir(os.getenv('SCRATCH')+'/pickles')
        if num is None:
            num = tt2[0]
        fname = os.getenv('SCRATCH')+'/pickles/bad_'+name+'_'+str(num)+'.pickle'
        if os.access(fname, os.F_OK):
            print ("OUTPUT ERROR: "+fname+" already exists")
            return
        print ("OUTPUTTING "+fname)
        print ("N images = ", len(tr.images))
        tt2[0] += 1
        #f = open(fname,'wb')
        #import pickle
        #pickle.dump(tr, f)
        #f.close()
        #print ("SIMILARITY BAD blobid="+str(tr.blobid)+" source="+str(tr.catalog[0]))
        #print ("SIMILARITY BAD  source="+str(tr.catalog[0]))
        return

    def gpuPointSourceUpdateDirections(self, tr, **kwargs):
        #Should call batch_pointsource to get derivs, propagates to batch_psf
        #This should be equivalent of G array and then propagate to A, B arrays as in galaxies.
        allderivs = tr.getDerivs()
        #ComputeUpdateDirections after calculation of G should then be performed
        return []

    def gpuSingleImageUpdateDirections(self, tr, **kwargs):
        
        print('Using GpuFriendly code')
        #if not (tr.isParamFrozen('images') and (len(tr.catalog) == 1) and isinstance(tr.catalog[0], ProfileGalaxy)):
        #    return gpuPointSourceUpdateDirections(tr, **kwargs)

        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        #assert(all([tim.sky.getConstant() == 0 for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)
        #img_pix = [tim.data for tim in tr.images]
        img_pix = [tim.getImage(use_gpu=True) for tim in tr.images]
        img_ie  = [tim.getInvError(use_gpu=True) for tim in tr.images]
        # Assume galaxy
        src = tr.catalog[0]
        assert(isinstance(src, ProfileGalaxy))
        psfs = [tim.getPsf() for tim in tr.images]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))

        # Assume ConstantSky models, grab constant sky levels
        # NOTE - instead of building this list and passing it around in ImageDerivs, etc,
        # we could perhaps just subtract it off img_pix at the start...
        img_sky = [tim.getSky().getConstant() for tim in tr.images]

        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        assert(all([m is not None for m in masks]))

        # Assume we *are* fitting for the (RA,Dec) position
        assert(src.isParamThawed('pos'))

        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tr.images]
        # WCS inv(CD) matrix
        img_cdi = [tim.getWcs().cdInverseAtPosition(src.getPosition(), src=src)
                   for tim in tr.images]
        # Current counts
        img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                      for tim in tr.images]
        bands = src.getBrightness().getParamNames()
        #print('Bands:', bands)
        img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]
        #print('ibands', img_bands)

        # (x0,x1,y0,y1) in image coordinates
        extents = [mm.extent for mm in masks]

        inner_real_nsigma = 3.
        outer_real_nsigma = 4.

        nimages = len(masks)
        gpu_px = np.zeros(nimages, dtype=np.float32)
        gpu_py = np.zeros(nimages, dtype=np.float32)
        gpu_halfsize = np.zeros(nimages, dtype=np.float32)
        sourceOut = np.zeros(nimages, dtype=bool)
        i = 0
        for mm,(px,py),(x0,x1,y0,y1),psf,pix,ie,counts,cdi,tim in zip(
                masks, pxy, extents, psfs, img_pix, img_ie, img_counts, img_cdi, tr.images):

            psfH,psfW = psf.shape
            halfsize = max([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2])
            gpu_px[i] = px
            gpu_py[i] = py
            gpu_halfsize[i] = halfsize
            sourceOut[i] = (px < x0 or px > x1 - 1 or py < y0 or py > y1 - 1)
            i += 1
        #print (f'{px=} {x0=} {x1=} {py=} {y0=} {y1=} {sourceOut=}')
        #print (f'{gpu_halfsize}')

        #print (f'{sourceOut=}')
        """
        if np.any(sourceOut):
            print ("running CPU version")
            tct[7] += 1
            sout[0] += sourceOut.sum()
            sout[1] += nimages
            t = time.time()
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            tt[7] += time.time()-t
            return xout 
        """

        # PSF Fourier transforms
        #print ("Nimges = ",nimages, gpu_px.shape, gpu_py.shape)
        #for psf in psfs:
        #    print ("PSF type", type(psf), psf)
        #    print ("PIX type", type(psf.pix), psf.pix)
        batch_psf = BatchPixelizedPSF(psfs)
        t1 = time.time()
        #print('Getting Fourier transform of PSF at', gpu_px,gpu_py)
        P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(gpu_px, gpu_py, gpu_halfsize)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (nimages,len(w),len(v)))

        t1 = time.time()
        img_params = BatchImageParams(P, v, w, batch_psf.psf_mogs)

        #Not optimal but for now go back into loop
        pi = 0
        for mm,(px,py),(x0,x1,y0,y1),psf,pix,ie,counts,sky,cdi,tim in zip(
                masks, pxy, extents, psfs, img_pix, img_ie, img_counts, img_sky, img_cdi, tr.images):
            #Subtract 1 from y0 and x0 CW 1/22/25 -- unknown why but rectifies difference with CPU IE
            y_delta = 1
            x_delta = 1

            # sub-pixel shift we have to do at the end...
            dx = px - cx
            dy = py - cy
            mux = dx - x0
            muy = dy - y0
            sx = int(np.round(mux))
            sy = int(np.round(muy))

            if mm.y0 == 0 or sy == 0:
                y_delta = 0
            if mm.x0 == 0 or sx == 0:
                x_delta = 0
            if mm.y1 > pix.shape[0]:
                print ("Updating Y1")
                mm.h = pix.shape[0]-mm.y0
            if mm.x1 > pix.shape[1]:
                print ("Updating X1")
                mm.w = pix.shape[1]-mm.x0
            #mmpix = pix[mm.y0-1:mm.y1, mm.x0-1:mm.x1]
            #mmie =   ie[mm.y0-1:mm.y1, mm.x0-1:mm.x1]
            mmpix = pix[mm.y0-y_delta:mm.y1, mm.x0-x_delta:mm.x1]
            mmie =   ie[mm.y0-y_delta:mm.y1, mm.x0-x_delta:mm.x1]

            # PSF Fourier transforms
            #P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)
            mh,mw = mm.shape

            # the subpixel portion will be handled with a Lanczos interpolation
            mux -= sx
            muy -= sy
            dxi = x0+sx
            dyi = y0+sy
            assert(np.abs(mux) <= 0.5)
            assert(np.abs(muy) <= 0.5)

            # Embed pix and ie in images the same size as pW,pH.
            padpix = cp.zeros((pH,pW), cp.float32)
            padie  = cp.zeros((pH,pW), cp.float32)
            assert(sy <= 0 and sx <= 0)
            #print (f'{sy=} {y_delta=} {mh=} {sx=} {x_delta=} {mw=}')

            padpix[-sy-y_delta: -sy+mh, -sx-x_delta: -sx+mw] = mmpix
            padie [-sy-y_delta: -sy+mh, -sx-x_delta: -sx+mw] = mmie
            roi = (-sx, -sy, mw, mh)
            mmpix = padpix
            mmie  = padie
            sx = sy = 0
            mh = pH
            mw = pW

            # Compute the mixture-of-Gaussian components for this galaxy model
            # (at its current parameter values)
            amix = src._getShearedProfile(tim, px, py)

            # Get derivatives for each galaxy shape parameter.
            amixes = src.getDerivativeShearedProfiles(tim, px, py)
            amixes = [('current', amix, 0.)] + amixes

            # Split "amix" into terms that we will evaluate using MoG vs FFT.
            # (we'll use that same split for the derivatives also)
            nvar =  amix.var.shape[0]
            for i in range(len(amixes)):
                assert(amixes[i][1].var.shape[0] == nvar)
                #maxnvar = max(maxnvar, amixes[i][1].var.shape[0])
            vv = np.zeros((len(amixes), amix.var.shape[0]))
            #vv = np.zeros((len(amixes), maxnvar))
            for i, am in enumerate(amixes):
                #print ("VV", vv.shape, "AMV", am[1].var.shape)
                vv[i] = am[1].var[:,0,0]+am[1].var[:,1,1]
                #vv[i,:am[1].var[:,0,0].size] = am[1].var[:,0,0]+am[1].var[:,1,1]
            #vv = amix.var[:, 0, 0] + amix.var[:, 1, 1]
            # Ramp between:
            nsigma1 = inner_real_nsigma
            nsigma2 = outer_real_nsigma
            # Terms that will wrap-around significantly if evaluated
            # with FFT...  We want to know: at the outer edge of this
            # patch, how many sigmas out are we?  If small (ie, the
            # edge still has a significant fraction of the flux),
            # render w/ MoG.
            IM = ((pW/2)**2 < (nsigma2**2 * vv))
            IF = ((pW/2)**2 > (nsigma1**2 * vv))
            ramp = np.any(IM*IF)
            #mogweights = 1.
            #fftweights = 1.
            mogweights = np.ones(vv.shape, dtype=np.float32)
            fftweights = np.ones(vv.shape, dtype=np.float32)
            if ramp:
                # ramp
                ns = (pW/2) / np.maximum(1e-6, np.sqrt(vv))
                #mogweights = np.minimum(1., (nsigma2 - ns[IM]) / (nsigma2 - nsigma1))
                #fftweights = np.minimum(1., (ns[IF] - nsigma1) / (nsigma2 - nsigma1))
                mogweights = np.minimum(1., (nsigma2 - ns) / (nsigma2 - nsigma1))*IM
                fftweights = np.minimum(1., (ns - nsigma1) / (nsigma2 - nsigma1))*IF
                assert(np.all(mogweights[IM] > 0.))
                assert(np.all(mogweights[IM] <= 1.))
                assert(np.all(fftweights[IF] > 0.))
                assert(np.all(fftweights[IF] <= 1.))

            K = amix.var.shape[0]
            D = amix.var.shape[1]
            # are we fitting for the position of this source?
            fit_pos = (src.getPosition().numberOfParams() > 0)
            #print ("LEN AMIXES", len(amixes))
            img_derivs = ImageDerivs(amixes, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, cdi, roi, sky, dxi, dyi, fit_pos)
            img_params.add_image_deriv(img_derivs)
            #Commented out print below
            #img_derivs.tostr()

            assert(sx == 0 and sy == 0)
            pi += 1

        #Call collect_params() to finalize BatchImageParams object with all ImageDerivs
        img_params.collect_params()
        #nbands = 1 + max(img_bands)
        nbands = len(bands)

        priorVals = tr.getLogPriorDerivatives()
        if priorVals is not None:
            # Adjust the priors to handle the single-band case that we consider...
            bright = src.getBrightness()
            pnames = bright.getThawedParams()
            assert(len(pnames) > 0)
            bright.freezeAllBut(pnames[0])
            priorVals = tr.getLogPriorDerivatives()
            print('Prior vals with one band:', priorVals)
            bright.thawParams(*pnames)

        Xic = self.computeUpdateDirections(img_params, priorVals)
        #print ("LEN Xic", len(Xic))

        if nbands > 1:
            t1 = time.time()
            full_xic = []
            fullN = tr.numberOfParams()
            # number of source position parameters
            # - these would be RA and Dec, except for when we're fitting Gaia stars,
            #   whose positions are assumed correct and not re-fit!
            npos = src.getPosition().numberOfParams()
            assert(npos in [0,2])
            # The GPU-based fitting was done assuming 2 positions being fit
            # (the coefficients are zeroed out, but the resulting arrays "x" below are still sized to
            #  hold this many positional parameters)
            npos_fit = 2
            # fitting was done on a single band
            nbands_fit = 1
            for iband,(x,ic) in zip(img_bands, Xic):
                #print ("X1", x.shape, x, "IC1", ic)
                #assert(fullN == len(x) + nbands - 1)
                # the "x" here are from fitting on a single image, *assuming* 2 positions are being fit
                assert(fullN == len(x) + nbands - nbands_fit + npos - npos_fit)
                x2 = cp.zeros(fullN, cp.float32)
                ic2 = cp.zeros((fullN,fullN), cp.float32)
                # source params are ordered: position, brightness, others
                #npos = 2
                #nothers = len(x)-3
                nothers = len(x) - npos_fit - nbands_fit

                # Where aa is a block of npos elements
                #       cc is a block of nothers elements
                #       b is a single element
                #       z1,z2 are some number of zeros
                #
                # x = [ aa b cc ] ---> x2 = [ aa z1 b z2 cc ]
                
                # ic = [ A B C ]      ic2 = [ A z B z C ]
                #      [ B D E ]  -->       [ z z z z z ]
                #      [ C E F ]            [ B z D z E ]
                #                           [ z z z z z ]
                #                           [ C z E z F ]
                # (note, sometimes npos=0, so then we have x2[:0] = x[:0], ie a no-op, but that's okay)
                #print (f'{nothers=} {npos_fit=} {nbands_fit=} {npos=} {nbands=} {iband=}')

                # aa
                x2[:npos] = x[:npos]
                # b
                if npos == 0:
                    x2[npos + iband] = x[npos_fit]
                else:
                    x2[npos + iband] = x[npos]
                # cc
                x2[-nothers:] = x[-nothers:]

                # A
                ic2[:npos,:npos] = ic[:npos,:npos]
                # B
                ic2[npos + iband, :npos] = ic[npos, :npos]
                ic2[:npos, npos + iband] = ic[:npos, npos]
                # C
                ic2[:npos, -nothers:] = ic[:npos,    -nothers:]
                ic2[-nothers:, :npos] = ic[-nothers:, :npos]
                # D
                if npos == 0:
                    ic2[npos + iband, npos + iband] = ic[npos_fit, npos_fit]
                else:
                    ic2[npos + iband, npos + iband] = ic[npos, npos]
                # E
                if npos == 0:
                    ic2[npos + iband, -nothers:] = ic[npos_fit, -nothers:]
                    ic2[-nothers:, npos + iband] = ic[-nothers:, npos_fit]
                else:
                    ic2[npos + iband, -nothers:] = ic[npos, -nothers:]
                    ic2[-nothers:, npos + iband] = ic[-nothers:, npos]
                # F
                ic2[-nothers:,-nothers:] = ic[-nothers:,-nothers:]

                # print('x')
                # print(x)
                # print('x2')
                # print(x2)
                # print('ic')
                # print(ic)
                # print('ic2')
                # print(ic2)
                
                full_xic.append((x2.get(),ic2.get()))
                #print ("X2", x2, "IC2", ic2)
            Xic = full_xic
        else:
            print ("SKIPPING nbands branch - fullN=",fullN, "nbands=",nbands,"X",Xic[0][0])

        #
        #print('Calling original version...')
        #sXic = super().getSingleImageUpdateDirections(tr, **kwargs)

        return Xic

    def gpuSingleImageUpdateDirectionsVectorized_less_mem(self, tr, **kwargs):
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            p = self.ps
            self.ps = None
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            self.ps = p
            return R
        if len(tr.images) == 0:
            print (f"WARNING: len images == 0, running CPU version")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout
        tims = tr.images
        modelmasks = tr.modelMasks
        try:
            xx = []
            for i,(tim,mm) in enumerate(zip(tims, modelmasks)):
                tr.images = [tim]
                tr.modelMasks = [mm]
                src = tr.catalog[0]
                sx,sy = tim.getWcs().positionToPixel(src.getPosition(), src)
                h,w = tim.shape
                try:
                    x = self.gpuSingleImageUpdateDirectionsVectorized(tr, **kwargs)
                except Exception as e:
                    x = super().getSingleImageUpdateDirections(tr, **kwargs)
                xx.extend(x)
        finally:
            tr.images = tims
            tr.modelMasks = modelmasks
        return xx


    def gpuSingleImageUpdateDirectionsVectorized(self, tr, **kwargs):
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            p = self.ps
            self.ps = None
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            self.ps = p
            return R
        if len(tr.images) == 0: 
            print (f"WARNING: len images == 0, running CPU version")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout 

        #print('Using GpuFriendly vectorized code')
        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        #assert(all([tim.sky.getConstant() == 0 for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)
        t1 = time.time()
        Nimages = len(tr.images)

        #img_pix = [tim.data for tim in tr.images]
        img_pix = [tim.getImage(use_gpu=True) for tim in tr.images]
        img_ie  = [tim.getInvError(use_gpu=True) for tim in tr.images]
        # Assume galaxy
        src = tr.catalog[0]
        assert(isinstance(src, ProfileGalaxy))
        psfs = [tim.getPsf() for tim in tr.images]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))

        # Assume ConstantSky models, grab constant sky levels
        # NOTE - instead of building this list and passing it around in ImageDerivs, etc,
        # we could perhaps just subtract it off img_pix at the start...
        img_sky = [tim.getSky().getConstant() for tim in tr.images]

        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        masks = [tr._getModelMaskByIdx(i, src) for i in range(len(tr.images))]
        #masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        if any(m is None for m in masks):
            print (f"WARNING: One or more modelMasks is None; running CPU version.")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout
        assert(all([m is not None for m in masks]))

        assert(src.isParamThawed('pos'))

        # FIXME -- must handle priors (ellipticity)!!

        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tr.images]
        # WCS inv(CD) matrix
        img_cdi = [tim.getWcs().cdInverseAtPosition(src.getPosition(), src=src)
                   for tim in tr.images]
        # Current counts
        img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                      for tim in tr.images]
        bands = src.getBrightness().getParamNames()
        #print('Bands:', bands)
        try:
            img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]
        except ValueError as ex:
            print (f'WARNING: Image bands error - {bands=} - '+str(ex))
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout 
        #print('ibands', img_bands)

        #print ("Calling VECTORIZED version")
        img_params, cx,cy,pW,pH = self._getBatchImageParams(tr, masks, pxy)
        # Dustin FIXME - cx,cy,pW,pH should probably be in img_params, they're about the PSF sizes
        # and centering.
        
        px, py = np.array(pxy).T
        px = px.astype(np.float32)
        py = py.astype(np.float32)
        amix_gpu = getShearedProfileGPU(src, tr.images, px, py)
        amixes_gpu = getDerivativeShearedProfilesGPU(src, tr.images, px, py)
        amixes_gpu = [('current', amix_gpu, 0.)] + amixes_gpu

        img_derivs = self._getBatchGalaxyProfiles(amixes_gpu, masks, px, py, cx, cy, pW, pH,
                                                  img_counts, img_sky, img_pix, img_ie)
        img_params.addBatchGalaxyProfiles(img_derivs)

        # are we fitting for the position of this source?
        fit_pos = np.asarray([(src.getPosition().numberOfParams() > 0)]*Nimages)
        cdi = cp.asarray(img_cdi)
        img_params.addBatchGalaxyDerivs(cdi, fit_pos)

        #img_derivs.tostr()

        i = 0
        #Call collect_params() to finalize BatchImageParams object with all ImageDerivs
        #img_params.collect_params()
        #nbands = 1 + max(img_bands)
        nbands = len(bands)

        """
        #Comment out priorVals
        priorVals = tr.getLogPriorDerivatives()
        if priorVals is not None:
            # Adjust the priors to handle the single-band case that we consider...
            bright = src.getBrightness()
            pnames = bright.getThawedParams()
            assert(len(pnames) > 0)
            bright.freezeAllBut(pnames[0])
            priorVals = tr.getLogPriorDerivatives()
            print('Prior vals with one band:', priorVals)
            bright.thawParams(*pnames)
        """
        priorVals = None


        if img_params.ffts is None:
            print ("Warning> img_params.ffts is None!  Running CPU version")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout

        Xic = self.computeUpdateDirectionsVectorized(img_params, priorVals)
        print (f"{Xic=}")

        mpool = cp.get_default_memory_pool()
        del img_params
        del amixes_gpu
        mpool.free_all_blocks()
        #print (f'{nbands=}')

        if nbands >= 1:
            full_xic = []
            fullN = tr.numberOfParams()
            # number of source position parameters
            # - these would be RA and Dec, except for when we're fitting Gaia stars,
            #   whose positions are assumed correct and not re-fit!
            npos = src.getPosition().numberOfParams()
            assert(npos in [0,2])
            # The GPU-based fitting was done assuming 2 positions being fit
            # (the coefficients are zeroed out, but the resulting arrays "x" below are still sized to
            #  hold this many positional parameters)
            npos_fit = 2
            # fitting was done on a single band
            nbands_fit = 1
            (X, Xicov) = Xic
            X = X.get()
            Xicov = Xicov.get()
            print (f"{fullN=} {npos=} {nbands=}")
            for iband,x,ic in zip(img_bands, X, Xicov):
                #assert(fullN == len(x) + nbands - 1)
                # the "x" here are from fitting on a single image, *assuming* 2 positions are being fit
                assert(fullN == len(x) + nbands - nbands_fit + npos - npos_fit)
                x2 = np.zeros(fullN, np.float32)
                ic2 = np.zeros((fullN,fullN), np.float32)
                # source params are ordered: position, brightness, others
                #npos = 2
                #nothers = len(x)-3
                #print (f'{npos_fit=} {nbands_fit=} {npos=} {nbands=} {fullN=}')
                nothers = len(x) - npos_fit - nbands_fit

                # Where aa is a block of npos elements
                #       cc is a block of nothers elements
                #       b is a single element
                #       z1,z2 are some number of zeros
                #
                # x = [ aa b cc ] ---> x2 = [ aa z1 b z2 cc ]

                # ic = [ A B C ]      ic2 = [ A z B z C ]
                #      [ B D E ]  -->       [ z z z z z ]
                #      [ C E F ]            [ B z D z E ]
                #                           [ z z z z z ]
                #                           [ C z E z F ]
                # (note, sometimes npos=0, so then we have x2[:0] = x[:0], ie a no-op, but that's okay)

                # aa
                x2[:npos] = x[:npos]
                # b
                if npos == 0:
                    x2[npos + iband] = x[npos_fit]
                else:
                    x2[npos + iband] = x[npos]
                # cc
                x2[-nothers:] = x[-nothers:]

                # A
                ic2[:npos,:npos] = ic[:npos,:npos]
                # B
                ic2[npos + iband, :npos] = ic[npos, :npos]
                ic2[:npos, npos + iband] = ic[:npos, npos]
                # C
                ic2[:npos, -nothers:] = ic[:npos,    -nothers:]
                ic2[-nothers:, :npos] = ic[-nothers:, :npos]
                # D
                if npos == 0:
                    ic2[npos + iband, npos + iband] = ic[npos_fit, npos_fit]
                else:
                    ic2[npos + iband, npos + iband] = ic[npos, npos]
                # E
                if npos == 0:
                    ic2[npos + iband, -nothers:] = ic[npos_fit, -nothers:]
                    ic2[-nothers:, npos + iband] = ic[-nothers:, npos_fit]
                else:
                    ic2[npos + iband, -nothers:] = ic[npos, -nothers:]
                    ic2[-nothers:, npos + iband] = ic[-nothers:, npos]
                # F
                ic2[-nothers:,-nothers:] = ic[-nothers:,-nothers:]

                # print('x')
                # print(x)
                # print('x2')
                # print(x2)
                # print('ic')
                # print(ic)
                # print('ic2')
                # print(ic2)

                #full_xic.append((x2.get(),ic2.get()))
                full_xic.append((x2, ic2))
            Xic = full_xic
        else:
            (X, Xicov) = Xic
            full_xic = []
            X = X.get()
            Xicov = Xicov.get()
            for iband,x,ic in zip(img_bands, X, Xicov):
                full_xic.append((x, ic))
            Xic = full_xic

        #
        #print('Calling original version...')
        #sXic = super().getSingleImageUpdateDirections(tr, **kwargs)
        print (f"Final {Xic=}")
        return Xic

    def computeUpdateDirections(self, img_params, priorVals):
        '''
        This is the function you'll want to GPU-ify!

        "imgs" is a list with one element per image to process.  These can all be performed in parallel.
        A typical length for this would be ~ 10.

        The return value is a list of (vector, matrix) pairs, one per
        image.  The vector,matrix sizes are roughly (6, 6x6), giving
        the parameter update direction and its inverse-covariance.

        Each element in "imgs" is a giant tuple:

        (img_derivs, pix, ie, P, mux, muy, mw, mh, counts, cdi, roi)

        - img_derivs is a list (typical length ~ 4) of tuples,
        describing the parameters that we want to adjust in this
        parameter update.  (Plus two spatial ones that we handle specially.)
        Each tuple has (name, stepsize, mogs, fftmix),
        - name (string) is just the name of the parameter
        - stepsize (float) is the finite-difference step size we took in this parameter;
          it is used to scale the derivative
        - mogs will be None in this setting
        - fftmix is a MixtureOfGaussians objects (see
          mixture_profiles.py); in this setting, the "mean" values are
          zero, so only the "amp" and "var" (variance) fields are
          relevant.  The elements in this list correspond to the shape
          of the galaxy at the current parameters, and after stepping in
          each of a few different galaxy shape parameters.  We evaluate
          the Fourier transform of this weighted sum of Gaussians,
          directly in Fourier space; see mp_fourier.i :
          gaussian_fourier_transform_zero_mean_2 for the implementation.

        - pix is a (float32) matrix of pixels with size = (mh,mw), typically 64x64.

        - ie is a (float32) matrix of inverse-uncertainty ("inverse-error") pixels,
        same size as "pix".  Some of these will be zero, corresponding to zero-weight pixels.

        - P is the Fourier transform of the point-spread function (how the atmosphere blurs
        a point source in this image), of shape typically 64x33, and type numpy.complex64
        (ie, two float32 values for real and imaginary)

        - mux, muy are floating-point values giving the sub-pixel shift to apply to this
        galaxy.  They will be in the range [-0.5, +0.5].

        - mw, mh are the integer size of the images

        - counts is the floating-point brightness of the galaxy model.

        - cdi is a 2x2 matrix giving the transformation between sky coordinates (RA,Dec) and
        pixel coordinates.  We use this to produce the parameter update in sky coordinates,
        but instead of computing the derivatives by finite differences in those parameters,
        we compute them directly from the model image.

        - roi ("region of interest") is a tuple of 4 integers, (x0, y0, w, h), of the
        sub-region within "pix" and "ie" that contain non-zero values.
        I padded "pix" and "ie" out to be the same size as the galaxy
        model (ie, size "w x w", = the size of the ifft(P)) to make
        life easier in GPU land.  The code below shows how you might use
        the ROI information to process fewer pixels, but at the expense of
        shuffling them around a bit.

        '''
        use_roi = False
        Xic = []

        Npriors = 0
        if priorVals is not None:
            rA, cA, vA, pb, mub = priorVals
            Npriors = max(Npriors, max([1+max(r) for r in rA]))

        #assert(img_params.mogs is None)
        assert(img_params.ffts is not None)
        assert(img_params.mux is not None)
        assert(img_params.muy is not None)
        v = img_params.v
        w = img_params.w
        #img_params.ffts is a BatchMixtureOfGaussians (see batch_mixture_profiles.py)
        Fsum = img_params.ffts.getFourierTransform(v, w, zero_mean=True)
        #Fsum shape (Nimages, maxNd, nw, nv)
        # P is created in psf.getFourierTransform - this should be done in batch on GPU
        # resulting in P already being a CuPy array
        #P shape (Nimages, nw, nv)
        P = img_params.P[:,cp.newaxis,:,:]
        #print ("FSUM", Fsum.shape, "P", P.shape, P.max())
        #cp.savetxt('gfsum.txt', Fsum.ravel())
        #cp.savetxt('gp.txt', P.ravel())
        # Multiply P into Fsum in-place to save one massive array allocation
        Fsum *= P
        del P # Free P immediately
        G = cp.fft.irfft2(Fsum).astype(cp.float32)
        del Fsum
        #Do Lanczos shift
        G = lanczos_shift_image_batch_gpu(G, img_params.mux, img_params.muy)
        #cp.savetxt('gg.txt', G.ravel())
        #G should be (nimages, maxNd, nw, nv) and mux and muy should be 1d vectors
        assert (G.shape == (img_params.Nimages, img_params.maxNd, img_params.mh, img_params.mw))


        if img_params.mogs is not None:
            psfmog = img_params.psf_mogs
            #print('Img_params.mogs:', img_params.mogs)
            #print('PSF mogs:', img_params.psf_mogs)
            # assert trivial mixture of Gaussians - single circular Gaussian
            # If we relax this, then convolution becomes much harder!
            assert(psfmog.K == 1)
            assert(np.all(psfmog.mean == 0.))
            assert(np.all(psfmog.amp == 1.))
            assert(np.all(psfmog.var[..., 0, 0,1] == 0))
            assert(np.all(psfmog.var[..., 0, 1,0] == 0))
            assert(np.all(psfmog.var[..., 0, 0,0] == psfmog.var[..., 0, 1,1]))

            # Trivial convolution
            mogs = img_params.mogs
            varcopy = mogs.var.copy()
            # varcopy shape: (13, 4, 1, 2, 2)
            # psfmog.var shape: (13, 1, 2, 2)
            #varcopy[..., 0, 0, 0] += psfmog.var[..., cp.newaxis, 0, 0, 0]
            #varcopy[..., 0, 1, 1] += psfmog.var[..., cp.newaxis, 0, 1, 1]
            #print ("VARCOPY", varcopy.shape, psfmog.var.shape)
            varcopy[..., :, 0, 0] += psfmog.var[..., cp.newaxis, cp.newaxis, 0, 0, 0]
            varcopy[..., :, 1, 1] += psfmog.var[..., cp.newaxis, cp.newaxis, 0, 1, 1]

            #print('mogs.amp', mogs.amp.shape)
            #print('mogs.mean', mogs.mean.shape)
            #print('varcopy', varcopy.shape)

            conv_mog = BatchMixtureOfGaussians(mogs.amp, mogs.mean, varcopy, quick=True)
            #print('Convolved MoG:', conv_mog)

            ### Could also assert that the Gaussians are all concentric... means for all K are equal

            #print('var shape', conv_mog.var.shape)
            #print('mogs K', mogs.K)
            #print('maxNmogs', img_params.maxNmogs)
            #print(img_params.Nimages, img_params.maxNd, img_params.maxNmogs)

            # Evaluate MoG on pixel grid, add to G
            use_roi = True
            xx = cp.arange(img_params.mw)
            yy = cp.arange(img_params.mh)
            det = conv_mog.var[:,:,:,0,0] * conv_mog.var[:,:,:,1,1] - conv_mog.var[:,:,:,0,1] * conv_mog.var[:,:,:,1,0]
            iv0 = conv_mog.var[:,:,:,1,1] / det
            iv1 = -(conv_mog.var[:,:,:,0,1] + conv_mog.var[:,:,:,1,0]) / det
            iv2 = conv_mog.var[:,:,:,0,0] / det
            scale = conv_mog.amp / (2.*cp.pi*cp.sqrt(det))
            #print ("IV", iv0.max(), iv1.max(), iv2.max(), iv0.shape, np.where(iv0 == iv0.max()), np.where(iv1 == iv1.max()), np.where(iv2 == iv2.max()))

            #print('conv_mog.mean shape:', conv_mog.mean.shape)
            #print('xx shape:', xx.shape)

            #print('conv_mog.means:', conv_mog.mean)
            #print('img_params.mux,muy:', img_params.mux, img_params.muy)

            # conv_mog.mean is, eg, (13 x 4 x 2 x 2)
            # (nimages x nderivs x nmog x 2), where the 2 is x,y coordinates.
            # BUT, it's really only (nimages x 2), the values for all the derivs and mogs are equal!
            #print (conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:].shape)
            #b1 = np.where(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] != conv_mog.mean)
            #print ("B1", b1)
            #print (conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:][b1])
            #print ("ORIG", conv_mog.mean[b1])

            ###  TAG: ISSUE 2 4/9/25 - uncomment assert to see crash
            #assert(np.all(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] == conv_mog.mean))
            #means = conv_mog.mean[:, 0, 0, :].copy()
            means = conv_mog.mean.copy()
            #print ("MEANS", means.shape, means)
            # now "means" is (nimages x 2)

            # xx, yy are each 64 elements long.

            #print('img_derivs length:', len(img_params.img_derivs))

            #dx = xx - means[:,0]
            #dy = yy - means[:,1]
            #TODO - @Dustin - if I make the following changes, I then need to add another axis to distsq below
            # conv_mog.mean[:,:,:,0].shape == (13,4,1) in one pass through run-one-blob.py and (13,4,2) in another
            #dx = xx - conv_mog.mean[:,:,:,cp.newaxis,0]
            #dy = yy - conv_mog.mean[:,:,:,cp.newaxis,1]
            #if use_roi:

                # rois: (rx0,ry0, rw,rh)
                #means[:,0] += cp.array([d.roi[0] for d in img_params.img_derivs])
                #means[:,1] += cp.array([d.roi[1] for d in img_params.img_derivs])

                # #Loop over img_params.img_derivs and correct mean so that mogs are centered with G
                # for img_i, imderiv in enumerate(img_params.img_derivs):
                #     (rx0,ry0,rw,rh) = imderiv.roi
                #     dx[img_i] -= rx0
                #     dy[img_i] -= ry0
            # distsq = (iv0[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dx[:,:,cp.newaxis,:] +
            #           iv1[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dy[:,:,:,cp.newaxis] +
            #           iv2[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis])

            for i, deriv in enumerate(img_params.img_derivs):
                #means[i,0] -= deriv.dx
                #means[i,1] -= deriv.dy
                means[i,:,:,0] -= deriv.dx
                means[i,:,:,1] -= deriv.dy
            #print ("MEANS2", means.shape, means)
            #print ("ROI", img_params.roi)

            # (13,4,1) = (nimages, nderivs, nmog)
            #print('scale:', scale.shape)
            #print (np.where(scale == scale.max()))
            #print('iv shapes:', iv0.shape, iv1.shape, iv2.shape)

            # (13,4,64,64) = (nimages, nderivs, ny,nx)
            #print('G:', G.shape)

            # The distsq array is going to be nimages x nderivs x nmog x ny=64 x nx=64

            # distsq = (iv0[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dx[:,:,cp.newaxis,:] +
            #           iv1[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dy[:,:,:,cp.newaxis] +
            #           iv2[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis])

            n = cp.newaxis
            #distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2 +
            #          iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]) +
            #          iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)

            iv0 = iv0.astype(cp.float32)
            iv1 = iv1.astype(cp.float32)
            iv2 = iv2.astype(cp.float32)
            xx = xx.astype(cp.float32)
            yy = yy.astype(cp.float32)
            means = means.astype(cp.float32)

            distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:])**2 +
                      iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n]) +
                      iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n])**2)
            distsq *= -0.5
            # t1 = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2)
            # print('t1', t1.shape)
            # t3 = (iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # print('t3', t3.shape)
            # t2 = (iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]))
            # print('t2', t2.shape)
            distsq = cp.exp(distsq)
            scale = scale.astype(cp.float32)
            distsq *= scale[:,:,:,cp.newaxis,cp.newaxis]
            mog_g = cp.sum(distsq, axis=2)
            # Sum over the nmog
            #mog_g = cp.sum(scale[:,:,:,cp.newaxis,cp.newaxis] * cp.exp(distsq), axis=2).astype(cp.float32)
            del means, distsq, varcopy
            #print ("MOGG", mog_g.shape, np.where(mog_g == mog_g.max()))
            #cp.savetxt('gmogpatch.txt',mog_g.ravel())
            G += mog_g

        #Do no use roi since images are padded to be (mh, mw)
        #use_roi = False
        t1 = time.time()

        Npix = img_params.mh*img_params.mw
        #Npix is a scalar
        Nd = img_params.maxNd
        cdi = img_params.cdi
        A = cp.zeros((img_params.Nimages, Npix + Npriors, Nd+2), cp.float32)
        # A is of shape (Nimages, Npix+Npriors, Nd+2)
        # The first element in img_derivs is the current galaxy model parameters.
        mod0 = G[:,0,:,:]
        # mod0 should be (Nimages, nw, nv)
        assert (mod0.shape == (img_params.Nimages, img_params.mh, img_params.mw))
        ## 02-13-25 - make a mask with only the ROI as ones to mask out background
        roi = img_params.roi
        #roimask can be different for each image
        roimask = cp.zeros((img_params.Nimages, mod0.shape[1], mod0.shape[2]), dtype=bool)
        #print ("ROI", roi)
        for i, ri in enumerate(roi): 
            roimask[i, ri[1]:ri[1]+ri[3], ri[0]:ri[0]+ri[2]] = True
        ###############

        # boolean vector, are we fitting for positions?
        fit_pos = img_params.fit_pos
        # Shift this initial model image to get X,Y pixel derivatives
        ###  TAG: ISSUE 1 4/9/25 - comment / uncomment dx and dy to toggle fit_pos 
        dx = cp.zeros_like(mod0)
        # dx is of shape (Nimages, nw, nv)
        # X derivative -- difference between shifted-left and shifted-right arrays
        #dx[:,:,1:-1] = mod0[:,:, 2:] - mod0[:,:, :-2]
        dx[fit_pos,:,1:-1] = mod0[fit_pos,:, 2:] - mod0[fit_pos,:, :-2]
        #print ("FITPOS", fit_pos)
        # Y derivative -- difference between shifted-down and shifted-up arrays
        dy = cp.zeros_like(mod0)
        # dy is of shape (Nimages, nw, nv)
        #dy[:,1:-1, :] = mod0[:,2:, :] - mod0[:,:-2, :]
        dy[fit_pos,1:-1, :] = mod0[fit_pos,2:, :] - mod0[fit_pos,:-2, :]
        # Push through local WCS transformation to get to RA,Dec param derivatives
        assert(cdi.shape == (img_params.Nimages,2,2))

        ##Multiply everything by roimask 02-13-25
        dx *= roimask
        dy *= roimask
        mod0 *= roimask
        #Need to reshape for G
        G *= roimask[:,None,:,:]

        #cp.savetxt('roi.txt', roimask.ravel())

        #cp.savetxt('dx.txt', dx.ravel())
        #cp.savetxt('dy.txt', dy.ravel())
        #cp.savetxt('cdi.txt', cdi.ravel())
        #print ("DX", dx.shape, "CDI", cdi.shape, cdi)
        #print ("COUNTS", img_params.counts, "SKY", img_params.sky)

        # divide by 2 because we did +- 1 pixel
        # negative because we shifted the *image*, which is opposite
        # from shifting the *model*
        A[:,:Npix, 0] = cp.reshape(-((dx * cdi[:,0, 0][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 0][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2),(img_params.Nimages, -1))
        A[:,:Npix, 1] = cp.reshape(-((dx * cdi[:,0, 1][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 1][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2), (img_params.Nimages, -1))
        del dx,dy
        A[:,:Npix,2] = cp.reshape(mod0,(img_params.Nimages, -1))
#
        #A[:Npix, i + 2] = counts / stepsizes[i] * (Gi[i,:,:] - mod0).ravel()
        stepsizes = img_params.steps
        #print (stepsizes.shape, img_params.counts.shape)
        #print (img_params.counts[:,cp.newaxis, cp.newaxis].shape)
        #print (stepsizes[:,cp.newaxis,1:].shape)
        #print (cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).shape)
        A[:,:Npix, 3:] = img_params.counts[:,cp.newaxis, cp.newaxis] / stepsizes[:,cp.newaxis,1:] * cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1))

        #A[:Npix,:] *= ie.ravel()[:, cp.newaxis]
        A[:,:Npix,:] *= img_params.ie.reshape((img_params.Nimages, Npix))[:,:,cp.newaxis]
        del G

        B = cp.zeros((img_params.Nimages, Npix + Npriors), cp.float32)
        B[:,:Npix] = ((img_params.pix - (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis])) * img_params.ie).reshape((img_params.Nimages, Npix))
        del mod0

        #print ("Stepsizes", stepsizes)
        #print ("Counts", img_params.counts[0])
        #cp.savetxt('mod0.txt', mod0.ravel())
        #cp.savetxt("gmod.txt", (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis]).ravel())
        #cp.savetxt("gie.txt", img_params.ie.ravel())
        #cp.savetxt("gpix.txt", img_params.pix.ravel())
        #cp.savetxt("ga1.txt", A.ravel())
        #cp.savetxt("gb1.txt", B.ravel())

        # B should be of shape (Nimages, :)                           
        #B = cp.append(((pix - counts*mod0) * ie).ravel(),
        #                cp.zeros(Npriors, cp.float32))


        # Append priors --do priors depend on which image I am looking at?
        #TODO not sure if this is correct for priors? 

        if priorVals is not None:
            print ("Using PRIORS")
            rA, cA, vA, pb, mub = priorVals
            #print (priorVals)
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                if fit_pos[0] == False:
                    ci += 2
                    print (f"Updating ci to {ci=}")
                for rij,vij,bij in zip(ri, vi, bi):
                    #print (f'{vij=} {bij=} {rij=} {ci=}')
                    A[:,Npix + rij, ci] = vij
                    B[:,Npix + rij] += bij
        else:
            print ("NO PRIORS")

        # Compute the covariance matrix
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)
        #print ("Xicov", Xicov)

        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        colscales[colscales == 0] = 1.
        #print ("COLSCALES", colscales.shape)
        #cp.savetxt("gcols.txt", colscales.ravel())
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        A /= colscales[:,cp.newaxis, :]
        #cp.savetxt("ga2.txt", A.ravel())
        #cp.savetxt("gb1.txt", B.ravel())

        # Solve the least-squares problem!
        #X,_,_ = cp.linalg.lstsq(A, B, rcond=None)
        #A_T_dot_A = cp.einsum("...ji,...jk", A, A)
        #A_T_dot_B = cp.einsum("...ji,...j", A, B)
        #X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
        #print ("X DOT = ", X)
        X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)
        #print ("ATA", A_T_dot_A)
        #print ("ATB", A_T_dot_B)
        #print ("X", X)
        #X[fit_pos,0:2] = 0
        #print ("OUTPUT SHAPES GPU ", A.shape, B.shape, X.shape)
        #X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)

        if self.ps is not None:

            N,_,_ = A.shape
            xx = []
            for i in range(N):
                xi,_,_,_ = cp.linalg.lstsq(A[i,:,:], B[i,:], rcond=None)
                xx.append(xi)
            print('X:', X)
            print('xx:', xx)

            import pylab as plt
            plt.clf()
            myA = A.get()
            myA = myA[0,:,:]
            print('A:', myA.shape)
            myX = X.get()
            print('X', myX.shape)
            myX = myX[0,:]
            myB = B.get()
            print('B', myB.shape)
            myB = myB[0,:]

            import fitsio
            fitsio.write('gpu-x.fits', myX, clobber=True)
            fitsio.write('gpu-x-scaled.fits', myX / colscales.get(), clobber=True)
            fitsio.write('gpu-a.fits', myA.reshape((64,64,-1)) * colscales.get()[np.newaxis,:], clobber=True)
            fitsio.write('gpu-a-scaled.fits', myA.reshape((64,64,-1)), clobber=True)
            fitsio.write('gpu-b.fits', myB.reshape((64,64)), clobber=True)
            
            ax = np.dot(myA, myX)
            print('AX', ax.shape)
            print('AX', np.percentile(np.abs(ax), [1,99]))
            print('B ', np.percentile(np.abs(myB), [1,99]))
            #lo,hi = np.percentile(np.abs(myB), [1,99])
            lo,hi = myB.min(), myB.max()
            plt.clf()
            plt.subplot(1,3,1)
            imx = dict(interpolation='nearest', origin='lower', vmin=lo, vmax=hi)
            plt.imshow(ax.reshape((64,64)), **imx)
            plt.colorbar()
            plt.title('AX')
            plt.subplot(1,3,2)
            plt.imshow(myB.reshape((64,64)), **imx)
            plt.colorbar()
            plt.title('B')
            plt.subplot(1,3,3)
            plt.imshow(ax.reshape((64,64)), interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('AX')
            self.ps.savefig()

        # Undo pre-scaling
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        X /= colscales
        #print ("NANs", cp.isnan(X).sum())
        X[cp.isnan(X)] = 0
        #print ("X NORM", X)
        #else:
        #    X[cp.isnan(X)] = 0
        # del A, B
        #Have to corectly make Xic a list of tuples
        for i in range(img_params.Nimages):
            #print ("I", i, X[i], "Xicov", Xicov[i])
            Xic.append((X[i], Xicov[i]))

        """
        for img_i, imderiv in enumerate(img_params.img_derivs):
            pix, ie, mw, mh, counts, cdi, roi = imderiv.mmpix, imderiv.mmie, imderiv.mw, imderiv.mh, imderiv.counts, imderiv.cdi, imderiv.roi 
            assert(pix.shape == (mh,mw))
            # We're going to build a tall matrix A, whose number of
            # rows = number of pixels and cols = number of parameters
            # to update.  We special-case the two spatial derivatives,
            # the rest are in the 'img_derivs' list.

            # number of derivatives
            #Nd = len(img_derivs)
            #Nd = img_derivs_batch.N
            Nd = imderiv.N
            if use_roi:
                (rx0,ry0,rw,rh) = roi
                roi_slice = slice(ry0, ry0+rh), slice(rx0, rx0+rw)
                pix = pix[roi_slice]
                ie = ie[roi_slice]
                Npix = rh*rw
            else:
                Npix = mh*mw

            print (f'{Npix=}, {Npriors=}, {Nd=}')
            A = cp.zeros((Npix + Npriors, Nd+2), cp.float32)

            mod0 = None

            #assert(img_derivs_batch.mogs is None)
            #assert(img_derivs_batch.ffts is not None)
            ## "img_derivs_batch.ffts" is a BatchMixtureOfGaussians (see batch_mixture_profiles.py)
            #Fsum = img_derivs_batch.ffts.getFourierTransformBatchGPU(v, w, zero_mean=True)
            #print ("FSUM", Fsum.shape, "P", P.shape)
            ## Copy P to GPU
            ## TODO P is created in psf.getFourierTransform - this should be done in batch on GPU
            ## resulting in P already being a CuPy array
            #P = cp.asarray(P)
            #G = cp.fft.irfft2(Fsum*P[img_i,:,:])
            ## lanczos_shift_image_batch_gpu has a Python implementation in psf.py
            #print ("G", G.shape, mux, muy)
            ### TODO G should be (nimages, nx, ny) and mux and muy should be 1d vectors
            #lanczos_shift_image_batch_gpu(G, mux, muy)
            #del Fsum
            Gi = G[img_i]
            print (f'{Nd=}, {mh=}, {mw=}', Gi.shape)
            if use_roi:
                assert(Gi.shape == (Nd, rh, rw))
            else:
                assert (Gi.shape == (Nd,mh,mw))
            #if use_roi:
            #    #Gi = Gi[:,roi_slice]
            #    Gi = Gi[:,ry0:ry0+rh, rx0:rx0+rw]

            # The first element in img_derivs is the current galaxy model parameters.
            mod0 = Gi[0,:,:]
            print ("MOD0", mod0.shape, mod0)

            # Shift this initial model image to get X,Y pixel derivatives
            dx = cp.zeros_like(mod0)
            # X derivative -- difference between shifted-left and shifted-right arrays
            dx[:,1:-1] = mod0[:, 2:] - mod0[:, :-2]
            # Y derivative -- difference between shifted-down and shifted-up arrays
            dy = cp.zeros_like(mod0)
            dy[1:-1, :] = mod0[2:, :] - mod0[:-2, :]
            # Push through local WCS transformation to get to RA,Dec param derivatives
            assert(cdi.shape == (2,2))
            # divide by 2 because we did +- 1 pixel
            # negative because we shifted the *image*, which is opposite
            # from shifting the *model*
            A[:Npix, 0] = -((dx * cdi[0, 0] + dy * cdi[1, 0]) * counts / 2).ravel()
            A[:Npix, 1] = -((dx * cdi[0, 1] + dy * cdi[1, 1]) * counts / 2).ravel()
            del dx,dy

            # The derivative with respect to flux (brightness) = the unit-flux current model
            A[:Npix, 2] = mod0.ravel()

            #stepsizes = [s for _,s,_,_ in img_derivs]
            stepsizes = imderiv.steps
            #print ("STEPSIZES", len(stepsizes), Nd, imderiv.N)
            # The first element is mod0, so the stepped parameters start at 1 here.
            for i in range(1, Nd):
                # For other parameters, compute the numerical derivative.
                # mod0 is the unit-brightness model at the current position
                # G[i,*] is the unit-brightness model after stepping the parameter
                # The i+2 here is because the first two params are the spatial derivs
                # (A[:,0] and A[:,1] are filled above)
                # And the next param is the flux deriv
                # (A[:,2] is filled)
                # So the first time through this loop, i=1 and we fill column A[:,3]
                A[:Npix, i + 2] = counts / stepsizes[i] * (Gi[i,:,:] - mod0).ravel()

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
            #
            # Parameters to optimize go in the columns of matrix A
            # Pixels go in the rows.

            # Scale by IE (inverse-error) weighting to get to units of chi
            A[:Npix,:] *= ie.ravel()[:, cp.newaxis]
            # The current residuals = the observed image "pix" minus the current model (counts*mod0),
            # weighted by the inverse-errors.
            print (f'{Npriors=}')
            print ("A", A.shape, pix.shape, ie.shape)
            B = cp.append(((pix - counts*mod0) * ie).ravel(),
                          cp.zeros(Npriors, cp.float32))
            print ("B", B.shape)

            # Append priors
            if priorVals is not None:
                rA, cA, vA, pb, mub = priorVals
                for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                    for rij,vij,bij in zip(ri, vi, bi):
                        A[Npix + rij, ci] = vij
                        B[Npix + rij] += bij

            if False:
                import pylab as plt
                n,m = A.shape
                for i in range(m):
                    plt.clf()
                    plt.imshow(A[:Npix,i].get().reshape((mh,mw)), interpolation='nearest', origin='lower')
                    plt.savefig('gpu-img%i-d%i.png' % (img_i, i))
                for i in range(Nd):
                    plt.clf()
                    plt.imshow(Fsum[img_i,i,:,:].get(), interpolation='nearest', origin='lower')
                    plt.savefig('gpu-imgF%i-d%i.png' % (img_i, i))

            # Compute the covariance matrix
            Xicov = cp.matmul(A.T, A)

            # Pre-scale the columns of A
            colscales = cp.sqrt(cp.diag(Xicov))
            A /= colscales[cp.newaxis, :]

            # Solve the least-squares problem!
            X,_,_,_ = cp.linalg.lstsq(A, B, rcond=None)

            # Undo pre-scaling
            X /= colscales

            if self.ps is not None:
                import pylab as plt
                plt.clf()
                ima = dict(interpolation='nearest', origin='lower')
                rr,cc = 3,4
                plt.subplot(rr,cc,1)
                plt.imshow(mod0.get(), **ima)
                plt.title('mod0')
                sh = mod0.shape
                plt.subplot(rr,cc,2)
                mx = max(cp.abs(B))
                imx = ima.copy()
                imx.update(vmin=-mx, vmax=+mx)
                plt.imshow(B[:Npix].get().reshape(sh), **imx)
                plt.title('B')
                AX = np.dot(A, X)
                plt.subplot(rr,cc,3)
                plt.imshow(AX[:Npix].get().reshape(sh), **imx)
                plt.title('A X')
                for i in range(min(Nd+2, 8)):
                    plt.subplot(rr,cc,5+i)
                    plt.imshow(A[:Npix,i].get().reshape(sh), **ima)
                    if i == 0:
                        plt.title('dx')
                    elif i == 1:
                        plt.title('dy')
                    elif i == 2:
                        plt.title('dflux')
                    else:
                        #plt.title(img_derivs[i-2][0])
                        plt.title("I = "+str(i))
                #plt.suptitle('GPU: image %i/%i' % (img_i+1, img_params.N))
                self.ps.savefig()

            del A,B
            Xic.append((X, Xicov))
            print ("X2", X.shape, Xicov.shape)
        """
        del img_params, A, B, X
        #print_timer()
        return Xic

    def computeUpdateDirectionsVectorized(self, img_params, priorVals):
        '''
        This is the function you'll want to GPU-ify!

        "imgs" is a list with one element per image to process.  These can all be performed in parallel.
        A typical length for this would be ~ 10.

        The return value is a list of (vector, matrix) pairs, one per
        image.  The vector,matrix sizes are roughly (6, 6x6), giving
        the parameter update direction and its inverse-covariance.

        Each element in "imgs" is a giant tuple:

        (img_derivs, pix, ie, P, mux, muy, mw, mh, counts, cdi, roi)

        - img_derivs is a list (typical length ~ 4) of tuples,
        describing the parameters that we want to adjust in this
        parameter update.  (Plus two spatial ones that we handle specially.)
        Each tuple has (name, stepsize, mogs, fftmix),
        - name (string) is just the name of the parameter
        - stepsize (float) is the finite-difference step size we took in this parameter;
          it is used to scale the derivative
        - mogs will be None in this setting
        - fftmix is a MixtureOfGaussians objects (see
          mixture_profiles.py); in this setting, the "mean" values are
          zero, so only the "amp" and "var" (variance) fields are
          relevant.  The elements in this list correspond to the shape
          of the galaxy at the current parameters, and after stepping in
          each of a few different galaxy shape parameters.  We evaluate
          the Fourier transform of this weighted sum of Gaussians,
          directly in Fourier space; see mp_fourier.i :
          gaussian_fourier_transform_zero_mean_2 for the implementation.

        - pix is a (float32) matrix of pixels with size = (mh,mw), typically 64x64.

        - ie is a (float32) matrix of inverse-uncertainty ("inverse-error") pixels,
        same size as "pix".  Some of these will be zero, corresponding to zero-weight pixels.

        - P is the Fourier transform of the point-spread function (how the atmosphere blurs
        a point source in this image), of shape typically 64x33, and type numpy.complex64
        (ie, two float32 values for real and imaginary)

        - mux, muy are floating-point values giving the sub-pixel shift to apply to this
        galaxy.  They will be in the range [-0.5, +0.5].

        - mw, mh are the integer size of the images

        - counts is the floating-point brightness of the galaxy model.

        - cdi is a 2x2 matrix giving the transformation between sky coordinates (RA,Dec) and
        pixel coordinates.  We use this to produce the parameter update in sky coordinates,
        but instead of computing the derivatives by finite differences in those parameters,
        we compute them directly from the model image.

        - roi ("region of interest") is a tuple of 4 integers, (x0, y0, w, h), of the
        sub-region within "pix" and "ie" that contain non-zero values.
        I padded "pix" and "ie" out to be the same size as the galaxy
        model (ie, size "w x w", = the size of the ifft(P)) to make
        life easier in GPU land.  The code below shows how you might use
        the ROI information to process fewer pixels, but at the expense of
        shuffling them around a bit.

        '''
        use_roi = False
        Xic = []

        Npriors = 0
        if priorVals is not None:
            rA, cA, vA, pb, mub = priorVals
            Npriors = max(Npriors, max([1+max(r) for r in rA]))

        G = self.computeGalaxyModelsVectorized(img_params)

        t1 = time.time()
        Npix = img_params.mh*img_params.mw
        #Npix is a scalar
        Nd = img_params.maxNd
        Nimages = img_params.Nimages
        cdi = img_params.cdi
        A = cp.zeros((img_params.Nimages, Npix + Npriors, Nd+2), cp.float32)
        # A is of shape (Nimages, Npix+Npriors, Nd+2)
        # The first element in img_derivs is the current galaxy model parameters.
        mod0 = G[:,0,:,:]
        # mod0 should be (Nimages, nw, nv)
        assert (mod0.shape == (img_params.Nimages, img_params.mh, img_params.mw))
        ## 02-13-25 - make a mask with only the ROI as ones to mask out background
        #print (img_params.roi.shape)
        roi = img_params.roi
        roimask = cp.zeros((img_params.Nimages, mod0.shape[1], mod0.shape[2]), dtype=bool)
        for i, ri in enumerate(roi):
            roimask[i, ri[1]:ri[1]+ri[3], ri[0]:ri[0]+ri[2]] = True
        #roimask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1
        ###############

        # boolean vector, are we fitting for positions?
        fit_pos = img_params.fit_pos
        #print ("FITPOS V", fit_pos)
        # Shift this initial model image to get X,Y pixel derivatives
        ###  TAG: ISSUE 1 4/9/25 - comment / uncomment dx and dy to toggle fit_pos 
        dx = cp.zeros_like(mod0)
        # dx is of shape (Nimages, nw, nv)
        # X derivative -- difference between shifted-left and shifted-right arrays
        dx[fit_pos,:,1:-1] = mod0[fit_pos,:, 2:] - mod0[fit_pos,:, :-2]
        #dx[:,:,1:-1] = mod0[:,:, 2:] - mod0[:,:, :-2]
        # Y derivative -- difference between shifted-down and shifted-up arrays
        dy = cp.zeros_like(mod0)
        # dy is of shape (Nimages, nw, nv)
        #dy[:,1:-1, :] = mod0[:,2:, :] - mod0[:,:-2, :]
        dy[fit_pos,1:-1, :] = mod0[fit_pos,2:, :] - mod0[fit_pos,:-2, :]
        # Push through local WCS transformation to get to RA,Dec param derivatives
        assert(cdi.shape == (img_params.Nimages,2,2))

        ##Multiply everything by roimask 02-13-25
        dx *= roimask
        dy *= roimask
        mod0 *= roimask
        #Need to reshape for G
        G *= roimask[:,None,:,:]

        # divide by 2 because we did +- 1 pixel
        # negative because we shifted the *image*, which is opposite
        # from shifting the *model*
        A[:,:Npix, 0] = cp.reshape(-((dx * cdi[:,0, 0][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 0][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2),(img_params.Nimages, -1))
        A[:,:Npix, 1] = cp.reshape(-((dx * cdi[:,0, 1][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 1][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2), (img_params.Nimages, -1))
        del dx,dy
        A[:,:Npix,2] = cp.reshape(mod0,(img_params.Nimages, -1))

        #A[:Npix, i + 2] = counts / stepsizes[i] * (Gi[i,:,:] - mod0).ravel()
        stepsizes = img_params.steps
        #print ("A", img_params.counts[:,cp.newaxis, cp.newaxis].shape, stepsizes[:,cp.newaxis,1:].shape, cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1)).shape)
        #print (A[:,:Npix, 3:].shape)
        A[:,:Npix, 3:] = img_params.counts[:,cp.newaxis, cp.newaxis] / stepsizes[:,cp.newaxis,1:] * cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1))

        #A[:Npix,:] *= ie.ravel()[:, cp.newaxis]
        A[:,:Npix,:] *= img_params.ie.reshape((img_params.Nimages, Npix))[:,:,cp.newaxis]
        del G

        B = cp.zeros((img_params.Nimages, Npix + Npriors), cp.float32)
        B[:,:Npix] = ((img_params.pix - (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis])) * img_params.ie).reshape((img_params.Nimages, Npix))
        del mod0, img_params
        #B[:,:Npix] = ((img_params.pix - img_params.counts[:,cp.newaxis, cp.newaxis]*mod0) * img_params.ie).reshape((img_params.Nimages, Npix))
        # B should be of shape (Nimages, :)                           
        #B = cp.append(((pix - counts*mod0) * ie).ravel(),
        #                cp.zeros(Npriors, cp.float32))

        #cp.savetxt('vmod0.txt', mod0.ravel())
        #cp.savetxt("vgmod.txt", (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis]).ravel())
        #cp.savetxt("vgie.txt", img_params.ie.ravel())
        #cp.savetxt("vgpix.txt", img_params.pix.ravel())
        #cp.savetxt("vga1.txt", A.ravel())
        #cp.savetxt("vgb1.txt", B.ravel())

        # Append priors --do priors depend on which image I am looking at?
        #TODO not sure if this is correct for priors? 

        """
        Priors should NOT be added in here - this will be removed
        if priorVals is not None:
            print ("Using PRIORS")
            rA, cA, vA, pb, mub = priorVals
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                if fit_pos[0] == False:
                    ci += 2
                    print (f"Updating ci to {ci=}")
                for rij,vij,bij in zip(ri, vi, bi):
                    A[:,Npix + rij, ci] = vij
                    B[:,Npix + rij] += bij
        else:
            print ("NO PRIORS")
        """

        # Compute the covariance matrix
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)
        #print ("Xicov V", Xicov)

        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        colscales[colscales == 0] = 1.
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        A /= colscales[:,cp.newaxis, :]

        # Solve the least-squares problem!
        #X,_,_ = cp.linalg.lstsq(A, B, rcond=None)
        #A_T_dot_A = cp.einsum("...ji,...jk", A, A)
        #A_T_dot_B = cp.einsum("...ji,...j", A, B)
        #X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
        #print ("X DOT V", X)
        #X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)
        #X[fit_pos,0:2] = 0
        #print ("A", A)
        #print ("B", B)
        #print ("XV", X)
        #X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)

        ## TAG: indidious hang issue 10/17/25 - pinv can occasionally hang.
        # But we had the issue where if fit_pos is False for all images, and
        # first two columns of A are all zero, the ATA ATB method fails with
        # NaNs.  But we can strip out those rows and zero-pad the output!
        if not np.any(fit_pos):
            print ("WARNING: fit_pos is False for all images, stripping off rows to prevent NaNs before fitting")
            A_T_dot_A = cp.einsum("...ji,...jk", A[:,:,2:], A[:,:,2:])
            A_T_dot_B = cp.einsum("...ji,...j", A[:,:,2:], B)
            X = cp.zeros((Nimages, Nd+2), dtype=cp.float32)
            X[:,2:] = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
        else:
            A_T_dot_A = cp.einsum("...ji,...jk", A, A)
            A_T_dot_B = cp.einsum("...ji,...j", A, B)
            print (f"{A_T_dot_A=} {A_T_dot_B=}")
            X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)

        print (f"GPU {X=} {A.sum(axis=1)=} {B.sum()=}")
        print (f"{X.shape=} {A.shape=} {B.shape=}")
        # Undo pre-scaling
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        X /= colscales
        #print ("NANs", cp.isnan(X).sum())
        #if cp.any(cp.isinf(X)):
        #    print ("NAN", X)
        X[cp.isnan(X)] = 0
        #X[cp.isinf(X)] = 0
        #print ("X NORM", X.shape)
        #    print ("X norm", X)
        #else:
        #    X[cp.isnan(X)] = 0
        # del A, B
        #Have to corectly make Xic a list of tuples
        #for i in range(img_params.Nimages):
        #    Xic.append((X[i], Xicov[i]))
        ##Change return args to keep as cupy arrays
        Xic = (X, Xicov)
        del A, B, X
        return Xic

    def evaluate_mog_gpu(self, conv_mog, xx, yy, img_params, G):
        import cupy as cp
        from tractor.miscutils import get_mog_eval_kernel
        
        N, Nd, K = conv_mog.amp.shape
        H, W = yy.size, xx.size
        
        # Heuristic for A100 40GB:
        # 208 images * 10 derivs * 6 components * 64 * 64 is small,
        # but we chunk anyway for scalability.
        from tractor.miscutils import get_safe_chunk_size
        chunk_size = get_safe_chunk_size(N, H, W, vram_gb=40)
        
        kernel = get_mog_eval_kernel()
        
        # Pre-calculate inversion terms
        det = conv_mog.var[:,:,:,0,0] * conv_mog.var[:,:,:,1,1] - \
              conv_mog.var[:,:,:,0,1] * conv_mog.var[:,:,:,1,0]
        
        # Broadcastable components
        iv0 = (conv_mog.var[:,:,:,1,1] / det).astype(cp.float32)[:,:,:,None,None]
        iv1 = (-(conv_mog.var[:,:,:,0,1] + conv_mog.var[:,:,:,1,0]) / det).astype(cp.float32)[:,:,:,None,None]
        iv2 = (conv_mog.var[:,:,:,0,0] / det).astype(cp.float32)[:,:,:,None,None]
        scale = (conv_mog.amp / (2.*cp.pi*cp.sqrt(det))).astype(cp.float32)[:,:,:,None,None]
        
        mx = (conv_mog.mean[:,:,:,0] - img_params.dx[:,None,None]).astype(cp.float32)[:,:,:,None,None]
        my = (conv_mog.mean[:,:,:,1] - img_params.dy[:,None,None]).astype(cp.float32)[:,:,:,None,None]

        # Grid (1, 1, 1, H, W)
        XX = xx[None, None, None, None, :]
        YY = yy[None, None, None, :, None]

        for i in range(0, N, chunk_size):
            e = min(i + chunk_size, N)
            # kernel computes (chunk, Nd, K, H, W)
            # we sum over K (axis 2) immediately to keep it 4D (chunk, Nd, H, W)
            res = kernel(iv0[i:e], iv1[i:e], iv2[i:e], scale[i:e], mx[i:e], my[i:e], XX.astype(cp.float32), YY.astype(cp.float32))
            G[i:e] += cp.sum(res, axis=2)
            
            del res
            cp.get_default_memory_pool().free_all_blocks()
        return G

    def predict_fft_memory(self, tr, masks, pxy, max_nd):
        # 1. Replicate the gpu_halfsize logic
        # (Simplified version of your _getBatchImageParams logic)
        px, py = np.array(pxy).T
        extents = [mm.extent for mm in masks]
        x0, x1, y0, y1 = np.asarray(extents).T
        x0 = x0.astype(np.int32) 
        x1 = x1.astype(np.int32) 
        y0 = y0.astype(np.int32) 
        y1 = y1.astype(np.int32)

        # This is the "radius" that determines pH/pW
        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                                1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py]), axis=0)

        # 2. Calculate the Power-of-Two padding
        radius_max = gpu_halfsize.max()
        try:
            sz = 2**int(np.ceil(np.log2(radius_max * 2.)))
        except Exception as ex:
            print (f'predict_fft_memory> {gpu_halfsize=} {radius_max=} {x0=} {x1=} {y0=} {y1=} {px=} {py=} '+str(ex))
            return 1.e+12

        pH, pW = sz, sz
        n_images = len(tr.images)

        # 3. Calculate Actual Bytes
        # Complex64 = 8 bytes, Float32 = 4 bytes
        # Assume 8 arrays of complex64
        # Assume 3 x arrays needed
        est_mem = n_images * max_nd * pH * pW * 8. * 3 / (1024.**3)
        print (f'{est_mem=} {n_images=} {max_nd=} {pH=} {pW=}')
        #fsum_bytes = n_images * max_nd * pH * (pW // 2 + 1) * 8
        #p_bytes = n_images * 1 * pH * (pW // 2 + 1) * 8
        #g_bytes = n_images * max_nd * pH * pW * 4

        # Total predicted for the FFT block (including a safety factor for IRFFT workspace)
        #total_predicted_gb = (fsum_bytes + p_bytes + g_bytes * 2) / (1024**3)
        #return total_predicted_gb, pH, pW
        return est_mem

    def computeGalaxyModelsVectorized(self, img_params):
        # Note, this does *not* scale by *counts*, it produces unit-flux models
        t1 = time.time()
        #assert(img_params.mogs is None)
        assert(img_params.ffts is not None)
        assert(img_params.mux is not None)
        assert(img_params.muy is not None)
        v = img_params.v
        w = img_params.w
        #img_params.ffts is a BatchMixtureOfGaussians (see batch_mixture_profiles.py)
        Fsum = img_params.ffts.getFourierTransform(v, w, zero_mean=True)
        #Fsum shape (Nimages, maxNd, nw, nv)
        #print ("Fsum", Fsum.shape, img_params.Nimages, img_params.maxNd, w.shape, v.shape)
        # P is created in psf.getFourierTransform - this should be done in batch on GPU
        # resulting in P already being a CuPy array
        #P shape (Nimages, nw, nv)
        P = img_params.P[:,cp.newaxis,:,:]
        #print ("FSUM", Fsum.shape, "P", P.shape)
        # Multiply P into Fsum in-place to save one massive array allocation
        Fsum *= P
        del P # Free P immediately
        # Perform IRFFT on the modified Fsum
        G = cp.fft.irfft2(Fsum).astype(cp.float32)
        del Fsum
        #Do Lanczos shift
        G = lanczos_shift_image_batch_gpu(G, img_params.mux, img_params.muy)
        #cp.savetxt('vgg.txt', G.ravel())
        #G should be (nimages, maxNd, nw, nv) and mux and muy should be 1d vectors
        #print ("G shape", G.shape, img_params.Nimages,img_params.maxNd,img_params.mh,img_params.mw)
        assert (G.shape == (img_params.Nimages, img_params.maxNd, img_params.mh, img_params.mw))
        
        if img_params.mogs is not None:
            psfmog = img_params.psf_mogs
            #print('Img_params.mogs:', img_params.mogs)
            #print('PSF mogs:', img_params.psf_mogs)
            # assert trivial mixture of Gaussians - single circular Gaussian
            # If we relax this, then convolution becomes much harder!
            assert(psfmog.K == 1)
            assert(np.all(psfmog.mean == 0.))
            assert(np.all(psfmog.amp == 1.))
            assert(np.all(psfmog.var[..., 0, 0,1] == 0))
            assert(np.all(psfmog.var[..., 0, 1,0] == 0))
            assert(np.all(psfmog.var[..., 0, 0,0] == psfmog.var[..., 0, 1,1]))

            # Trivial convolution
            mogs = img_params.mogs
            varcopy = mogs.var.copy()
            # varcopy shape: (13, 4, 1, 2, 2)
            # psfmog.var shape: (13, 1, 2, 2)
            #varcopy[..., 0, 0, 0] += psfmog.var[..., cp.newaxis, 0, 0, 0]
            #varcopy[..., 0, 1, 1] += psfmog.var[..., cp.newaxis, 0, 1, 1]
            varcopy[..., :, 0, 0] += psfmog.var[..., cp.newaxis, cp.newaxis, 0, 0, 0]
            varcopy[..., :, 1, 1] += psfmog.var[..., cp.newaxis, cp.newaxis, 0, 1, 1]

            #print('mogs.amp', mogs.amp.shape)
            #print('mogs.mean', mogs.mean.shape)
            #print('varcopy', varcopy.shape)

            conv_mog = BatchMixtureOfGaussians(mogs.amp, mogs.mean, varcopy, quick=True)
            #print('Convolved MoG:', conv_mog)

            ### Could also assert that the Gaussians are all concentric... means for all K are equal

            #print('var shape', conv_mog.var.shape)
            #print('mogs K', mogs.K)
            #print('maxNmogs', img_params.maxNmogs)
            #print(img_params.Nimages, img_params.maxNd, img_params.maxNmogs)

            # Evaluate MoG on pixel grid, add to G
            use_roi = True
            xx = cp.arange(img_params.mw)
            yy = cp.arange(img_params.mh)

            G = self.evaluate_mog_gpu(conv_mog, xx, yy, img_params, G)

            """
            det = conv_mog.var[:,:,:,0,0] * conv_mog.var[:,:,:,1,1] - conv_mog.var[:,:,:,0,1] * conv_mog.var[:,:,:,1,0]
            iv0 = conv_mog.var[:,:,:,1,1] / det
            iv1 = -(conv_mog.var[:,:,:,0,1] + conv_mog.var[:,:,:,1,0]) / det
            iv2 = conv_mog.var[:,:,:,0,0] / det
            scale = conv_mog.amp / (2.*cp.pi*cp.sqrt(det))

            #print('conv_mog.mean shape:', conv_mog.mean.shape)
            #print('xx shape:', xx.shape)

            #print('conv_mog.means:', conv_mog.mean)
            #print('img_params.mux,muy:', img_params.mux, img_params.muy)

            # conv_mog.mean is, eg, (13 x 4 x 2 x 2)
            # (nimages x nderivs x nmog x 2), where the 2 is x,y coordinates.
            # BUT, it's really only (nimages x 2), the values for all the derivs and mogs are equal!
            ###  TAG: ISSUE 2 4/9/25 - uncomment assert to see crash
            #assert(np.all(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] == conv_mog.mean))
            #means = conv_mog.mean[:, 0, 0, :].copy()
            means = conv_mog.mean.copy()
            # now "means" is (nimages x 2)

            # xx, yy are each 64 elements long.

            #print('img_derivs length:', len(img_params.img_derivs))

            #dx = xx - means[:,0]
            #dy = yy - means[:,1]
            #TODO - @Dustin - if I make the following changes, I then need to add another axis to distsq below
            # conv_mog.mean[:,:,:,0].shape == (13,4,1) in one pass through run-one-blob.py and (13,4,2) in another
            #dx = xx - conv_mog.mean[:,:,:,cp.newaxis,0]
            #dy = yy - conv_mog.mean[:,:,:,cp.newaxis,1]
            #if use_roi:

                #print ("MEANS", means[:,0].shape, means[:,1].shape, img_params.roi.shape)
                ## rois: (rx0,ry0, rw,rh)
                #means[:,0] += cp.array([d.roi[0] for d in img_params.img_derivs])
                #means[:,1] += cp.array([d.roi[1] for d in img_params.img_derivs])
                #means[:,0] += img_params.roi[:,0]
                #means[:,1] += img_params.roi[:,1]

            #means[:,0] -= img_params.dx
            #means[:,1] -= img_params.dy
            #print ("MEANS", means.shape, img_params.dx.shape, means[:,:,:,0].shape)
            means[:,:,:,0] -= img_params.dx[:,None,None]
            means[:,:,:,1] -= img_params.dy[:,None,None]

            # (13,4,1) = (nimages, nderivs, nmog)
            #print('scale:', scale.shape)
            #print('iv shapes:', iv0.shape, iv1.shape, iv2.shape)

            # (13,4,64,64) = (nimages, nderivs, ny,nx)
            #print('G:', G.shape)

            # The distsq array is going to be nimages x nderivs x nmog x ny=64 x nx=64
            n = cp.newaxis
            #distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2 +
            #          iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]) +
            #          iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:])**2 +
                      iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n]) +
                      iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n])**2)
            # t1 = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2)
            # print('t1', t1.shape)
            # t3 = (iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # print('t3', t3.shape)
            # t2 = (iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]))
            # print('t2', t2.shape)
            # Sum over the nmog
            mog_g = cp.sum(scale[:,:,:,cp.newaxis,cp.newaxis] * cp.exp(-0.5*distsq), axis=2).astype(cp.float32)
            del means, distsq, varcopy
            #cp.savetxt('vgmogpatch.txt',mog_g.ravel())
            G += mog_g
            """

        #Do no use roi since images are padded to be (mh, mw)
        use_roi = False
        return G

    def tryUpdates(self, tractor, X, alphas=None, check_step=None):
        """
        Attempts to find the optimal step size (alpha) along the update direction X
        to maximize the log probability. Leverages GPU for step calculation and
        likelihood evaluation.

        Parameters
        ----------
        tractor : object
            An instance of the Tractor class (or similar) with methods:
            - getLogProb(use_gpu=False, **kwargs): Returns scalar log probability.
              Must have 'use_gpu=True' option that uses getLogLikelihood with GPU.
            - getParams(): Returns current parameters as a CuPy array.
            - setParams(p): Sets parameters. Expects CuPy array, handles GPU->CPU transfer.
            - getLowerBounds(): Returns lower bounds as NumPy array.
            - getUpperBounds(): Returns upper bounds as NumPy array.
            - getMaxStep(): Returns max step sizes as NumPy array.
        X : cupy.ndarray
            The update direction (from getUpdateDirection, expected to be CuPy array).
        alphas : list or numpy.ndarray or None
            List of alpha values to test. If None, default values are used.

        Returns
        -------
        dlogprob : float
            The change in log probability from the initial state to the best found state.
        alphaBest : float
            The optimal alpha (step size) found.
        """

        # 1. Fallback if GPU is disabled
        if self._gpumode == 0 or self._gpumode == 4:
            return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
        import cupy as cp
        import datetime
        #return super().tryUpdates(tractor, X, alphas=alphas)
        #lp1 = super().tryUpdates(tractor, X, alphas=alphas)
        #print (f'{lp1=}')
        t = time.time()
        print ("GPU FACTORED TRY UPDATES")
        print ("X", X.shape)

        # 2. Check if we can use the PointSource GPU path
        is_ptsrc = (len(tractor.catalog) == 1 and 
                    isinstance(tractor.catalog[0], PointSource))
        
        # 3. Check if we can use the Galaxy GPU path
        is_galaxy = (len(tractor.catalog) == 1 and 
                     isinstance(tractor.catalog[0], ProfileGalaxy))

        if tractor.isParamFrozen('images'):
            if is_ptsrc:
                # Use the new GPU Point Source Engine
                # Check if we have a cached fitter for this source
                if hasattr(self, 'current_fitter'):
                    # Pass the work to the engine
                    print ("Using new GPU PointSource Engine")

                    tuc[0] += 1
                    p0 = tractor.getParams()
                    #t = time.time()
                    #s_d, s_a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
                    #tu[1] += time.time()-t
                    #tractor.setParams(p0)


                    #Estimate memory needed with new helper method based on padded image sizes and nd
                    src = tractor.catalog[0]

                    # 1. Get masks and find which images actually overlap
                    all_tims = tractor.images
                    all_masks = tractor.modelMasks

                    # Filter out the "None" masks
                    # Store the original full list of dicts to restore later
                    all_mask_dicts = tractor.modelMasks 
                    
                    valid_tims = []
                    valid_mask_objects = []
                    valid_mask_dicts = []

                    for tim in all_tims:
                        m = tractor._getModelMaskFor(tim, src)
                        if m is not None:
                            valid_tims.append(tim)
                            valid_mask_objects.append(m)
                            # Re-wrap the mask in a dict so tractor can still query it by source
                            valid_mask_dicts.append({src: m})

                    if len(valid_tims) == 0:
                        print("Source has no overlap with any image; log-likelihood is 0.")
                        # Handle edge case: no overlap at all
                        return 0.0, 0.0 

                    # 2. Update context with the expected data structures
                    tractor.images = valid_tims
                    # This is now a list of dicts, keeping tractor happy
                    tractor.modelMasks = valid_mask_dicts
                    if len(valid_tims) != len(all_tims):
                        print (f'TryUpdates PS: Warning {len(valid_tims)=} {len(all_tims)=} - ONE OR MORE model masks are None.')
                    Nimages = len(valid_tims) # The new, filtered count
                    #Get free memory
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    nd = tractor.numberOfParams()+2

                    try:
                        # Estimate memory using ONLY the images we will actually render
                        # pixel position of the source in each image
                        xy_valid = [tim.getWcs().positionToPixel(src.pos) for tim in valid_tims]
                        px, py = np.array(xy_valid).T
                        if np.any(np.isnan(px)):
                            xy2 = [tim.getWcs().positionToPixel(src.pos) for tim in all_tims]
                            px2, py2 = np.array(xy2).T
                            print (f'NAN PS {px=} {py=} {px2=} {py2=}')

                        est_mem = self.predict_fft_memory(tractor, valid_mask_objects, xy_valid, nd)
                        free_mem /= 1.e+9
                        print (f'TryUpdates PS: Estimated memory {est_mem} GB free memory {free_mem} for {Nimages=} {nd=} at ', datetime.datetime.now(), "src = ", tractor.catalog[0])
                        use_less_mem = False
                        if free_mem < est_mem:
                            print(f"Warning: TryUpdates PS Estimated memory {est_mem} GB is greater than free memory {free_mem} GB; Running less-mem GPU mode instead!", datetime.datetime.now(), "src = ", tractor.catalog[0])
                            use_less_mem = True

                        t = time.time()
                        self.current_fitter.ie_batch = None
                        best_dlnp, best_alpha = self.current_fitter.tryUpdates_gpu(X, alphas, check_step=check_step, use_less_mem=use_less_mem)
                        tu[0] += time.time()-t
                    except Exception as ex:
                        print ("Exception in TRY UPDATES PS GPU - running CPU version instead: "+str(ex))
                        import sys
                        sys.stderr.write(f"\n----Traceback below----\n")
                        import traceback 
                        traceback.print_exc()
                        sys.stderr.flush()
                        sys.exit(-1)
                        tu[0] += time.time()-t
                        t = time.time()
                        tractor.images = all_tims
                        tractor.modelMasks = all_masks
                        tractor.setParams(p0)
                        s_d, s_a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
                        tu[1] += time.time()-t
                        return s_d, s_a
                    finally:
                        # CRITICAL: Restore the original image list so Tractor doesn't
                        # "forget" the other images for the rest of the brick
                        tractor.images = all_tims
                        tractor.modelMasks = all_masks

                    #import sys
                    #sys.exit(0)
                    # Clean up after the line search is done
                    del self.current_fitter 
                    #print (f'TU PS COMP {best_dlnp=} {s_d=} {best_alpha=} {s_a=}')
                    #if s_a != best_alpha:
                    #    print ("PS MISMATCH")
                    #print (f'TU PS GPU {best_dlnp=} {best_alpha=}')
                    #print (f'TU PS CPU {s_d=} {s_a=}')
                    return best_dlnp, best_alpha
                    #return s_d, s_a
                t = time.time()
                s_d, s_a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
                #return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step) 
                tu[2] += time.time()-t
                return s_d, s_a
            
        if not (is_galaxy and tractor.isParamFrozen('images')):
            # 4. Fallback for everything else
            print('Calling superclass tryUpdates')
            t = time.time()
            a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
            tuc[1] += 1
            tu[3] += time.time()-t
            return a
            #return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)

        if not ((len(tractor.catalog) == 1) and
                isinstance(tractor.catalog[0], ProfileGalaxy) and
                tractor.isParamFrozen('images')):
            print('Calling superclass tryUpdates')
            t = time.time()
            #return super().tryUpdates(tractor, X, alphas=alphas)
            s_d, s_a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
            tuc[2] += 1
            tu[4] += time.time()-t
            return s_d, s_a
 

        # Dustin's FIXME improvement ideas
        # - pass in / cache initial logprob / model?
        # - pass in / cache BatchPixelizedPSF ??

        #t = time.time()
        #return super().tryUpdates(tractor, X, alphas=alphas)
        tuc[3] += 1
        #p0 = tractor.getParams()
        #s_d, s_a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
        #tu[5] += time.time()-t
        #return s_d, s_a
        #tractor.setParams(p0)
        t = time.time()

        #print('number of images:', len(tractor.images))
        Nimages = len(tractor.images)
        src = tractor.catalog[0]
        steps = self.getParameterSteps(tractor, X, alphas)
        #print('number of steps (alphas) to try:', len(steps))
        #print ("Alphas", alphas, "STEPS", steps)
        if len(steps) == 0:
            self.last_step_hit_limit = True
            self.hit_limit = True
            return 0., 0.

        p0 = tractor.getParams()
        # Also add the current parameter values (zero step size) for a baseline comparison.
        # (it would be nice to be able to track this and pass it in rather than recomputing,
        # but with modelmasks etc I think it will be simpler to just recompute it here...)
        # steps: list of (alpha, params, step_lim, param_lim)
        steps.append((0., p0, False, False))

        # Filter out the "None" masks
        # Store the original full list of dicts to restore later
        all_tims = tractor.images
        all_masks = tractor.modelMasks
        
        valid_tims = [] 
        valid_mask_objects = []
        valid_mask_dicts = []

        for tim in all_tims:
            m = tractor._getModelMaskFor(tim, src)
            if m is not None:
                valid_tims.append(tim)
                valid_mask_objects.append(m)
                # Re-wrap the mask in a dict so tractor can still query it by source
                valid_mask_dicts.append({src: m})

        if len(valid_tims) == 0:
            print("Source has no overlap with any image; log-likelihood is 0.")
            # Handle edge case: no overlap at all
            return 0.0, 0.0

        # 2. Update context with the expected data structures
        tractor.images = valid_tims
        # This is now a list of dicts, keeping tractor happy
        tractor.modelMasks = valid_mask_dicts

        if len(valid_tims) != len(all_tims):
            print (f'TryUpdates Gal: Warning {len(valid_tims)=} {len(all_tims)=} - ONE OR MORE model masks are None.')
        Nimages = len(valid_tims)
        nd = tractor.numberOfParams() + 2

        try:
            # Position in valid images only
            xy = [tim.getWcs().positionToPixel(src.pos) for tim in valid_tims]
            px, py = np.array(xy).T
            if np.any(np.isnan(px)):
                xy2 = [tim.getWcs().positionToPixel(src.pos) for tim in all_tims]
                px2, py2 = np.array(xy2).T
                print (f'NAN {px=} {py=} {px2=} {py2=}')

            #Get free memory
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            #New helper method predicts memory needed
            est_mem = self.predict_fft_memory(tractor, valid_mask_objects, xy, nd)
            free_mem /= 1.e+9
            print (f'TryUpdates: Estimated memory {est_mem} GB free memory {free_mem} for {Nimages=} {nd=} at ', datetime.datetime.now(), "src = ", tractor.catalog[0])
            use_less_mem = False
            if free_mem < est_mem:
                print(f"Warning: TryUpdates Estimated memory {est_mem} GB is greater than free memory {free_mem} GB; Running less-mem GPU mode instead!", datetime.datetime.now(), "src = ", tractor.catalog[0])
                use_less_mem = True
            img_sky = [tim.getSky().getConstant() for tim in tractor.images]
            img_pix = [tim.getImage(use_gpu=True) for tim in tractor.images]
            img_ie  = [tim.getInvError(use_gpu=True) for tim in tractor.images]
            profiles = []
            fluxes = []
            logpriors = []
            for _,p,_,_ in steps:
                #print ("P", p)
                tractor.setParams(p)
                # for _getBatchGalaxyProfiles we need tuples of (name, amix, step)...
                profiles.append(('', getShearedProfileGPU(src, tractor.images, px, py), 0.))
                # grab the brightnesses
                fluxes.append([tim.getPhotoCal().brightnessToCounts(src.brightness) for tim in tractor.images])
                # While we're stepping through the parameters, compute log-priors...
                lp = tractor.getLogPrior()
                logpriors.append(lp)

            # Unfortunately, Sersic galaxies can get a different number of Gaussian mixture components
            # as we step the Sersic index parameter, so in the "profiles" above, they can have different
            # sizes; _getBatchGalaxyProfiles requires them to be the same sizes - so check and pad if
            # necessary.
            ng = np.unique([pro.var.shape[1] for _,pro,_ in profiles])
            if len(ng) != 1:
                # pad
                padsize = max(ng)
                for _,pro,_ in profiles:
                    shape = pro.var.shape
                    # needs padding?
                    oldsize = shape[1]
                    if shape[1] == padsize:
                        continue
                    newshape = (shape[0], padsize, shape[2], shape[3])
                    # var: size (nimages, ng, 2, 2) - cupy
                    padded = cp.zeros(newshape)
                    padded[:,:oldsize,:,:] = pro.var
                    pro.var = padded
                    # amp: size (ng,) -- a numpy array (not cupy) for some reason
                    padded = np.zeros(padsize)
                    padded[:oldsize] = pro.amp
                    pro.amp = padded
                    # mean: size (ng,2) -- also a numpy array
                    shape = pro.mean.shape
                    newshape = (padsize, shape[1])
                    padded = np.zeros(newshape)
                    padded[:oldsize,:] = pro.mean
                    pro.mean = padded

            if use_less_mem:
                # Initialize the accumulator for all alpha steps
                # This stays on the GPU so we don't transfer back and forth in the loop
                total_chisq = cp.zeros(len(steps), dtype=cp.float32)

                # Pre-convert fluxes to GPU once (it's a small array)
                # Shape: (nsteps, Nimages)
                fluxes_gpu = cp.asarray(fluxes, dtype=cp.float32)

                tims = tractor.images
                modelmasks = valid_mask_objects 

                try:
                    for i in range(Nimages):
                        # 1. Extract single-image data
                        # We wrap these in lists so your existing Batch methods still work
                        s_masks = [modelmasks[i]]
                        s_xy = [xy[i]]
                        s_sky = [img_sky[i]]
                        s_pix = [img_pix[i]]
                        s_ie = [img_ie[i]]


                        # 1. Create single-image profile data
                        s_profiles = []
                        for name, amix, step in profiles:
                            #print (f'{amix.mean.shape=} {amix.amp.shape=} {amix.var.shape=}')
                            #print (f'{type(amix)=}')
                            #s_mean = amix.mean[i:i+1] # (1, Nd, K, D)
                            #s_amp  = amix.amp[i:i+1]  # (1, Nd, K)
                            s_var  = amix.var[i:i+1]  # (1, Nd, K, D, D)
                            # amix.amp and amix.mean are usually shared (ng, ...) 
                            # but if they were per-image, you'd slice them here too.
                            #s_amix = BatchMixtureOfGaussians(s_amp, s_mean, s_var)
                            s_amix = BatchMixtureOfGaussians(amix.amp, amix.mean, s_var, unbalanced=True)
                            s_profiles.append((name, s_amix, step))

                        tractor.images = [tims[i]]
                        tractor.modelMasks = [valid_mask_dicts[i]]
                        # 2. Setup single-image parameters and profiles
                        # This keeps pH/pW consistent with the single image's needs
                        img_params_i, cx_i, cy_i, pW_i, pH_i = self._getBatchImageParams(tractor, s_masks, s_xy)

                        # Slicing fluxes_gpu for just this image: fluxes_gpu[:, i]
                        gals_i = self._getBatchGalaxyProfiles(s_profiles, s_masks, px[i:i+1], py[i:i+1],
                                                              cx_i, cy_i, pW_i, pH_i,
                                                              fluxes_gpu[0, i:i+1], s_sky, s_pix, s_ie)

                        img_params_i.addBatchGalaxyProfiles(gals_i)

                        # 3. Render and Calculate Chi2 for this image
                        # G_i shape: (1, nsteps, pH_i, pW_i)
                        G_i = self.computeGalaxyModelsVectorized(img_params_i)

                        # Call a modified version of your chi2 function or inline it:
                        # We sum over the image axes (2 and 3) and the single image axis (0)
                        # to get a (nsteps,) result to add to total_chisq.
                        total_chisq += self.calculate_chi2_cupy(G_i, fluxes_gpu[:, [i]],
                                                                gals_i.mmpix, gals_i.mmie,
                                                                cp.asarray(s_sky))

                        # 4. CRITICAL: Clear the deck for the next image
                        del G_i, img_params_i, gals_i, s_pix, s_ie, s_amix, s_var
                        cp.get_default_memory_pool().free_all_blocks()

                finally:
                    # Always restore state even if the loop fails
                    tractor.images = tims
                    tractor.modelMasks = modelmasks
                # Move final result back to CPU for the logprob logic
                chisq = total_chisq
            else:
                img_params, cx,cy,pW,pH = self._getBatchImageParams(tractor, valid_mask_objects, xy)
                gals = self._getBatchGalaxyProfiles(profiles, valid_mask_objects, px, py, cx, cy, pW, pH,
                                                    fluxes[0], img_sky, img_pix, img_ie)
                img_params.addBatchGalaxyProfiles(gals)
                if img_params.ffts is None:
                    print ("Warning> img_params.ffts is None!  Calling superclass tryUpdates (on CPU)")
                    #print (f'TU Z2 0 {s_d=} 0 {s_a=}')
                    tu[6] += time.time()-t
                    return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
                G = self.computeGalaxyModelsVectorized(img_params)
                assert(G.shape == (Nimages, len(steps), pH, pW))
                mmpix = gals.mmpix
                mmie = gals.mmie
                fluxes = cp.asarray(fluxes, dtype=cp.float32)
                sky = cp.asarray(img_sky, dtype=cp.float32)
                #print ("MMPIX", mmpix.shape, mmie.shape, mmpix.max(), mmpix.sum())
                #print ("FLUXES", fluxes.shape, fluxes)
                #print ("SKY", sky.shape, sky)

                #print ("G", G.shape)
                #chisq = cp.sum([((G[i_image] * fluxes[i_step, i_image] - mmpix[i_image]) * mmie[i_image])**2 for i_image in range(Nimages)])
                chisq = self.calculate_chi2_cupy(G, fluxes, mmpix, mmie, sky)
                del G, img_params
                mpool = cp.get_default_memory_pool()
                mpool.free_all_blocks()
                #print ("CHISQ", chisq.shape)

            #print ("PRIOR", logpriors)
            logprob = np.array(logpriors)-0.5*chisq.get()
        except Exception as ex:
            print(f"Exception in Galaxy TRY UPDATES GPU: {ex}. Falling back to CPU.")
            tractor.setParams(p0)
            # Restore full image list for CPU fallback
            tractor.images = all_tims
            tractor.modelMasks = all_masks
            return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)

        finally:
            # CRITICAL: Restore the full list
            tractor.images = all_tims
            tractor.modelMasks = all_masks

        #print ("LOGPROB", logprob.shape)

        # CRAIG -- at this point, we have rendered the galaxy models, in G.
        # We should have the (padded) image pixels and inverse-errors in 'mmpix' and 'mmie' on the GPU.
        # (ie, I think they're all the same shape and ready to go...)
        # At this point, we need to compute the log-probs for each step in `steps`.
        # The log-probs are equal to the logpriors plus the log-likelihoods, which are
        # equal to the -0.5 * the sum of chi-squared values: (data - model) * inverr.

        # Once the log-probs for each step have been computed, we want to select the best one,
        # set the tractor params to the corresponding params, and return the
        # (delta-logprob, alpha)
        # of the best one.  delta-logprob will be the log-prob relative to the last one
        # (corresponding to the "step" we added at p0).

        # copy-pasting some additional description from slack:

        # I don’t know exactly what shape G is at this point, but conceptually it is
        #   N_steps x N_images x postage-stamp-size
        # For each step, we want to compute
        #    chisq = sum([((G[i_step, i_image] * fluxes[i_step, i_image] - mm_pix[i_image]) * mm_ie[i_image])**2 for i_image in range(N_images)])
        # that is,
        #    chisq = sum over i_image:
        #                 of chi**2  (pixelwise)
        #                 where chi = (model - data) * inverr
        #                 and data = mm_pix[i_image,...], and inverr = mm_ie[i_image...]
        #                 and model = G * flux[i_step,i_image] + sky[i_image]
        # and then
        #    logprob = logprior[i_step] + -0.5*chisq

        max_idx = np.argmax(logprob)
        #print ("LOGPROB", logprob)
        #print ("MAX", max_idx, logprob[max_idx])
        #print ("Pbest", steps[max_idx][1])
        pbest = steps[max_idx][1]
        hit_limit = steps[max_idx][2]
        tractor.setParams(pbest)
        for i in range(len(steps)):
            if steps[i][2]:
                self.hit_limit = True #set if any hits limit

        tt[6] += time.time()-t
        if max_idx == len(logprob)-1:
            print ("Best is previous")
            #print (f'TU Z1 0 {s_d=} 0 {s_a=}')
            tu[6] += time.time()-t
            return (0., 0.)

        lp_best = logprob[max_idx]
        lp_last = logprob[-1]
        alpha = steps[max_idx][0] #get alpha from steps not array of alphas!
        if lp_best > lp_last:
            # Best we've found so far -- accept this step!
            self.last_step_hit_limit = hit_limit
        tu[6] += time.time()-t
        #g_d = lp_best-lp_last
        #print (f'TU G {g_d=} {s_d=} {alpha=} {s_a=}')
        #if s_a != alpha:
        #    print ("GAL MISMATCH")
        return (lp_best-lp_last, alpha)

    def calculate_chi2_cupy(self, G, fluxes, mmpix, mmie, sky):
        """
        Calculates the chi-squared efficiently using CuPy.

        Args:
            G (cupy.ndarray): Shape (Nimages, nsteps, height, width).
            fluxes (cupy.ndarray): Shape (nsteps, Nimages).
            mmpix (cupy.ndarray): Shape (height, width).
            mmie (cupy.ndarray): Shape (height, width).
            sky (cupy.ndarray): Shape (Nimages,).

        Returns:
            cupy.ndarray: chi2 with shape (Nimages, nsteps).
        """
        Nimages, nsteps, height, width = G.shape

        # 1. Reshape sky and fluxes for broadcasting
        # sky needs to broadcast as (Nimages, 1, 1, 1)
        sky_reshaped = sky[:, cp.newaxis, cp.newaxis, cp.newaxis]

        # fluxes needs to broadcast as (1, nsteps, 1, 1) across height/width for G,
        # but the i_image index needs to align with G's first dimension.
        # It's better to think of fluxes as (nsteps, Nimages)
        # and then carefully expand its dimensions for the multiplication.

        # We need fluxes to be (Nimages, nsteps, 1, 1) for element-wise multiplication with G
        # Let's reorder fluxes to (Nimages, nsteps) for easier broadcasting with G
        # If fluxes is (nsteps, Nimages), we should transpose it for this logic
        # Or, we can reshape fluxes to (1, nsteps, 1, 1) and G to (Nimages, nsteps, height, width)
        # Let's adjust for the given fluxes shape (nsteps, Nimages)
        # Model = G * fluxes[i_step, i_image] + sky[i_image]

        # To align dimensions:
        # G: (Nimages, nsteps, height, width)
        # fluxes: (nsteps, Nimages) -> we want to multiply G[i,j,:,:] by fluxes[j,i]
        # sky: (Nimages,) -> we want to add sky[i]

        # The most straightforward way to handle fluxes given its shape (nsteps, Nimages)
        # is to transpose it and then expand dimensions for broadcasting with G.
        # We want fluxes_broadcastable[i_image, i_step, 1, 1]
        # So, fluxes.T will be (Nimages, nsteps).
        fluxes_broadcastable = fluxes.T[:, :, cp.newaxis, cp.newaxis] # Shape (Nimages, nsteps, 1, 1)

        # Calculate the model: G * flux + sky
        # G: (Nimages, nsteps, height, width)
        # fluxes_broadcastable: (Nimages, nsteps, 1, 1) - broadcasts across height and width
        # sky_reshaped: (Nimages, 1, 1, 1) - broadcasts across nsteps, height, and width
        model = G * fluxes_broadcastable + sky_reshaped

        # Calculate (model - data) * inverr
        # mmpix: (Nimages, height, width)
        # mmie: (Nimags, height, width)
        diff = model - mmpix[:,cp.newaxis,:,:]
        chi = diff * mmie[:,cp.newaxis,:,:] # This is (model - data) * inverr (mmie is inverr)

        # Square chi and sum over Nimages, height and width
        # Sum over Nimages (axis 0), height (axis 2), and width (axis 3)
        # The result will be a 1D array of shape (nsteps,)
        chi2 = cp.sum(chi**2, axis=(0, 2, 3))
        #print ("CHI2", cp.sum(chi**2, axis=(2, 3)), chi2.dtype)
        return chi2

    def _getBatchImageParams(self, tr, masks, pxy):
        Nimages = len(tr.images)
        # pxy: [(x,y), ...] length tr.images
        psfs = [tim.getPsf() for tim in tr.images]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))

        extents = [mm.extent for mm in masks]

        px, py = np.array(pxy).T
        px = px.astype(np.float32)
        py = py.astype(np.float32)
        imgH, imgW = np.array([tim.shape for tim in tr.images]).T
        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        x0, x1, y0, y1 = np.asarray(extents).T
        #print (f'{x0=} {x0.dtype=}')
        """
        if np.any(px < 0):
            print (f"Warning: {px=} is outside of image")
            px[px < 0] = 0
        if np.any(py < 0):
            print (f"Warning: {py=} is outside of image")
            py[py < 0] = 0
        if np.any(px > imgW):
            print (f"Warning: {px=} is outside of image")
            px[px > imgW] = imgW[px > imgW]
        if np.any(py > imgH):
            print (f"Warning: {py=} is outside of image")
            py[py > imgH] = imgH[py > imgH]
        """
        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2]), axis=0)

        # PSF Fourier transforms
        batch_psf = BatchPixelizedPSF(psfs)
        P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(px, py, gpu_halfsize)
        #print ("pH", pH, pW)
        #print ("Cx", cx, cy)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (Nimages,len(w),len(v)))

        img_params = BatchImageParams(P, v, w, batch_psf.psf_mogs)
        return img_params, cx,cy, pH,pW

    def _getBatchPointSourceProfiles(self, derivs, masks, px, py, cx, cy, pW, pH,
                                     img_counts, img_sky, img_pix, img_ie):
        Nimages = len(img_counts)
        mx0, mx1, my0, my1, mh, mw = np.array([(mm.x0, mm.x1, mm.y0, mm.y1)+mm.shape for mm in masks]).T
        counts = cp.asarray(img_counts, dtype=cp.float32)

        extents = [mm.extent for mm in masks]
        x0, x1, y0, y1 = np.asarray(extents).T

        # sub-pixel shift we have to do at the end...
        dx = px - cx
        dy = py - cy
        mux = dx - x0
        muy = dy - y0
        sx = mux.round().astype(np.int32)
        sy = muy.round().astype(np.int32)
        # the subpixel portion will be handled with a Lanczos interpolation
        mux -= sx
        muy -= sy
        dxi = cp.asarray(x0+sx)
        dyi = cp.asarray(y0+sy)
        assert(np.abs(mux).max() <= 0.5)
        assert(np.abs(muy).max() <= 0.5)

        # Embed pix and ie in images the same size as pW,pH.
        padpix = cp.zeros((Nimages, pH,pW), cp.float32)
        padie  = cp.zeros((Nimages, pH,pW), cp.float32)
        assert(np.all(sy <= 0) and np.all(sx <= 0))

        x_delta = np.ones(mx0.shape, np.int32)
        y_delta = np.ones(my0.shape, np.int32)
        x_delta[mx0 == 0] = 0
        y_delta[my0 == 0] = 0
        x_delta[sx == 0] = 0
        y_delta[sy == 0] = 0
        for i, pix in enumerate(img_pix):
            padpix[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = pix[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
        for i, ie in enumerate(img_ie):
            padie[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = ie[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
        roi = cp.asarray([-sx, -sy, mw, mh]).T
        mmpix = cp.asarray(padpix)
        mmie = cp.asarray(padie)
        sky = cp.asarray(img_sky, dtype=cp.float32)


    def _getBatchGalaxyProfiles(self, amixes_gpu, masks, px, py, cx, cy, pW, pH,
                                img_counts, img_sky, img_pix, img_ie):
        Nimages = len(img_counts)
        #Nimages = len(img_pix)

        #Not optimal but for now go back into loop
        mx0, mx1, my0, my1, mh, mw = np.array([(mm.x0, mm.x1, mm.y0, mm.y1)+mm.shape for mm in masks]).T
        counts = cp.asarray(img_counts, dtype=cp.float32)

        extents = [mm.extent for mm in masks]
        x0, x1, y0, y1 = np.asarray(extents).T

        # sub-pixel shift we have to do at the end...
        dx = px - cx
        dy = py - cy
        mux = dx - x0
        muy = dy - y0
        sx = mux.round().astype(np.int32)
        sy = muy.round().astype(np.int32)
        # the subpixel portion will be handled with a Lanczos interpolation
        mux -= sx
        muy -= sy
        dxi = cp.asarray(x0+sx)
        dyi = cp.asarray(y0+sy)
        assert(np.abs(mux).max() <= 0.5)
        assert(np.abs(muy).max() <= 0.5)

        # Embed pix and ie in images the same size as pW,pH.
        padpix = cp.zeros((Nimages, pH,pW), cp.float32)
        padie  = cp.zeros((Nimages, pH,pW), cp.float32)
        assert(np.all(sy <= 0) and np.all(sx <= 0))

        x_delta = np.ones(mx0.shape, np.int32)
        y_delta = np.ones(my0.shape, np.int32)
        x_delta[mx0 == 0] = 0
        y_delta[my0 == 0] = 0
        x_delta[sx == 0] = 0
        y_delta[sy == 0] = 0
        for i, pix in enumerate(img_pix):
            #padpix[i, -sy[i]-1:-sy[i]+mh[i], -sx[i]-1:-sx[i]+mw[i]] = pix[my0[i]-1:my1[i], mx0[i]-1:mx1[i]]
            padpix[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = pix[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
            #print ("PIX", pix.max(), pix[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]].max(), padpix[i].max(), pix[my0[i]-y_delta[i], mx0[i]-x_delta[i]], padpix[i, -sy[i]-y_delta[i], -sx[i]-x_delta[i]])
        for i, ie in enumerate(img_ie):
            #padie[i, -sy[i]-1:-sy[i]+mh[i], -sx[i]-1:-sx[i]+mw[i]] = ie[my0[i]-1:my1[i], mx0[i]-1:mx1[i]]
            padie[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = ie[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
            #print ("IE",ie.max(), padie[i].max())
        roi = cp.asarray([-sx, -sy, mw, mh]).T
        mmpix = cp.asarray(padpix)
        mmie = cp.asarray(padie)
        sky = cp.asarray(img_sky, dtype=cp.float32)

        # Split "amix" into terms that we will evaluate using MoG vs FFT.
        # (we'll use that same split for the derivatives also)
        #vv = amix_gpu.var[:,:,0,0] + amix_gpu.var[:,:,1,1]
        _,amix_gpu,_ = amixes_gpu[0]
        nvar =  amix_gpu.var.shape[1]
        for i in range(len(amixes_gpu)):
            assert(amixes_gpu[i][1].var.shape[1] == nvar)

        vv = cp.zeros((Nimages, len(amixes_gpu), amix_gpu.var.shape[1]))
        for i, am in enumerate(amixes_gpu):
            vv[:,i] = am[1].var[:,:,0,0]+am[1].var[:,:,1,1]

        # Ramp between:
        inner_real_nsigma = 3.
        outer_real_nsigma = 4.
        nsigma1 = inner_real_nsigma
        nsigma2 = outer_real_nsigma

        # Terms that will wrap-around significantly if evaluated
        # with FFT...  We want to know: at the outer edge of this
        # patch, how many sigmas out are we?  If small (ie, the
        # edge still has a significant fraction of the flux),
        # render w/ MoG.
        IM = ((pW/2)**2 < (nsigma2**2 * vv))
        IF = ((pW/2)**2 > (nsigma1**2 * vv))
        ramp = np.any(IM*IF)
        #mogweights = cp.ones(Nimages, dtype=np.float32)
        #fftweights = cp.ones(Nimages, dtype=np.float32)
        mogweights = cp.ones(vv.shape, dtype=cp.float32)
        fftweights = cp.ones(vv.shape, dtype=cp.float32)
        if ramp:
            # ramp
            ns = (pW/2) / cp.maximum(1e-6, cp.sqrt(vv))
            #mogweights = cp.minimum(1., (nsigma2 - ns[IM]) / (nsigma2 - nsigma1))
            #fftweights = cp.minimum(1., (ns[IF] - nsigma1) / (nsigma2 - nsigma1))
            mogweights = cp.minimum(1., (nsigma2 - ns) / (nsigma2 - nsigma1))*IM
            fftweights = cp.minimum(1., (ns - nsigma1) / (nsigma2 - nsigma1))*IF
            assert(cp.all(mogweights[IM] > 0.))
            assert(cp.all(mogweights[IM] <= 1.))
            assert(cp.all(fftweights[IF] > 0.))
            assert(cp.all(fftweights[IF] <= 1.))

        K = amix_gpu.var.shape[1]
        D = amix_gpu.var.shape[2]
        mh = pH
        mw = pW

        #print ("amixes_gpu", type(amixes_gpu), type(IF), type(IM), type(K), type(D))
        #print ("mogweights", type(mogweights), type(fftweights), fftweights.shape, type(px), type(py), type(mux), type(muy))
        #print ("MH", type(mh), type(mw), type(counts), type(cdi), type(roi))

        #img_derivs = BatchImageDerivs(amixes_gpu, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, cdi, roi, sky, dxi, dyi, fit_pos)
        #img_params.addBatchImageDerivs(img_derivs)
        return BatchGalaxyProfiles(amixes_gpu, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, roi, sky, dxi, dyi)

class MyAwesomeGpuImplementation(GPUFriendlyOptimizer):
    def computeUpdateDirections(self, imgs):
        return super().computeUpdateDirections(imgs)

    
if __name__ == '__main__':

    import pylab as plt
    from tractor import Image, PixPos, Flux, Tractor, NullWCS, NCircularGaussianPSF, PointSource
    from tractor import ExpGalaxy, PixelizedPSF, HybridPixelizedPSF, GaussianMixturePSF
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor import ModelMask, ConstantFitsWcs, RaDecPos
    from astrometry.util.util import Tan

    n_ims = 2
    sig1s = [3., 10.] * 5
    psf_sigmas = [2., 1.] * 5
    #pixscales = [0.25, 0.25] * 5
    pixscales = [1, 1] * 5
    H,W = 50,50
    cx,cy = 23,27
    #H,W = 100,100
    #cx,cy = 53,47
    # True source...
    true_flux = 1000.
    true_shape = [3., 0.5, 0.3]

    pixscale = 1.0
    #wcs_rot = np.linspace(0, 2.*np.pi, n_ims)
    wcs_rot = np.array([0, np.pi/4.])
    ra,dec = 100., 42.

    ra_off,dec_off = 0.0003 * np.random.normal(size=(2,))

    true_pos = RaDecPos(ra + ra_off, dec + dec_off)
    true_src = ExpGalaxy(true_pos, Flux(true_flux), EllipseE(*true_shape))

    tims = []
    for i in range(n_ims):
        #x = np.arange(W)
        #y = np.arange(H)
        #data = np.exp(-0.5 * ((x[np.newaxis,:] - cx)**2 + (y[:,np.newaxis] - cy)**2) /
        #              psf_sigmas[i]**2)
        #data *= fluxes[i] / (2. * np.pi * psf_sigmas[i]**2)
        data = np.random.normal(size=(H,W)) * sig1s[i]

        pW = pH = 63
        pp = np.arange(pW)
        psf_stamp = np.exp(-0.5 * ((pp[np.newaxis,:] - pW//2)**2 + (pp[:,np.newaxis] - pH//2)**2) /
                           psf_sigmas[i]**2) / (2. * np.pi * psf_sigmas[i]**2)
        gpsf = GaussianMixturePSF(np.array([1.]), np.array([[0.,0.]]),
                                  np.eye(2)[np.newaxis,:,:] * psf_sigmas[i]**2)
        pix = PixelizedPSF(psf_stamp)
        psf = HybridPixelizedPSF(pix, gauss=gpsf)

        c,s = np.cos(wcs_rot[i]), np.sin(wcs_rot[i])
        pixsc = pixscales[i] / 3600.
        tan = Tan(ra, dec, float(cx+i), float(cy), c*pixsc, s*pixsc, -s*pixsc, c*pixsc,
                  float(W), float(H))
        #print('Pixel scale:', tan.pixel_scale())
        wcs = ConstantFitsWcs(tan)

        tims.append(Image(data=data, inverr=np.ones_like(data) / sig1s[i],
                          psf=psf, wcs=wcs))

        tr = Tractor([tims[i]], [true_src])
        true_mod = tr.getModelImage(0)
        data += true_mod

    #src = PointSource(PixPos(W//2, H//2), Flux(100.))
    e = EllipseE(2., 0., 0.)
    src  = ExpGalaxy(RaDecPos(ra, dec), Flux(100.), EllipseESoft.fromEllipseE(e))
    src2 = ExpGalaxy(RaDecPos(ra, dec), Flux(100.), EllipseESoft.fromEllipseE(e))

    opt = ConstrainedDenseOptimizer()
    #opt2 = GPUFriendlyOptimizer()
    opt2 = MyAwesomeGpuImplementation()

    tr = Tractor(tims, [src], optimizer=opt)
    tr.setModelMasks([{src: ModelMask(0, 0, W, H)} for tim in tims])

    tr2 = Tractor(tims, [src2], optimizer=opt2)
    tr2.setModelMasks([{src2: ModelMask(0, 0, W, H)} for tim in tims])

    tr.freezeParam('images')
    tr2.freezeParam('images')

    mods = list(tr.getModelImages())

    fit_kwargs = dict(shared_params=False, priors=False)
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **fit_kwargs)
    up2 = tr2.optimizer.getLinearUpdateDirection(tr2, **fit_kwargs)
    print('Update directions:')
    print('Orig:', up1)
    print('GPU :', up2)

    print('True source:', true_src)
    print('Initial source:', src)

    R = tr.optimize_loop(**fit_kwargs)
    print('Normal fitter: took', R['steps'], 'steps')

    R2 = tr2.optimize_loop(**fit_kwargs)
    print('GPU-friendly fitter: took', R2['steps'], 'steps')

    # for step in range(20):
    #     dlnp1,x1,alpha1 = tr.optimize(**fit_kwargs)
    #     dlnp2,x2,alpha2 = tr2.optimize(**fit_kwargs)
    #     print('Step', step)
    #     print('  dlnp1:', dlnp1)
    #     print('  dlnp2:', dlnp2)
    #     print('  alpha1:', alpha1)
    #     print('  alpha2:', alpha2)
    #     print('  x1:', x1)
    #     print('  x2:', x2)
    #     print('  step1:', x1*alpha1)
    #     print('  step2:', x2*alpha2)
    #     print('  Source:', src)
    #     print('  Source2:', src2)
        
    mods_after = list(tr.getModelImages())
    print('Source:', src)
    
    mods2_after = list(tr2.getModelImages())
    print('Source:', src2)

    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods[i], **ima)
    plt.suptitle('Original optimizer, before')
    plt.savefig('1.png')

    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods_after[i], **ima)
    plt.suptitle('Original optimizer, after')
    plt.savefig('2.png')
    
    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods2_after[i], **ima)
    plt.suptitle('GPU-friendly optimizer, after')
    plt.savefig('3.png')

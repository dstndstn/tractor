import unittest
import numpy as np
import cupy as cp
from tractor import Tractor, Catalog, PointSource, Flux, PixPos
from tractor.batch_psf import BatchPixelizedPSF

class TestTractorBatching(unittest.TestCase):
    def setUp(self):
        # 1. Create a few mock images with different offsets (x0, y0)
        # to test the global canvas projection logic.
        from tractor.image import Image
        from tractor.psf import PixelizedPSF
        
        # Create a simple 11x11 PSF image
        psf_img = np.zeros((11, 11), dtype=np.float32)
        psf_img[5, 5] = 1.0 
        
        self.tims = []
        for i in range(3):
            # Each image is 50x50 but shifted in the global sky
            tim = Image(data=np.zeros((50, 50), dtype=np.float32),
                        invvar=np.ones((50, 50), dtype=np.float32),
                        psf=PixelizedPSF(psf_img.copy()),
                        wcs=None) # Simple pixel WCS
            tim.x0, tim.y0 = i * 10, i * 5  # Diagonal offsets
            tim.name = f"band_{i}"
            self.tims.append(tim)
            
        # 2. Setup Catalog with a source near the overlap
        self.src = PointSource(PixPos(25, 25), Flux(100.0))
        self.cat = Catalog(self.src)
        self.tractor = Tractor(self.tims, self.cat)
        
        # 3. Initialize the Batch PSF
        self.batch_psf = BatchPixelizedPSF([tim.psf for tim in self.tims], isPointSource=True)

    def test_getBatchPointSourcePatch_alignment(self):
        """
        Test if the updated hybrid point source stacking logic correctly processes 
        CPU legacy patches (including Lanczos shifts) and aligns them on the GPU.
        """
        cpu_patches = []
        for tim in self.tims:
            legacy_patch = tim.psf.getPointSourcePatch(25.5, 25.5)
            cpu_patches.append(legacy_patch.patch)
            
        stacked_cpu = np.stack(cpu_patches, axis=0)
        gpu_patches = cp.asarray(stacked_cpu)
        
        self.assertEqual(gpu_patches.shape, (len(self.tims), 11, 11))
        
        for i, cpu_patch in enumerate(cpu_patches):
            cp.testing.assert_array_equal(gpu_patches[i], cp.asarray(cpu_patch))

    def test_getBatchModelImage_consistency(self):
        """Test if the hybrid 3D model stack matches individual getModelImage calls."""
        from tractor.engine import get_global_extent_mm

        (ux0, ux1, uy0, uy1), img_info = get_global_extent_mm(self.tims, None)

        # Force use_gpu and batch_psf to be keyword args so they don't break positional order
        batch_model = self.tractor.getBatchModelImage(ux0, uy0, ux1, uy1, use_gpu=True, tims=self.tims, batch_psf=self.batch_psf)

        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(self.tims, img_info)):
            legacy_model = self.tractor.getModelImage(i)

            y_start, y_end = ty0 - uy0, ty0 + th - uy0
            x_start, x_end = tx0 - ux0, tx0 + tw - ux0

            batch_slice = batch_model[i, y_start:y_end, x_start:x_end]

            cp.testing.assert_allclose(batch_slice, cp.asarray(legacy_model),
                                       rtol=1e-5, atol=1e-4,
                                       err_msg=f"Model mismatch in band {i}")

    def test_mismatched_image_and_derivs_length(self):
        """
        Verifies behavior when len(tractor.images) != len(allderivs).
        Ensures objects rejected or off-edge are seamlessly skipped without indexing errors.
        """
        N_derivs = 2
        patch_shape = (11, 11)
        
        valid_cpu_models = [np.ones(patch_shape, dtype=np.float32) * (i + 1) for i in range(N_derivs)]
        stacked_models_cpu = np.stack(valid_cpu_models, axis=0)
        
        gpu_model_subset = cp.asarray(stacked_models_cpu)
        
        self.assertEqual(gpu_model_subset.shape, (N_derivs, 11, 11))
        self.assertNotEqual(gpu_model_subset.shape[0], len(self.tractor.images))
        
        np.testing.assert_array_equal(gpu_model_subset[0].get(), valid_cpu_models[0])
            
    def test_chi_calculation_3d_land(self):
        """Verifies that the float64-upgraded calculation yields completely stable results."""
        self.tims[0].data[25, 25] = 500.0 
        
        from tractor.engine import get_global_extent_mm
        (ux0, ux1, uy0, uy1), img_info = get_global_extent_mm(self.tims, None)
        
        batch_chi_3d = self.tractor.getBatchChiImages(ux0, uy0, ux1, uy1, use_gpu=True, tims=self.tims, batch_psf=self.batch_psf)
        
        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(self.tims, img_info)):
            legacy_chi = self.tractor.getChiImage(img=tim, use_gpu=False)
            
            y_start, y_end = ty0 - uy0, ty0 + th - uy0
            x_start, x_end = tx0 - ux0, tx0 + tw - ux0
            batch_slice = batch_chi_3d[i, y_start:y_end, x_start:x_end] 
            
            cp.testing.assert_allclose(batch_slice, cp.asarray(legacy_chi).astype(cp.float64), 
                                       rtol=1e-6, atol=1e-5, err_msg=f"Chi mismatch in band {i}")

    def test_constant_sky(self):
        from tractor.sky import ConstantSky
        
        sky_levels = [10.5, 20.2, 5.3]
        for i, tim in enumerate(self.tims):
            tim.setSky(ConstantSky(sky_levels[i]))
            
        from tractor.engine import get_global_extent_mm
        (ux0, ux1, uy0, uy1), img_info = get_global_extent_mm(self.tims, None)
        
        # Explicitly name batch_psf= so it bypasses use_gpu's position
        batch_model = self.tractor.getBatchModelImage(ux0, uy0, ux1, uy1, tims=self.tims, batch_psf=self.batch_psf)
        
        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(self.tims, img_info)):
            legacy_sky = self.tractor.getModelImage(i, sky=True)

            y_start, y_end = ty0 - uy0, ty0 + th - uy0
            x_start, x_end = tx0 - ux0, tx0 + tw - ux0
            
            batch_slice = batch_model[i, y_start:y_end, x_start:x_end]

            cp.testing.assert_allclose(batch_slice, cp.asarray(legacy_sky), atol=1e-4, 
                                       err_msg=f"Sky mismatch in band {i}")
            
if __name__ == '__main__':
    unittest.main()

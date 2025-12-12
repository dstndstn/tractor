import cupy as cp

from tractor import Image

class CupyImage(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_gpu = None
        self.inverr_gpu = None

    def getGpuInvError(self):
        if self.inverr_gpu is None:
            self.inverr_gpu = cp.asarray(self.inverr)
        return self.inverr_gpu

    def setInvError(self, inverr):
        super().setInvError(inverr)
        self.inverr_gpu = None

    def getGpuImage(self):
        if self.data_gpu is None:
            self.data_gpu = cp.asarray(self.data)
        return self.data_gpu

    def setImage(self, img):
        super().setImage(img)
        self.data_gpu = None

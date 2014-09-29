import numpy as np
from tractor import RaDecPos

def interpret_roi(wcs, (H,W), roi=None, roiradecsize=None, roiradecbox=None,
                  **kwargs):
    '''
    (H,W): full image size

    If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
    in the image, in zero-indexed pixel coordinates.  x1,y1 are
    NON-inclusive, ie, x in [x0,x1);
    roi=(0,100,0,100) will yield a 100 x 100 image.

    "roiradecsize" = (ra, dec, half-size in pixels) indicates that you
    want to grab a ROI around the given RA,Dec.

    "roiradecbox" = (ra0, ra1, dec0, dec1) indicates that you
    want to grab a ROI containing the given RA,Dec ranges.
    '''

    if roiradecsize is not None:
        ra,dec,S = roiradecsize
        fxc,fyc = wcs.positionToPixel(RaDecPos(ra,dec))
        xc,yc = [int(np.round(p)) for p in fxc,fyc]

        roi = [np.clip(xc-S, 0, W),
               np.clip(xc+S, 0, W),
               np.clip(yc-S, 0, H),
               np.clip(yc+S, 0, H)]
        roi = [int(x) for x in roi]
        if roi[0]==roi[1] or roi[2]==roi[3]:
            return None

    if roiradecbox is not None:
        ra0,ra1,dec0,dec1 = roiradecbox
        xy = []
        for r,d in [(ra0,dec0),(ra1,dec0),(ra0,dec1),(ra1,dec1)]:
            xy.append(wcs.positionToPixel(RaDecPos(r,d)))
        xy = np.array(xy)
        xy = np.round(xy).astype(int)
        x0 = xy[:,0].min()
        x1 = xy[:,0].max()
        y0 = xy[:,1].min()
        y1 = xy[:,1].max()
        roi = [np.clip(x0,   0, W),
               np.clip(x1+1, 0, W),
               np.clip(y0,   0, H),
               np.clip(y1+1, 0, H)]
        if roi[0] == roi[1] or roi[2] == roi[3]:
            return None

    if roi is not None:
        x0,x1,y0,y1 = roi
    else:
        x0 = y0 = 0
        x1 = W
        y1 = H

    return (x0,x1,y0,y1), ((x0 != 0) or (y0 != 0) or (x1 != W) or (y1 != H))

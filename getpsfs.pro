
; Hahaha, I don't know how to do loops in IDL

; be sure to:
;    export WISE_DATA=~/WISE/etc
;    export IDL_PATH="~/WISE/pro:+${IDL_PATH}"


psf = wise_psf_cutout(0, 0, allsky=1, band=1)

mwrfits,psf,'psf-1-0-0.fits'

psf = wise_psf_cutout(0, 500, band=1, /allsky)
mwrfits,psf,'psf-1-0-500.fits'

psf = wise_psf_cutout(0, 1000, band=1, /allsky)
mwrfits,psf,'psf-1-0-1000.fits'

psf = wise_psf_cutout(500, 0, band=1, /allsky)
mwrfits,psf,'psf-1-500-0.fits'

psf = wise_psf_cutout(500, 500, band=1, /allsky)
mwrfits,psf,'psf-1-500-500.fits'

psf = wise_psf_cutout(500, 1000, band=1, /allsky)
mwrfits,psf,'psf-1-500-1000.fits'

psf = wise_psf_cutout(1000, 0, band=1, /allsky)
mwrfits,psf,'psf-1-1000-0.fits'

psf = wise_psf_cutout(1000, 500, band=1, /allsky)
mwrfits,psf,'psf-1-1000-500.fits'

psf = wise_psf_cutout(1000, 1000, band=1, /allsky)
mwrfits,psf,'psf-1-1000-1000.fits'


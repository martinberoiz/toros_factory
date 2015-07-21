import toros
import numpy as np    
    
if __name__ == "__main__":
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import os

    test_hdulist = fits.open("test.fits")
    test_data = test_hdulist[0].data
    test_mask = test_data < 0
    test_header_wcs = test_hdulist[0].header
    test_data_mask = np.ma.array(test_data, mask=test_mask)

    ref_hdulist = fits.open("ref.fits")
    ref_data = ref_hdulist[0].data
    ref_mask = ref_hdulist[1].data.astype('bool')
    ref_header_wcs = ref_hdulist[0].header

    #Create a ref file with a WCS header, but no mask
    hdu_nowcs_nomask = fits.PrimaryHDU(ref_data, header=ref_header_wcs)
    hdu_nowcs_nomask.writeto('ref_wcs_nomask.fits', clobber=True)

    #create a ref file with no WCS and no mask
    hdu_nowcs_nomask = fits.PrimaryHDU(ref_data)
    hdu_nowcs_nomask.writeto('ref_nowcs_nomask.fits', clobber=True)

    #create a ref file with no WCS and mask
    hdulist_nowcs_masked = fits.HDUList([fits.PrimaryHDU(ref_data), fits.ImageHDU(ref_mask.astype('uint8'), name='mask')])
    hdulist_nowcs_masked.writeto('ref_nowcs_mask.fits', clobber=True)

    #Create a bare header with no WCS
    header_nowcs = fits.PrimaryHDU(data=test_data).header


    for test_label, test_h in zip(["with wcs", "without wcs", "with no header"], [test_header_wcs, header_nowcs, None]):
        for ref_label, ref_file in zip(["with mask, with wcs", "with no wcs, no mask", "with wcs, no mask", "with no wcs, with mask"], \
            ["ref.fits", "ref_nowcs_nomask.fits", "ref_wcs_nomask.fits", "ref_nowcs_mask.fits"]):
            print("Test image %s, Ref Image %s" % (test_label, ref_label))
            try:
                ref = toros.skygoldmaster.getReference(test_data_mask, test_h, ref_file)
                print("Success")
            except:
                print("Fail")


    for afile in ["ref_nowcs_nomask.fits", "ref_wcs_nomask.fits", "ref_nowcs_mask.fits"]:
        os.remove(afile)
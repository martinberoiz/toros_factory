"""@package skygoldmaster

    Sky Gold Master module
    ----------------------

    A collection of tools to create a reference image for the sky
    for the Transient Optical Robotic Observatory of the South (TOROS).

    Martin Beroiz - 2014

    email: <martinberoiz@phys.utb.edu>

    University of Texas at San Antonio
"""

import numpy as np
from astropy.io import fits
import math
import pkg_resources
from scipy import ndimage
from skimage import exposure
import registration
from astropy import wcs
from reproject import reproject_interp

_REF_IMAGE_NAME = 'master2010.fits'

def getReference(image_in, header_in = None, reference_fits_file = None, ref_maskfile=None):
    """Return the reference image aligned with the input image.

    getReference accepts a numpy array (masked or not) or a header with WCS information and optionally a master reference fits file and return
    a reprojected reference image array for the same portion of the sky.
    Return reference_image"""

    #It's assumed that the default master has WCS info
    if reference_fits_file is None:
        refHasWCS = True
    else:
        refHasWCS = _checkIfWCS(fits.getheader(reference_fits_file))

    #Check if there is a header available
    if (header_in is not None) and _checkIfWCS(header_in) and refHasWCS:
        #reproject with reproject here...

        if reference_fits_file is None:
            reference_fits_file = pkg_resources.resource_filename('toros.resources', _REF_IMAGE_NAME)

        refhdu = fits.open(reference_fits_file)[0]
        if ref_maskfile is None:
            ref_mask = refhdu.data < 0
            refhdu_mask = fits.PrimaryHDU(ref_mask.astype('float'), header=refhdu.header)
        else:
            refhdu_mask = fits.open(ref_maskfile)[0]
            ref_mask = refhdu_mask.data
        ref_reproj_data, __ = reproject_interp(refhdu, header_in)
        ref_reproj_mask, __ = reproject_interp(refhdu_mask, header_in)
        gold_master = np.ma.array(data=ref_reproj_data, mask=ref_reproj_mask)
    else:
        gold_master = _no_wcs_available(image_in, reference_fits_file)

    return gold_master


def _checkIfWCS(header):
    my_wcs = wcs.WCS(header)
    return True if my_wcs.wcs.ctype[0] else False


def _no_wcs_available(image_in, reference_fits_file):

    if not isinstance(image_in, np.ma.MaskedArray):
        image_in_ma = np.ma.array(image_in)
    else:
        image_in_ma = image_in
    test_srcs = findSources(image_in_ma)[:50]

    if reference_fits_file is None:
        ref_image = fits.getdata(pkg_resources.resource_filename('toros.resources', _REF_IMAGE_NAME))
        ref_mask  = ref_image < 0
        ref_image = np.ma.array(ref_image, mask=ref_mask)
        ref_sources = None
    else:
        ref_image = fits.getdata(reference_fits_file)
        ref_mask  = ref_image < 0
        ref_image = np.ma.array(ref_image, mask=ref_mask)
        ref_sources = findSources(ref_image)[:70]

    M = registration.findAffineTransform(test_srcs, ref_srcs = ref_sources)

    #SciPy Affine transformation transform a (row,col) pixel according to pT+s where p is in the _output_ image,
    #T is the rotation and s the translation offset, so some mathematics is required to put it into a suitable form
    #In particular, affine_transform() requires the inverse transformation that registration returns but for (row, col) instead of (x,y)
    def inverseTransform(M):
        M_rot_inv = np.linalg.inv(M[:2,:2])
        M_offset_inv = -M_rot_inv.dot(M[:2,2])
        Minv = np.zeros(M.shape)
        Minv[:2,:2] = M_rot_inv
        Minv[:2,2] = M_offset_inv
        if M.shape == (3,3): Minv[2,2] = 1
        return Minv

    Minv = inverseTransform(M)
    #P will transform from (x,y) to (row, col)=(y,x)
    P = np.array([[0,1],[1,0]])
    Mrcinv_rot = P.dot(Minv[:2,:2]).dot(P)
    Mrcinv_offset = P.dot(Minv[:2,2])

    #M_rot = M[:2,:2]
    #M_offset = M[:2,2]
    #M_rot_inv = np.linalg.inv(M_rot)
    #Mrcinv_rot = P.dot(M_rot_inv).dot(P)
    #Mrcinv_offset = -P.dot(M_rot_inv).dot(M_offset)

    gold_master = ndimage.interpolation.affine_transform(ref_image, Mrcinv_rot, offset=Mrcinv_offset, output_shape=image_in.shape)
    #gold_master_mask = ndimage.interpolation.affine_transform(ref_img.mask, M_rot, offset=M_offset, output_shape=image_in.shape)
    gold_master = np.ma.array(gold_master, mask=gold_master < 0)

    return gold_master


def makeSourcesMask(dataImg, noiseLvl = 3.):

    m, s = bkgNoiseSigma(dataImg, noiseLvl = noiseLvl)
    srcsMask = dataImg > m + noiseLvl*s
    if isinstance(srcsMask, np.ma.MaskedArray):
        srcsMask = srcsMask.astype(bool)
        srcsMask.set_fill_value(False)
        srcsMask = srcsMask.filled()
    return srcsMask


def bkgNoiseSigma(dataImg, noiseLvl = 3.0):
    """Return background mean and std. dev. of sky background.

    Calculate the background (sky) mean value and a measure of its standard deviation.
    Background is considered anything below 'noiseLvl' sigmas.
    goodPixMask is a mask containing the good pixels that should be considered.
    Return mean, std_dev
    """
    m = dataImg.mean()
    s = dataImg.std()

    prevSgm = 2*s #This will make the first while condition true
    tol = 1E-2
    while abs(prevSgm - s)/s > tol:
        prevSgm = s
        bkgMask = np.logical_and(dataImg < m + noiseLvl*s, dataImg > m - noiseLvl*s)
        if isinstance(bkgMask, np.ma.MaskedArray):
            bkgMask = bkgMask.astype(bool)
            bkgMask.set_fill_value(False)
            bkgMask = bkgMask.filled()
        m, s = dataImg[bkgMask].mean(), dataImg[bkgMask].std()

    return m, s


def findSources(image):
    """Return sources sorted by brightness.
    """

    img1 = image.copy()
    src_mask = makeSourcesMask(img1)
    img1[~src_mask] = img1[src_mask].min()
    img1 = exposure.rescale_intensity(img1)
    img1[~src_mask] = 0.
    img1.set_fill_value(0.)

    def obj_params_with_offset(img, labels, aslice, label_idx):
        y_offset = aslice[0].start
        x_offset = aslice[1].start
        thumb = img[aslice]
        lb = labels[aslice]
        yc, xc = ndimage.center_of_mass(thumb, labels=lb, index=label_idx)
        br = thumb[lb == label_idx].sum() #the intensity of the source
        return [br, xc + x_offset, yc + y_offset]

    srcs_labels, num_srcs = ndimage.label(img1)

    if num_srcs < 10:
        print("WARNING: Only %d sources found." % (num_srcs))

    #Eliminate here all 1 pixel sources
    all_objects = [[ind + 1, aslice] for ind, aslice in enumerate(ndimage.find_objects(srcs_labels))
                                                if srcs_labels[aslice].shape != (1,1)]
    lum = np.array([obj_params_with_offset(img1, srcs_labels, aslice, lab_idx)
                for lab_idx, aslice in all_objects])

    lum = lum[lum[:,0].argsort()[::-1]]  #sort by brightness highest to smallest

    return lum[:,1:]



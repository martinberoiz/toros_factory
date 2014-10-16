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
import cv2
from astropy.io import fits
import os
#import datetime as d
import ransac
import math
import pkg_resources

__version__ = '0.3'

_REF_IMAGE_NAME = 'cstar_ref_image.fits'

def getReference(hdulist):
    """Accept an HDU list object and return an array with the same piece of sky in the reference image."""

    ref_path = pkg_resources.resource_filename('toros', _REF_IMAGE_NAME)
    ref_image = fits.getdata(ref_path).astype('float32')
    ref_mask = fits.getdata(ref_path, 1).astype('float32')
    
    image = hdulist[0].data
    if isinstance(image, np.ma.MaskedArray):
        image_mask = image.mask
    else:
        image_mask = None
    
    refU8 = convertToU8(ref_image, ref_mask.astype('bool'))
    imgU8 = convertToU8(image.data, ~image_mask)
    
    M, pointsMask = blindAlign(refU8, imgU8)
    #alignImages(refImgU8, imgU8)
    
    registered_ref = cv2.warpAffine(ref_image, M, ref_image.shape, borderMode = cv2.BORDER_CONSTANT, borderValue = 0.)
    registered_ref_mask = cv2.warpAffine(ref_mask, M, ref_mask.shape, borderMode = cv2.BORDER_CONSTANT, borderValue = 0.)
    
    reg_ref = np.ma.array(registered_ref, mask = ~(registered_ref_mask.astype('bool')))
        
    return reg_ref
    
    
def convertToU8(image, goodPixMask):
    
    reduced_im = makeSourcesImage(image, mask=goodPixMask)
    xmin = reduced_im.min()
    xmax = reduced_im.max()
    reduced_im = (reduced_im - xmin)*(255.0/(xmax - xmin))
    reduced_im = reduced_im.astype('uint8', copy=False)
    
    return reduced_im
    
    
def makeSourcesImage(dataImg, noiseLvl = 3., mask = None):
    """Return an image where the background and the given mask, is set to zero.
        
    Background is anything below 'noiseLvl' sigmas.
    """
    
    m, s = bkgNoiseSigma(dataImg, noiseLvl = noiseLvl, goodPixMask = mask)
    
    srcsImg = dataImg.copy()
    if mask is not None:
        srcsMask = np.logical_and(mask, dataImg > m + noiseLvl*s)
    else:
        srcsMask = dataImg > m + noiseLvl*s
    srcsImg[~srcsMask] = noiseLvl*s
    return srcsImg
    

def bkgNoiseSigma(dataImg, noiseLvl = 3.0, goodPixMask = None):
    """Return background mean and std. dev. of sky background.
        
    Calculate the background (sky) mean value and a measure of its standard deviation.
    Background is considered anything below 'noiseLvl' sigmas.
    goodPixMask is a mask containing the good pixels that should be considered.
    Return mean, std_dev
    """
    m = dataImg[goodPixMask].mean()
    s = dataImg[goodPixMask].std()
    
    prevSgm = 2*s #This will make the first while condition true
    tol = 1E-2
    while abs(prevSgm - s)/s > tol:
        prevSgm = s
        bkgMask = np.logical_and(dataImg < m + noiseLvl*s, dataImg > m - noiseLvl*s)
        if goodPixMask is not None: bkgMask = np.logical_and(bkgMask, goodPixMask)
        m = dataImg[bkgMask].mean()
        s = dataImg[bkgMask].std()
    
    return m, s


def findSources(img):
    """Return sources sorted by brightness.
        
    img should be an image with background set to zero.
    Anything above zero is considered part of a source.
    """

    img1 = img.copy()
    cnt, hier1 = cv2.findContours(img1, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
    img1 = 0

    #This could fail if the enclosing rectangle for one contour intersects
    #  another contour
    #  Improve: set zero outside contour (use drawContours to mask elements outside)
    centroids = []
    brightness = []
    for aCont in cnt:
        #xmin, ymin = map(min, np.squeeze(aCont).T)
        #xmax, ymax = map(max, np.squeeze(aCont).T)

        xy = zip(*[tuple(cc[0]) for cc in aCont])
        xmin = min(xy[0])
        ymin = min(xy[1])
        imext = img[ymin: max(xy[1])+1, xmin:max(xy[0])+1].astype('float32')
        #imext = img[ymin: ymax+1, xmin:xmax+1].astype('float32')

        m01 = sum([(rowind + ymin)*sum(row) for (rowind,row) in enumerate(imext)])
        m10 = sum([(colind + xmin)*sum(col) for (colind,col) in enumerate(imext.T)])
        m00 = np.sum(imext)
        centroids.append([m10/m00, m01/m00])
        brightness.append(m00)

    if len(centroids) < 10:
        print("Warning: Only %d sources found." % (len(centroids)))

    centroids = np.array(centroids)
    brightness = np.array(brightness)
    bestStars = centroids[np.argsort(-brightness)]

    return bestStars


def findIsolatedSources(img, minRad = 50.):
    """Find sources that are at least minRad pixels from any other.
        
    minRad: the circular radius of the exclusion zone.
    """
    
    centroids = findSources(img)
    
    #This loop makes N^2 operations that can be done in NLgN with a proper data structure (a Kd tree)
    loneStars = []
    for ct in centroids:
        #find neighbor stars
        for ct1 in centroids:
            if np.array_equal(ct1,ct):
                continue
            if np.linalg.norm(ct1 - ct) < minRad:
                break
        
        #if no neighbor found, add centroids to loneStars
        else:
            loneStars.append(ct)
    
    return loneStars


def alignImages(refImgU8, imgU8, approxM = np.array([[1,0,0],[0,1,0]]), refStars = [], minRad = 100.):
    """Find the transformation matrix that takes imgU8 into the reference image refImgU8.
        
    An approximate transformation matrix approxM should be provided to find possible 
    candidates for matching stars.
    refStars is a set of reference stars in the reference image to find companions in imgU8.
    If not provided, they will be found using findSources.
    """

    if len(refStars) == 0:
        refStars = findSources(refStars)
    refStars = refStars[:50]
    
    centroids2 = findSources(imgU8)[:70]
    if len(centroids2) < 4 or len(refStars) < 4:
        raise ValueError("Insufficient sources to estimate a transformation.")

    #rotate centroids in second image
    rotCentroids2 = cv2.transform(np.array([centroids2]), approxM)[0]
    src_pts = []
    dst_pts = []
    for ct in refStars:
        #find corresponding stars
        for (i, ct1) in enumerate(rotCentroids2):
            if np.linalg.norm(ct1 - ct) < minRad:
                src_pts.append(centroids2[i])
                dst_pts.append(ct)

    if len(dst_pts) < 4:
        raise ValueError("Not enough matches found.")

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    try:
        M, mask = findAffineTransform(src_pts, dst_pts)
    except:
        M, mask = blindAlign(refImgU8, imgU8, refStars, minRad)

    return M


def blindAlign(refImgU8, imgU8, refStars = [], minRad = 100.):
    """Brute force try to approximate transformations to register 2 images.
        
    Align two images by trying 36 different rotations as
    the starting approximate transformation to match star pairs.
    It will then pick the one that gave the best fit between the images. 
    This best transformation includes rotation and translation.
    """
    
    if len(refStars) == 0:
        refStars = findSources(refImgU8)
    refStars = refStars[:50]

    centroids2 = findSources(imgU8)[:70]
    if len(centroids2) < 4 or len(refStars) < 4:
        raise ValueError("Insufficient sources to estimate a transform.")

    bestM = np.zeros(shape=(2,3), dtype='float32')
    bestMask = []
    bestModel = 0
    for alpha in np.arange(0.,360.,10.):
        approxM = cv2.getRotationMatrix2D((imgU8.shape[0]//2, imgU8.shape[1]//2), alpha, 1.0)
        
        rotCentroids2 = cv2.transform(np.array([centroids2]), approxM)[0]
        src_pts = []
        dst_pts = []
        for ct in refStars:
            #find corresponding stars
            for (i, ct1) in enumerate(rotCentroids2):
                if np.linalg.norm(ct1 - ct) < minRad:
                    src_pts.append(centroids2[i])
                    dst_pts.append(ct)
        
        assert(len(src_pts) == len(dst_pts))
        if len(dst_pts) < 4:
            continue
        
        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        M, mask = findAffineTransform(src_pts, dst_pts)
        if mask.sum() > bestModel:
            bestM = M
            bestMask = mask

    return bestM, bestMask


def findAffineTransform(src_pts, dst_pts):
    """Find a rotation + translation transformation between pairs of points using a RANSAC algorithm."""
    assert src_pts.shape == dst_pts.shape
    data = np.hstack((src_pts, dst_pts))  #a set of observed data points
    transfModel = LinearTransformModel()  #a model that can be fitted to data points

    minNDataPoints = 2     #the minimum number of data values required to fit the model
    maxPixDist     = 5.    #a threshold value for determining when a data point fits a model
    minNInliers    = 10    #the number of close data values required to assert that a model fits well to data

    #w is the probability of choosing an inlier each time a single point is selected
    #w = number of inliers in data / number of points in data
    w = 45./len(src_pts)
    
    #p is the probability that the RANSAC algorithm in some iteration selects only inliers
    #from the input data set when it chooses the n points from which the model parameters are estimated
    p = 0.99
    
    #the maximum number of iterations allowed in the algorithm
    maxNIterations = 2 * int(math.log(1. - p) / math.log(1. - w ** minNDataPoints))
    if maxNIterations > 1000: maxNIterations = 1000

    mask = np.zeros(len(src_pts), dtype = 'bool')
    try:
        M, maskDict = ransac.ransac(data,
                                    transfModel,
                                    minNDataPoints,
                                    maxNIterations * 2,
                                    maxPixDist,
                                    minNInliers,
                                    debug=False,
                                    return_all=True)
    except:
        M = None

    else:
        mask[maskDict['inliers']] = True
        #improve M with all inliers
        M = transfModel.fit(data[mask])
    
    return M, mask


class LinearTransformModel:
    def __init__(self): pass
    def fit(self, data):
        A = []
        y = []
        for pt in data:
            x1 = pt[0]
            x2 = pt[1]
            A.append([x1, x2, 1, 0])
            A.append([x2, -x1, 0, 1])
            y.append(pt[2])
            y.append(pt[3])
        
        A = np.matrix(A)
        y = np.matrix(y)
        
        sol, resid, rank, sv = np.linalg.lstsq(A,y.T)
        c = sol.item(0)
        s = sol.item(1)
        t1 = sol.item(2)
        t2 = sol.item(3)
        approxM = np.array([[c, s, t1],[-s, c, t2]])
        return approxM
    
    def get_error(self, data, approxM):
        error = []
        for pt in data:
            spt = pt[:2]
            dpt = pt[2:]
            dpt_fit = cv2.transform(np.array([[spt]]), approxM)[0]
            error.append(np.linalg.norm(dpt - dpt_fit))
        return np.array(error)

"""@package photometry
    
    Photometry module
    -----------------
    
    A collection of tools to perform photometry
    for the Transient Optical Robotic Observatory of the South (TOROS).
    
    Martin Beroiz - 2014
    
    email: <martinberoiz@phys.utb.edu>
    
    University of Texas at San Antonio
"""

from astropy.time import Time
import datetime as d
import numpy as np
import math
from photutils import CircularAperture, aperture_photometry


def photoCalibrate(hdulist, catalogFileName):
    """Calibrate photometrically the FITS file by calculating and adding the MAGZERO keyword to the header."""
    img = hdulist[0].data
    img_mask = hdulist[1].data.astype('bool')
    
    #get ra, dec from catalog file
    data = ascii.read(catalogFileName)
    ra_dec = np.array([[aline[0], aline[1]] for aline in data])
    mags = np.array([aline[2] for aline in data])
    
    head = hdulist[0].header
    theWCS = wcs.WCS(head)
    
    apR = 3.5
    
    mm, ss = bkgNoiseSigma(img, noiseLvl = 3.0, goodPixMask = img_mask)
    fluxes, gmask = getFluxesForRADec(ra_dec, img, img_mask, theWCS, bkg_level=mm, apRadius=apR)
    mags = mags[gmask]
    
    #Remove flux too small
    minflux = ss * math.pi * apR**2
    mags    = mags[fluxes > 2*minflux]
    fluxes  = fluxes[fluxes > 2*minflux]
    
    best_intercept = sigmaClipIntercepFit(mags, np.log10(fluxes), nsig = 3, a = -0.4)
    
    head['magzero'] = (2.5*best_intercept, 'Added by Martin Beroiz.')
    
    return


def getFluxesForRADec(radecList, image, goodMask, the_wcs, bkg_level = 0, apRadius = 3.5):
    """Get the aperture flux at the (RA, Dec) position for the radecList array"""
    pix_ctr = []
    good_pix_mask = []
    for aPoint in radecList:
        try:
            pixPoint = the_wcs.all_world2pix([aPoint],0,tolerance=1E-3)[0]
            x,y = pixPoint.astype('int')
            h, w = image.shape
            if(x > w or x < 0 or y < 0 or y > h):
                good_pix_mask.append(False)
            elif(goodMask[y,x] == False):
                good_pix_mask.append(False)
            else:
                pix_ctr.append(pixPoint)
                good_pix_mask.append(True)
        except:
            good_pix_mask.append(False)
    
    pix_ctr = np.array(pix_ctr)
    good_pix_mask = np.array(good_pix_mask)
    
    apertures = CircularAperture(pix_ctr, apRadius)
    fluxes = aperture_photometry(image, apertures)
    fluxes -= bkg_level * np.pi * apRadius**2
    
    return fluxes, good_pix_mask


def sigmaClipIntercepFit(x, y, nsig = 3, a = -2.5):
    """Return the best fit to the y-intercept of the x,y data assuming a is fixed, ignoring outliers.
        
    This is a least square problem to solve the equation y=ax+b when a is fixed (not fitted)
    to some value. It will also clip out recursively data points that are away from nsig std dev
    from the current iteration.
    It stops when it doesn't find any more outliers away from nsig std. deviations.
    Return b
    """
    b = -sum((a*x - y))/len(x)
    sig = math.sqrt(sum((x*a + b - y)**2)/len(x))
    gpts_mask = np.abs(x*a + b - y) < nsig*sig
    
    gx = x.copy()
    gy = y.copy()
    MAX_ITER = 1000
    i = 0
    while(sum(gpts_mask) != len(gx) and i < MAX_ITER):
        gx = gx[gpts_mask]
        gy = gy[gpts_mask]
        b = -sum((a*gx - gy))/len(gx)
        sig = math.sqrt(sum((gx*a + b - gy)**2)/len(gx))
        gpts_mask = np.abs(gx*a + b - gy) < nsig*sig
        i += 1
    return b


def bkgNoiseSigma(dataImg, noiseLvl = 3.0, goodPixMask = None):
    """Return background mean and std-dev of sky background.
        
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
        if goodPixMask != None: bkgMask = np.logical_and(bkgMask, goodPixMask)
        m = dataImg[bkgMask].mean()
        s = dataImg[bkgMask].std()
    
    return (m, s)


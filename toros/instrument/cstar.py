"""@package CSTAR 
    
    CSTAR Specific tools
    --------------------
    
    A collection of tools to handle files in the CSTAR data set.
    
    Martin Beroiz - 2014
    
    email: <martinberoiz@phys.utb.edu>
    
    University of Texas at San Antonio
"""

from astropy.io import fits
import numpy as np
import numpy.ma as ma

def reduce(image_file_name):
    
    image = fits.getdata(image_file_name).astype('float32')
    
    bleedMask = maskBleeding(image, badPixMask = image < 0)
    badPixMask = bleedMask + (image < 0) #combine bad pixels and bled pixels mask
    new_image = ma.array(image, mask=badPixMask)
    
    return new_image
        
    
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
        if goodPixMask != None: bkgMask = np.logical_and(bkgMask, goodPixMask)
        m = dataImg[bkgMask].mean()
        s = dataImg[bkgMask].std()
    
    return m, s


def cleanBleeding(img, sigmas = 1.):
    """Fill bled columns with minimum value of image.
        
    Clean vertical bleeding by filling the whole column where bleeding occurs
    with the minimum value of the image.
    'sigmas' is the threshold (in std dev units) to mark a colum as containing bleeding.
    """
    colSums = np.array([np.sum(img[:,i]) for i in range(0,img.shape[1])])
    colSums = colSums - colSums.min()
    blIndx = colSums > sigmas * colSums.std()
    img[:,blIndx] = img.min()


def maskBleeding(scidata, badPixMask = None):
    """Return a mask for the bad and bled columns (True on bleeding).
        
    Identify the columns where bleeding appears (using excess of counts method)
    and return a mask that covers (True on bleeding) the bad columns.
    When supplied a pre-existing badPixMask, use it and
    return the union of both masks.
    """
    
    if badPixMask is None:
        goodPixMask = None
        colSums = scidata.sum(axis=0)

        mm = colSums.mean()
        ss = colSums.std()
        
        colSums = colSums - mm
        bleedIndx = colSums > 2.0 * ss
    
    else:
        goodPixMask = ~badPixMask
        colSums = np.array([aCol[aMaskCol].sum() for aCol, aMaskCol in zip(scidata.T, goodPixMask.T)])

        #colNorms[emptyColsMask] = 1.
        colNorms = goodPixMask.sum(axis=0)
        colNorms[colNorms == 0] = 1
        colSums = colSums / colNorms
        
        nonemptyColsMask = (goodPixMask.sum(axis=0) > 0)
        
        mm = colSums[nonemptyColsMask].mean()
        ss = colSums[nonemptyColsMask].std()
        
        colSums[nonemptyColsMask] = colSums[nonemptyColsMask] - mm
        bleedIndx = nonemptyColsMask * (colSums > 2.0 * ss)

    #Widen here the bleeding colums to bborder pixels to each side.
    idx = np.where(bleedIndx == True)[0]
    bborder = 3
    for i in idx:
        for j in range(-bborder, bborder + 1, 1):
            if i + j >= 0 and i + j < bleedIndx.shape[0]:
                bleedIndx[i + j] = True
    
    bleedMask = [bleedIndx] * badPixMask.shape[1]

    return bleedMask


def processCSTARImage(scidata):
    """Helper function that process a CSTAR image to be used in the coadd method.
    
    Create a mask for bled columns, bad pixels, and create a uint8 numpy
    array for the image. Return the uint8 image and the mask.
    """
    
    badPixMask = np.array(scidata < 0.)
    bleedMask = maskBleeding(scidata, badPixMask)
    
    #Add bleeding colums to the bad pixels mask
    badPixMask = np.logical_or(bleedMask, badPixMask)
    goodPixMask = np.logical_not(badPixMask)
    
    #Set bad pixels to zero
    scidata[badPixMask] = 0.
    
    reduced_im = makeSourcesImage(scidata, mask=goodPixMask)
    xmin = reduced_im.min()
    xmax = reduced_im.max()
    reduced_im = (reduced_im - xmin)*(255.0/(xmax - xmin))
    reduced_im = reduced_im.astype('uint8', copy=False)
    
    return reduced_im, badPixMask


def correctDate(hdulist):
    """Correct the CSTAR date format (a string) for the DATE-OBS header keyword and replace with an ISOT format."""
    head = hdulist[0].header
    
    dateStr = head['DATE-OBS'].strip()
    timeStr = head['TIME'].strip()
    t = d.datetime.strptime(" ". join([dateStr, timeStr]), '%Y %b %d %H:%M:%S.%f')
    #change to UTC by subtracting 5 hours:
    t -= d.timedelta(hours = 5)
    
    #Create an astropy time to correct using the polynomials
    thead = Time(t.strftime('%Y-%m-%d %H:%M:%S.%f'), format='iso', scale='utc')
    
    #2455391.5 corresponds to the UTC date 2010 Jul 14 00:00:00
    dt = thead.jd - 2455391.5
    
    #corr is the correction in seconds
    if dt < 0:
        corr = 1.70607*dt + 4.39088e-2*dt**2 + 7.60477e-4*dt**3 + 4.36212e-6*dt**4
    else:
        corr = 1.17962*dt - 2.74530e-2*dt**2 + 5.65247e-4*dt**3 - 2.98125e-6*dt**4
    
    #change corr from seconds to days
    corr /= 86400
    
    #tcorr is the corrected time by adding corr in days
    tcorr = Time(thead.jd + corr, scale='utc', format='jd')
    
    head['date-obs'] = (tcorr.isot, 'Julian UTC datetime with correction')
    del head['time']

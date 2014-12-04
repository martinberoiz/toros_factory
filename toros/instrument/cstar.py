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
    
    image = np.ma.array(fits.getdata(image_file_name))
    image[image < 0] = np.ma.masked
    maskSaturation(image)
    return image
        
    
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


def maskSaturation(image_in, sat_level = None):
    
    if not isinstance(image_in, np.ma.MaskedArray):
        image = np.ma.array(image_in)
    else:
        image = image_in
    
    def findSaturationLevel(image):
        old_fill_value = image.fill_value
        image.set_fill_value(0.)
        colSums = (image.filled()).sum(axis=0)
        colNorms = (~image.mask).sum(axis=0)
        colSums[colNorms > 0.] /= colNorms[colNorms > 0.]
        mm, ss = colSums[colSums != 0].mean(), colSums[colSums != 0].std()
        sat_level = min(map(max, (image.T)[colSums > mm + 3.*ss]))
        image.set_fill_value(old_fill_value)
        return sat_level
        
    if sat_level is None:
        sat_level = 0.9 * findSaturationLevel(image)
    
    image[image > sat_level] = np.ma.masked
    return image
    

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

"""@package subtraction
    
    Subtraction module
    -----------------
    
    A collection of tools to perform optimal image differencing
    for the Transient Optical Robotic Observatory of the South (TOROS).
    
    ### Usage example (from python):
    
        >>> from toros import subtraction
        >>> conv_image, optimalKernel, background = ois.getOptimalKernelAndBkg(image, referenceImage)
    
    (conv_image is the least square optimal approximation to image)
    
    See getOptimalKernelAndBkg docstring for more options.
    
    ### Command line arguments:
    * -h, --help: Prints this help and exits.
    * -v, --version: Prints version information and exits.
        
    Martin Beroiz - 2014
    
    email: <martinberoiz@phys.utb.edu>
    
    University of Texas at San Antonio
"""

import numpy as np
import cv2
import math
from scipy import signal

def __gauss(shape = (10,10), center=None, sx=2, sy=2):
    h, w = shape
    if center is None: center = ((h-1)/2., (w-1)/2.)
    x0,y0 = center
    x,y = np.meshgrid(range(w),range(h))
    norm = math.sqrt(2*math.pi*(sx**2)*(sy**2))
    return np.exp(-0.5*((x-x0)**2/sx**2 + (y-y0)**2/sy**2))/norm

def __CVectorsForGaussBasis(refImage, kernelShape, gaussList, modBkgDeg, badPixMask):
    #degMod is the degree of the modulating polyomial
    kh, kw = kernelShape
    
    v,u = np.mgrid[:kh,:kw]
    C = []
    for aGauss in gaussList:
        if 'modPolyDeg' in aGauss: degMod = aGauss['modPolyDeg']
        else: degMod = 2
        
        allUs = [pow(u,i) for i in range(degMod + 1)]
        allVs = [pow(v,i) for i in range(degMod + 1)]
        
        if 'center' in aGauss: center = aGauss['center']
        else: center = None
        gaussK = __gauss(shape=(kh,kw), center=center, sx=aGauss['sx'], sy=aGauss['sy'])
        
        if badPixMask is not None:
            newC = [np.ma.array(signal.convolve2d(refImage, gaussK * aU * aV, mode='same'), \
                                mask = badPixMask) \
                    for i, aU in enumerate(allUs) \
                    for aV in allVs[:degMod+1-i]]
        else:
            newC = [signal.convolve2d(refImage, gaussK * aU * aV, mode='same') \
                    for i, aU in enumerate(allUs) for aV in allVs[:degMod+1-i]]
        
        C.extend(newC)
    return C

def __CVectorsForDeltaBasis(refImage, kernelShape, badPixMask):
    kh, kw = kernelShape
    h, w = refImage.shape

    C = []
    for i in range(kh):
        for j in range(kw):
            Cij = np.zeros(refImage.shape)
            Cij[max(0,i-kh//2):min(h,h-kh//2+i), max(0,j-kw//2):min(w,w-kw//2+j)] = \
                refImage[max(0,kh//2-i):min(h,h-i+kh//2), max(0,kw//2-j):min(w,w-j+kw//2)]
            if badPixMask is not None:
                C.extend([np.ma.array(Cij, mask=badPixMask)])
            else:
                C.extend([Cij])

    # This is more pythonic but could be slower. I didn't test speed.
    #canonBasis = np.identity(kw*kh).reshape(kh*kw,kh,kw)
    #C.extend([signal.convolve2d(refImage, kij, mode='same')
    #                 for kij in canonBasis])
    #    canonBasis = None

    return C

def __getCVectors(refImage, kernelShape, gaussList, modBkgDeg = 2, badPixMask = None):

    C = []
    if gaussList is not None:
        C.extend(__CVectorsForGaussBasis(refImage, kernelShape, gaussList, modBkgDeg, badPixMask))
    else:
        C.extend(__CVectorsForDeltaBasis(refImage, kernelShape, badPixMask))
    
    #finally add here the background variation coefficients:
    h, w = refImage.shape
    y, x = np.mgrid[:h,:w]
    allXs = [pow(x,i) for i in range(modBkgDeg + 1)]
    allYs = [pow(y,i) for i in range(modBkgDeg + 1)]

    if badPixMask is not None:
        newC = [np.ma.array(anX * aY, mask = badPixMask) \
                for i, anX in enumerate(allXs) for aY in allYs[:modBkgDeg+1-i]]
    else:
        newC = [anX * aY for i, anX in enumerate(allXs) for aY in allYs[:modBkgDeg+1-i]]

    C.extend(newC)
    return C

def __coeffsToKernel(coeffs, gaussList, kernelShape = (10,10)):
    kh, kw = kernelShape
    if gaussList is None:
        kernel = coeffs[:kw*kh].reshape(kh,kw)
    else:
        v,u = np.mgrid[:kh,:kw]
        kernel = np.zeros((kh,kw))
        for aGauss in gaussList:
            if 'modPolyDeg' in aGauss: degMod = aGauss['modPolyDeg']
            else: degMod = 2
        
            allUs = [pow(u,i) for i in range(degMod + 1)]
            allVs = [pow(v,i) for i in range(degMod + 1)]
        
            if 'center' in aGauss: center = aGauss['center']
            else: center = None
            gaussK = __gauss(shape=kernelShape, center=center, sx=aGauss['sx'], sy=aGauss['sy'])
        
            ind = 0
            for i, aU in enumerate(allUs):
                for aV in allVs[:degMod+1-i]:
                    kernel += coeffs[ind] * aU * aV
                    ind += 1
            kernel *= gaussK
    return kernel

def __coeffsToBackground(shape, coeffs, bkgDeg = None):
    if bkgDeg is None: bkgDeg = int(-1.5 + 0.5*math.sqrt(9 + 8*(len(coeffs) - 1)))

    h, w = shape
    y, x = np.mgrid[:h,:w]
    allXs = [pow(x,i) for i in range(bkgDeg + 1)]
    allYs = [pow(y,i) for i in range(bkgDeg + 1)]

    mybkg = np.zeros(shape)

    ind = 0
    for i, anX in enumerate(allXs):
        for aY in allYs[:bkgDeg+1-i]:
            mybkg += coeffs[ind] * anX * aY
            ind += 1

    return mybkg

def getOptimalKernelAndBkg(image, refImage, gaussList = None, bkgDegree = 3, kernelShape = (11,11)):
    """Do Optimal Image Subtraction and return optimal kernel and background.
        
    This is an implementation of the Optimal Image Subtraction algorith of Alard&Lupton.
    It returns the best kernel and background fit that match the two images.
    
    gaussList is a list of dictionaries containing data of the gaussians used in the decomposition
    of the kernel. Dictionary keywords are:
    center, sx, sy, modPolyDeg
    If gaussList is None (default value), the OIS will try to optimize the value of each pixel in
    the kernel.
    
    bkgDegree is the degree of the polynomial to fit the background.
    
    kernelShape is the shape of the kernel to use.
    
    Return (optimal_image, kernel, background)
    """
    
    kh, kw = kernelShape
    if kw % 2 != 1 or kh % 2 != 1:
        print("This can only work with kernels of odd sizes.")
        return None, None, None
    
    badPixMask = None
    if isinstance(refImage, np.ma.MaskedArray):
        refMask = cv2.dilate(refImage.mask.astype('uint8'), np.ones(kernelShape))
        badPixMask = refMask.astype('bool')
        if isinstance(image, np.ma.MaskedArray):
             badPixMask += image.mask
    elif isinstance(image, np.ma.MaskedArray):
        badPixMask = image.mask

    C = __getCVectors(refImage, kernelShape, gaussList, bkgDegree, badPixMask)
    m = np.array([[(ci*cj).sum() for ci in C] for cj in C])
    b = np.array([(image*ci).sum() for ci in C])
    coeffs = np.linalg.solve(m,b)
    
    #nKCoeffs is the number of coefficients related to the kernel fit, not the background fit
    if gaussList is None:
        nKCoeffs = kh*kw
    else:
        nKCoeffs = 0
        for aGauss in gaussList:
            if 'modPolyDeg' in aGauss: degMod = aGauss['modPolyDeg']
            else: degMod = 2
            nKCoeffs += (degMod + 1)*(degMod + 2)//2
    
    kernel = __coeffsToKernel(coeffs[:nKCoeffs], gaussList, kernelShape)
    background = __coeffsToBackground(image.shape, coeffs[nKCoeffs:])
    if isinstance(refImage, np.ma.MaskedArray) or isinstance(image, np.ma.MaskedArray):
        background = np.ma.array(background, mask=badPixMask)
    optimal_image = signal.convolve2d(refImage, kernel, mode='same') + background

    return optimal_image, kernel, background



import toros
import numpy as np


def retrieveReferenceImage():
    import urllib, cStringIO
    from PIL import Image

    # http://homepages.cae.wisc.edu/~ece533/images/cameraman.tif
    # http://links.uwaterloo.ca/Repository/TIF/camera.tif
    f = cStringIO.StringIO(urllib.urlopen('http://homepages.cae.wisc.edu/~ece533/images/cameraman.tif').read())
    ref_img = np.array(Image.open(f), dtype='float32')
    return ref_img
    

def degradeReference(ref_img):
    from scipy import signal
    import math
    
    #Set some arbitrary kernel to convolve with
    def gauss(shape = (11,11), center=None, sx=2, sy=2):
        h, w = shape
        if center is None: center = ((h-1)/2., (w-1)/2.)
        x0,y0 = center
        x,y = np.meshgrid(range(w),range(h))
        norm = math.sqrt(2*math.pi*(sx**2)*(sy**2))
        return np.exp(-0.5*((x-x0)**2/sx**2 + (y-y0)**2/sy**2))/norm

    def createKernel(coeffs, gaussList, kernelShape = (10,10)):
        kh, kw = kernelShape

        v,u = np.mgrid[:kh,:kw]
    
        mykernel = np.zeros((kh,kw))
        for aGauss in gaussList:
            if 'modPolyDeg' in aGauss: degMod = aGauss['modPolyDeg']
            else: degMod = 2
        
            allUs = [pow(u,i) for i in range(degMod + 1)]
            allVs = [pow(v,i) for i in range(degMod + 1)]
        
            if 'center' in aGauss: center = aGauss['center']
            else: center = None
            gaussK = gauss(shape=kernelShape, center=center, sx=aGauss['sx'], sy=aGauss['sy'])
                
            ind = 0
            for i, aU in enumerate(allUs):
                for aV in allVs[:degMod+1-i]:
                    mykernel += coeffs[ind] * aU * aV
                    ind += 1
            mykernel *= gaussK
            
        #mykernel /= mykernel.sum()

        return mykernel


    #myGaussList = [{'sx':2., 'sy':2., 'modPolyDeg':3},{'sx':1., 'sy':3.},{'sx':3., 'sy':1.}]
    myGaussList = [{'sx':2., 'sy':2., 'modPolyDeg':3}]
    #myKCoeffs = np.random.rand(10)*90 + 10
    myKCoeffs = np.array([0., -7.3, 0., 0., 0., 2., 0., 1.5, 0., 0.])

    myKernel = createKernel(myKCoeffs, myGaussList, kernelShape=(11,11))
    #myKernel = gauss()
    kh, kw = myKernel.shape

    image = signal.convolve2d(ref_img, myKernel, mode='same')

    #Add a varying background:
    bkgDeg = 2

    h, w = ref_img.shape
    y, x = np.mgrid[:h,:w]
    allXs = [pow(x,i) for i in range(bkgDeg + 1)]
    allYs = [pow(y,i) for i in range(bkgDeg + 1)]

    mybkg = np.zeros(ref_img.shape)
    myBkgCoeffs = np.random.rand(6) * 1E-3

    ind = 0
    for i, anX in enumerate(allXs):
        for aY in allYs[:bkgDeg+1-i]:
            mybkg += myBkgCoeffs[ind] * anX * aY
            ind += 1

    image += mybkg
    
    return image
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Retrieving Reference Image")
    refImage = retrieveReferenceImage()
    print("Degrading Reference Image")
    image = degradeReference(refImage)
    print("Getting Optimal Image for Subtraction")
    ruined_image, optKernel, bkg = toros.subtraction.getOptimalKernelAndBkg(image, 
                                                                            refImage, 
                                                                            bkgDegree=2, 
                                                                            kernelShape=(11,11))
    
    print("Plotting results")    
    #plot the results
    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(refImage, interpolation='none', cmap='gray')
    axes[0,1].imshow(image, interpolation='none', cmap='gray')
    axes[1,0].imshow(ruined_image, interpolation='none', cmap='gray')
    axes[1,1].imshow(ruined_image - image, interpolation='none', cmap='gray')
    plt.savefig("subtraction_test.png")
    
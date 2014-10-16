import toros
from toros.instrument import cstar as telescope
import sys
import matplotlib.pyplot as plt
from astropy.io import fits

if __name__ == '__main__':
    
    image_file_name = sys.argv[1]
    image = telescope.reduce(image_file_name)

    hdulist = fits.open(image_file_name)
    hdulist[0].data = image
    
    #image = toros.trackremoval.removeTracks(image) #Still needs to be implemented
    
    #toros.photometry.photoCalibrate(hdulist, catalogFileName)

    #toros.registration()

    refImage = toros.skygoldmaster.getReference(hdulist)

    optimal_image, kernel, background = toros.subtraction.getOptimalKernelAndBkg(image, refImage)

    plt.imshow(optimal_image)
    plt.colorbar()
    plt.show()

    plt.imshow(kernel, interpolation='None')
    plt.colorbar()
    plt.show()

    plt.imshow(background)
    plt.colorbar()
    plt.show()

    #subtracted_image = optimal_image - refImage
    #Keep working with subtracted_image...

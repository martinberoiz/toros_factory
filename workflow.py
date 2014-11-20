import toros
from toros.instrument import cstar as telescope
import sys
import matplotlib.pyplot as plt
from astropy.io import fits

if __name__ == '__main__':
    
    image_file_name = sys.argv[1]
    image = telescope.reduce(image_file_name)
    
    #image = toros.trackremoval.removeTracks(image) #Still needs to be implemented
    
    #toros.photometry.photoCalibrate(hdulist, catalogFileName)

    refImage = toros.skygoldmaster.getReference(image)

    optimal_image, kernel, background = toros.subtraction.getOptimalKernelAndBkg(image, refImage)
    
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.savefig('test_image.png')

    plt.figure()
    plt.imshow(refImage)
    plt.colorbar()
    plt.savefig('gold_reference.png')

    plt.figure()
    plt.imshow(optimal_image)
    plt.colorbar()
    plt.savefig('optimal_image2subtract.png')

    subtracted_image = optimal_image - refImage
    #Keep working with subtracted_image...

    plt.figure()
    plt.imshow(subtracted_image)
    plt.colorbar()
    plt.savefig('subtraction.png')

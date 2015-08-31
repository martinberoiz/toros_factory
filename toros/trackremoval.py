"""@package trackremoval

    Satellite track removal module
    ------------------------------

    Function to find satellite tracks in astro images.
    The function needs to be feed with a NP array from a FITS file
    The result will be an array with zeros and ones, in which the ones are the borders of the satellite tracks

    Americo Hinojosa - 2015

    email: <americo.hinojosalee01@utrgv.edu>

    University of Texas at Rio Grande Valley
"""

import numpy as np
from skimage.filter import canny
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from astropy.stats import sigma_clip
from skimage import draw

def removeLines(image):

    #Sigma Clipping
    clipped = sigma_clip(image, 5, 1)

    img = clipped
    satracks = np.zeros_like(img)
    median = np.median(img)

    #Edge finding function
    edges = canny(img, sigma=3, low_threshold=median*0.1666, high_threshold=median*.25)

    #Identify current lines
    lines = probabilistic_hough_line(edges, threshold=30, line_length=20, line_gap=3)
    for line in lines:
        x, y = line
        x0, y0 = x[0], y[0]
        x1, y1 = x[1], y[1]
        rr, cc = draw.line(y0, y1, x0, x1)
        satracks[cc, rr] = 1

    return satracks

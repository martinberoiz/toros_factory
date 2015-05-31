TOROS Factory pipeline
--------------------------------

A collection of tools to process images
for the Transient Optical Robotic Observatory of the South (TOROS).

These tools are intended to be used in scripts.
Minimal example to get a subtraction image from a FITS file:

    >>> import toros
    >>> from toros.instrument import your_favorite_telescope as telescope
    >>> header, image = telescope.reduce("myImage.fits")
    >>> refImage = toros.skygoldmaster.getReference(image, header)
    >>> subtraction_image = toros.subtract.optimalSubtractOnGrid(image, refImage)

For more information refer to the docstrings in the package.

Martin Beroiz - 2014

email: <martinberoiz@phys.utb.edu>

(c) University of Texas at San Antonio

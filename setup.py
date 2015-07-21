from distutils.core import setup

setup(name='toros',
      version='1.0a0.1.1',
      description='TOROS Project pipeline',
      author='Martin Beroiz',
      author_email='martinberoiz@phys.utb.edu',
      url='http://toros.phys.utb.edu',
      packages=['toros','toros.instrument', 'toros.resources'],
      #package_data={'': ['master2010.fits', '*.npy', 'cstar_mags_catalog.txt',]},
      install_requires=["numpy", "scipy", "astropy", "photutils", "reproject", "sep"],
     )

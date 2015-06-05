"""TOROS Transient Factory Pipeline"""

import pkg_resources
__package__ = 'toros'
__version__ = pkg_resources.get_distribution('toros').version

import firstweed
import skygoldmaster
import trackremoval
import photometry
import registration
import subtraction


import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
import os
from astropy.io import fits
import ransac
import shlex, subprocess

class InvariantTriangleMapping():
    
    def invariantFeat(self, sources, ind1, ind2, ind3):
        x1,x2,x3 = sources[[ind1,ind2,ind3]]
        sides = np.sort([np.linalg.norm(x1-x2),np.linalg.norm(x2-x3),np.linalg.norm(x1-x3)])
        return [sides[2]/sides[1],sides[1]/sides[0]]
    
    def generateInvariants(self, sources, nearest_neighbors = 5):
        #Helping function
        def arrangeTriplet(sources, vertex_indices):
            side1 = np.array([vertex_indices[0], vertex_indices[1]])
            side2 = np.array([vertex_indices[1], vertex_indices[2]])
            side3 = np.array([vertex_indices[0], vertex_indices[2]])

            sideLengths = [np.linalg.norm(sources[p1_ind]-sources[p2_ind]) for p1_ind, p2_ind in [side1, side2, side3]]
            lengths_arg = np.argsort(sideLengths)
    
            #Sides sorted from shortest to longest
            sides = np.array([side1, side2, side3])[lengths_arg]
        
            #now I order the points inside the side this way: [(x2,x0),(x0,x1),(x1,x2)]
            for i in range(-1,2):
                if sides[i][0] in sides[i+1]: 
                    #swap the points
                    sides[i] = sides[i][[1,0]]
            
            return sides[:,1]
        
        inv = []
        triang_vrtx = []
        coordTree = KDTree(sources)
        for asrc in sources:
            __, indx = coordTree.query(asrc, 5)
            all_asterism_triang = [list(acomb) for acomb in combinations(indx, 3)]
            inv.extend([self.invariantFeat(sources, *triplet) for triplet in all_asterism_triang])
            triang_vrtx.extend(all_asterism_triang)

        #Remove here many duplicate triangles for close-tight neighbors
        inv_uniq = np.array([elem for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1:]])
        triang_vrtx_uniq = [triang_vrtx[pos] for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1:]]
   
        #This will order the vertices in the triangle in a determined way to make a point to point correspondance with other triangles
        #(basically going around the triangle from smallest side to largest side)
        triang_vrtx_uniq = np.array([arrangeTriplet(sources, triplet) for triplet in triang_vrtx_uniq])

        return inv_uniq, triang_vrtx_uniq

    class matchTransform:
        def __init__(self, ref_srcs, target_srcs):
            self.ref = ref_srcs
            self.target = target_srcs
        
        def fit(self, data):
            #numpy arrays require an explicit 'in' method
            def in_np_array(elem, arr):
                return np.any([np.all(elem == el) for el in arr])
        
            #Collect all matches, forget triangle info
            d1, d2, d3 = data.shape
            point_matches = data.reshape(d1*d2,d3)
            A = []; b = [];
            for match_ind, amatch in enumerate(point_matches):
                #add here the matches that don't repeat
                if not in_np_array(amatch, point_matches[match_ind + 1:]):
                    ind_r, ind_t = amatch
                    x_r, y_r = self.ref[ind_r]
                    x_t, y_t = self.target[ind_t]
                    A.extend([[x_r, y_r, 1, 0],[y_r, -x_r, 0, 1]])
                    b.extend([x_t, y_t])
            A = np.array(A)
            b = np.array(b)
            sol, resid, rank, sv = np.linalg.lstsq(A,b.T)
            #lc,s is l (scaling) times cos,sin(alpha); alpha is the rotation angle
            #ltx,y is l (scaling) times the translation in the x,y direction
            lc = sol.item(0)
            ls = sol.item(1)
            ltx = sol.item(2)
            lty = sol.item(3)
            approxM = np.array([[lc, ls, ltx],[-ls, lc, lty]])
            return approxM
    
        def get_error(self, data, approxM):
            error = []
            for amatch in data:
                max_err = 0.
                for ind_r, ind_t in amatch:
                    x = self.ref[ind_r]
                    y = self.target[ind_t]
                    y_fit = approxM.dot(np.append(x,1))
                    max_err = max(max_err, np.linalg.norm(y - y_fit))
                error.append(max_err)
            return np.array(error)


def findAffineTransform(test_srcs, ref_srcs, max_pix_tol = 2., min_matches_fraction = 0.8, invariantMap=None):
    if len(test_srcs) < 3:
        raise Exception("Test sources has less than the minimum value of points (3).")
    
    if invariantMap is None:
        invMap = InvariantTriangleMapping()
    
    if len(ref_srcs) < 3:
        raise Exception("Test sources has less than the minimum value of points (3).")
    #generateInvariants should return a list of the invariant tuples for each asterism and 
    # a corresponding list of the indices that make up the asterism 
    ref_invariants, ref_asterisms = invMap.generateInvariants(ref_srcs, nearest_neighbors = 7)
    ref_invariant_tree = KDTree(ref_invariants)

    test_invariants, test_asterisms = invMap.generateInvariants(test_srcs, nearest_neighbors = 5)
    test_invariant_tree = KDTree(test_invariants)

    #0.03 is just an empirical number that returns about the same number of matches than inputs
    matches_list = test_invariant_tree.query_ball_tree(ref_invariant_tree, 0.03)

    matches = []
    #t1 is an asterism in test, t2 in ref
    for t1, t2_list in zip(test_asterisms, matches_list):
        for t2 in np.array(ref_asterisms)[t2_list]:
            matches.append(zip(t2, t1))
    matches = np.array(matches)
    
    invModel = invMap.matchTransform(ref_srcs, test_srcs)
    nInvariants = len(matches)
    max_iter = nInvariants
    min_matches = min(10, int(nInvariants * min_matches_fraction))
    bestM = ransac.ransac(matches, invModel, 1, max_iter, max_pix_tol, min_matches)
    return bestM

 
def addAstrometryNet(image, header, ra = None, dec = None, radius=None):
    output_dir = 'astrometrynet_temp_output'
    #Clean up whatever might be left from a previous run
    os.system('rm -rf %s' % (output_dir))
    os.system('mkdir -p %s' % (output_dir))

    #Create a temporary file for Astrometry.net to process
    temp_fits_filename = 'temp_file.fits'
    if isinstance(image, np.ma.MaskedArray):
        myhdulist = fits.HDUList(fits.PrimaryHDU(data=image.filled(fill_value = np.median(image)), header=header))
    else:
        myhdulist = fits.HDUList(fits.PrimaryHDU(data=image, header=header))   
    myhdulist.writeto(os.path.join(output_dir, temp_fits_filename), clobber=True)

    centerpos = ""
    if ra is not None and dec is not None and radius is not None:
        centerpos = "--ra %g --dec %g --radius %g" % (ra, dec, radius)

    #Execute the astrometry.net command (solve_field)
    cmd = 'solve-field --dir %s --no-plots --index-xyls none --match none --corr none ' \
          '--rdls none --solved none %s --new-fits none %s' % (output_dir, centerpos, os.path.join(output_dir, temp_fits_filename))

    process = subprocess.Popen(shlex.split(cmd))
    output_data, error_data = process.communicate()

    #Retrieve the WCS header
    pos = temp_fits_filename.find('.')
    temp_base = temp_fits_filename[:pos]
    newWCSHeader = fits.open(os.path.join(output_dir, temp_base + '.wcs'))[0].header

    #Add WCS info to header
    header.extend(newWCSHeader)

    #clean up temporary files
    os.system('rm -rf %s' % (output_dir))

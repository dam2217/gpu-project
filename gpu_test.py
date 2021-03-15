#Using part of the code that already was written by Peter for now 

import numpy as np
import tifffile
import scipy.interpolate as interp
import scipy
import cupy

#Load an example cell image

#----------------------Change this location of this-----------------#
im = tifffile.imread('/content/drive/MyDrive/beads_660nm.tif')

#Parameters for inter-lens vector, r, and lens center, center. 
r,center = (np.array([0.117, 19.525]), np.array([1020.4, 1024.1]))

Nnum = 19 # Number of pixels under microlenses
new_center = (1023,1023) # where to place new center when warping the data to standard template
rad_spots = 50 # number of microlenses in frame
lamda = 0.01 #this is the regularization factor


#Gets mapping between LF image pixels and recon volume pixels
def get_locs(center,Nnum,rad_spots = 50):
    '''
    Gets the locations of the pixels in the 'native microlens space' in terms
    of pixel locations in the LF image space.
    
    '''    
    
    sampling_grid = np.arange(-rad_spots,rad_spots+1)*Nnum #creating evenly spaced intervals
    ii,jj = np.meshgrid(sampling_grid,sampling_grid,indexing = 'ij') #returns coord matrices
    ii += center[0]
    jj += center[1]
    locs = [ii,jj]
    
    return locs


locs = get_locs(new_center, Nnum, rad_spots = rad_spots)


#Save intermediate iterations to find best value
iterations = [1,5] #why do we want to do this

#Warp image so MLA is straight, use backward projection as start guess
total_iterations = 0
sum_ = np.sum(im)

def get_warped_grid(r,center,new_center,Nnum,im_size = 2048):
    '''
    
    Calculates a mapping between an 'aligned' LF image with exactly Nnum pixels
    between the lenslets and what we actually have.

    '''
    theta = np.arctan(r[0]/r[1])
    diff = np.array(new_center) - np.array(center)
    
    x = np.arange(im_size).astype(float)
    
    ii,jj = np.meshgrid(x-center[0],x-center[1],indexing = 'ij')

    ii0 = np.copy(ii)
    jj0 = np.copy(jj)
    #rotate
    jj -= ii0*np.sin(theta)
    ii += jj0*np.sin(theta)
    
    #add offset
    ii -= diff[0]
    jj -= diff[1]
    
    #dilate/contract
    factor = np.sqrt(r[0]**2+r[1]**2)/Nnum
    ii *= factor
    jj *= factor
    
    return np.array([ii,jj])

def rectify_image(im,r,center,new_center,Nnum):
    '''
    Warps the collected LF image onto an ideal LF image
    
    '''
    im_size = im.shape[0]
    x = np.arange(im_size)
    interped = interp.RectBivariateSpline(x-center[0],x-center[1],im)
    
    warped_coords = get_warped_grid(r,center,new_center,Nnum,im_size = im_size)
    testttt = interped(warped_coords[0,...],warped_coords[1,...],grid = False)
    return interped(warped_coords[0,...],warped_coords[1,...],grid = False)


rectified = rectify_image(im,r,center,new_center,Nnum)


#Transforming into cupy array 
gpu_rectified = cp.array(rectified)

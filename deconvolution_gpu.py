# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:28:50 2021

@author: Dimitra Marmaropoulou klk
adapted from deconvolution code written by Peter Quicke 
"""
import deconvolve as de
import numpy as np
import pyqtgraph as pg
import tifffile
import scipy.interpolate as interp
import scipy

#Load an example cell image
im = tifffile.imread('./beads_660nm.tif')

#Parameters for inter-lens vector, r, and lens center, center. 
r,center = (np.array([0.117,19.525]),np.array([1020.4,1024.1]))

Nnum = 19 # Number of pixels under microlensess
new_center = (1023,1023) # where to place new center when warping the data to standard template
rad_spots = 50 # number of microlenses in frame
lamda = 0.01 #this is the regularization factor


#Whether to use RL schemes
RL = 0
Reg = 1

#saving location 
RL_save_loc = './RL_result.npy'
Reg_save_loc = './Reg_result.npy'

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

#PSF IS ASSUMED TO BE CALCULATED AT 1UM STEPS
#this is why the zmin,max are in m, zstep is an int in um
H = de.load_H_part(df_path = './psf/sim_df.xlsx',
                        folder_to_data = './psf/',
                        zmin = -60*10**-6,
                        zmax = 60*10**-6,
                        zstep = 10)

#Save intermediate iterations to find best value
iterations = [1,5] #why do we want to do this

#Allocate arrays   
if RL:
    result_rl = np.zeros((len(iterations), H.shape[0], locs[0].shape[-2],locs[0].shape[-1]))

if Reg:
    result_reg = np.zeros((len(iterations), H.shape[0], locs[0].shape[-2],locs[0].shape[-1]))



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
    
    return interped(warped_coords[0,...],warped_coords[1,...],grid = False)


rectified = rectify_image(im,r,center,new_center,Nnum)



def forward_project(volume,H,locs):
    '''
    Projects forward from the object space to the LF camera image    
    
    '''
    
    result = np.zeros((2048,2048))
    volume_upsamp = np.zeros((volume.shape[0],2048,2048))
    volume_upsamp[:,locs[0],locs[1]] = volume
    for i in range(H.shape[0]):
        result += scipy.signal.fftconvolve(volume_upsamp[i,...],H[i,...],mode = 'same')
    return result
    
def backward_project(image,H,locs):
    '''
    projects backward from the camera image to the object space

    '''
    result = np.zeros((H.shape[0],2048,2048))
    for i in range(H.shape[0]):
        result[i,...] = scipy.signal.fftconvolve(image,H[i,::-1,::-1],mode = 'same')
    volume = result[:,locs[0],locs[1]]
    return volume
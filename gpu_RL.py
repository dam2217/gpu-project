#Using part of the code that already was written by Peter for now 

import numpy as np
import tifffile
import scipy.interpolate as interp
import scipy
import cupy as cp
import cusignal
import deconvolve as de

#Load an example cell image

#----------------------Change this location of this-----------------#
im = tifffile.imread('./beads_660nm.tif')

#Parameters for inter-lens vector, r, and lens center, center. 
r,center = (np.array([0.117, 19.525]), np.array([1020.4, 1024.1]))

Nnum = 19 # Number of pixels under microlenses
new_center = (1023,1023) # where to place new center when warping the data to standard template
rad_spots = 50 # number of microlenses in frame
lamda = 0.01 #this is the regularization factor
iterations = [1,5]


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

H = de.load_H_part(df_path = './psf/sim_df.xlsx',
                        folder_to_data = './psf/',
                        zmin = -60*10**-6,
                        zmax = 60*10**-6,
                        zstep = 10)


#Warp image so MLA is straight, use backward projection as start guess
total_iterations = 0
sum_ = np.sum(im)
gpu_sum_ = cp.array(sum_)
gpu_H = cp.array(H)

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
    #testttt = interped(warped_coords[0,...],warped_coords[1,...],grid = False)
    return interped(warped_coords[0,...],warped_coords[1,...],grid = False)


rectified = rectify_image(im,r,center,new_center,Nnum)


#Transforming into cupy array 
gpu_rectified = cp.array(rectified)


def forward_project(volume,H,locs):
  result = cp.zeros((2048,2048))
  volume_upsamp = cp.zeros((volume.shape[0],2048,2048)) 
  '''
  makes a 3D array with the dimensions of the row number volume, and then 2048,2048, which is the cam_size - this is kinda like the length function in matlab
  '''
  volume = volume_upsamp[:,locs[0],locs[1]]

  for i in range(H.shape[0]):
    result += cusignal.fftconvolve(volume_upsamp[i,...],H[i,...],mode = 'same')
    
    cp.save('./forward_result.npy',result)
  return result

def backward_project(image,H,locs):
    '''
    projects backward from the camera image to the object space
    '''
    result = cp.zeros((H.shape[0],2048,2048))
    for i in range(H.shape[0]):
        result[i,...] = cusignal.fftconvolve(image,H[i,::-1,::-1],mode = 'same')
    volume = result[:,locs[0],locs[1]]
    
    cp.save('./backward_result.npy',result)
    
    return volume

def RL_(start_guess,measured,H,iterations,locs):
    #richardson lucy iteration scheme
    
    norm_fac = cp.sum(cp.sum(H,-1),-1)[:,None,None]

    result = cp.copy(start_guess)
    for _ in range(iterations):
        div = measured/(forward_project(result,H,locs)+1*10**-7)
        div[cp.isnan(div)] = 0
        error = backward_project(div,H,locs)
        result *= error/norm_fac
    return result


start_guess = backward_project(gpu_rectified/gpu_sum_,gpu_H,locs)

result = cp.zeros((len(iterations), gpu_H.shape[0], locs[0].shape[-2],locs[0].shape[-1]))

for idx,iter_number in enumerate(iterations):
    
        #calculate number of iterations to do
        iterations_this_go = iter_number - total_iterations

    
        #if we have not done any iterations this go use start guess, else use result of previous iteration
        if total_iterations == 0:
            #array indexed: iteration level, z, t, i,j
            result[idx,:, :,:] = RL_(start_guess, gpu_rectified/gpu_sum_, gpu_H, iterations_this_go,locs)
        else:
            result[idx,:,:,:] = RL_(result[idx-1,:, :,:], gpu_rectified/gpu_sum_, gpu_H, iterations_this_go,locs)                
    
        #save intermediate results because this can take a long time
        total_iterations += iterations_this_go
        
        cp.save('./result.npy',result)

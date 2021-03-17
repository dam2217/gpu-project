# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:11:25 2019

@author: peq10
"""

import numpy as np
import scipy.signal
import pandas as pd
import scipy.interpolate as interp
import os

def get_locs(center,Nnum,rad_spots = 50):
    '''
    Gets the locations of the pixels in the 'native microlens space' in terms
    of pixel locations in the LF image space.
    
    '''    
    
    sampling_grid = np.arange(-rad_spots,rad_spots+1)*Nnum
    ii,jj = np.meshgrid(sampling_grid,sampling_grid,indexing = 'ij')
    ii+= center[0]
    jj += center[1]
    locs = [ii,jj]
    
    return locs
    


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




def RL(start_guess,measured,H,iterations,locs):
    #richardson lucy iteration scheme
    
    norm_fac = np.sum(np.sum(H,-1),-1)[:,None,None]

    result = np.copy(start_guess)
    for _ in range(iterations):
        div = measured/(forward_project(result,H,locs)+1*10**-7)
        div[np.isnan(div)] = 0
        error = backward_project(div,H,locs)
        result *= error/norm_fac
        
    return result

def ISRA(start_guess,measured,H,iterations,locs):    
    #ISRA iteration scheme
    
    measured_projection = backward_project(measured,H,locs)
    result = np.copy(start_guess)
    for _ in range(iterations):
        error = backward_project(forward_project(result,H,locs),H,locs)
        result *= measured_projection/(error+1*10**-7)
        
    return result


def load_H_part(df_path = 'psf/sim_df.xlsx', #'psf/new/sim_df.xlsx'
                folder_to_data = 'psf',
                zmax = 50.5*10**-6,
                zmin = -50.5*10**-6,
                zstep = 5,
                thresh = 1,
                cam_size = 2048):  
    '''
    
    Loads the PSF over a specific range of um.
    '''
    
    
    #loads parts of H 
    df = pd.read_excel(df_path)
    df = df[df.z>=zmin]
    df = df[df.z<=zmax]
    len_z = len(np.unique(df.z))
    un_z = np.unique(df.z)
    un_x = np.unique(df.x)
    un_x.sort()
    window = np.hamming(len(un_x))
    
    H = np.zeros((len_z,2048,2048))
    
    test  = []
    for _,data in df.iterrows():     
        psf = np.load(folder_to_data+os.path.split(data.file)[-1]+'.npy')
        
        z_idx = np.where(un_z == data.z)[0][0]
        
        i,j = np.argmin(np.abs(data.x-un_x)),np.argmin(np.abs(data.y-un_x))
        filt_val = window[i]*window[j]
        
        values = psf[0,:]*filt_val
        indices = psf[1,:].astype(int)
        
        i,j = np.divmod(indices,2048)
        H[z_idx,i,j] += values
        
    #now take only rows in step
    H = H[::zstep,...]
    
    s = np.sum(np.sum(H,-1),-1)
    H /= s[:,None,None]
    
    test = np.sum(H,0)
    
    w = np.where(test)
    
    firsti,lasti = w[0][0],w[0][-1]
    firstj,lastj = w[0][0],w[0][-1]
    
    cent = int(2048/2)
    
    len_H = max(int(lasti-firsti),int(lastj-firstj))
    
    if len_H%2 == 0:
        fastlen_2 = int(len_H/2)
        cropH = H[:,cent - fastlen_2:cent+fastlen_2,cent - fastlen_2:cent+fastlen_2]
    else:
        len_H += 1
        fastlen_2 = int(len_H/2)
        cropH = H[:,cent - fastlen_2:cent+fastlen_2,cent - fastlen_2:cent+fastlen_2]
        
    return cropH   
     
def norm(array):
    ma = array.max()
    mi = array.min()
    return (array-mi)/(ma-mi)

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

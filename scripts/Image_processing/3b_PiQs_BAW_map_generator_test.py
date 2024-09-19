#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mag 21 14:31:30 2024

@author: Thomas Bernard

This script is the same than 3_PiQs_BAW_map_generator_v2.py but it is made to perform 
test of the active thresholds and the size fo minimum objects.
The aim is to compare with the manual mapping of some images.
"""
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import cv2
from skimage import morphology
from scipy import ndimage
from scipy.ndimage import convolve
from PIL import Image
from PiQs_BAW_func_v1 import *

# -------------------------------- SELECT RUN -------------------------------- #

runs = ['W06_Q05rgm42']
diff_name_map = ['Img0035.png','Img0514.png','Img1943.png'] # This is the diff image name from which the manual mapping has been done.


# -------------------------------- SCRIPT MODE ------------------------------- #
    
plt_show = 0

# ----------------------------- SCRIPT PARAMETERS ---------------------------- #
survey_length = 6.140 # [meters]
dt = 1 # Time between shoots in minutes


folder_home = os.getcwd() # Setup home folder 


# ----------------------------- INITIALIZE ARRAYS ---------------------------- #
BAW_segm_array = []

# for run, env_timescale in zip(runs, Txnr):

for run in runs:

    if run[5:7] == '05':
        skip=0

    # -------------------- LONG TO REGIME SPECIFIC PARAMETERS -------------------- #
    if 'rgmf' in run:
        if run[1:3] == '05':
            mov_avg = 35
            eps = 0.2
            skip=0
        if run[1:3] == '07':
            mov_avg = 35
            eps = 0.2
            skip=0
        if run[1:3] == '10':
            mov_avg = 20
            eps = 0.2
            skip=0
        if run[1:3] == '15':
            mov_avg = 20
            eps = 0.2
            skip=0
        if run[1:3] == '20':
            mov_avg = 20
            eps = 0.2
            skip=0
        
    else:
         mov_avg = 5
         eps = 0.4
    
    # LOAD MASK
    mask_path = folder_home+ '/Border_mask_W06.tif' # Define image path
    mask = Image.open(mask_path) # Open image as image
    # # Set the new resolution
    # new_resolution = (mask.width//5, mask.height//5) # reduce resolution to half
    # mask = mask.resize(new_resolution, Image.ANTIALIAS) # Resize the image to the new resolution
    mask_arr = np.array(mask)

    # SET COUNTERS
    k=0
    j=1
    r=0
    m=0
    tscale_counter = 1
     
    # SETUP DATA FOLDER
    diff_path_out = folder_home+ '/3b_Image_filtering_BAW_map_test_param'
    path_diff = folder_home+ '/2_Differences/Output/'+ run +'/' # Set the directory path where to pick up images
    path_img = folder_home+ '/1_Fused_images/'+ run + '/'
    path_report = folder_home+ '/output_report/'+ run + '/'

    path_blurry_area = path_img+'/Blurry_areas/'
    
    # Check if the folders already exist and create them
    if not(os.path.exists(path_diff)):
        os.mkdir(path_diff)
    if not(os.path.exists(path_img)):
        os.mkdir(path_img)
    if not(os.path.exists(path_report)):
        os.mkdir(path_report)
    if not(os.path.exists(path_blurry_area)):
        os.mkdir(path_blurry_area)
    if not(os.path.exists(os.path.join(path_blurry_area, run))):
        os.mkdir(os.path.join(path_blurry_area, run))
    if not(os.path.exists(os.path.join(diff_path_out, run))):
        os.mkdir(os.path.join(diff_path_out, run))

    
    # Create a file list with all the diff name
    diff_name_files = sorted(os.listdir(path_diff))
    diff_names = []
    for name in diff_name_files:
        if name.endswith('.png') and not(name.endswith('rsz.png')):
            diff_names = np.append(diff_names, name)
        else:
            pass
        
    # Create a file list with all the photo name
    photo_name_files = sorted(os.listdir(path_img))
    photo_names = []
    for name in photo_name_files:
        if name.endswith('.jpg') and not(name.endswith('rsz.jpg')):
            photo_names = np.append(photo_names, name)
        else:
            pass
        
        
    diff_path = os.path.join(path_diff, diff_names[1]) # Define image path
    diff = Image.open(diff_path) # Open image as image

    diff_arr = np.array(diff) # Convert image as numpy array
    dim_y, dim_x = diff_arr.shape
    
    # # MASK DIFFERENCE
    mask_arr = mask_arr[:dim_y,:dim_x]
    diff_arr = diff_arr*mask_arr
    
    # Load UPSTREAM-DOVNSTREAM MASK
    path_img = folder_home + '/1_Fused_images/'+ run
    print(path_img)
    maskUD_path = os.path.join(path_img, 'Mask.tif') # Define image path
    maskUD = Image.open(maskUD_path) # Open image as image
    maskUD_arr = np.array(maskUD)
    maskUD_arr = np.where(maskUD_arr==255, 1, 0)
    maskUD_arr = maskUD_arr[:dim_y,:dim_x]
    
    
    BAW_report = []  # Actiwe width report
    envBAA_report = [] # Envelope active width report
    diff_names = diff_names[skip:]
    photo_names = photo_names[skip:]
    BAA_stack_bool = []
    
    active_map_stack=[]
    
    env_mgrt_rate = []
    
    act_mean_array=[]
    act_stdev_array=[]
    diff_mean_array=[]
    diff_stdev_array=[]
    env_disc_BAW_array=[]
    ist_mgrt_rate_array=[]
    ist_mgrt_rate_positive_array=[]
    ist_mgrt_rate_negative_array=[]
    ist_mgrt_rate_abs_array=[]
    ist_mgrt_rate_negative_rel_array=[]
    ist_mgrt_rate_positive_rel_array=[]
    ist_mgrt_rate_rel_array=[]
    ist_mgrt_rate_negative_abs_array=[]
    ist_mgrt_rate_positive_abs_array=[]
    
    matrix_BAW = np.zeros((len(diff_names),dim_x))

    
    print('****************')
    print(run)
    print('****************')
    
  
    #  CHECK OUTLIERS
    # THIS SECTION CHECK ALL THE IMAGES AND FIND THE IMAGES WITH FLASH AND OTHER DISTURBANCY. THIS SECTION PROVIDES A NUMPY ARRAY WHERE THE INDEX OF THE IMAGES OUT
    # OF A CERTAIN BOUNDARY IS STORED. 
    for name in diff_names: # For each images in the folder...

        diff_path = os.path.join(path_diff, name) # Define image path
        diff = Image.open(diff_path) # Open image as image
        # Set the new resolution
        # new_resolution = (diff.width//5, diff.height//5) # reduce resolution to half
        # diff = diff.resize(new_resolution, Image.ANTIALIAS) # Resize the image to the new resolution
        diff_arr = np.array(diff) # Convert image in array
        diff_mean_array=np.append(diff_mean_array, np.mean(diff))
        diff_stdev_array=np.append(diff_stdev_array, np.std(diff))
        
        # # MASK DIFFERENCE
        diff_arr = diff_arr*mask_arr


    sig = moving_average_filter(diff_mean_array, mov_avg)

    
    
    r_samp = resample_array(diff_mean_array, sig)
    elements = np.where(abs(r_samp-diff_mean_array)>eps)
    print('Over saturated images: ', int(np.array(elements).shape[1]), 'over ',np.array(diff_names).shape[0], ' images.' )
    
    # Clean diff_mena_array: outliers set as np.nan
    diff_mean_arr_cleaned = np.where(abs(r_samp-diff_mean_array)>eps, np.nan, diff_mean_array)

    
    
    
    diff_mean_arr_cleaned_c = np.copy(diff_mean_arr_cleaned) # Create a new array where perform linear interpolation of np.nan
    diff_mean_arr_lin_interp = interpolate_nans(diff_mean_arr_cleaned_c) # Linear interpolated array

    
    
    if plt_show==1:
        plt.plot(np.linspace(0,len(diff_mean_array),len(diff_mean_array)), diff_mean_array, label='raw')
        # plt.plot(np.linspace(0,len(diff_mean_array),len(sig)), sig)
        plt.plot(np.linspace(0,len(diff_mean_array),len(sig)), sig+eps, '--', c='black')
        plt.plot(np.linspace(0,len(diff_mean_array),len(sig)), sig-eps, '--', c='black')
        plt.plot(np.linspace(0,len(r_samp),len(r_samp)), r_samp, label='resampling')
        plt.plot(np.linspace(0,len(diff_mean_arr_lin_interp),len(diff_mean_arr_lin_interp)), diff_mean_arr_lin_interp, '.', c='red', label='interp')
        plt.plot(np.linspace(0,len(diff_mean_arr_cleaned_c),len(diff_mean_arr_cleaned_c)), diff_mean_arr_cleaned_c, '--', label='cleaned')
        plt.title(run + ' - moving average: ' + str(mov_avg))
        plt.legend()

        plt.show(block=False)
        plt.close()
        # time.sleep(1)
        
    

    # DEFINE THE THRESHOLS
    # Set the threshold:

    if run[5:7] == '05':
        # print('SKIPPING BLURRY ANALYSIS...')
        thrs_actDS_list = [5, 5.5, 6, 6.5, 7, 7.5] # Active threshold - For values equal or grater than thrs_act bedeload is considered as active
        thrs_rso1_list = [5000, 10000, 15000, 20000]
        thrs_rso2_list = thrs_rso1_list
    
    for i in range(0,len(thrs_actDS_list)):
        for ii in range(0,len(thrs_rso1_list)):
            print('****************')
            print('thrs_act = '+ str(thrs_actDS_list[i]))
            print('thrs_rso1 = '+ str(thrs_rso1_list[ii]))
            print('****************')
            for name in diff_name_map: # For each images in the folder...
                # print('****************')
                print(name)
                # print('****************')
                
                # 1. OPEN DIFFERENCES AND IMAGES, AND THEN CONVERT TO NP.ARRAY
                diff_path = os.path.join(path_diff, name) # Define image path
                diff = Image.open(diff_path) # Open image as image
                diff_arr = np.array(diff) # Convert image as numpy array
                
                # # MASK ARRAY DOMAIN
                diff_arr = diff_arr*mask_arr
                
                index = int(np.where(diff_names==name)[0]) # Define a index that mach the image name
                
                # PERFORM THE IMAGE CORRECTION
                if index in np.array(elements): # Check if image is outlier (elements array contains all the outliers name)
                    diff_arr = diff_arr-(diff_mean_array[index]-diff_mean_arr_lin_interp[index]) # Correct the differece
        
                #Image.fromarray(np.array(diff_arr)).save(os.path.join(diff_path_out, run, run + '_' + str(name)[:-4] + 'diff.tiff')) # Save image    
    
                
                if run[5:7] == '05':
                    # print('SKIPPING BLURRY ANALYSIS...')
                    thrs_actDS = thrs_actDS_list[i] # Active threshold - For values equal or grater than thrs_act bedeload is considered as active
                    thrs_actUS = thrs_actDS# Active threshold - For values equal or grater than thrs_act bedeload is considered as active
                    thrs_act = thrs_actDS
                    shad_coeff = 2
                    thrs_rso1 = thrs_rso1_list[ii]
                    thrs_rso2 = thrs_rso1
        
                # 1. NOISE REMOVING
                shaded_area_image_path = os.path.join(path_img, 'shaded_area_images', run, run + '_shaded_area.tiff')
                banks_noise_bool = Image.open(shaded_area_image_path)
                banks_noise_bool = np.array(banks_noise_bool)[:dim_y,:dim_x]
                banks_noise_bool = (banks_noise_bool>0)*shad_coeff
                
                # REMOVE BANKS NOSIE
                diff_arr_msk = diff_arr-banks_noise_bool*1  # Remove noise array  (Difference between integer)
                
                diff_arr_msk = diff_arr_msk*(diff_arr_msk>0) # Trim negative values
                diff_arr_msk = diff_arr_msk.astype(np.uint16)  # Convert image as uint16
                
                # Convert and save
                # diff_msk = Image.fromarray(diff_arr_msk.astype(np.uint16)) # Convert array to image
                # diff_msk.save(os.path.join(diff_path_out, run, run + '_' + str(name)[:-4] + '_noise_rm.tiff'))
                
        
                # 1. DIFF AVERAGING
                diff_arr_avg0 = cv2.GaussianBlur(diff_arr_msk, (11,11), 0)
                
                n_row0, n_col0 = 11, 11  # 11x11 is fine
                kernel = np.ones((n_row0, n_col0), np.float32)/(n_row0*n_col0)
                diff_arr_avg0 = convolve(diff_arr_avg0, kernel, mode='same')
                
                # n_row0, n_col0 =11,11  # 7,7 is fine
                # kernel0=np.ones((n_row0,n_col0), np.float32)/(n_row0*n_col0)
                # diff_arr_avg0=cv2.filter2D(src=diff_arr_msk,ddepth=-1, kernel=kernel0) 
                
                # Convert and save
                # diff_avg0=Image.fromarray(np.array(diff_arr_avg0))
                # diff_avg0.save(os.path.join(diff_path_out, run, run + '_' + str(name)[:-4] + '_avg0.tiff'))
                
                # 2. THRESHOLDING
                # diff_arr_thrs = np.where(diff_arr_avg0>=thrs_act, diff_arr_avg0,0)
                # th, diff_arr_thrs = cv2.threshold(diff_arr_avg0, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
                
                # APPLY THRESHOLD UPSTREAM
                th, diff_arr_thrsUS = cv2.threshold(diff_arr_avg0, thrs_actUS, 255, cv2.THRESH_TOZERO)
                diff_arr_thrsUS = diff_arr_thrsUS*maskUD_arr
                
                # APPLY THRESHOLD DOWNSTREAM
                th, diff_arr_thrsDS = cv2.threshold(diff_arr_avg0, thrs_actDS, 255, cv2.THRESH_TOZERO)
                diff_arr_thrsDS = diff_arr_thrsDS*abs(maskUD_arr-1)
                
                #MERGE THE TWO IMAGES
                diff_arr_thrs = diff_arr_thrsUS + diff_arr_thrsDS
                
                # Convert and save
                # diff_thrs = Image.fromarray(diff_arr_thrs.astype(np.uint16))
                # diff_thrs.save(os.path.join(diff_path_out, run, run + '_' + str(name)[:-4] + '_thrs.tiff'))
                
                # BLURRY AREAS ARE COMPUTED ONLY FOR q20_2 AND q15_2 RUN
                diff_arr_blur = diff_arr_thrs
                
                # 4. MORPHOLOGICAL ANALYSIS
                # 4.a FILL SMALL HOLES
                diff_arr_rm_hls, diff_arr_rm_hls_target = fill_small_holes(
                    matrix=diff_arr_blur, avg_target_kernel=51, area_threshold=1000,
                    connectivity=1, value=10)
                
                # ERODE AREAS TO FIND PENINSULA
                diff_arr_rm_hls = diff_arr_rm_hls*(ndimage.binary_erosion(diff_arr_rm_hls, iterations=2))
                
                # 4.d REMOVE SMALL OBJECTS
                diff_arr_rsm_mask = morphology.remove_small_objects(diff_arr_rm_hls>0, min_size=thrs_rso1, connectivity=1)
                diff_arr_rsm = diff_arr_rm_hls*diff_arr_rsm_mask
                
                # Convert and save
                # diff_rsm = Image.fromarray(np.array(diff_arr_rsm).astype(np.uint16))
                # diff_rsm.save(os.path.join(diff_path_out, run, run + '_' + str(name)[:-4] + '_rm_obj.tiff'))
                
                
                # AVERAGE
                n_row0, n_col0 = 21,51  # 7,7 is fine
                kernel0=np.ones((n_row0,n_col0), np.float32)/(n_row0*n_col0)
                diff_arr_avg1 = cv2.filter2D(src=diff_arr_rsm,ddepth=-1, kernel=kernel0)
                
                
                
                # REAPPLY THRESHOLD
                # APPLY THRESHOLD UPSTREAM
                th, diff_arr_thrs2US = cv2.threshold(diff_arr_avg1, thrs_actUS, 255, cv2.THRESH_TOZERO)
                diff_arr_thrs2US = diff_arr_thrs2US*maskUD_arr
                
                # APPLY THRESHOLD DOWNSTREAM
                th, diff_arr_thrs2DS = cv2.threshold(diff_arr_avg1, thrs_actDS, 255, cv2.THRESH_TOZERO)
                diff_arr_thrs2DS = diff_arr_thrs2DS*abs(maskUD_arr-1)
                
                #MERGE THE TWO IMAGES
                diff_arr_avg1 = diff_arr_thrs2US + diff_arr_thrs2DS
                
                diff_arr_avg1_mask = morphology.remove_small_objects(diff_arr_avg1>0, min_size=thrs_rso2, connectivity=1)
                diff_arr_avg1 = diff_arr_avg1*diff_arr_avg1_mask
                
                # Save the ultimate image as a numpy array
                #np.save(os.path.join(diff_path_out, run, run + '_' + str(name)[:-4]+ '_ultimate_map.npy'), diff_arr_avg1)
                
                # PERFORM LINEAR DOWNSAMPLING TO OBTAIN THE LOW RESOLUTION VERSION
                BAA_map_LR5 = non_overlapping_average(diff_arr_avg1, kernel_size=5)
        
                # Remove border effects where values can be < thrs_act
                maskUD_LR5  = non_overlapping_average(maskUD_arr, kernel_size=5)
                maskUD_LR5  = np.where(maskUD_LR5!=0, 255, 0)
                BAA_map_LR5 = np.where(np.logical_and(maskUD_LR5!=0,BAA_map_LR5<thrs_actUS),0,BAA_map_LR5)
                BAA_map_LR5 = np.where(np.logical_and(maskUD_LR5==0,BAA_map_LR5<thrs_actDS),0,BAA_map_LR5)
                BAA_map_LR5 = np.float32(np.round(BAA_map_LR5,decimals=2))

                if not(os.path.exists(os.path.join(diff_path_out, run,str(name)[:-4]))):
                    os.mkdir(os.path.join(diff_path_out, run,str(name)[:-4]))
                
                np.save(os.path.join(diff_path_out, run,str(name)[:-4], run + '_' + str(name)[:-4]+ '_map_LR5_actDS_'+str(thrs_actDS)+'_rso1_'+str(thrs_rso1)+'.npy'), BAA_map_LR5)
                
                
                # Convert and save
                #diff_avg1 = Image.fromarray(np.array(diff_arr_avg1).astype(np.uint16))
                #diff_avg1.save(os.path.join(diff_path_out, run, run + '_' + str(name)[:-4]
                                         #   + '_cld_avg.tiff'))

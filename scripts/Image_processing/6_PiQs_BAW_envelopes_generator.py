b#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:44:32 2023

@author: erri
"""

# Import packages
import numpy as np
# import matplotlib.pyplot as plt
# import time
import os
# import cv2
# from skimage import morphology
# from scipy import ndimage
# from scipy.ndimage import convolve
from PIL import Image
from PiQs_BAW_func_v1 import *

# -------------------------------- SELECT RUN -------------------------------- #

runs = ['W06_Q07rgm']


# ---------------------------- ANALYSIS PARAMETERS --------------------------- #
analysis_list = [
    # 'total_envelope',
    # 'total_envelope_stack',
    'envelope_timescale'
    ]


# -------------------------------- SCRIPT MODE ------------------------------- #
plt_show = 1

# ----------------------------- SCRIPT PARAMETERS ---------------------------- #
survey_length = 6.140 # [meters]
dt = 1 # Time between shoots in minutes
downsampling_dim = 5


folder_home = os.getcwd() # Setup home folder 

# ----------------------------- INITIALIZE ARRAYS ---------------------------- #
# BAW_segm_array = []

# for run, env_timescale in zip(runs, Txnr):

for run in runs:
    env_tscale_array = [5,7,12,18]  # in minutes

    
    # ---------------------------- ENVELOPE TIMESCALE ---------------------------- #
    if run[1:3] == '07':
        skip=1
        env_tscale = env_tscale_array[3]
    if run[1:3] == '10':
        skip=1
        env_tscale = env_tscale_array[2]
    if run[1:3] == '15':
        skip=1
        env_tscale = env_tscale_array[1]
    if run[1:3] == '20':
        skip=1
        env_tscale = env_tscale_array[0]


    # SET COUNTERS
    k=0
    # j=1
    # r=0
    m=0
    tscale_counter = 1
     
    # SETUP DATA FOLDER
    diff_path_out = os.path.join(folder_home, 'Maps')

    path_folder_stacks = os.path.join(folder_home, 'activity_stack/activity_stack_cleaned')
    path_folder_envelopes = os.path.join(path_folder_stacks,'envelopes_cleaned','space_discr_' + str(downsampling_dim), run)
    path_partial_envelopes = os.path.join(path_folder_envelopes, run + '_envTscale' + str(env_tscale)) # Path where to stock the envelopes taken at a given timescale
    if not(os.path.exists(path_folder_envelopes)):
        os.mkdir(path_folder_envelopes)
    if not(os.path.exists(path_partial_envelopes)):
        os.mkdir(path_partial_envelopes)
    if not(os.path.exists(path_partial_envelopes)):
        os.mkdir(path_partial_envelopes)
    
    

    BAW_report = []  # Actiwe width report
    envBAA_report = [] # Envelope active width report
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

    meanBAW_array=[]
    
    print('****************')
    print(run, '  Timescale: ', env_tscale)
    print('****************')
        

    # DEFINE THE FOLDER WHERE THE BAA MAPS ARE STORED
    BAA_maps_path = os.path.join(diff_path_out, run)

    # CREATE THE FILE NAME LIST OF ALL THE FILES IN THE FOLDER
    BAA_maps_folder_names = sorted(os.listdir(BAA_maps_path))
    BAA_maps_names = []
    
    # CREATE A LIST OF THE NUMPY ARRAY OF THE BAA MAPS
    for name in BAA_maps_folder_names:
        if name.endswith('_ultimate_map.npy'):
            BAA_maps_names = np.append(BAA_maps_names, name)
        else:
            pass
    
    # LOAD THE TOTAL LOW RESOLUTION STACK
    stack_path = os.path.join(path_folder_stacks, run + '_BAA_stack_LR' + str(downsampling_dim) + '_cld.npy')
    
    # LOAD THE DATA STACK
    stack = np.load(stack_path)
    # CONVERT THE STACK TO BOOLEAN
    stack_bool = np.where(stack>0, 1, stack)
        
    
    
    for i in range(stack_bool.shape[0]): # For each images in the folder...
        
        BAA_map_LR = stack_bool[i,:,:]
        

        '''
        THIS SECTION WILL FILL STACKS OF BAA MAPS AT A GIVEN TIMESCALE
        NB: BAA_map_LR can be used instead of BAA_map to get low resolution stack or 
        In this section an envelope stack will be created every timescale.
        This procedure became necessary to compute the bedload migration rate from images.
        First because I would like to scale the migration rate calculation for different runs,
        second to have slots that are easy to handle an where to perform a few statistics.
        '''
        if 'envelope_timescale' in analysis_list:
            # print('check')
            # print('timescale counter: ', tscale_counter)

            if tscale_counter == 1:
                # DEFINE THE FIRST ITERATION STACK
                partial_stack = np.expand_dims(BAA_map_LR, axis=0)

                tscale_counter+=1 # Counter update

            elif tscale_counter%env_tscale!=0: # If the residual of the division is not equal to zero tscale_counter 
                # STACK ALL THE DATA
                partial_stack = np.vstack((partial_stack, np.expand_dims(BAA_map_LR, axis=0)))

                tscale_counter+=1 # Counter update

            elif tscale_counter%env_tscale==0:
                # FILL THE PARTIAL STACK
                partial_stack = np.vstack((partial_stack, np.expand_dims(BAA_map_LR, axis=0)))
                
                partial_stack_bool = np.where(partial_stack>0,1,0)
 
                # ----------------------------- MAKE THE ENVELOPE ---------------------------- #
                partial_envelope_sum = np.nansum(partial_stack_bool, axis=0) # As the sum of active times
                partial_envelope = np.nansum(partial_stack, axis=0)
                partial_envelope_bool = np.where(partial_envelope>0, 1, 0) # As a boolean map
                # ---------------------------------------------------------------------------- #
                
                
                # COMPUTE THE BEDLOAD ACTIVE WIDTH AND SAVE IT INTO A NUMPY ARRAY
                channel_width = 120
                meanBAW = np.nansum(partial_envelope_bool)/partial_envelope_bool.shape[1]/channel_width
                meanBAW_array = np.append(meanBAW_array, meanBAW)
                
                
                
                
                
                # # # TRIM THE CELLS THAT ARE ACTIVE LESS THAN 1/5 OF THE PARTAL STACK LENGTH
                # t_lenght_thrs = int(round(env_tscale*0.2))
                # partial_envelope_thrs_mask = np.where(partial_envelope_sum>t_lenght_thrs, 1, 0)
                partial_envelope_thrs = partial_envelope #*partial_envelope_thrs_mask
                partial_envelope_thrs_bool = np.where(partial_envelope_thrs>0, 1, 0) # As a boolean map

                # # -------------------- SAVE THE PARTIAL ENVELOPE AS IMAGE -------------------- #
                # partial_envelop_thrs_bool_img = Image.fromarray(np.array(partial_envelop_thrs_bool*1).astype(np.uint16))
                # partial_envelop_thrs_bool_img.save(os.path.join(path_partial_envelopes, run + '_'+ str(m)+'_envBAA_partial.tiff'))

                # ---------------------- SAVE THE BOOL ENVELOPE AS .npy ---------------------- #
                np.save(os.path.join(path_partial_envelopes, str(m) + '_' + run + '_partialBAA_cld_env.npy'), np.around(partial_envelope_thrs_bool, decimals=2))
                

                # ---------------------------------------------------------------------------- #

                # Update counter
                m +=1
                tscale_counter = 1
                
                
        
        
        
        
        
        # if 'total_envelope' in analysis_list:
        #     # Compute the dimensionless BAW (BAW*):
        #     arr_envBAA = np.sum(envBAA>0,axis=0)/np.nansum(mask_arr, axis=0) # Calculate cross section dimensionless bedload active width
            
        #     # Append the BAW* mean for the BAW* report
        #     envBAA_report = np.append(envBAA_report, np.mean(arr_envBAA)) # Append the average active width to build up the report
            
        # Update counter
        k += 1
    
    
    # GIVEN ALL THE _partialBAA_cld_env.npy FILES, BUILD THE STACK FOR EACH RUN
    # --- INPUT DATA: A SERIES OF PARTIAL ENVELOPES TAKEN AT DIFFERENT TIMESCALE -- #
    if 'envelope_timescale' in analysis_list:
        
        '''
        The input file of this section are the _partialBAA_cld_env.npy files.
        These maps are the result of the envelope process of a series of
        partial stack take at a give timescale.
        The maps are already provided as boolean map and the envelope is still
        epurated from the pixels that are active less than 20% of the partial
        envelope length.
        '''
        # Get a list of all files in the folder
        file_list = os.listdir(path_partial_envelopes)
        # List the envelope in the partial envelopes folder:
        file_names = [file for file in file_list if file.endswith(run + '_partialBAA_cld_env.npy')]

        # Sort the list using the custom sorting key function
        file_names = sorted(file_names, key=custom_sort_key)

        for i, file_name in enumerate(file_names):
            path = os.path.join(path_partial_envelopes, file_name)
            partial_env_bool = np.load(path)

            # # APPLY DOMAIN MASK
            # dim_y, dim_x = partial_env_bool.shape  # Define dimension
            # # print(stack_partial_bool.shape)

            # # RESHAPE MASK
            # mask_arr_LR = mask_arr_LR[:dim_y,:dim_x]  # Reshape
            # # Set np.nan outside the domain
            # mask_arr_LR = np.where(mask_arr_LR > 0, 1, np.nan)

            # # APPLY DOMAIN MASK
            # partial_env_bool = partial_env_bool*mask_arr_LR

            # # ------------- FROM PARTIAL STACK TO PARTIAL ENVELOPE - ANALYSIS ------------ #
            # # COMPUTE THE ENVELOPE OF THE PARTIAL STACK
            # envBAA_partial = np.nansum(stack_partial, axis=0) # Perform the envelope
            # envBAA_partial_bool = np.where(envBAA_partial>0, 1, envBAA_partial) # Convert in boolean maps
            
            # STACK ALL THE ENVELOPES IN A STACK
            if i == 0:
                stack_bool = np.expand_dims(partial_env_bool, axis=0)
            else:
                stack_bool = np.vstack((stack_bool,np.expand_dims(partial_env_bool, axis=0)))

        # SAVE THE PARTIAL STACK AS A BINARY FILE
        # TODO THIS DO NOT WORK
        np.save(os.path.join(path_partial_envelopes, run+ '_envT' + str(env_tscale) + '_partial_envBAA_stack.npy'), stack_bool)
    
    
    # Save the meanBAW_array:
    np.savetxt(os.path.join(path_partial_envelopes, run + '_' + str(env_tscale) + '_meanBAW.txt'), np.around(meanBAW_array, decimals=2))
    print(np.round(np.nanmean(meanBAW_array), decimals=3))
    print(np.round(np.nanstd(meanBAW_array), decimals=3))
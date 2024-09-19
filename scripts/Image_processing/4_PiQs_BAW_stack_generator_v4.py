#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:54:30 2023

@author: erri

INPUT:
    BAA_map
    TYPE: NUMPY 2D ARRAY
    NOTE: BEDLOAD ACTIVE AREA MAP AS A 2D NUMPY ARRY WHERE EVERY CELL CONTAINS
        BEDLOAD INTENSITY VALUE
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
runs = ['W06_Q05r1']


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
    


    # SET COUNTERS
    k=0
    # j=1
    # r=0
    m=0
    tscale_counter = 1
     
    # SETUP DATA FOLDER
    diff_path_out = os.path.join(folder_home, '3_Image_filtering_BAW_map')

    path_report = os.path.join(folder_home, 'output_report', run)
    path_partial_envelopes_folder = os.path.join(path_report, 'partial_envelopes')

    if not(os.path.exists(path_partial_envelopes_folder)):
        os.mkdir(path_partial_envelopes_folder)

    

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


    
    print('****************')
    print(run)
    print('****************')
        

    # DEFINE THE FOLDER WHERE THE BAA MAPS ARE STORED
    BAA_maps_path = os.path.join(diff_path_out, run)

    # CREATE THE FILE NAME LIST OF ALL THE FILES IN THE FOLDER
    BAA_maps_folder_names = sorted(os.listdir(BAA_maps_path))
    BAA_maps_names = []
    
    # CREATE A LIST OF THE NUMPY ARRAY OF THE BAA MAPS
    for name in BAA_maps_folder_names:
        if name.endswith('_ultimate_map_LR' + str(downsampling_dim) + '.npy'):
            BAA_maps_names = np.append(BAA_maps_names, name)
        else:
            pass
    
    # FOR ALL THE FILES IN THE LIST:
    
    for name in BAA_maps_names: # For each images in the folder...
        # LOAD THE BAA MAP (map of the bedload intensity values)
        BAA_map_path = os.path.join(diff_path_out,'thrs_act_6.5', run, name) # Define the BAA map path
        BAA_map = np.array(np.load(BAA_map_path)) # Load the .npy file
        
        # PERFORM LINEAR DOWNSAMPLING TO OBTAIN THE LOW RESOLUTION VERSION
        BAA_map_LR = BAA_map
        
        
        
        '''
        THIS SECTION COMPUTES THE TOTAL ENVELOPE AND SAVE ALL THE ACTIVITY MAP IN A NUMPY STACK
        envBAA: this is the envelope of all the maps in boolean format
        BAA_stack_bool: this is the stack of all the activity map in boolean format
        envBAA_act_cumulative; this is the sum of all the images keeping the intensity value
        BAA_stack_LR: this is the stack of the low resolution (LR) data 
        '''
                
        if name == BAA_maps_names[0]: # For the first image:
            
            # CREATE THE NATIVE AND LOW RESOLUTION FIRST LAYER OF THE STACK
            envBAA = np.where(BAA_map>0, 1, 0) # First iteration to build the boolean envelope of the activity map
            envBAA_LR = np.where(BAA_map_LR>0, 1, 0) # First iteration to build the boolean envelope of the activity map
            
            
            # CREATE THE FIRST LAYER OF THE:
            #   1. BAA_stack_bool that is the overall boolean stack
            #   2. envBAA_act_cumulative that is a map where each cell is the sum of the bedload intensity
            #   3. BAA_stack_LR that is the low resolution stack of the intensity values

            # BUILD THE DATA STACK
            BAA_stack_bool = np.expand_dims(envBAA, axis=0) # First iteration to create the boolean stack

            # BUILD THE CUMULATIVE ENVELOPE OF BEDLOAD ACTIVITY INTENSITY
            envBAA_act_cumulative = BAA_map # First iteration to create the cumulative envelope of the bedload activity intensity
        
            # BUILD THE LOW RESOLUTION BAA STACK KEEPING THE INTENSITY VALUE
            BAA_stack_LR = np.expand_dims(BAA_map_LR, axis=0) # First iteration to create the  Low Resolution cumulative envelope of the bedload activity intensity
            
        else: # For the others images

            # CREATE THE ENVELOPE SUMMING THE PREVIOUS AND THE CURRENT IMAGE AS BOOLEAN MAPS
            BAA_map_bool = np.where(BAA_map>0, 1, 0) # Convert current BAA map in type=bool
            envBAA = envBAA + BAA_map_bool # Add the new map to the previous one: compute the envelope

            # BUILD THE DATA STACK
            BAA_stack_bool = np.vstack((BAA_stack_bool, np.expand_dims(BAA_map_bool, axis=0))) # Build the boolean stack

            # BUILD THE CUMULATIVE ENVELOPE OF BEDLOAD ACTIVITY INTENSITY
            envBAA_act_cumulative = envBAA_act_cumulative + BAA_map
        
            # BUILD THE LOW RESOLUTION BAA STACK KEEPING THE INTENSITY VALUE
            BAA_stack_LR = np.vstack((BAA_stack_LR, np.expand_dims(BAA_map_LR, axis=0)))
        
        
    BAA_stack_LR_bool = np.where(BAA_stack_LR>0, 1, 0) # Convert the stack in bool stack

    # SAVE COMPLETE LOW RESOLUTION STACK
    np.save(os.path.join(folder_home, '4_BAW_stack', run + '_BAA_stack_LR' + str(downsampling_dim) + '.npy'),np.around(BAA_stack_LR, decimals=2))

    
    # ----------------------- SAVE BAA act mean intensity ENVELOPE ----------------------- #
    BAA_stack_LR_mean_intensity = np.nanmean(BAA_stack_LR,axis=0) 
    np.save(os.path.join(folder_home, '4_BAW_stack', run + '_envBAA_act_mean_intensity_LR' + str(downsampling_dim) +'.npy'),np.around(BAA_stack_LR_mean_intensity, decimals=2))

    # ------------------------------ STACK FILTERING ----------------------------- #
    #'''FILTER THE STACK TO REMOVE PIXELS THAT ARE ACTIVE LESS THAT THE
    #20% OF THE STACK LENGTH IN THE ENVELOPE TIME'''
    #BAA_stack_LR_bool_envelope = np.nansum(BAA_stack_LR_bool, axis=0) # With the envelope, find where a pixel is active at least once
    #BAA_stack_LR_bool_envelope_thrs = BAA_stack_LR_bool_envelope
    
    
    # BAA_stack_LR_bool_cld = (BAA_stack_LR_bool_envelope!=1)*BAA_stack_LR_bool[:] # Trim pixels active at least once
    # ---------------------------------------------------------------------------- #

    # --------------------- SAVE BAA STACK IN LOW RESOLUTION --------------------- #
    # Save BAA stack at low resolution (_LR)
    #np.save(os.path.join(folder_home, 'activity_stack', run + '_BAA_stack_envelope_LR' + str(downsampling_dim) + '.npy'),np.around(BAA_stack_LR_bool_envelope_thrs, decimals=2))



    # ----------------------- SAVE BAA CUMULATIVE ENVELOPE ----------------------- #
    # AS THE SUM OF THE TIMES A CERTAIN PIXEL HAS BEEN ACTIVE
    # as txt...
    #np.savetxt(os.path.join(diff_path_out, run, run + '_envBAA_act_cumulative.txt'), envBAA_act_cumulative, fmt='%.{}f'.format(0))
    
    np.save(os.path.join(folder_home,  '4_BAW_stack', run + '_envBAA_act_cumulative_LR' + str(downsampling_dim) +'.npy'),np.around(envBAA_act_cumulative, decimals=2))

    #envBAA_act_cumulative_img = Image.fromarray(np.array(envBAA_act_cumulative*1).astype(np.uint16))
    #envBAA_act_cumulative_img.save(os.path.join(diff_path_out, run, run + '_envBAA_act_cumulative.tiff'))
    # envBAA_act_cumulative_img = Image.fromarray(np.array(envBAA_act_cumulative*1))
    # envBAA_act_cumulative_img.save(os.path.join(diff_path_out, run, run + '_envBAA_act_cumulative.tiff'))


    # This is the envelop as a boolean map where 1 is active and 0 is not active
    # where all the pixel active only once are trimmed
    #TODO from the overall envelope remove the pixels that are active at least 1 time
    #envBAA_cld = envBAA*(envBAA>1)
    #envBAA_cld_img = Image.fromarray(np.array(envBAA_cld*1).astype(np.uint16))
    #envBAA_cld_img.save(os.path.join(diff_path_out, run, run + '_envBAA_map_cleaned.tiff'))
    
    # Save the envBAA as tiff image (keep the number of pixel activation)
    #envBAA_img = Image.fromarray(np.array(envBAA*1).astype(np.uint16))
    #envBAA_img.save(os.path.join(diff_path_out, run, run + '_envBAA_map_activ_history.tiff'))
    
    # Save the envBAA as tiff image (set 1 where active at least 1, zero elsewhere)
    #envBAA_img = envBAA*(envBAA>0)
    #envBAA_img = Image.fromarray(np.array(envBAA*1).astype(np.uint16))
    #envBAA_img.save(os.path.join(diff_path_out, run, run + '_envBAA_map.tiff'))


    # Save report in .txt files
    #np.savetxt(os.path.join(path_report,run + '_BAW_report.txt'), BAW_report, delimiter=',') # Save active width report
    #np.savetxt(os.path.join(path_report,run + '_envBAW_report.txt'), envBAA_report, delimiter=',') # Save envelope active width report
    # np.savetxt(os.path.join(path_report,run + '_env_migratio_rate_report.txt'), env_mgrt_rate, delimiter=',') # Save envelope active width report
    

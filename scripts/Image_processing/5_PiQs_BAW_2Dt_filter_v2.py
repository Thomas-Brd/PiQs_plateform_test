'''
# ---------------------------------------------------------------------------- #
                            PiQs Pixel Age Analysis                           #
 ---------------------------------------------------------------------------- #
August 2023
Author: Enrico Pandrin
#* INPUT FILES
    - stack as a 3D array [t,y,x] where low resolution (LR) bedload acrtivity map are stored
#* OUTPUT
#* GRAPHICAL OUTPUTS
    - Histograms of the distribution of frequency of the activity periods length
       divided in active periods and inactive periods
       ```_Active_Inactive_DoF_his.pdf```
    -   
        1. the number ot times a pixel is active
        2. The number of activity periods
        3. The number of inactivity periods
        ```_Activity_MAP.pdf```
    - An interactive map and a series of maps for each runs where the activated
        and deactivated area along the run duration are displayed.
        ```_activation_deactivation_BAA.pdf```


'''


# IMPORT LIBRARIES
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

from PIL import Image

from PiQs_BAW_func_v1 import *




# GET CURRENT FOLDER
folder_home = os.getcwd() # Setup home folder

# -------------------------------- SCRIPT MODE ------------------------------- #


downsampling_dim = 5


# ----------------------------------- RUNS ----------------------------------- #
runs = ['W06_Q05r1']


# runs = ['q07rgm', 'q10rgm2', 'q15rgm2', 'q20rgm2']
# runs = ['q10rgm2', 'q15rgm2', 'q20rgm2']
# runs = ['q07rgm', 'q10rgm2', 'q15rgm2', 'q20rgm2']
# runs = ['q07rgm']
# runs = ['q10rgm2']
# runs = ['q20rgm2']

# runs = ['q20rgm2']
# runs = ['q20r9']

# ----------------------------- INITIALIZE ARRAYS ---------------------------- #


for run in runs:
    
    
    
    print()
    print(run)

    index = runs.index(run)


    # --------------------------------- LOAD MASK -------------------------------- #
    mask_path = os.path.join(folder_home, 'Border_mask_'+run+'.tif') # Define image path
    mask = Image.open(mask_path) # Open image as image
    
    mask_arr = np.array(mask)
    mask_arr_LR = non_overlapping_average(mask_arr, kernel_size=downsampling_dim) # perform linear downsampling

    # ------------------------------- DEFINE PATHS ------------------------------- #
    path_report = os.path.join(folder_home, 'output_report', run)
    
    path_report_graphic = os.path.join(folder_home, 'output_report', run, 'plot')
    


    # Check if the folders already exist and create them
    if not(os.path.exists(path_report)):
        os.mkdir(path_report)
    if not(os.path.exists(path_report_graphic)):
        os.mkdir(path_report_graphic)

    # ---------------------------------------------------------------------------- #
    #                             DEFINE THE INPUT DATA                            #
    # ---------------------------------------------------------------------------- #

    # -------------------- INPUT DATA: THE TOTAL RUN ENVELOPE -------------------- #
    stack_path = os.path.join(folder_home, '4_BAW_stack/', run + '_BAA_stack_LR' + str(downsampling_dim) +  '.npy')
    
    # LOAD THE DATA STACK
    stack = np.load(stack_path)
    # CONVERT THE STACK TO BOOLEAN
    stack_bool = np.where(stack>0, 1, stack)

    # APPLY DOMAIN MASK
    dim_t, dim_y, dim_x = stack_bool.shape # Define dimension

    # RESHAPE MASK
    mask_arr_LR = mask_arr_LR[:dim_y,:dim_x] # Reshape
    mask_arr_LR = np.where(mask_arr_LR>0, 1, np.nan) # Set np.nan outside the domain

    # APPLY DOMAIN MASK
    stack_bool[:] = stack_bool[:]*mask_arr_LR
    stack_bool = np.where(stack_bool<0, np.nan, stack_bool)
        
        
    # # #TODO FOR TESTING PURPOSES ONLY
    # # stack_bool = stack_bool[:200, :200, :200]
    
    
    stack_bool_raw = np.copy(stack_bool)
        
    # ---------------------------------------------------------------------------- #
    #                 FROM HERE BELOW STACK BOOL IS THE INPUT DATA                 #
    # ---------------------------------------------------------------------------- #
    
    #################################################
    # FOR TESTING ONLY
    # stack_bool_raw = stack_bool_raw[:,:50,:50]
    #################################################
    
    
    thrs = 0.4
    stack_bool_cld = spatial_temporal_activation_filtering(stack_bool_raw, (3,3,3), thrs)

    # Apply spatio temporal filterin to intensity stack
    mask_nan = np.copy(stack)
    mask_nan = np.where(np.isnan(stack),1,0)
    stack_cld = np.where(stack_bool_cld==1,stack,0)
    stack_cld = np.where(mask_nan==1, np.nan, stack_cld)
    # # # MANUALLY CREATE THE HISTOGRAM
    # data_cld = np.copy(stack_bool_cld)
    # data_raw= np.copy(stack_bool_raw)
    
    # data_cld = data_cld[data_cld!=0] # Trim zero values
    # data_raw = data_raw[data_raw!=0] # Trim zero values
    
    # data_cld = data_cld[data_cld==1] # Trim zero values
    # data_raw = data_raw[data_raw==1] # Trim zero values
    
    
    # hist_array = []
    
    # for n in range(int(np.nanmin(data)), data.shape[0]+1):
    #     # print(n)
    #     count = np.count_nonzero(data == n)/(len(data))
    #     # count = np.count_nonzero(data == n)
    #     hist_array = np.append(hist_array, count)
        
        
    env_stack_cld = np.nansum(stack_bool_cld, axis=0)
    env_stack_bool_cld = np.where(env_stack_cld>0,1,0)
    np.save(os.path.join(os.path.join(folder_home, '5_Time_filtering/'), run + '_BAA_stack_LR' + str(downsampling_dim) + '_cld_bool.npy'),np.around(stack_bool_cld, decimals=2))
    np.save(os.path.join(os.path.join(folder_home, '5_Time_filtering/'), run + '_BAA_stack_LR' + str(downsampling_dim) + '_cld.npy'),np.around(stack_cld, decimals=2))
    np.save(os.path.join(os.path.join(folder_home, '5_Time_filtering/'), run + '_envBAA_stack_LR' + str(downsampling_dim) + '_cld.npy'),np.around(env_stack_bool_cld, decimals=2))

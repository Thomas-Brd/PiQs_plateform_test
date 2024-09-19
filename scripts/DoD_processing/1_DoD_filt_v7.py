#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:44:30 2021

@author: erri
"""
import os
import shutil
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from DoD_analysis_functions_4 import *
from morph_quantities_func_v2 import morph_quantities

###############################################################################
# TODO
###############################################################################
# 1. 

start = time.time() # Set initial time

###############################################################################
# SETUP SCRIPT PARAMETERS and RUN MODE
###############################################################################

'''
run mode:
    0 = runs in the runs list
    1 = one run at time
    2 = bath process 'all the runs in the folder'
data_interpolatuon_mode:
    0 = no interpolation
    1 = data interpolation
windows_mode:
    0 = fixed windows (all the channel)
    1 = floating windows
    2 = fixed windows (WxW, Wx2W, Wx3W, ...) without overlapping
    3 = fixed windows (WxW, Wx2W, Wx3W, ...) with overlapping
mask mode:
    1 = mask the flume edge
    2 = mask the upstream half flume
    3 = mask the downstream half flume
process mode: (NB: set DEMs name)
    1 = batch process
    2 = single run process
save mode:
    0 = save only reports
    1 = save all chart and figure
DoD_plot_save_mode:
    0 = do not save DoD plots
    1 = save plot
DoD_plot_show_mode:
    0 = do not show DoD plots
    1 = show DoD plots
'''
run_mode = 0
mask_mode = 1
process_mode = 1
filt_analysis = 0 # Print the morphological metrics for each filtering process stage

plot_mode = [ 'mDoD_plot']

# SINGLE RUN NAME
run = 'W06_Q05'
# ARRAY OF RUNS
runs = ['W06_Q05']

# Set DEM single name to perform process to specific DEM
if len(run) ==1:
    DEM1_single_name = 'matrix_bed_norm_' + str(run) +'s'+'0'+'.txt' # DEM1 name
    DEM2_single_name = 'matrix_bed_norm_' + str(run) +'s'+'1'+'.txt' # DEM2 name

# Filtering process thresholds values
thrs_zeros = 7 # [-] isolated_killer function threshold
thrs_nature = 5 # [-] nature_checker function threshold
thrs_fill = 7 # [-] isolated_filler function threshold
thrs_1 = 1.3  # [mm] # Lower threshold
thrs_2 = 15.0  # [mm] # Upper threshold

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]

# Not a number raster value (NaN)
NaN = -999

#%%
###############################################################################
# SETUP FOLDERS and RUNS
###############################################################################
# setup working directory and DEM's name
home_dir = os.getcwd()
out_dir = os.path.join(home_dir, run)
plot_out_dir = os.path.join(home_dir, run, 'DoDs', 'plot')

# Create folders
if not(os.path.exists(out_dir)):
    os.mkdir(out_dir)
if not(os.path.exists(plot_out_dir)):
    os.mkdir(plot_out_dir)
    



run_dir = os.path.join(home_dir, run, 'DEMs')



# Create the run name list
RUNS=[]

if run_mode==0:
    RUNS=runs
elif run_mode==2: # batch run mode
    for RUN in sorted(os.listdir(run_dir)): # loop over surveys directories
        if RUN.startswith('q'): # Consider only folder names starting wit q
            RUNS = np.append(RUNS, RUN) # Append run name at RUNS array
elif run_mode==1: # Single run mode
    RUNS=run.split() # RUNS as a single entry array, provided by run variable

#%%
###############################################################################
# INITIALIZE OVERALL REPORT ARRAYS
###############################################################################

# Define volume time scale report matrix:
# B_dep, SD(B_dep), B_sco, SD(B_sco)
volume_temp_scale_report=np.zeros((len(RUNS), 4))

# Define morphW time scale report matrix:
# B_morphW [min], SD(B_morphW)
morphW_temp_scale_report = np.zeros((len(RUNS), 2))

# # Define Engelund Gauss model report matrix:
# # D [m], Q [m^3/s], Wwet/W [-]
# engelund_model_report=np.zeros((len(RUNS),3))

# Array that collect all the morphWact_array dimension.
# It will be used to create the morphWact_matrix
morphWact_dim = [] # Array with the dimensions of morphWact_values array

DoD_length_array=[] # DoD length array


#%%
###############################################################################
# MAIN LOOP OVER RUNS
###############################################################################
for run in RUNS:

    ###########################################################################
    # SETUP FOLDERS
    ###########################################################################
    print('######')
    print(run)
    print('######')
    print()
    # setup working directory and DEM's name
    DoDs_dir = os.path.join(out_dir,'DoDs')
    input_dir = os.path.join(run_dir, 'survey_')
    report_dir = os.path.join(DoDs_dir, 'reports')
    plot_dir = plot_out_dir
    path_out = os.path.join(DoDs_dir,'DoDs') # path where to save DoDs
    DoDs_plot = os.path.join(plot_out_dir,'DoDs_maps')
    
    # Save a report with xData as real time in minutes and the value of scour and deposition volumes for each runs
    # Check if the file already exists
    if os.path.exists(os.path.join(report_dir, 'volume_over_time.txt')):
        os.remove(os.path.join(report_dir, 'volume_over_time.txt'))
    else:
        pass

    # CREATE FOLDERS
    
    if not(os.path.exists(DoDs_dir)):
        os.mkdir(DoDs_dir)
#if os.path.exists(os.path.join(DoDs_dir)): # If the directory exist, remove it to clean all the old data
 #       shutil.rmtree(os.path.join(DoDs_dir), ignore_errors=False, onerror=None)
    if not(os.path.exists(os.path.join(DoDs_dir))): # If the directory does not exist, create it
        os.mkdir(os.path.join(DoDs_dir))
    if not(os.path.exists(os.path.join(DoDs_dir, 'DoDs_stack'))):
        os.mkdir(os.path.join(DoDs_dir, 'DoDs_stack'))
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)
    if not(os.path.exists(DoDs_plot)):
        os.mkdir(DoDs_plot)
    if not(os.path.exists(report_dir)):
        os.mkdir(report_dir)


    # IMPORT RUN PARAMETERS from file parameters.txt
    # variable run must be as 'q' + discharge + '_' repetition number
    # Parameters.txt structure:
    # discharge [l/s],repetition,run time [min],Texner discretization [-], Channel width [m], slope [m/m]
    # Load parameter matrix:
    parameters = np.loadtxt(os.path.join(home_dir, 'parameters.txt'),
                            delimiter=',',
                            skiprows=1)

    # Extract run parameter depending by run name
    run_param = parameters[np.intersect1d(np.argwhere(parameters[:,1]==float(run[-1:])),np.argwhere(parameters[:,0]==float(run[5:7])/10)),:]

    # Run time data
    dt = run_param[0,2] # dt between runs [min] (real time)
    dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)
    
    # Flume geometry parameters
    W = run_param[0,4] # Flume width [m]
    S = run_param[0,5] # Flume slope
    
    # Sediment parameters
    d50 = run_param[0,6]

    # Run discharge
    Q = run_param[0,0] # Run discharge [l/s]
    
    # Create the list of surveys files.
    # To do so, two system array will be created for the run with more than 10 surveys due to the behaviour of the sort python function that sort files in this way:
    # matrix_bed_norm_q10_3s0.txt matrix_bed_norm_q10_3s1.txt matrix_bed_norm_q10_3s11.txt matrix_bed_norm_q10_3s12.txt matrix_bed_norm_q10_3s2.txt
    files = []
    files1=[] # Files list with surveys from 0 to 9
    files2=[] # Files list with surveys from 10
    # Creating array with file names:
    for survey_nb in range(0,10):
        folder = input_dir+str(survey_nb)
        print(folder)
        for f in os.listdir(folder):
            path = os.path.join(input_dir+str(survey_nb),f)
            if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm_'+run+'_s'):
                files = np.append(files, f)
            else: 
                pass
    print(files)
    for f in files:
        if len(f)==len(files[1]):
            files1 = np.append(files1, f)
        elif len(f)==len(files[1])+1:
            files2 = np.append(files2, f)
            
    files = np.append(files1,files2) # Files has been overwritten with a list of file names sorted in the right way :) 
    print(files)
    if process_mode==1:
        pass
    elif process_mode == 2:
        files=[]
        files=np.append(files,(DEM1_single_name, DEM2_single_name))

    # INITIALIZE ARRAYS
    comb = np.array([]) # combination of differences
    matrix_DEM_analysis = np.zeros((len(files), len(files)))
    
    # Analysis on the effect of the spatial filter on the morphological changes
    # Initialize array
    DoD_raw_morph_quant = []
    DoD_filt_mean_morph_quant = []
    DoD_filt_isol_morph_quant = []
    DoD_filt_fill_morph_quant = []
    DoD_filt_nature_morph_quant = []
    DoD_filt_isol2_morph_quant = []
    DoD_filt_ult_morph_quant = []

    ###########################################################################
    # CHECK DEMs SHAPE
    ###########################################################################
    # Due to differences between DEMs shape (not the same ScanArea.txt laser survey file)
    # a preliminary loop over the all DEMs is required in order to define the target
    # dimension of the reshaping operation
    array_dim_x = []
    array_dim_y = []
    survey_nb=0

    for f in files:
        
        path_DEM = os.path.join(input_dir+str(survey_nb), f)
        survey_nb= survey_nb + 1
        DEM = np.loadtxt(path_DEM,
                          # delimiter=',',
                          skiprows=8
                          )
        array_dim_x = np.append(array_dim_x, DEM.shape[0])
        array_dim_y = np.append(array_dim_y, DEM.shape[1])

    # Define target dimension:
    shp_target_x, shp_target_y = int(min(array_dim_x)), int(min(array_dim_y))

    arr_shape = np.array([shp_target_x, shp_target_y]) # Define target shape


    ###########################################################################
    # SETUP MASKS
    ###########################################################################
    # array mask for filtering data outside the channel domain
    # Different mask will be applied depending on the run due to different ScanArea
    # used during the laser surveys
    runs_list = ['q10_1', 'q10_2', 'q15_1', 'q20_1', 'q20_2'] # Old runs with old ScanArea
    array_mask_name, array_mask_path = 'mask_borders_'+run+'.txt', os.path.join(home_dir, run) # Mask for runs 07 onwards

    if run in runs_list:
        array_mask_name, array_mask_path = 'array_mask_0.txt', home_dir
        print(array_mask_name)


    # Load mask
    array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask_name),skiprows=8)
    # Reshape mask:
    array_mask_rshp = array_mask[:shp_target_x,:shp_target_y] # Array mask reshaped

    # Create array mask:
    # - array_mask: np.array with 0 and 1
    # - array_mask_nan: np.array with np.nan and 1
    array_mask_rshp_nan = np.where(array_mask_rshp==0, np.nan, 1) # Convert in mask with np.nan and 1

    # Here we can split in two parts the DEMs or keep the entire one
    if mask_mode==1:
        pass
    elif mask_mode==2: # Working downstream, masking upstream
       array_mask_rshp[:,:-int(array_mask_rshp.shape[1]/2)] = NaN
       array_mask_rshp=np.where(array_mask_rshp==NaN, np.nan, array_mask_rshp)

    elif mask_mode==3: # Working upstream, masking downstream
        array_mask_rshp[:,int(array_mask_rshp.shape[1]/2):] = NaN
        array_mask_rshp=np.where(array_mask_rshp==NaN, np.nan, array_mask_rshp)

#%%
    ###########################################################################
    # LOOP OVER ALL DEMs COMBINATIONS
    ###########################################################################
    nn=0
    # Perform difference between DEMs over all the possible combination of surveys in the survey directory
    for h in range (0, len(files)-1):
        for k in range (0, len(files)-1-h):
            print(h)
            nn+=1
            DEM1_name=files[h] # Extract the DEM1 name...
            DEM2_name=files[h+1+k] #...and the DEM2 name
            comb = np.append(comb, DEM2_name + '-' + DEM1_name) # Create a list with all the available combinations of DEMs

            # Overwrite DEM1 and DEM2 names in case of single DoD analysis
            if process_mode==1:
                pass
            elif process_mode==2:
                DEM1_name = DEM1_single_name
                DEM2_name = DEM2_single_name

            # Create DEMs paths...
            path_DEM1 = os.path.join(input_dir+str(h), DEM1_name)
            path_DEM2 = os.path.join(input_dir+str(h+1+k), DEM2_name)
            
            # ...and DOD name. The DoD name extraction depends by the length of
            # the DoD name sice for runs with more than 10 surveys the DEM's name is larger  
            if len(DEM1_name)==int(len(files[0])):
                DEM1_num = DEM1_name[-5:-4]
            elif len(DEM1_name)==int(len(files[0])+1):
                DEM1_num = DEM1_name[-6:-4]
                
            if len(DEM2_name)==int(len(files[0])):
                DEM2_num = DEM2_name[-5:-4]
            elif len(DEM2_name)==int(len(files[0])+1):
                DEM2_num = DEM2_name[-6:-4]
                
            DoD_name = 'DoD_' + DEM2_num + '-' + DEM1_num + '_'
            
            print(run)
            print('=========')
            print(DoD_name[:-1])
            print('=========')
            
            # TODO UPDATE
            delta=int(DEM2_num)-int(DEM1_num) # Calculate delta between DEM
            
            print('delta = ', delta)
            print('----------')
            print()
            
            # # Setup output folder
            # output_name = 'script_outputs_' + DEM2_DoD_name + '-' + DEM1_DoD_name # Set outputs name

            # Set DoD outputs directory where to save DoD as ASCII grid and numpy matrix
            
            if not(os.path.exists(path_out)):
                os.mkdir(path_out)


            ###################################################################
            # DATA LOADING...
            ###################################################################
            # Load DEMs
            DEM1 = np.loadtxt(path_DEM1,
                              # delimiter=',',
                              skiprows=8
                              )
            DEM2 = np.loadtxt(path_DEM2,
                              # delimiter=',',
                              skiprows=8)


            # DEMs reshaping according to arr_shape...
            DEM1=DEM1[0:arr_shape[0], 0:arr_shape[1]]
            DEM2=DEM2[0:arr_shape[0], 0:arr_shape[1]]
            
            # Raster dimension
            dim_y, dim_x = DEM1.shape
            print('dim_x: ', dim_x, '    dim_y: ', dim_y)
            
            ###################################################################
            # HEADER
            ###################################################################
            # Lines array and header array initialization and extraction:
            lines = []
            header = []

            with open(path_DEM1, 'r') as file:
                for line in file:
                    lines.append(line)  # lines is a list. Each item is a row of the input file
                # Header extraction...
                for i in range(0, 7):
                    header.append(lines[i])
            
            # Update header columns and row number:
            header[0] = header[0].replace(header[0][18:25], ' '+str(int(dim_y*px_y))+str('.00'))
            header[1] = header[1].replace(header[1][17:25], '    '+str('0.00'))
            header[2] = header[2].replace(header[2][17:25], str(int(dim_x*px_x))+str('.00'))
            header[3] = header[3].replace(header[3][17:25], '    '+str('0.00'))
            header[4] = header[4].replace(header[4][22:25], str(dim_y))
            header[5] = header[5].replace(header[5][22:25], str(dim_x))
            
            # Header printing in a file txt called header.txt
            with open(path_out + '/' + DoD_name + 'header.txt', 'w') as head:
                head.writelines(header)
            
            ###################################################################
            # PERFORM DEM OF DIFFERENCE - DEM2-DEM1
            ###################################################################
            # Print DoD name
            print(DEM2_name, '-', DEM1_name)

            # Calculate the DoD length in meters:
            DoD_length = DEM1.shape[1]*px_x/1000 # DoD length [m]
            
            # DoD CREATION:
            # Creating DoD array with np.nan instead of NaN
            DoD_raw = np.zeros(DEM1.shape)
            DoD_raw = np.where(np.logical_or(DEM1 == NaN, DEM2 == NaN), np.nan, DEM2 - DEM1)
        
            
            # Masking with array mask
            DoD_raw = DoD_raw*array_mask_rshp_nan
            # Scale array for plotting
            DoD_raw_plot = rescaling_plot(DoD_raw)
            
            # Creating GIS readable DoD array (np.nan as -999)
            DoD_raw_gis = np.zeros(DoD_raw.shape)
            DoD_raw_gis = np.where(np.isnan(DoD_raw), NaN, DoD_raw)
            


            # Count the number of pixels in the channel area
            DoD_count = np.count_nonzero(np.where(np.isnan(DoD_raw), 0, 1))
            print('Number of channel pixel pixels:', DoD_count)
            
            # Append for each DoD the number of active pixels to the DoD_act_px_count_array
            # DoD_act_px_count_array = np.append(DoD_act_px_count_array, DoD_count)



            ###################################################################
            # DATA FILTERING...
            ###################################################################
            
            
            # 1- PERFORM DOMAIN-WIDE WEIGHTED AVERAGE:
            # -------------------------------------
            
            # kernel=np.array([[1],
            #                  [1],
            #                  [2],
            #                  [1],
            #                  [1]])
            
            # kernel=np.array([[1],
            #                   [1],
            #                   [1],
            #                   [1],
            #                   [1]])
            
            kernel=np.array([[1],
                              [2],
                              [1]])
            
            # kernel=np.array([[1],
            #                   [1],
            #                   [1]])
            
            ker=kernel/np.sum(kernel)
            
            DoD_filt_mean=cv2.filter2D(src=DoD_raw,ddepth=-1, kernel=ker)
            DoD_filt_mean_gis = np.where(np.isnan(DoD_filt_mean), NaN, DoD_filt_mean)
            DoD_filt_mean_plot = rescaling_plot(DoD_filt_mean)
            
            
            # 2- PERFORM UNDER THRESHOLD ZEROING:
            DoD_filt_thrs = np.where(np.abs(DoD_filt_mean)<=thrs_1, 0, DoD_filt_mean)
            DoD_filt_thrs_gis = np.where(np.isnan(DoD_filt_thrs), NaN, DoD_filt_thrs)
            # Scale array for plotting
            DoD_filt_thrs_plot = rescaling_plot(DoD_filt_thrs)
            
            
            # PLOT A CROSS-SECTION BEFORE AND AFTER THE AVERAGE PROCESS
            plot1 = plt.plot(DoD_raw[:,50], label = 'raw')
            plot2 = plt.plot(DoD_filt_mean[:,50], label = 'avg')
            plot3 = plt.plot(DoD_filt_thrs[:,50], label = 'thrs')
            # Set title and show the plot
           # plt.title(run + ' - ' + DoD_name)
           # plt.legend()
            # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'plot.png'), dpi=600 )
            #plt.savefig(os.path.join(DoDs_plot, DoD_name + 'section_plot.pdf'), dpi=600 )
           # plt.show()
            
            
            # 3- PERFORM ISOLATED PIXEL REMOVAL:
            #--------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero in the DoD_filt_mean matrix
            counter0 = np.count_nonzero(DoD_filt_thrs[np.logical_not(np.isnan(DoD_filt_thrs))])
            
            # Perform the forst iteration
            DoD_filt_isol = remove_small_objects(DoD_filt_thrs, 10, 2) # take into account diagonal pixels for connectivity
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol matrix
            counter1 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore 
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol = remove_small_objects(DoD_filt_isol, 10, 2) # take into account diagonal pixels for connectivity
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
            
            # Convert matrix to be GIS-readable
            DoD_filt_isol_gis = np.where(np.isnan(DoD_filt_isol), NaN, DoD_filt_isol)
            
            # Scale array for plotting
            DoD_filt_isol_plot = rescaling_plot(DoD_filt_isol)

            
            # 4- PERFORM PITTS FILLING PROCEDURE:
            #---------------------------------------
            # Initialize the counter from the previous step of the filtering process
            counter0 = np.count_nonzero(DoD_filt_isol[np.logical_not(np.isnan(DoD_filt_isol))])
            # Perform the first step of the filling procedure
            DoD_filt_fill, matrix_target = fill_small_holes(DoD_filt_isol, avg_target_kernel=5, area_threshold=8, connectivity=1, filt_threshold=thrs_1)
            # Calculate the current counter
            counter1 = np.count_nonzero(DoD_filt_fill[np.logical_not(np.isnan(DoD_filt_fill))])
            # Perform the loop of the filtering process
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_fill, matrix_target = fill_small_holes(DoD_filt_fill, avg_target_kernel=5, area_threshold=8, connectivity=1, filt_threshold=thrs_1)
                # Update counters:
                counter0=counter1
                counter1 = np.count_nonzero(DoD_filt_fill[np.logical_not(np.isnan(DoD_filt_fill))])   
            
            # Convert matrix to be GIS-readable
            DoD_filt_fill_gis = np.where(np.isnan(DoD_filt_fill), NaN, DoD_filt_fill)
                
            # Scale array for plotting
            DoD_filt_fill_plot = rescaling_plot(DoD_filt_fill)
            
            
            # 5- PERFORM NATURE CHECKER PIXEL PROCEDURE:
            #----------------------------------------
            # TODO THIS IS NOT ACTIVE!!
            DoD_filt_nature, DoD_filt_nature_gis = nature_checker_resolution(DoD_filt_fill, thrs_nature, 1, NaN)
            DoD_filt_nature_plot = rescaling_plot(DoD_filt_nature)
            
            
            # 6- RE-PERFORM ISOLATED PIXEL REMOVAL:
            #-----------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero o fthe DoD_filt_fill matrix
            counter0 = np.count_nonzero(DoD_filt_nature[np.logical_not(np.isnan(DoD_filt_nature))])
            
            # Perform the very first isolated_killer procedure
            DoD_filt_isol2 = remove_small_objects(DoD_filt_nature, 30, 2)
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol2 matrix
            counter1 = np.count_nonzero(DoD_filt_isol2[np.logical_not(np.isnan(DoD_filt_isol2))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore

            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol2 = remove_small_objects(DoD_filt_isol2, 30, 2)
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol2[np.logical_not(np.isnan(DoD_filt_isol2))])

            DoD_filt_isol2_gis = np.where(np.isnan(DoD_filt_isol2), NaN, DoD_filt_isol2)
            
            DoD_filt_isol2_plot = rescaling_plot(DoD_filt_isol2)
            
            
            
            # 6- PERFORM ISOLATED PIXEL REMOVAL:
            #-----------------------------------------------------
            DoD_filt_isol3      = test(DoD_filt_isol2)
            DoD_filt_isol3_gis  = np.where(np.isnan(DoD_filt_isol3), NaN, DoD_filt_isol3)
            DoD_filt_isol3_plot = rescaling_plot(DoD_filt_isol3)
            
            
            
            # 7- RE-PERFORM ISOLATED ANISOTROPIC PIXEL REMOVAL:
            #-----------------------------------------------------
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero o fthe DoD_filt_fill matrix
            counter0 = np.count_nonzero(DoD_filt_isol3[np.logical_not(np.isnan(DoD_filt_isol3))])
            
            # Perform the first iteration
            DoD_filt_isol4 = test2(DoD_filt_isol3)
            
            # After trimming al np.nan values, counter represent the number of
            # pixel not equal to zero of the DoD_filt_isol2 matrix
            counter1 = np.count_nonzero(DoD_filt_isol4[np.logical_not(np.isnan(DoD_filt_isol4))])
            # Perform the isolated_killer procedure until the number of active
            # pixel will not change anymore
            
            while counter0-counter1!=0:
                # Filtering...
                DoD_filt_isol4 = test2(DoD_filt_isol3)
                # Update counters:
                counter0 = counter1
                counter1 = np.count_nonzero(DoD_filt_isol4[np.logical_not(np.isnan(DoD_filt_isol4))])
                
            DoD_filt_ult = DoD_filt_isol4
            
            # Set all the value between +/- 2mm at 2mm and keep zero as zero
            DoD_filt_ult = np.where(np.logical_and(abs(DoD_filt_isol4)<thrs_1, DoD_filt_isol4!=0), thrs_1,DoD_filt_isol4)
            
            # DoD_filt_ult       = test2(DoD_filt_isol4)
            # DoD_filt_ult = np.where(DoD_filt_ult>0, remove_small_objects(DoD_filt_ult>0, 40, 1), DoD_filt_ult)
            # DoD_filt_ult = np.where(DoD_filt_ult<0, remove_small_objects(DoD_filt_ult<0, 40, 1), DoD_filt_ult)
            # DoD_filt_ult = DoD_filt_ult*DoD_filt_isol4
            DoD_filt_ult_gis   = np.where(np.isnan(DoD_filt_ult), NaN, DoD_filt_ult)
            
            
            # Scale array real size
            DoD_filt_ult_plot = rescaling_plot(DoD_filt_ult)
            
            
            ################
            # Filtering process visual check
            ################
            
            if 'DoD_plot' in plot_mode:
            
                img2 = plt.imshow(DoD_raw_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_mean_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'raw-thrs.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'raw-mean.pdf'), dpi=2000)
                plt.show()
                
                img2 = plt.imshow(DoD_filt_mean_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_thrs_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'raw-thrs.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'mean_thrs.pdf'), dpi=2000)
                plt.show()
                
                img2 = plt.imshow(DoD_filt_thrs_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_isol_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'thrs-isol.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'thrs-isol.pdf'), dpi=2000)
                plt.show()
                
                img2 = plt.imshow(DoD_filt_isol_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_fill_plot, cmap='RdBu', alpha=0.5, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol-fill.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol-fill.pdf'), dpi=2000)
                plt.show()
                
                img2 = plt.imshow(DoD_filt_fill_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_nature_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'fill-nature.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'fill-nature.pdf'), dpi=2000)
                plt.show()
     
                img2 = plt.imshow(DoD_filt_nature_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_isol2_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'nature-isol2.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'nature-isol2.pdf'), dpi=2000)
                plt.show()
                
                img2 = plt.imshow(DoD_filt_isol2_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_isol3_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol2-isol3.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol2-isol3.pdf'), dpi=2000)
                plt.show()
                
                img2 = plt.imshow(DoD_filt_isol3_plot, cmap='binary', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                img1 = plt.imshow(DoD_filt_ult_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                # plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol3-isol4.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + 'isol3-ult.pdf'), dpi=2000)
                plt.show()
                
                
                
    
                
                # Plot the result for double check
                img2 = plt.imshow(DoD_filt_thrs_plot, cmap='binary', alpha=0.4, vmin=-20, vmax=+20)
                # img1 = plt.imshow(np.where(DoD_filt_ult_plot==0, np.nan, DoD_filt_ult_plot), cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20)
                img1 = plt.imshow(DoD_filt_ult_plot, cmap='RdBu', alpha=1.0, vmin=-20, vmax=+20, interpolation_stage='rgba')
    
                # Set title and show the plot
                plt.title(run + ' - ' + DoD_name[:-1])
                plt.axis('off')
                plt.savefig(os.path.join(DoDs_plot, DoD_name + '_plot.png'), dpi=2000)
                plt.savefig(os.path.join(DoDs_plot, DoD_name + '_plot.pdf'), dpi=2000)
            
            

            ###################################################################
            # TOTAL VOLUMES, DEPOSITION VOLUMES AND SCOUR VOLUMES
            ###################################################################
            # DoD filtered name: DoD_filt
            # Create new raster where apply volume calculation
            # DoD>0 --> Deposition, DoD<0 --> Scour
            # =+SUMIFS(A1:JS144, A1:JS144,">0")*5*50(LibreCalc function)
            
            
            # SCOUR AND DEPOSITION MATRIX, DEPOSITION ONLY MATRIX AND SCOUR ONLY MATRIX:
            DoD_vol = np.where(np.isnan(DoD_filt_ult), 0, DoD_filt_ult) # Total volume matrix
            dep_DoD = (DoD_vol>0)*DoD_vol # Deposition only matrix
            sco_DoD = (DoD_vol<0)*DoD_vol # Scour only matrix
            
            # ...as boolean active pixel matrix:
            act_px_matrix = np.where(DoD_filt_ult!=0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active pixel matrix, both scour and deposition
            act_px_matrix_dep = np.where(DoD_filt_ult>0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active deposition matrix 
            act_px_matrix_sco = np.where(DoD_filt_ult<0, 1, 0)*np.where(np.isnan(DoD_filt_ult), np.nan, 1) # Active scour matrix
            
            # GIS readable matrix where np.nan is NaN
            act_px_matrix_gis = np.where(np.isnan(act_px_matrix), NaN, act_px_matrix) # Active pixel matrix, both scour and deposition
            act_px_matrix_dep_gis = np.where(np.isnan(act_px_matrix_dep), NaN, act_px_matrix_dep) # Active deposition matrix 
            act_px_matrix_sco_gis = np.where(np.isnan(act_px_matrix_sco), NaN, act_px_matrix_sco) # Active scour matrix

#%%         ###################################################################
            # MORPHOLOGICAL QUANTITIES:
            ###################################################################
            
            tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_ult)
            
            ###################################################################
            # Filtering process stage analysis
            if filt_analysis == 1:
                # Analysis to investigate the role of the application of spatial filters at the DoD
                # in the morphological changes calculation
                if delta==1:
                    # DoD_raw
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_raw)
                    if len(DoD_raw_morph_quant)==0:
                        DoD_raw_morph_quant=np.append(DoD_raw_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_raw_morph_quant=np.vstack((DoD_raw_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_mean
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_mean)
                    if len(DoD_filt_mean_morph_quant)==0:
                        DoD_filt_mean_morph_quant=np.append(DoD_filt_mean_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_mean_morph_quant=np.vstack((DoD_filt_mean_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_isol
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_isol)
                    if len(DoD_filt_isol_morph_quant)==0:
                        DoD_filt_isol_morph_quant=np.append(DoD_filt_isol_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_isol_morph_quant=np.vstack((DoD_filt_isol_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_fill
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_fill)
                    if len(DoD_filt_fill_morph_quant)==0:
                        DoD_filt_fill_morph_quant=np.append(DoD_filt_fill_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_fill_morph_quant=np.vstack((DoD_filt_fill_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_nature
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_nature)
                    if len(DoD_filt_nature_morph_quant)==0:
                        DoD_filt_nature_morph_quant=np.append(DoD_filt_nature_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_nature_morph_quant=np.vstack((DoD_filt_nature_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_isol2
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_isol2)
                    if len(DoD_filt_isol2_morph_quant)==0:
                        DoD_filt_isol2_morph_quant=np.append(DoD_filt_isol2_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_isol2_morph_quant=np.vstack((DoD_filt_isol2_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                    
                    # DoD_filt_ult
                    tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri = morph_quantities(DoD_filt_ult)
                    if len(DoD_filt_ult_morph_quant)==0:
                        DoD_filt_ult_morph_quant=np.append(DoD_filt_ult_morph_quant, (tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))
                    else:
                        DoD_filt_ult_morph_quant=np.vstack((DoD_filt_ult_morph_quant, np.hstack((tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri))))
                


            

            ###################################################################
            # STACK CONSECUTIVE DoDS IN A 3D ARRAY
            ###################################################################
            # Initialize 3D array to stack DoDs
            if h==0 and k==0: # initialize the first array with the DEM shape
                # stack["DoD_stack_{0}".format(run, delta)] = np.zeros([len(files)-1, dim_y, dim_x])
                DoD_stack = np.ones([len(files)-1, dim_y, dim_x, len(files)-1])*NaN
              # DoD_stack[time, X, Y, delta]
            else:
                pass
            
            
            # STACK ALL THE DoDS INSIDE THE 3D ARRAY
            DoD_stack[h,:,:, delta-1] = DoD_filt_ult[:,:]
            DoD_stack = np.where(DoD_stack==NaN, np.nan, DoD_stack)
            
            
            
            # CREATE STACK BOOL
            DoD_stack_bool = np.where(DoD_stack>0, 1, DoD_stack)
            DoD_stack_bool = np.where(DoD_stack_bool<0, -1, DoD_stack_bool)
            
            '''
            DoD input stack structure:
                
                DoD_stack[time,x,y,delta]
                DoD_stack_bool[time,x,y,delta]
                
                - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >    delta
                |  DoD 1-0  DoD 2-0  DoD 3-0  DoD 4-0  DoD 5-0  DoD 6-0  DoD 7-0  DoD 8-0  DoD 9-0
                |  DoD 2-1  DoD 3-1  DoD 4-1  DoD 5-1  DoD 6-1  DoD 7-1  DoD 8-1  DoD 9-1
                |  DoD 3-2  DoD 4-2  DoD 5-2  DoD 6-2  DoD 7-2  DoD 8-2  DoD 9-2
                |  DoD 4-3  DoD 5-3  DoD 6-3  DoD 7-3  DoD 8-3  DoD 9-3
                |  DoD 5-4  DoD 6-4  DoD 7-4  DoD 8-4  DoD 9-4
                |  DoD 6-5  DoD 7-5  DoD 8-5  DoD 9-5
                |  DoD 7-6  DoD 8-6  DoD 9-6
                |  DoD 8-7  DoD 9-7
                |  DoD 9-8
                |
                v
                time
                    
            '''
            

            ###################################################################
            # DoDs SAVING...
            ###################################################################

            
            # SAVE DOD FILES...
            
            # RAW DoD
            # Print raw DoD in txt file (NaN as np.nan)
            np.savetxt(os.path.join(path_out, DoD_name + 'raw.txt'), DoD_raw, fmt='%0.1f', delimiter='\t')
            # Printing raw DoD in txt file (NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'raw_gis.txt'), DoD_raw_gis, fmt='%0.1f', delimiter='\t')

            # WEIGHTED AVERAGED DoD
            # Print DoD mean in txt file (NaN as np.nan)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_mean.txt'), DoD_filt_mean , fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_mean_gis.txt'), DoD_filt_mean_gis , fmt='%0.1f', delimiter='\t')

            # ISOLATE KILLER FUNCTION APPLIED DoD
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol.txt'), DoD_filt_isol, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol_gis.txt'), DoD_filt_isol_gis, fmt='%0.1f', delimiter='\t')
            
            # NATURE CHECKER FUNCTION APPLIED DoD
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_nature.txt'), DoD_filt_nature, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt'), DoD_filt_nature_gis, fmt='%0.1f', delimiter='\t')
            
            # ISOLATE FILLER FUNCTION APPLIED DoD
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_fill.txt'), DoD_filt_fill, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_fill_gis.txt'), DoD_filt_fill_gis, fmt='%0.1f', delimiter='\t')

            # SECOND ROUND OF ISOLATE KILLER FUNCTION APPLIED DoD (This is the ultimate DoD)
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol2.txt'), DoD_filt_isol2, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_isol2_gis.txt'), DoD_filt_isol2_gis, fmt='%0.1f', delimiter='\t')
            
            # ISLAND KILLER FUNCTION APPLIED DoD (...)
            # Print filtered DoD (with np.nan)...
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_ult.txt'), DoD_filt_ult, fmt='%0.1f', delimiter='\t')
            # Print filtered DoD (with NaN as -999)
            np.savetxt(os.path.join(path_out, DoD_name + 'filt_ult_gis.txt'), DoD_filt_ult_gis, fmt='%0.1f', delimiter='\t')

            # ACTIVE PIXEL DoD
            # Print boolean map of active pixel: 1=active, 0=not active
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map.txt'), act_px_matrix, fmt='%0.1f', delimiter='\t')
            # Print boolean GIS readable map of active pixel as above (np.nan is NaN)
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_gis.txt'), act_px_matrix_gis , fmt='%0.1f', delimiter='\t')
            
            # ACTIVE DEPOSITION PIXEL DoD
            # Print boolean map of active pixel: 1=deposition, 0=not active or scour
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_dep.txt'), act_px_matrix_dep, fmt='%0.1f', delimiter='\t')
            # Print boolean GIS readable map of active pixel as above (np.nan is NaN)
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_dep_gis.txt'), act_px_matrix_dep_gis , fmt='%0.1f', delimiter='\t')
            
            # ACTIVE SCOUR PIXEL DoD
            # Print boolean map of active pixel: 1=scour, 0=not active or deposition
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_sco.txt'), act_px_matrix_sco, fmt='%0.1f', delimiter='\t')
            # Print boolean GIS readable map of active pixel as above (np.nan is NaN)
            np.savetxt(os.path.join(path_out, DoD_name + 'activity_map_sco_gis.txt'), act_px_matrix_sco_gis, fmt='%0.1f', delimiter='\t')


            # Print DoD and filtered DoD (with NaN as -999) in a GIS readable format (ASCII grid):
            with open(os.path.join(path_out, DoD_name + 'header.txt')) as f_head:
                w_header = f_head.read()    # Header
            with open(os.path.join(path_out, DoD_name + 'raw_gis.txt')) as f_DoD:
                w_DoD_raw= f_DoD.read()
            with open(os.path.join(path_out, DoD_name + 'filt_mean_gis.txt')) as f_DoD_mean:
                w_DoD_mean = f_DoD_mean.read()
            with open(os.path.join(path_out, DoD_name + 'filt_isol_gis.txt')) as f_DoD_isol:
                w_DoD_isol = f_DoD_isol.read()
            with open(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt')) as f_DoD_nature:
                w_DoD_nature = f_DoD_nature.read()
            with open(os.path.join(path_out, DoD_name + 'filt_fill_gis.txt')) as f_DoD_fill:
                w_DoD_fill = f_DoD_fill.read()
            with open(os.path.join(path_out, DoD_name + 'filt_isol2_gis.txt')) as f_DoD_isol2:
                w_DoD_isol2 = f_DoD_isol2.read()
            with open(os.path.join(path_out, DoD_name + 'filt_ult_gis.txt')) as f_DoD_ult:
                w_DoD_ult = f_DoD_ult.read()
            with open(os.path.join(path_out, DoD_name + 'activity_map_gis.txt')) as f_DoD_act_map:
                w_DoD_act_map = f_DoD_act_map.read()
            with open(os.path.join(path_out, DoD_name + 'activity_map_dep_gis.txt')) as f_DoD_act_map_dep:
                w_DoD_act_map_dep = f_DoD_act_map_dep.read()
            with open(os.path.join(path_out, DoD_name + 'activity_map_sco_gis.txt')) as f_DoD_act_map_sco:
                w_DoD_act_map_sco = f_DoD_act_map_sco.read()

                # Print GIS readable raster
                DoD_raw_gis = w_header + w_DoD_raw
                DoD_mean_gis = w_header + w_DoD_mean
                DoD_isol_gis = w_header + w_DoD_isol
                DoD_nature_gis = w_header + w_DoD_nature
                DoD_fill_gis = w_header + w_DoD_fill
                DoD_isol2_gis = w_header + w_DoD_isol2
                DoD_ult_gis = w_header + w_DoD_ult
                DoD_act_map_gis = w_header + w_DoD_act_map
                DoD_act_map_dep_gis = w_header + w_DoD_act_map_dep
                DoD_act_map_sco_gis = w_header + w_DoD_act_map_sco

            with open(os.path.join(path_out, DoD_name + 'raw_gis.txt'), 'w') as fp:
                fp.write(DoD_raw_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_mean_gis.txt'), 'w') as fp:
                fp.write(DoD_mean_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_isol_gis.txt'), 'w') as fp:
                fp.write(DoD_isol_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_nature_gis.txt'), 'w') as fp:
                fp.write(DoD_nature_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_fill_gis.txt'), 'w') as fp:
                fp.write(DoD_fill_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_isol2_gis.txt'), 'w') as fp:
                fp.write(DoD_isol2_gis)
            with open(os.path.join(path_out, DoD_name + 'filt_ult_gis.txt'), 'w') as fp:
                fp.write(DoD_ult_gis)
            with open(os.path.join(path_out, DoD_name + 'activity_map_gis.txt'), 'w') as fp:
                fp.write(DoD_act_map_gis)
            with open(os.path.join(path_out, DoD_name + 'activity_map_dep_gis.txt'), 'w') as fp:
                fp.write(DoD_act_map_dep_gis)
            with open(os.path.join(path_out, DoD_name + 'activity_map_sco_gis.txt'), 'w') as fp:
                fp.write(DoD_act_map_sco_gis)
    
    ###################################################################
    # DoDs STACK SAVING...
    ###################################################################
    '''
    INPUTS:
        DoD_stack1 : 3D numpy array stack
            Stack on which each 1-step DoD has been saved (with extra domain cell as np.nan)
    OUTPUTS SAVED FILES:
        DoD_stack1 : 3D numpy array stack
            Stack on which DoDs are stored as they are, with np.nan
        DoD_stack1_bool : 3D numpy array stack
            Stack on which DoDs are stored as -1, 0, +1 data, also with np.nan
    '''

    # Save 3D array as binary file
    np.save(os.path.join(DoDs_dir, 'DoDs_stack',"DoD_stack_"+run+".npy"), DoD_stack)
    
    
    # Save 3D "boolean" array as binary file
    np.save(os.path.join(DoDs_dir, 'DoDs_stack',"DoD_stack_bool_"+run+".npy"), DoD_stack_bool)


    # Fill DoD lenght array
    DoD_length_array = np.append(DoD_length_array, DoD_length)



    

    if filt_analysis==1:
        n=0
        for matrix in (DoD_raw_morph_quant, DoD_filt_mean_morph_quant, DoD_filt_isol_morph_quant,DoD_filt_fill_morph_quant,DoD_filt_nature_morph_quant,DoD_filt_isol2_morph_quant,DoD_filt_ult_morph_quant):
            n+=1
            matrix_mean = np.nanmean(matrix, axis=0)
            matrix_std = np.nanstd(matrix, axis=0)
            if n==1:
                matrix_stack = np.vstack((matrix_mean, matrix_std))
            else:
                matrix_stack = np.vstack((matrix_stack,matrix_mean, matrix_std))
            np.savetxt(os.path.join(report_dir, 'morph_quant_report.txt'), matrix_stack, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
            
        # Save report for the analysis of the effects of spatial filter application on DoD, at different stages.
        np.savetxt(os.path.join(report_dir, 'DoD_raw_morph_quant.txt'), DoD_raw_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_mean_morph_quant.txt'), DoD_filt_mean_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_isol_morph_quant.txt'), DoD_filt_isol_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_fill_morph_quant.txt'), DoD_filt_fill_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_nature_morph_quant.txt'), DoD_filt_nature_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_isol2_morph_quant.txt'), DoD_filt_isol2_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        np.savetxt(os.path.join(report_dir, 'DoD_filt_ult_morph_quant.txt'), DoD_filt_ult_morph_quant, header='tot_vol, sum_vol, dep_vol, sco_vol, morph_act_area, morph_act_area_dep, morph_act_area_sco, act_width_mean, act_width_mean_dep, act_width_mean_sco, act_thickness, act_thickness_dep, act_thickness_sco, bri')
        
DoD_length_array = np.append(DoD_length_array, DoD_length)


end = time.time()
print()
print('Execution time: ', (end-start), 's')

# --------------------------------------------------
#
#     Copyright (C) {2020} Kevin Bronik
#
#     UCL Medical Physics and Biomedical Engineering
#     https://www.ucl.ac.uk/medical-physics-biomedical-engineering/
#     UCL Queen Square Institute of Neurology
#     https://www.ucl.ac.uk/ion/

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#     {Multi-Label Multi/Single-Class Image Segmentation}  Copyright (C) {2020}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.

# This program uses piece of source code from:
# Title: nicMSlesions
# Author: Sergi Valverde
# Date: 2017
# Code version: 0.2
# Availability: https://github.com/NIC-VICOROB/nicMSlesions


import os
import shutil
import sys
import signal
import subprocess
import time
import platform
import nibabel as nib
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion as ans_dif
from nibabel import load as load_nii
from libs.mclahe import utils
from libs.mclahe.core import *


CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'

def zscore_normalize(img, mask=None):
    """
    normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        mask (nibabel.nifti1.Nifti1Image): brain mask for img
    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_data()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_data()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        mask_data = img_data > img_data.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized

def myflatten(l):

    return myflatten(l[0]) + (myflatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]


def get_set_input_images(current_folder, settings):


    if settings['train_test'] == 'training':
        modalities = settings['input_modality'][:] + ['lesion']
        image_tags = settings['image_tags'][:] + settings['InputLabel'][:]
    else:
        modalities = settings['input_modality'][:]
        image_tags = settings['image_tags'][:]

    if settings['debug']:
        print("> DEBUG:", "number of input sequences to find:", len(modalities))
    scan = settings['input']

    print("> PRE:", scan, "identifying input modalities")

    found_modalities = 0

    masks = [m for m in os.listdir(current_folder) if m.find('.nii') > 0]

    for t, m in zip(image_tags, modalities):

        # check first the input modalities
        # find tag

        found_mod = [mask.find(t) if mask.find(t) >= 0
                     else np.Inf for mask in masks]

        if found_mod[np.argmin(found_mod)] is not np.Inf:
            found_modalities += 1
            index = np.argmin(found_mod)
            # generate a new output image modality
            # check for extra dimensions
            input_path = os.path.join(current_folder, masks[index])
            input_sequence = nib.load(input_path)
            input_image = np.squeeze(input_sequence.get_data())
            output_sequence = nib.Nifti1Image(input_image,
                                              affine=input_sequence.affine)
            output_sequence.to_filename(
                os.path.join(settings['tmp_folder'], m + '.nii.gz'))
            if m == 'FLAIR':
                images = nib.load(os.path.join(settings['tmp_folder'], m + '.nii.gz'))

                data = images.get_data()
                # data = mclahe(data)

                ni_img = nib.Nifti1Image(data, images.affine, images.header)
                print(CYELLOW + "Replacing the", CRED + m + CEND,
                      CGREEN + "with a new enhanced version of it in tmp folder" + CEND)

                ni_img.to_filename(os.path.join(settings['tmp_folder'], m + '.nii.gz'))

            if settings['debug']:
                print("    --> ", masks[index], "as", m, "image")
            masks.remove(masks[index])

    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
        print("> ERROR:", scan, \
            "does not contain all valid input modalities")
        sys.stdout.flush()
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)


def register_masks(settings):
    """
    - to doc
    - moving all images to the MPRAGE+192 space

    """

    scan = settings['input']
    # rigid registration
    os_host = platform.system()
    if os_host == 'Windows':
        reg_exe = 'reg_aladin.exe'
    elif os_host == 'Linux' or 'Darwin':
        reg_exe = 'reg_aladin'
    else:
        print("> ERROR: The OS system", os_host, "is not currently supported.")
    reg_aladin_path=''

    if os_host == 'Windows':
          reg_aladin_path = os.path.join(settings['niftyreg_path'], reg_exe)
    elif os_host == 'Linux':
          reg_aladin_path = os.path.join(settings['niftyreg_path'], reg_exe)
    elif os_host == 'Darwin':
          reg_aladin_path = reg_exe
    else:
          print('Please install first  NiftyReg in your mac system and try again!')
          sys.stdout.flush()
          time.sleep(1)
          os.kill(os.getpid(), signal.SIGTERM)




    print ('running ....> ',reg_aladin_path)
    if settings['reg_space'] == 'FlairtoT1':
        for mod in settings['input_modality']:
            if mod == 'T1':
                continue

            try:
                print("> PRE:", scan, "registering",  mod, " --> T1 space")

                subprocess.check_output([reg_aladin_path, '-ref',
                                         os.path.join(settings['tmp_folder'], 'T1.nii.gz'),
                                         '-flo', os.path.join(settings['tmp_folder'], mod + '.nii.gz'),
                                         '-aff', os.path.join(settings['tmp_folder'], mod + '_transf.txt'),
                                         '-res', os.path.join(settings['tmp_folder'], 'r' + mod + '.nii.gz')])
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    # if training, the lesion mask is also registered through the T1 space.
    # Assuming that the refefence lesion space was FLAIR.
    if settings['reg_space'] == 'FlairtoT1':
        if settings['train_test'] == 'training':
            # rigid registration
            os_host = platform.system()
            if os_host == 'Windows':
                reg_exe = 'reg_resample.exe'
            elif os_host == 'Linux' or 'Darwin':
                reg_exe = 'reg_resample'
            else:
                print("> ERROR: The OS system", os_host, "is not currently supported.")

            reg_resample_path = ''

            if os_host == 'Windows':
                reg_resample_path = os.path.join(settings['niftyreg_path'], reg_exe)
            elif os_host == 'Linux':
                reg_resample_path = os.path.join(settings['niftyreg_path'], reg_exe)
            elif os_host == 'Darwin':
                reg_resample_path = reg_exe
            else:
                print('Please install first  NiftyReg in your mac system and try again!')
                sys.stdout.flush()
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)
            print('running ....> ', reg_resample_path)


            try:
                print("> PRE:", scan, "resampling the lesion mask --> T1 space")
                subprocess.check_output([reg_resample_path, '-ref',
                                         os.path.join(settings['tmp_folder'], 'T1.nii.gz'),
                                         '-flo', os.path.join(settings['tmp_folder'], 'lesion'),
                                         '-trans', os.path.join(settings['tmp_folder'], 'FLAIR_transf.txt'),
                                         '-res', os.path.join(settings['tmp_folder'], 'lesion.nii.gz'),
                                         '-inter', '0'])
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if settings['reg_space'] == 'T1toFlair':
        for mod in settings['input_modality']:
            if mod == 'FLAIR':
                continue

            try:
                print("> PRE:", scan, "registering", mod, " --> Flair space")

                subprocess.check_output([reg_aladin_path, '-ref',
                                         os.path.join(settings['tmp_folder'], 'FLAIR.nii.gz'),
                                         '-flo', os.path.join(settings['tmp_folder'], mod + '.nii.gz'),
                                         '-aff', os.path.join(settings['tmp_folder'], mod + '_transf.txt'),
                                         '-res', os.path.join(settings['tmp_folder'], 'r' + mod + '.nii.gz')])
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        # if training, the lesion mask is also registered through the T1 space.
        # Assuming that the refefence lesion space was FLAIR.
    if settings['reg_space'] == 'T1toFlair':
        if settings['train_test'] == 'training':
            # rigid registration
            os_host = platform.system()
            if os_host == 'Windows':
                reg_exe = 'reg_resample.exe'
            elif os_host == 'Linux' or 'Darwin':
                reg_exe = 'reg_resample'
            else:
                print("> ERROR: The OS system", os_host, "is not currently supported.")

            reg_resample_path = ''

            if os_host == 'Windows':
                reg_resample_path = os.path.join(settings['niftyreg_path'], reg_exe)
            elif os_host == 'Linux':
                reg_resample_path = os.path.join(settings['niftyreg_path'], reg_exe)
            elif os_host == 'Darwin':
                reg_resample_path = reg_exe
            else:
                print('Please install first  NiftyReg in your mac system and try again!')
                sys.stdout.flush()
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)
            print('running ....> ', reg_resample_path)

            try:
                print("> PRE:", scan, "resampling the lesion mask --> Flair space")
                subprocess.check_output([reg_resample_path, '-ref',
                                         os.path.join(settings['tmp_folder'], 'FLAIR.nii.gz'),
                                         '-flo', os.path.join(settings['tmp_folder'], 'lesion'),
                                         '-trans', os.path.join(settings['tmp_folder'], 'T1_transf.txt'),
                                         '-res', os.path.join(settings['tmp_folder'], 'lesion.nii.gz'),
                                         '-inter', '0'])
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if  settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':
        print("registration to standard space:", settings['reg_space'])
        print('using ....> ', settings['standard_lib'])
        for mod in settings['input_modality']:

            try:
                print("> PRE:", scan, "registering", mod, "--->",  settings['reg_space'])

                subprocess.check_output(['reg_aladin', '-ref',
                                         os.path.join(settings['standard_lib'], settings['reg_space']),
                                         '-flo', os.path.join(settings['tmp_folder'], mod + '.nii.gz'),
                                         '-aff', os.path.join(settings['tmp_folder'], mod + '_transf.txt'),
                                         '-res', os.path.join(settings['tmp_folder'], 'r' + mod + '.nii.gz')])
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if  settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':
        print("resampling the lesion mask ----->:", settings['reg_space'])
        if settings['train_test'] == 'training':
            # rigid registration
            os_host = platform.system()
            if os_host == 'Windows':
                reg_exe = 'reg_resample.exe'
            elif os_host == 'Linux' or 'Darwin':
                reg_exe = 'reg_resample'
            else:
                print("> ERROR: The OS system", os_host, "is not currently supported.")

            reg_resample_path = ''

            if os_host == 'Windows':
                reg_resample_path = os.path.join(settings['niftyreg_path'], reg_exe)
            elif os_host == 'Linux':
                reg_resample_path = os.path.join(settings['niftyreg_path'], reg_exe)
            elif os_host == 'Darwin':
                reg_resample_path = reg_exe
            else:
                print('Please install first  NiftyReg in your mac system and try again!')
                sys.stdout.flush()
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            print('running ....> ', reg_resample_path)

            try:
                print("> PRE:", scan, "resampling the lesion mask -->",settings['reg_space'] )
                subprocess.check_output(['reg_resample', '-ref',
                                         os.path.join(settings['standard_lib'], settings['reg_space']),
                                         '-flo', os.path.join(settings['tmp_folder'], 'lesion'),
                                         '-trans', os.path.join(settings['tmp_folder'], 'FLAIR_transf.txt'),
                                         '-res', os.path.join(settings['tmp_folder'], 'lesion.nii.gz'),
                                         '-inter', '0'])
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

def bias_correction(settings):
    """
    Bias correction of  masks [if large differences, bias correction is needed!]
    Using FSL (https://fsl.fmrib.ox.ac.uk/)

    """
    scan = settings['input']
    if settings['train_test'] == 'training':
         current_folder = os.path.join(settings['training_folder'], scan)
         settings['bias_folder'] = os.path.normpath(os.path.join(current_folder,'bias'))
    else:
        current_folder = os.path.join(settings['inference_folder'], scan)
        settings['bias_folder'] = os.path.normpath(os.path.join(current_folder,'bias'))    
    try:
        # os.rmdir(os.path.join(current_folder,  'tmp'))
        if settings['train_test'] == 'training':
           os.mkdir(settings['bias_folder'])
           print ("bias folder is created for training!")
        else: 
           os.mkdir(settings['bias_folder'])
           print ("bias folder is created for testing!")  
    except:
        if os.path.exists(settings['bias_folder']) is False:
            print("> ERROR:",  scan, "I can not create bias folder for", current_folder, "Quiting program.")

        else:
            pass

                                                              
   
    # os_host = platform.system()
    print('please be sure FSL is installed in your system, or install FSL in your system and try again!')

    it ='--iter=' + str(settings['bias_iter']) 
    smooth = str(settings['bias_smooth'])  
    type = str(settings['bias_type']) 
    
  
    if settings['bias_choice'] == 'All':
        BIAS = settings['input_modality']
    if settings['bias_choice'] == 'FLAIR':
        BIAS = ['FLAIR']
    if settings['bias_choice'] == 'T1':
        BIAS = ['T1']
    if settings['bias_choice'] == 'MOD3':
        BIAS = ['MOD3']  
    if settings['bias_choice'] == 'MOD4':
        BIAS = ['MOD4']              


    for mod in BIAS:

        # current_image = mod + '.nii.gz' if mod == 'T1'\  current_image = mod
            try:
                if settings['debug']:
                   print("> DEBUG: Bias correction ......> ", mod)
                print("> PRE:", scan, "Bias correction of", mod, "------------------------------->")
                input_scan = mod + '.nii.gz' 
            
                shutil.copy(os.path.join(settings['tmp_folder'],
                                         input_scan),
                            os.path.join(settings['bias_folder'],
                                         input_scan))
                                        
                fslm = 'fslmaths'
                ft = 'fast'
                fslsf = 'fslsmoothfill'
                output = subprocess.check_output([fslm, os.path.join(settings['bias_folder'], input_scan),
                                         '-mul', '0', os.path.join(settings['bias_folder'], mod+'lesionmask.nii.gz')], stderr=subprocess.STDOUT)
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod+'lesionmask.nii.gz'),
                                         '-bin', os.path.join(settings['bias_folder'], mod+'lesionmask.nii.gz')])
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod+'lesionmask.nii.gz'),
                                         '-binv', os.path.join(settings['bias_folder'], mod+'lesionmaskinv.nii.gz')])
                 
                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(step one is done!)" + CEND)                                                         


                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], input_scan),
                                         os.path.join(settings['bias_folder'], mod + '_initfast2_brain.nii.gz')])
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod + '_initfast2_brain.nii.gz'), '-bin', 
                                         os.path.join(settings['bias_folder'], mod + '_initfast2_brain_mask.nii.gz')])
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod + '_initfast2_brain.nii.gz'), 
                                         os.path.join(settings['bias_folder'], mod + '_initfast2_restore.nii.gz')]) 
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod + '_initfast2_restore.nii.gz'), '-mas', 
                                         os.path.join(settings['bias_folder'], mod+'lesionmaskinv.nii.gz'), 
                                         os.path.join(settings['bias_folder'], mod + '_initfast2_maskedrestore.nii.gz')]) 

                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(step two is done!)" + CEND) 


                # subprocess.check_output([ft, '-o', os.path.join(settings['bias_folder'], mod+'_fast'), '-l', '20', '-b', '-B', 
                #                          '-t', '1', '--iter=10', '--nopve', '--fixed=0', '-v', 
                #                          os.path.join(settings['bias_folder'], mod + '_initfast2_maskedrestore.nii.gz')])

                subprocess.check_output([ft, '-o', os.path.join(settings['bias_folder'], mod+'_fast'), '-l', smooth, '-b', '-B', 
                                         '-t', type , it , '--nopve', '--fixed=0', '-v', 
                                         os.path.join(settings['bias_folder'], mod + '_initfast2_maskedrestore.nii.gz')])

                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], input_scan), '-div',
                                         os.path.join(settings['bias_folder'], mod + '_fast_restore.nii.gz'), '-mas',
                                         os.path.join(settings['bias_folder'], mod + '_initfast2_brain_mask.nii.gz'),
                                         os.path.join(settings['bias_folder'], mod + '_fast_totbias.nii.gz')])
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod + '_initfast2_brain_mask.nii.gz'), 
                                        '-ero', '-ero', '-ero', '-ero', '-mas', 
                                        os.path.join(settings['bias_folder'], mod+'lesionmaskinv.nii.gz'),
                                        os.path.join(settings['bias_folder'], mod + '_initfast2_brain_mask2.nii.gz')]) 
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod + '_fast_totbias.nii.gz'), '-sub', '1',
                                        os.path.join(settings['bias_folder'], mod + '_fast_totbias.nii.gz')]) 


                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(step three is done!)" + CEND)



                subprocess.check_output([fslsf, '-i', os.path.join(settings['bias_folder'], mod + '_fast_totbias.nii.gz'), '-m',
                                        os.path.join(settings['bias_folder'], mod + '_initfast2_brain_mask2.nii.gz'),'-o',
                                        os.path.join(settings['bias_folder'], mod + '_fast_bias.nii.gz')]) 
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod + '_fast_bias.nii.gz'),'-add', '1',
                                        os.path.join(settings['bias_folder'], mod + '_fast_bias.nii.gz')])  
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], mod + '_fast_totbias.nii.gz'),'-add', '1',
                                        os.path.join(settings['bias_folder'], mod + '_fast_totbias.nii.gz')])  
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], input_scan),'-div', 
                                        os.path.join(settings['bias_folder'], mod + '_fast_bias.nii.gz'),     
                                        os.path.join(settings['bias_folder'], mod + '_biascorr.nii.gz')])
                subprocess.check_output([fslm, os.path.join(settings['bias_folder'], input_scan),'-div', 
                                        os.path.join(settings['bias_folder'], mod + '_fast_bias.nii.gz'),     
                                        os.path.join(settings['bias_folder'], mod + '_biascorr.nii.gz')])
                print(CYELLOW + "Replacing the", CRED + mod  + CEND, CGREEN+ "with a new bias corrected version of it in tmp folder" + CEND)                         

                shutil.copy(os.path.join(settings['bias_folder'], mod + '_biascorr.nii.gz'),
                            os.path.join(settings['tmp_folder'], mod + '.nii.gz'))
                # shutil.copy(os.path.join(settings['bias_folder'], mod + '_biascorr.nii.gz'),
                #             os.path.join(settings['tmp_folder'], 'bc' + mod + '.nii.gz'))              

                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(is completed!)" + CEND)                                                          


         
                                             
            except:
                
                # print("err: '{}'".format(output))
                print("> ERROR:", scan, "Bias correction of  ", mod,  "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)


def denoise_masks(settings):
    """
    Denoise input masks to reduce noise.
    Using anisotropic Diffusion (Perona and Malik)

    """
    # if settings['register_modalities_kind'] != 'FlairtoT1' and  settings['register_modalities_kind'] != 'T1toFlair':
    #     print("registration must be either FlairtoT1 or T1toFlair and not", settings['register_modalities_kind'])
    #     print("> ERROR:", "quiting program.")
    #     sys.stdout.flush()
    #     time.sleep(1)
    #     os.kill(os.getpid(), signal.SIGTERM)

    for mod in settings['input_modality']:

        # current_image = mod + '.nii.gz' if mod == 'T1'\
        #                 else 'r' + mod + '.nii.gz'

        if settings['reg_space'] == 'T1toFlair':
            current_image = mod + '.nii.gz' if mod == 'FLAIR' \
                else 'r' + mod + '.nii.gz'

        if settings['reg_space'] == 'FlairtoT1':
            current_image = mod + '.nii.gz' if mod == 'T1' \
                else 'r' + mod + '.nii.gz'
        if  settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':
            current_image ='r' + mod + '.nii.gz'

        tmp_scan = nib.load(os.path.join(settings['tmp_folder'],
                                         current_image))

        tmp_scan.get_data()[:] = ans_dif(tmp_scan.get_data(),
                                         niter=settings['denoise_iter'])

        tmp_scan.to_filename(os.path.join(settings['tmp_folder'],
                                          'd' + current_image))
        if settings['debug']:
            print("> DEBUG: Denoising ", current_image)


def skull_strip(settings):
    """
    External skull stripping using ROBEX: Run Robex and save skull
    stripped masks
    input:
       - settings: contains the path to input images
    output:
    - None
    """
    # if settings['register_modalities_kind'] != 'FlairtoT1' and  settings['register_modalities_kind'] != 'T1toFlair':
    #     print("registration must be either FlairtoT1 or T1toFlair and not", settings['register_modalities_kind'])
    #     print("> ERROR:", "quiting program.")
    #     sys.stdout.flush()
    #     time.sleep(1)
    #     os.kill(os.getpid(), signal.SIGTERM)

    os_host=platform.system()
    scan = settings['input']
    if settings['reg_space'] == 'FlairtoT1':

            # t1_im = os.path.join(settings['tmp_folder'], 'dT1.nii.gz')
            # t1_st_im = os.path.join(settings['tmp_folder'], 'T1_tmp.nii.gz')
            t1_im = os.path.join(settings['tmp_folder'], 'drFLAIR.nii.gz')
            t1_st_im = os.path.join(settings['tmp_folder'], 'FLAIR_tmp.nii.gz')

            try:
                print("> PRE:", scan, "skull_stripping the Flair modality")
                if os_host == 'Windows':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    subprocess.check_output([settings['robex_path'],
                                             t1_im,
                                             t1_st_im])
                elif os_host == 'Linux':
                    print("skull_stripping the Flair modality on ",os_host, "system")
                    # out = subprocess.check_output(["bash", settings['robex_path'],
                    #                           t1_im,
                    #                           t1_st_im])
                    bash = 'bash'
                    arg1 = str(bash) + '  ' + str(settings['robex_path'])+'  ' + str(t1_im)+'  ' + str(t1_st_im)
                    print(arg1)
                    os.system(arg1)


                elif os_host == 'Darwin':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    subprocess.check_output([bet,
                                             t1_im,
                                             t1_st_im, '-R', '-S', '-B'])
                else:
                    print('Please install first  FSL in your mac system and try again!')
                    sys.stdout.flush()
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGTERM)

            except:
                print("> ERROR:", scan, "registering masks, quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            brainmask = nib.load(t1_st_im).get_data() > 1
            for mod in settings['input_modality']:

                if mod == 'FLAIR':
                    continue

                # apply the same mask to the rest of modalities to reduce
                # computational time

                print('> PRE: ', scan, 'Applying skull mask to ', mod, 'image')
                current_mask = os.path.join(settings['tmp_folder'],
                                            'dr' + mod + '.nii.gz')
                current_st_mask = os.path.join(settings['tmp_folder'],
                                               mod + '_tmp.nii.gz')

                mask = nib.load(current_mask)
                mask_nii = mask.get_data()
                mask_nii[brainmask == 0] = 0
                mask.get_data()[:] = mask_nii
                mask.to_filename(current_st_mask)



    if settings['reg_space'] == 'T1toFlair':


            # t1_im = os.path.join(settings['tmp_folder'], 'dT1.nii.gz')
            # t1_st_im = os.path.join(settings['tmp_folder'], 'T1_tmp.nii.gz')
            t1_im = os.path.join(settings['tmp_folder'], 'drFLAIR.nii.gz')
            t1_st_im = os.path.join(settings['tmp_folder'], 'FLAIR_tmp.nii.gz')

            try:
                print("> PRE:", scan, "skull_stripping the Flair modality")
                if os_host == 'Windows':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    subprocess.check_output([settings['robex_path'],
                                             t1_im,
                                             t1_st_im])
                elif os_host == 'Linux':
                    print("skull_stripping the Flair modality on ",os_host, "system")
                    # out = subprocess.check_output(["bash", settings['robex_path'],
                    #                           t1_im,
                    #                           t1_st_im])
                    bash = 'bash'
                    arg1 =str(bash) + '  ' + str(settings['robex_path'])+'  ' + str(t1_im)+'  ' + str(t1_st_im)
                    print(arg1)
                    os.system(arg1)
                    # print(settings['robex_path'])
                    # p = subprocess.Popen([settings['robex_path'],
                    #                          t1_im,
                    #                          t1_st_im], stdout=subprocess.PIPE)

                    # print(p.communicate())

                elif os_host == 'Darwin':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    subprocess.check_output([bet,
                                             t1_im,
                                             t1_st_im, '-R', '-S', '-B'])
                else:
                    print('Please install first  FSL in your mac system and try again!')
                    sys.stdout.flush()
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGTERM)

            except:
                print("> ERROR:", scan, "registering masks, quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            brainmask = nib.load(t1_st_im).get_data() > 1
            for mod in settings['input_modality']:

                if mod == 'FLAIR':
                    continue

                # apply the same mask to the rest of modalities to reduce
                # computational time

                print('> PRE: ', scan, 'Applying skull mask to ', mod, 'image')
                current_mask = os.path.join(settings['tmp_folder'],
                                            'dr' + mod + '.nii.gz')
                current_st_mask = os.path.join(settings['tmp_folder'],
                                               mod + '_tmp.nii.gz')

                mask = nib.load(current_mask)
                mask_nii = mask.get_data()
                mask_nii[brainmask == 0] = 0
                mask.get_data()[:] = mask_nii
                mask.to_filename(current_st_mask)

    if  settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':    

            # t1_im = os.path.join(settings['tmp_folder'], 'dT1.nii.gz')
            # t1_st_im = os.path.join(settings['tmp_folder'], 'T1_tmp.nii.gz')
            t1_im = os.path.join(settings['tmp_folder'], 'drFLAIR.nii.gz')
            t1_st_im = os.path.join(settings['tmp_folder'], 'FLAIR_tmp.nii.gz')

            try:
                print("> PRE:", scan, "skull_stripping the Flair modality")
                if os_host == 'Windows':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    subprocess.check_output([settings['robex_path'],
                                             t1_im,
                                             t1_st_im])
                elif os_host == 'Linux':
                    print("skull_stripping the Flair modality on ",os_host, "system")
                    # out = subprocess.check_output(["bash", settings['robex_path'],
                    #                           t1_im,
                    #                           t1_st_im])
                    bash = 'bash'
                    arg1 =str(bash) + '  ' + str(settings['robex_path'])+'  ' + str(t1_im)+'  ' + str(t1_st_im)
                    print(arg1)
                    os.system(arg1)

                elif os_host == 'Darwin':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    subprocess.check_output([bet,
                                             t1_im,
                                             t1_st_im, '-R', '-S', '-B'])
                else:
                    print('Please install first  FSL in your mac system and try again!')
                    sys.stdout.flush()
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGTERM)

            except:
                print("> ERROR:", scan, "registering masks, quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            brainmask = nib.load(t1_st_im).get_data() > 1
            for mod in settings['input_modality']:

                if mod == 'FLAIR':
                    continue

                # apply the same mask to the rest of modalities to reduce
                # computational time

                print('> PRE: ', scan, 'Applying skull mask to ', mod, 'image')
                current_mask = os.path.join(settings['tmp_folder'],
                                            'dr' + mod + '.nii.gz')
                current_st_mask = os.path.join(settings['tmp_folder'],
                                               mod + '_tmp.nii.gz')

                mask = nib.load(current_mask)
                mask_nii = mask.get_data()
                mask_nii[brainmask == 0] = 0
                mask.get_data()[:] = mask_nii
                mask.to_filename(current_st_mask)

            image = load_nii(os.path.join(settings['tmp_folder'], 'FLAIR_tmp.nii.gz'))
            image_norm = zscore_normalize(image, mask=None)
            image_norm.to_filename(os.path.join(settings['tmp_folder'], 'normalized_' + '.nii.gz'))


def preprocess_run(current_folder, settings):
    """
    Preprocess scan taking into account user settings
    - input:
      current_folder = path to the current image
      settings: settings

    """
    preprocess_time = time.time()

    scan = settings['input']
    try:
        # os.rmdir(os.path.join(current_folder,  'tmp'))
        os.mkdir(settings['tmp_folder'])
    except:
        if os.path.exists(settings['tmp_folder']) is False:
            print("> ERROR:",  scan, "I can not create tmp folder for", current_folder, "Quiting program.")

        else:
            pass

    # --------------------------------------------------
    # find modalities
    # --------------------------------------------------
    id_time = time.time()
    get_set_input_images(current_folder, settings)
    print("> INFO:", scan, "elapsed time: ", round(time.time() - id_time), "sec")

    # --------------------------------------------------
    # bias_correction(settings)
    if settings['bias_correction'] is True:
        denoise_time = time.time()
        bias_correction(settings)
        print("> INFO: bias correction", scan, "elapsed time: ", round(time.time() - denoise_time), "sec")
    else:
        pass

    # --------------------------------------------------
    # register modalities  bias_correction(settings)



    if settings['register_modalities'] is True:
        print(CBLUE2 + "Registration started... moving all images to the MPRAGE+192 space" +  CEND) 
        reg_time = time.time()
        register_masks(settings)
        print("> INFO:", scan, "elapsed time: ", round(time.time() - reg_time), "sec")
        print(CBLUE2 + "Registration completed!" +  CEND)
    else:
        try:
            if settings['reg_space'] == 'FlairtoT1':
                for mod in settings['input_modality']:
                    if mod == 'T1':
                        continue
                    out_scan = mod + '.nii.gz' if mod == 'T1' else 'r' + mod + '.nii.gz'
                    shutil.copy2(os.path.join(settings['tmp_folder'],
                                              mod + '.nii.gz'),
                                 os.path.join(settings['tmp_folder'],
                                              out_scan))
            if settings['reg_space'] == 'T1toFlair':
                for mod in settings['input_modality']:
                    if mod == 'FLAIR':
                        continue
                    out_scan = mod + '.nii.gz' if mod == 'FLAIR' else 'r' + mod + '.nii.gz'
                    shutil.copy2(os.path.join(settings['tmp_folder'],
                                              mod + '.nii.gz'),
                                 os.path.join(settings['tmp_folder'],
                                              out_scan))
            if  settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':
                for mod in settings['input_modality']:
                    out_scan = 'r' + mod + '.nii.gz'
                    shutil.copy2(os.path.join(settings['tmp_folder'],
                                              mod + '.nii.gz'),
                                 os.path.join(settings['tmp_folder'],
                                              out_scan))


        except:
            print("> ERROR: registration ", scan, "I can not rename input modalities as tmp files. Quiting program.")

            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    # --------------------------------------------------
    # noise filtering
    # --------------------------------------------------
    if settings['denoise'] is True:
        print(CBLUE2 + "Denoising started... reducing noise using anisotropic Diffusion" +  CEND)
        denoise_time = time.time()
        denoise_masks(settings)
        print("> INFO: denoising", scan, "elapsed time: ", round(time.time() - denoise_time), "sec")
        print(CBLUE2 + "Denoising completed!" +  CEND)
    else:
        try:
            for mod in settings['input_modality']:
                if settings['reg_space'] == 'FlairtoT1':
                    input_scan = mod + '.nii.gz' if mod == 'T1' else 'r' + mod + '.nii.gz'
                if settings['reg_space'] == 'T1toFlair':
                    input_scan = mod + '.nii.gz' if mod == 'FLAIR' else 'r' + mod + '.nii.gz'
                if settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':
                    input_scan = 'r' + mod + '.nii.gz'
                shutil.copy(os.path.join(settings['tmp_folder'],
                                         input_scan),
                            os.path.join(settings['tmp_folder'],
                                         'd' + input_scan))
        except:
            print("> ERROR denoising:", scan, "I can not rename input modalities as tmp files. Quiting program.")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    # --------------------------------------------------
    # skull strip
    # --------------------------------------------------

    if settings['skull_stripping'] is True:
        print(CBLUE2 + "External skull stripping started... using ROBEX or BET(Brain Extraction Tool)" +  CEND)
        sk_time = time.time()
        skull_strip(settings)
        print("> INFO:", scan, "elapsed time: ", round(time.time() - sk_time), "sec")
        print(CBLUE2 + "External skull stripping completed!" +  CEND)

    else:
        try:
            for mod in settings['input_modality']:
                if settings['reg_space'] == 'FlairtoT1':
                    input_scan = 'd' + mod + '.nii.gz' if mod == 'T1' else 'dr' + mod + '.nii.gz'
                if settings['reg_space'] == 'T1toFlair':
                    input_scan = 'd' + mod + '.nii.gz' if mod == 'FLAIR' else 'dr' + mod + '.nii.gz'
                if settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':
                    input_scan = 'dr' + mod + '.nii.gz'    
                shutil.copy(os.path.join(settings['tmp_folder'],
                                         input_scan),
                            os.path.join(settings['tmp_folder'],
                                         mod + '_tmp.nii.gz'))
        except:
            print("> ERROR: Skull-stripping", scan, "I can not rename input modalities as tmp files. Quiting program.")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)


    image = load_nii(os.path.join(settings['tmp_folder'], 'FLAIR_tmp.nii.gz'))
    image_norm = zscore_normalize(image, mask=None)
    image_norm.to_filename(os.path.join(settings['tmp_folder'], 'normalized_FLAIR_X.nii.gz'))

    if settings['skull_stripping'] is True and settings['register_modalities'] is True:
        print("> INFO:", scan, "total preprocessing time: ", round(time.time() - preprocess_time))

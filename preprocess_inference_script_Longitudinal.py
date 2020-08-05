
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
import click
import shutil
import argparse
import os
import sys
import platform
from timeit import time
import configparser
import numpy as np
from sources.preprocess_Longitudinal import preprocess_run
from sources.postprocess import invert_registration
from sources.read_settings import load_settings, Train_Test_settings
THIS_PATH = THIS_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(THIS_PATH, 'libs'))
# load settings from input
parser = argparse.ArgumentParser()
parser.add_argument('--docker',
                    dest='docker',
                    action='store_true')
parser.set_defaults(docker=False)
args = parser.parse_args()
container = args.docker

# check and remove the folder which dose not contain the necessary modalities before prepossessing step
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

def check_inputs(current_folder, settings, choice):
    """
    checking input errors, fixing  and writing it into the Input Issue Report File


    """
    erf =os.path.join(THIS_PATH, 'InputIssueReportfile.txt')
    f = open(erf, "a")

    if os.path.isdir(os.path.join(settings['training_folder'], current_folder)):
        if len(os.listdir(os.path.join(settings['training_folder'], current_folder))) == 0:
           print(('Directory:', current_folder, 'is empty'))
           print('Warning: if the  directory is not going to be removed, the Training could be later stopped!')
           if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
             f.write("The empty directory: %s has been removed from Training set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['training_folder'], current_folder), ignore_errors=True)
             return
           return
    else:
        pass

    if choice == 'training':
        modalities = settings['input_modality'][:] + ['lesion']
        image_tags = settings['image_tags'][:] + settings['InputLabel'][:]
    else:
        modalities = settings['input_modality'][:]
        image_tags = settings['image_tags'][:]

    if settings['debug']:
        print("> DEBUG:", "number of input sequences to find:", len(modalities))


    print("> PRE:", current_folder, "identifying input modalities")

    found_modalities = 0
    if os.path.isdir(os.path.join(settings['training_folder'], current_folder)):
        masks = [m for m in os.listdir(os.path.join(settings['training_folder'], current_folder)) if m.find('.nii') > 0]
        pass  # do your stuff here for directory
    else:
        # shutil.rmtree(os.path.join(settings['training_folder'], current_folder), ignore_errors=True)
        print(('The file:', current_folder, 'is not part of training'))
        print('Warning: if the  file is not going to be removed, the Training could be later stopped!')
        if click.confirm('The file will be removed. Do you want to continue?', default=True):
          f.write("The file: %s has been removed from Training set!" % current_folder + os.linesep)
          f.close()
          os.remove(os.path.join(settings['training_folder'], current_folder))
          return
        return




    for t, m in zip(image_tags, modalities):

        # check first the input modalities
        # find tag

        found_mod = [mask.find(t) if mask.find(t) >= 0
                     else np.Inf for mask in masks]

        if found_mod[np.argmin(found_mod)] is not np.Inf:
            found_modalities += 1


    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
           print("> ERROR:", current_folder, \
            "does not contain all valid input modalities")
           print('Warning: if the  folder is  not going to be removed, the Training could be later stopped!')
           if click.confirm('The folder will be removed. Do you want to continue?', default=True):
             f.write("The folder: %s has been removed from Training set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['training_folder'], current_folder), ignore_errors=True)


        #return True


def overall_config():
    """
    Get the CNN configuration from file
    """
    traintest_config = configparser.SafeConfigParser()
    traintest_config.read(os.path.join(THIS_PATH, 'config', 'configuration.cfg'))

    # read user's configuration file
    settings = load_settings(traintest_config)
    settings['tmp_folder'] = THIS_PATH + '/tmp'
    settings['standard_lib'] = THIS_PATH + '/libs/standard'
    # set paths taking into account the host OS
    host_os = platform.system()
    if host_os == 'Linux' or 'Darwin':
        settings['niftyreg_path'] = THIS_PATH + '/libs/linux/niftyreg'
        settings['robex_path'] = THIS_PATH + '/libs/linux/ROBEX/runROBEX.sh'
        # settings['tensorboard_path'] = THIS_PATH + '/libs/bin/tensorboard'
        settings['test_slices'] = 256
    elif host_os == 'Windows':
        settings['niftyreg_path'] = os.path.normpath(
            os.path.join(THIS_PATH,
                         'libs',
                         'win',
                         'niftyreg'))

        settings['robex_path'] = os.path.normpath(
            os.path.join(THIS_PATH,
                         'libs',
                         'win',
                         'ROBEX',
                         'runROBEX.bat'))
        settings['test_slices'] = 256
    else:
        print("The OS system also here ...", host_os, "is not currently supported.")
        exit()

    # print settings when debugging
    if settings['debug']:
        Train_Test_settings(settings)

    return settings

def lib_config(settings):
    """
    Define the library backend and write settings
    """
    #
    #    if settings['backend'] == 'theano':
    #        device = 'cuda' + str(settings['gpu_number']) if settings['gpu_mode'] else 'cpu'
    #        os.environ['KERAS_BACKEND'] = settings['backend']
    #        os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=' + device + ',floatX=float32,optimizer=fast_compile'
    #    else:
    #        device = str(settings['gpu_number']) if settings['gpu_mode'] is not None else " "
    #        print "DEBUG: ", device
    #        os.environ['KERAS_BACKEND'] = 'tensorflow'
    #        os.environ["CUDA_VISIBLE_DEVICES"] = device


    # forcing tensorflow
    device = str(settings['gpu_number'])
    print("DEBUG: ", device)
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_VISIBLE_DEVICES"] = device




def check_oututs(current_folder, settings, choice='testing'):
    """
    checking input errors, fixing  and writing it into the Input Issue Report File


    """
    erf =os.path.join(THIS_PATH, 'OutputIssueReportfile.txt')
    f = open(erf, "a")

    if os.path.isdir(os.path.join(settings['inference_folder'], current_folder)):
        if len(os.listdir(os.path.join(settings['inference_folder'], current_folder))) == 0:
           print(('Directory:', current_folder, 'is empty'))
           print('Warning: if the  directory is not going to be removed, the Testing could be later stopped!')
           if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
             f.write("The empty directory: %s has been removed from Testing set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['inference_folder'], current_folder), ignore_errors=True)
             return
           return
    else:
        pass

    if choice == 'training':
        modalities = settings['input_modality'][:] + ['lesion']
        image_tags = settings['image_tags'][:] + settings['InputLabel'][:]
    else:
        modalities = settings['input_modality'][:]
        image_tags = settings['image_tags'][:]

    if settings['debug']:
        print("> DEBUG:", "number of input sequences to find:", len(modalities))


    print("> PRE:", current_folder, "identifying input modalities")

    found_modalities = 0
    if os.path.isdir(os.path.join(settings['inference_folder'], current_folder)):
        masks = [m for m in os.listdir(os.path.join(settings['inference_folder'], current_folder)) if m.find('.nii') > 0]
        pass  # do your stuff here for directory
    else:
        # shutil.rmtree(os.path.join(settings['training_folder'], current_folder), ignore_errors=True)
        print(('The file:', current_folder, 'is not part of testing'))
        print('Warning: if the  file is not going to be removed, the Testing could be later stopped!')
        if click.confirm('The file will be removed. Do you want to continue?', default=True):
          f.write("The file: %s has been removed from Testing set!" % current_folder + os.linesep)
          f.close()
          os.remove(os.path.join(settings['inference_folder'], current_folder))
          return
        return




    for t, m in zip(image_tags, modalities):

        # check first the input modalities
        # find tag

        found_mod = [mask.find(t) if mask.find(t) >= 0
                     else np.Inf for mask in masks]

        if found_mod[np.argmin(found_mod)] is not np.Inf:
            found_modalities += 1


    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
           print("> ERROR:", current_folder, \
            "does not contain all valid input modalities")
           print('Warning: if the  folder is  not going to be removed, the Testing could be later stopped!')
           if click.confirm('The folder will be removed. Do you want to continue?', default=True):
             f.write("The folder: %s has been removed from Testing set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['inference_folder'], current_folder), ignore_errors=True)





def pre_segmentation(settings):
    """
    Infer segmentation given the input settings passed as parameters
    """

    # define the training backend
    lib_config(settings)

    from sources.base_old import prediction_models
    from sources.build_model_Longitudinal_Homogeneous import build_and_compile_models_tensor_1

    # --------------------------------------------------
    # net configuration
    # take into account if the learnedmodel models have to be used
    # all images share the same network model
    # --------------------------------------------------
    settings['full_train'] = True
    settings['load_weights'] = True
    settings['model_saved_paths'] = os.path.join(THIS_PATH, 'models')
    settings['net_verbose'] = 0
    # model = build_and_compile_models_tensor_1(settings)

    # --------------------------------------------------
    # process each of the scans
    # - image identification
    # - image registration
    # - skull-stripping
    # - WM segmentation
    # --------------------------------------------------

    # all_folders = os.listdir(settings['inference_folder'])
    # all_folders.sort()
    # # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    # for check in all_folders:
    #    check_oututs(check, settings)

    # update scan list after removing  the unnecessary folders before prepossessing step


    settings['train_test'] = 'testing'
    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()

    for scan in all_folders:

        total_time = time.time()
        settings['input'] = scan
        # --------------------------------------------------
        # move things to a tmp folder before starting
        # --------------------------------------------------

        current_folder = os.path.join(settings['inference_folder'], scan)
        settings['tmp_folder'] = os.path.normpath(
            os.path.join(current_folder,  'tmp'))

        # --------------------------------------------------
        # preprocess scans
        # --------------------------------------------------
        preprocess_run(current_folder, settings)


if __name__ == '__main__':

# 

   try:
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
       print('\x1b[6;30;42m' + 'preprocessing of testing data started.......................' + '\x1b[0m')
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
       settings = overall_config()
       pre_segmentation(settings)
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
       print('\x1b[6;30;42m' + 'preprocessing of testing data completed.....................' + '\x1b[0m')
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
   except KeyboardInterrupt:
       print("KeyboardInterrupt has been caught.")
       time.sleep(1)
       os.kill(os.getpid(), signal.SIGTERM)    
    
   
    
    
    
    

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
import os
import sys
import platform
from timeit import time
import configparser
import numpy as np
import tensorflow as tf
from sources.preprocess_cross import preprocess_run
from sources.read_settings import load_settings, Train_Test_settings
from sources.postprocess import invert_registration
THIS_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(THIS_PATH, 'libs'))
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

# check and remove the folder which dose not contain the necessary modalities before prepossessing step

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

    device = str(settings['gpu_number'])
    print("DEBUG: ", device)
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_VISIBLE_DEVICES"] = device

def train_test_network_cross(settings):
    """
    Train the CNN network given the settings passed as parameter
    """

    # set GPU mode from the configuration file. Trying to update
    # the backend automatically from here in order to use either theano
    # or tensorflow backends

    from sources.main_cross import train_first_model, train_sec_model
    from sources.build_model_cross import build_and_compile_models_tensor_1, build_and_compile_models_tensor_2

    # define the training backend
    lib_config(settings)

    # all_folders = os.listdir(settings['training_folder'])
    # all_folders.sort()
    # # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    # for check in all_folders:
    #     check_inputs(check, settings, 'training')

    # update scan list after removing  the unnecessary folders before prepossessing step
    training_folders = os.listdir(settings['training_folder'])
    training_folders.sort()

    settings['train_test'] = 'training'
    settings['training_folder'] = os.path.normpath(settings['training_folder'])
    total_time = time.time()
    if settings['pre_processing'] is False:
        for scan in training_folders:
            # --------------------------------------------------
            # move things to a tmp folder before starting
            # --------------------------------------------------

            settings['input'] = scan
            current_folder = os.path.join(settings['training_folder'], scan)
            settings['tmp_folder'] = os.path.normpath(os.path.join(current_folder,
                                                                   'tmp'))
            print('Preprocessing:', CURL + current_folder + CEND)

            preprocess_run(current_folder, settings)

    cross_valid_folders = os.listdir(settings['cross_validation_folder'])
    cross_valid_folders.sort()

    settings['cross_validation_folder'] = os.path.normpath(settings['cross_validation_folder'])
    total_time = time.time()
    if settings['pre_processing'] is False:
        for scan in cross_valid_folders:
            # --------------------------------------------------
            # move things to a tmp folder before starting
            # --------------------------------------------------

            settings['input'] = scan
            current_folder = os.path.join(settings['cross_validation_folder'], scan)
            settings['tmp_folder'] = os.path.normpath(os.path.join(current_folder,
                                                                   'tmp'))
            print('Preprocessing:', CURL + current_folder + CEND)
            preprocess_run(current_folder, settings)

    if settings['pre_processing'] is False:
        traintest_config = configparser.ConfigParser()
        traintest_config.read(os.path.join(THIS_PATH, 'config', 'configuration.cfg'))
        traintest_config.set('completed', 'pre_processing', str(True))
        with open(os.path.join(THIS_PATH,
                               'config',
                               'configuration.cfg'), 'w') as configfile:
            traintest_config.write(configfile)

    seg_time = time.time()
    print("> CNN: Starting training session")
    # select training scans
    train_x_data = {f: {m: os.path.join(settings['training_folder'], f, 'tmp', n)
                        for m, n in zip(settings['input_modality'],
                                        settings['x_names'])}
                    for f in training_folders}

    train_y_data = {f: os.path.join(settings['training_folder'],
                                    f,
                                    'tmp',
                                    'lesion.nii.gz')
                    for f in training_folders}

    val_x_data = {f: {m: os.path.join(settings['cross_validation_folder'], f, 'tmp', n)
                      for m, n in zip(settings['input_modality'],
                                      settings['x_names'])}
                  for f in cross_valid_folders}

    val_y_data = {f: os.path.join(settings['cross_validation_folder'],
                                  f,
                                  'tmp',
                                  'lesion.nii.gz')
                  for f in cross_valid_folders}

    settings['model_saved_paths'] = os.path.join(THIS_PATH, 'models')
    settings['load_weights'] = False

    # train the model for the current scan

    print("> CNN: training net with %d subjects" % (len(list(train_x_data.keys()))))

    # --------------------------------------------------
    # initialize the CNN and train the classifier
    # --------------------------------------------------
    if tf.__version__ < "2.2.0":
        model = build_and_compile_models_tensor_1(settings)
        print('\x1b[6;30;44m' + 'Currently running TensorFlow version:' + '\x1b[0m', tf.__version__ )
    else:
        model = build_and_compile_models_tensor_2(settings)
        print('\x1b[6;30;44m' + 'Currently running TensorFlow version:' + '\x1b[0m', tf.__version__)
    print('train_x_data', train_x_data)
    print('train_y_data', train_y_data)
    print('val_x_data', val_x_data)
    print('val_y_data', val_y_data)
    first_model = train_first_model(model[0], model[1], train_x_data, train_y_data, val_x_data, val_y_data,
                                    settings, THIS_PATH)
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('\x1b[6;30;44m' + 'Training of first network done successfully' + '\x1b[0m')
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('')
    sec_model = train_sec_model(model[1], first_model, val_x_data, val_y_data, train_x_data, train_y_data, settings,
                                THIS_PATH)
    # model = train_cascaded_model(model, train_x_data, train_y_data,  settings, THIS_PATH)

    print("> INFO: training time:", round(time.time() - seg_time), "sec")
    print("> INFO: total pipeline time: ", round(time.time() - total_time), "sec")
    if settings['model_1_train'] is True and settings['model_2_train'] is True:
        print('\x1b[6;30;44m' + '...............................................' + '\x1b[0m')
        print('\x1b[6;30;44m' + 'First and second model are created successfully' + '\x1b[0m')
        print('\x1b[6;30;44m' + '...............................................' + '\x1b[0m')
        print('\x1b[6;30;41m' + 'Inference will be proceeded now!               ' + '\x1b[0m')

    else:
        print('\x1b[6;30;44m' + 'Training was not successfully done!' + '\x1b[0m')

    model[0] = first_model
    model[1] = sec_model

    from sources.main_cross import prediction_models
    settings['full_train'] = True
    settings['load_weights'] = True
    settings['model_saved_paths'] = os.path.join(THIS_PATH, 'models')
    settings['net_verbose'] = 0
    model = build_and_compile_models_tensor_1(settings)
    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    for check in all_folders:
       check_oututs(check, settings)

    settings['train_test'] = 'testing'
    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()

    for scan in all_folders:

        total_time = time.time()
        settings['input'] = scan
        current_folder = os.path.join(settings['inference_folder'], scan)
        settings['tmp_folder'] = os.path.normpath(
            os.path.join(current_folder,  'tmp'))
        preprocess_run(current_folder, settings)
        seg_time = time.time()
        sys.stdout.flush()
        settings['prediction'] = scan

        test_x_data = {scan: {m: os.path.join(settings['tmp_folder'], n)
                              for m, n in zip(settings['input_modality'],
                                              settings['x_names'])}}

        prediction_models(model, test_x_data, settings)

        if settings['register_modalities']:
             # print("> INFO:", scan, "Inverting lesion segmentation masks")
             print(CYELLOW + "Inverting lesion segmentation masks:", CRED + scan  + CEND , ".....started!" + CEND)
             invert_registration(current_folder, settings)


        print("> INFO:", scan, "CNN Segmentation time: ", round(time.time() - seg_time), "sec")
        print("> INFO:", scan, "total pipeline time: ", round(time.time() - total_time), "sec")

        # remove tmps if not set
        if settings['save_tmp'] is False:
            try:
                os.rmdir(settings['tmp_folder'])
                os.rmdir(os.path.join(settings['current_folder'],
                                      settings['modelname']))
            except:
                pass

    print('\x1b[6;30;41m' + 'Inference has been proceeded' + '\x1b[0m')


def train_network_cross(settings):
    """
    Train the CNN network given the settings passed as parameter
    """

    # set GPU mode from the configuration file. Trying to update
    # the backend automatically from here in order to use either theano
    # or tensorflow backends

    from sources.main_cross import train_first_model, train_sec_model
    from sources.build_model_cross import build_and_compile_models_tensor_1, build_and_compile_models_tensor_2

    # define the training backend
    lib_config(settings)

    # all_folders = os.listdir(settings['training_folder'])
    # all_folders.sort()
    # # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    # for check in all_folders:
    #     check_inputs(check, settings, 'training')

    # update scan list after removing  the unnecessary folders before prepossessing step
    training_folders = os.listdir(settings['training_folder'])
    training_folders.sort()

    settings['train_test'] = 'training'
    settings['training_folder'] = os.path.normpath(settings['training_folder'])
    total_time = time.time()
    if settings['pre_processing'] is False:
        for scan in training_folders:
            # --------------------------------------------------
            # move things to a tmp folder before starting
            # --------------------------------------------------

            settings['input'] = scan
            current_folder = os.path.join(settings['training_folder'], scan)
            settings['tmp_folder'] = os.path.normpath(os.path.join(current_folder,
                                                                   'tmp'))
            print('Preprocessing:', CURL + current_folder + CEND)

            preprocess_run(current_folder, settings)

    cross_valid_folders = os.listdir(settings['cross_validation_folder'])
    cross_valid_folders.sort()

    settings['cross_validation_folder'] = os.path.normpath(settings['cross_validation_folder'])
    total_time = time.time()
    if settings['pre_processing'] is False:
        for scan in cross_valid_folders:
            # --------------------------------------------------
            # move things to a tmp folder before starting
            # --------------------------------------------------

            settings['input'] = scan
            current_folder = os.path.join(settings['cross_validation_folder'], scan)
            settings['tmp_folder'] = os.path.normpath(os.path.join(current_folder,
                                                                   'tmp'))
            print('Preprocessing:', CURL + current_folder + CEND)
            preprocess_run(current_folder, settings)

    if settings['pre_processing'] is False:
        traintest_config = configparser.ConfigParser()
        traintest_config.read(os.path.join(THIS_PATH, 'config', 'configuration.cfg'))
        traintest_config.set('completed', 'pre_processing', str(True))
        with open(os.path.join(THIS_PATH,
                               'config',
                               'configuration.cfg'), 'w') as configfile:
            traintest_config.write(configfile)

    seg_time = time.time()
    print("> CNN: Starting training session")
    # select training scans
    train_x_data = {f: {m: os.path.join(settings['training_folder'], f, 'tmp', n)
                        for m, n in zip(settings['input_modality'],
                                        settings['x_names'])}
                    for f in training_folders}

    train_y_data = {f: os.path.join(settings['training_folder'],
                                    f,
                                    'tmp',
                                    'lesion.nii.gz')
                    for f in training_folders}

    val_x_data = {f: {m: os.path.join(settings['cross_validation_folder'], f, 'tmp', n)
                      for m, n in zip(settings['input_modality'],
                                      settings['x_names'])}
                  for f in cross_valid_folders}

    val_y_data = {f: os.path.join(settings['cross_validation_folder'],
                                  f,
                                  'tmp',
                                  'lesion.nii.gz')
                  for f in cross_valid_folders}

    settings['model_saved_paths'] = os.path.join(THIS_PATH, 'models')
    settings['load_weights'] = False

    # train the model for the current scan

    print("> CNN: training net with %d subjects" % (len(list(train_x_data.keys()))))

    # --------------------------------------------------
    # initialize the CNN and train the classifier
    # --------------------------------------------------
    if tf.__version__ < "2.2.0":
        model = build_and_compile_models_tensor_1(settings)
        print('\x1b[6;30;44m' + 'Currently running TensorFlow version:' + '\x1b[0m', tf.__version__ )
    else:
        model = build_and_compile_models_tensor_2(settings)
        print('\x1b[6;30;44m' + 'Currently running TensorFlow version:' + '\x1b[0m', tf.__version__)
    print('train_x_data', train_x_data)
    print('train_y_data', train_y_data)
    print('val_x_data', val_x_data)
    print('val_y_data', val_y_data)
    first_model = train_first_model(model[0], model[1], train_x_data, train_y_data, val_x_data, val_y_data,
                                    settings, THIS_PATH)
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('\x1b[6;30;44m' + 'Training of first network done successfully' + '\x1b[0m')
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('')
    sec_model = train_sec_model(model[1], first_model, val_x_data, val_y_data, train_x_data, train_y_data, settings,
                                THIS_PATH)
    # model = train_cascaded_model(model, train_x_data, train_y_data,  settings, THIS_PATH)

    print("> INFO: training time:", round(time.time() - seg_time), "sec")
    print("> INFO: total pipeline time: ", round(time.time() - total_time), "sec")
    if settings['model_1_train'] is True and settings['model_2_train'] is True:
        print('\x1b[6;30;44m' + '...............................................' + '\x1b[0m')
        print('\x1b[6;30;44m' + 'First and second model are created successfully' + '\x1b[0m')
        print('\x1b[6;30;44m' + '...............................................' + '\x1b[0m')
        print('\x1b[6;30;41m' + 'Inference can be proceeded now!                ' + '\x1b[0m')

    else:
        print('\x1b[6;30;44m' + 'Training was not successfully done!' + '\x1b[0m')


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





def infer_segmentation_cross(settings):
    """
    Infer segmentation given the input settings passed as parameters
    """

    # define the training backend
    lib_config(settings)

    from sources.main_cross import prediction_models
    from sources.build_model_cross import build_and_compile_models_tensor_1

    # --------------------------------------------------
    # net configuration
    # take into account if the learnedmodel models have to be used
    # all images share the same network model
    # --------------------------------------------------
    settings['full_train'] = True
    settings['load_weights'] = True
    settings['model_saved_paths'] = os.path.join(THIS_PATH, 'models')
    settings['net_verbose'] = 0
    model = build_and_compile_models_tensor_1(settings)

    # --------------------------------------------------
    # process each of the scans
    # - image identification
    # - image registration
    # - skull-stripping
    # - WM segmentation
    # --------------------------------------------------

    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    for check in all_folders:
       check_oututs(check, settings)

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

        # --------------------------------------------------
        # WM MS lesion inference
        # --------------------------------------------------
        seg_time = time.time()

        "> CNN:", scan, "running WM lesion segmentation"
        sys.stdout.flush()
        settings['prediction'] = scan

        test_x_data = {scan: {m: os.path.join(settings['tmp_folder'], n)
                              for m, n in zip(settings['input_modality'],
                                              settings['x_names'])}}

        prediction_models(model, test_x_data, settings)

        if settings['register_modalities']:
             # print("> INFO:", scan, "Inverting lesion segmentation masks")
             print(CYELLOW + "Inverting lesion segmentation masks:", CRED + scan  + CEND , ".....started!" + CEND)
             invert_registration(current_folder, settings)


        print("> INFO:", scan, "CNN Segmentation time: ", round(time.time() - seg_time), "sec")
        print("> INFO:", scan, "total pipeline time: ", round(time.time() - total_time), "sec")

        # remove tmps if not set
        if settings['save_tmp'] is False:
            try:
                os.rmdir(settings['tmp_folder'])
                os.rmdir(os.path.join(settings['current_folder'],
                                      settings['modelname']))
            except:
                pass

    print('\x1b[6;30;41m' + 'Inference has been proceeded' + '\x1b[0m')

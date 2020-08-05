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
import signal
import time
import numpy as np
from nibabel import load as load_nii
import nibabel as nib
from operator import itemgetter
from sources.build_model_Longitudinal_Homogeneous import define_training_layers, train_model, fit_thismodel
from operator import add
from keras.models import load_model
import tensorflow as tf
import configparser


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

def calcul_num_sample_multi(x_train, settings):

    this_size = len(settings['all_mod'])
    # train_val = {}
    # for label in range(0, this_size):
    #     # perm_indices = np.random.permutation(x_train[label].shape[0])
    #     # train_val[label] = int(len(perm_indices) * train_split_perc)
    #     train_val[label] = int(x_train[label].shape[0])
    #
    # x_train_ = {}
    # train_val = min(train_val.items(), key=lambda x: x[1])
    # train_val = train_val[1]
    #
    # for i in range(0, this_size):
    #     # print("x_train[", i, "]:", x_train[i].shape[0])
    #     x_train_[i] = x_train[i][:train_val]

    temp = {}
    for label in range(0, this_size):
        temp[label] = x_train[label].shape[0]
    # print("temp[label]",label, temp[label])

    num_samples_t = min(temp.items(), key=lambda x: x[1])
    num_samples = num_samples_t[1]

    return num_samples

def train_first_model(firstmodel, secmodel,  train_x_data, train_y_data, val_x_data, val_y_data, settings, thispath):
    """
    Train the model using a cascade of two CNN

    inputs:

    - CNN model: a list containing the two cascaded CNN models

    - train_x_data: a nested dictionary containing training image paths:
           train_x_data['scan_name']['modality'] = path_to_image_modality

    - train_y_data: a dictionary containing labels
        train_y_data['scan_name'] = path_to_label

    - settings: dictionary containing general hyper-parameters:

    Outputs:
    - trained model: list containing the 2 cascaded CNN models after training
    """

    # ----------
    # CNN1
    # ----------
    traintest_config = configparser.ConfigParser()
    traintest_config.read(os.path.join(thispath, 'config', 'configuration.cfg'))

    print(CSELECTED + "CNN: loading training data for first model" + CEND)
    #modeltest = fit_thismodel(model[0], X, Y, settings)
    X = {}
    Y = {}
    X_val = {}
    Y_val = {}
    x_data = {}
    y_data = {}
    xvl_data = {}
    yvl_data = {}
    scans_train = list(train_x_data.keys())
    scans_vaid = list(val_x_data.keys())
    sel_voxels_train = {}
    sel_voxels_val = {}
    # train_y_data = {f: os.path.join(settings['training_folder'],
    checked_run = False
    this_size = len(settings['all_mod'])
    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         x_data[i] = {s: train_x_data[s][n] for s in scans_train}

    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         y_data[i] = {s: train_y_data[s][n] for s in scans_train}

    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         xvl_data[i] = {s: val_x_data[s][n] for s in scans_vaid}

    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         yvl_data[i] = {s: val_y_data[s][n] for s in scans_vaid}



    for i in range(0, this_size):
        settings['balanced_training'] = False
        print("Loading data", CRED + "for training" + CEND, "of subjects:  modality,", CRED2 + str(x_data[i]) + CEND, " lesion,", CGREEN + str(y_data[i]) + CEND)
        X[i], Y[i], sel_voxels_train[i] = dataset_training(x_data[i], y_data[i], settings)
        print("Loading data", CGREEN + "for cross validation" + CEND, "of subjects:  modality,", CRED2 + str(xvl_data[i]) + CEND, " lesion,", CGREEN + str(yvl_data[i]) + CEND)
        settings['balanced_training'] = True
        X_val[i], Y_val[i], sel_voxels_val[i] = dataset_training(xvl_data[i], yvl_data[i], settings)


    net_model_name = firstmodel['special_name_1']
    net_model_name_2 = secmodel['special_name_2']


    if settings['full_train'] is False:
        max_epochs = settings['max_epochs']
        patience = 0
        best_val_loss = np.Inf
        numsamplex = int(calcul_num_sample_multi(X, settings))
        firstmodel = define_training_layers(model=firstmodel, settings=settings,
                                          num_layers=settings['num_layers'],
                                          number_of_samples=numsamplex)
        settings['max_epochs'] = 0
        for it in range(0, max_epochs, 10):
            settings['max_epochs'] += 10
            # model[0] = train_model(model[0], X, Y, settings,
            #                      initial_epoch=it)
            firstmodel = train_model(firstmodel, X, Y, X_val, Y_val, settings, initial_epoch=it)

            # evaluate if continuing training or not
            val_loss = min(firstmodel['history'].history['val_loss'])
            if val_loss > best_val_loss:
                patience += 10
            else:
                best_val_loss = val_loss

            if patience >= settings['patience']:
                break

            # X, Y, sel_voxels = dataset_training(train_x_data,
            #                                       train_y_data,
            #                                       settings)

            for i in range(0, this_size):
                settings['balanced_training'] = False
                print("Loading data", CRED + "for training" + CEND, "of subjects:  modality,",
                      CRED2 + str(x_data[i]) + CEND, " lesion,", CGREEN + str(y_data[i]) + CEND)
                X[i], Y[i], sel_voxels_train[i] = dataset_training(x_data[i], y_data[i], settings)
                print("Loading data", CGREEN + "for cross validation" + CEND, "of subjects:  modality,",
                      CRED2 + str(xvl_data[i]) + CEND, " lesion,", CGREEN + str(yvl_data[i]) + CEND)
                settings['balanced_training'] = True
                X_val[i], Y_val[i], sel_voxels_val[i] = dataset_training(xvl_data[i], yvl_data[i], settings)

        settings['max_epochs'] = max_epochs
    else:
        # model[0] = load_model(net_weights_1)
        # net_model_name = model[0]['special_name_1']
        if os.path.exists(os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name + '.hdf5')) and \
                not os.path.exists(os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name_2 + '.hdf5')):
            net_weights_1 = os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name + '.hdf5')
            try:
                checked_run = True
                firstmodel['model'].load_weights(net_weights_1, by_name=True)
                print("CNN has Loaded previous weights from the", net_weights_1)
            except:
                print("> ERROR: The model", \
                    settings['modelname'], \
                    'selected does not contain a valid network model')
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if not os.path.exists(os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name_2 + '.hdf5')) and \
                (settings['model_1_train'] is  False):
          checked_run = True
          firstmodel = train_model(firstmodel, X, Y,  X_val, Y_val, settings)
          traintest_config.set('completed', 'model_1_train', str(True))
          with open(os.path.join(thispath,
                                 'config',
                                 'configuration.cfg'), 'w') as configfile:
              traintest_config.write(configfile)
          settings['model_1_train'] = True
        # thismodelx = os.path.join(THIS_PATH
        #                        , 'SAVEDMODEL', net_model_name + '.h5')
        # model[0]['net'].save(thismodelx)
        M1 = traintest_config.get('completed', 'model_1_train')
        print('Was first model created successfully?', M1)

    if not checked_run and M1:
        # model = os.path.join(settings['model_saved_paths'],
        #                      settings['modelname'])
        net_weights_1 = os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name + '.hdf5')
        try:
            if settings['debug']:
                print("loading best weights of first model after training")
            firstmodel['model'].load_weights(net_weights_1, by_name=True)
            print("CNN has Loaded previous weights from the", net_weights_1)
        except:
            print("> ERROR: The model", settings['modelname'], 'selected does not contain a valid network model')
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    del X, Y, X_val, Y_val, x_data, y_data, xvl_data, yvl_data,  sel_voxels_train, sel_voxels_val

    return firstmodel


def train_sec_model(secmodel, firstmodel, train_x_data, train_y_data, val_x_data, val_y_data, settings, thispath):


    traintest_config = configparser.ConfigParser()
    traintest_config.read(os.path.join(thispath, 'config', 'configuration.cfg'))

    print(CSELECTED + "CNN: loading training data for second model" + CEND)
    #modeltest = fit_thismodel(model[0], X, Y, settings)
    X = {}
    Y = {}
    X_val = {}
    Y_val = {}
    x_data = {}
    y_data = {}
    xvl_data = {}
    yvl_data = {}
    scans_train = list(train_x_data.keys())
    scans_vaid = list(val_x_data.keys())
    sel_voxels_train = {}
    sel_voxels_val = {}
    # train_y_data = {f: os.path.join(settings['training_folder'],

    this_size = len(settings['all_mod'])
    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         x_data[i] = {s: train_x_data[s][n] for s in scans_train}

    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         y_data[i] = {s: train_y_data[s][n] for s in scans_train}

    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         xvl_data[i] = {s: val_x_data[s][n] for s in scans_vaid}

    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         yvl_data[i] = {s: val_y_data[s][n] for s in scans_vaid}

    net_model_name_2 = secmodel['special_name_2']
    net_model_name = firstmodel['special_name_1']
    if settings['model_2_train'] is False:
        # X, Y, sel_voxels = dataset_training(train_x_data,
        #                                     train_y_data,
        #                                     settings,
        #                                     model=model[0])

        for i in range(0, this_size):
            settings['balanced_training'] = False
            print("Loading data", CRED + "for training" + CEND, "of subjects:  modality,", CRED2 + str(x_data[i]) + CEND,
                  " lesion,", CGREEN + str(y_data[i]) + CEND)
            X[i], Y[i], sel_voxels_train[i] = dataset_training(x_data[i], y_data[i], settings,
                                                                 model=firstmodel, index=i)
            print("Loading data", CGREEN + "for cross validation" + CEND, "of subjects:  modality,",
                  CRED2 + str(xvl_data[i]) + CEND, " lesion,", CGREEN + str(yvl_data[i]) + CEND)
            settings['balanced_training'] = True
            X_val[i], Y_val[i], sel_voxels_val[i] = dataset_training(xvl_data[i], yvl_data[i], settings,
                                                                       model=firstmodel, index=i)

    # print('> CNN: train_x ', X.shape)

    # define training layers
    if settings['full_train'] is False:
        max_epochs = settings['max_epochs']
        patience = 0
        best_val_loss = np.Inf
        numsamplex = int(calcul_num_sample_multi(X, settings))
        secmodel = define_training_layers(model=secmodel, settings=settings,
                                          num_layers=settings['num_layers'],
                                          number_of_samples=numsamplex)

        settings['max_epochs'] = 0
        for it in range(0, max_epochs, 10):
            settings['max_epochs'] += 10
            # model[1] = train_model(model[1], X, Y, settings,
            #                      initial_epoch=it)
            secmodel = train_model(secmodel, X, Y, X_val, Y_val, settings,
                                 initial_epoch=it)

            # evaluate if continuing training or not
            val_loss = min(secmodel['history'].history['val_loss'])
            if val_loss > best_val_loss:
                patience += 10
            else:
                best_val_loss = val_loss

            if patience >= settings['patience']:
                break

            # X, Y, sel_voxels = dataset_training(train_x_data,
            #                                       train_y_data,
            #                                       settings,
            #                                       model=model[0],
            #                                       selected_voxels=sel_voxels)

            for i in range(0, this_size):
                settings['balanced_training'] = False
                print("Loading data", CRED + "for training" + CEND, "of subjects:  modality,",
                      CRED2 + str(x_data[i]) + CEND,
                      " lesion,", CGREEN + str(y_data[i]) + CEND)
                X[i], Y[i], sel_voxels_train[i] = dataset_training(x_data[i], y_data[i], settings,
                                                                     model=firstmodel, selected_voxels=sel_voxels_train[i], index=i)
                print("Loading data", CGREEN + "for cross validation" + CEND, "of subjects:  modality,",
                      CRED2 + str(xvl_data[i]) + CEND, " lesion,", CGREEN + str(yvl_data[i]) + CEND)
                settings['balanced_training'] = True
                X_val[i], Y_val[i], sel_voxels_val[i] = dataset_training(xvl_data[i], yvl_data[i], settings,
                                                                           model=firstmodel, selected_voxels=sel_voxels_val[i], index=i)
        settings['max_epochs'] = max_epochs
    else:
        # model[1] = train_model(model[1], X, Y, settings)
        if os.path.exists(os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name + '.hdf5'))  and  \
                os.path.exists(os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name_2 + '.hdf5')):
            net_weights_2 = os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name_2 + '.hdf5')
            try:
                secmodel['model'].load_weights(net_weights_2, by_name=True)
                print("CNN has Loaded previous weights from the", net_weights_2)
            except:
                print("> ERROR: The model", \
                    settings['modelname'], \
                    'selected does not contain a valid network model')
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if os.path.exists(os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name + '.hdf5')) and settings['model_1_train'] \
                and (settings['model_2_train'] is False):
          # model[1] = train_model(model[1], X, Y, settings)
          secmodel = train_model(secmodel, X, Y, X_val, Y_val, settings)
          traintest_config.set('completed', 'model_2_train', str(True))
          with open(os.path.join(thispath,
                                 'config',
                                 'configuration.cfg'), 'w') as configfile:
              traintest_config.write(configfile)
          settings['model_2_train'] = True
        M2 = traintest_config.get('completed', 'model_2_train')
        print('Was second model created successfully?', M2)


    return secmodel


def prediction_models(model, test_x_data, settings, scan):
    """
    Test the cascaded approach using a learned model

    inputs:

    - CNN model: a list containing the two cascaded CNN models

    - test_x_data: a nested dictionary containing testing image paths:
           test_x_data['scan_name']['modality'] = path_to_image_modality


    - settings: dictionary containing general hyper-parameters:

    outputs:
        - output_segmentation
    """

    this_size = len(settings['all_mod'])

    # test_x_data_tmp = {scan: {test_x_data[scan][n] for n in range(0, this_size)}}
    #
    # print("Found testing data:", test_x_data_tmp, "in testing folder:",scan)

    n_class = int(this_size)
    # organize experiments
    exp_folder = os.path.join(settings['inference_folder'],
                              settings['prediction'],
                              settings['modelname'])
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    # first network
    firstnetwork_time = time.time()
    first_model_seg = {}
    print('\x1b[6;30;41m' + 'Prediction of first model is started ...' + '\x1b[0m')
    # settings['prediction_name'] = 'First_model_probability_map.nii.gz'
    for i in range(0, this_size):
        test_x_data_tmp = {scan: test_x_data[scan][i+1]}
        save_nifti = True if settings['debug'] is True else False
        print(CSELECTED + "First model, input:", CRED + str(i + 1) + CEND, " given image:", CRED + str(test_x_data[scan][i+1]) + CEND,  "...probability map" + CEND)

        settings['prediction_name'] = 'First_model_class_' + str(int(i) + 1) + '_probability_map.nii.gz'
        # print(CSELECTED +"First model, label:", labels[i], "probability map"+ CEND)
        first_model_seg[i] = prediction(model[0],
                              test_x_data_tmp,
                              settings,
                              index=i,
                              save_nifti=save_nifti)

    print("> INFO:............",  "total pipeline time for first network ", round(time.time() - firstnetwork_time), "sec")

    # # second network
    secondnetwork_time = time.time()
    second_model_seg = {}
    Cvoxel = {}
    print('\x1b[6;30;41m' + 'Prediction of second model is started ...' + '\x1b[0m')


    for i in range(0, this_size):
        #Cvoxel[i] = first_model_seg[i] > 0.5  # <.... tested work well
        Cvoxel[i] = first_model_seg[i] > 0.15
        if np.sum(Cvoxel[i]) == 0:
            Cvoxel[i] = first_model_seg[i] > 0.0

    # settings['prediction_name'] = 'Second_model_probability_map.nii.gz'
    for i in range(0, this_size):
        test_x_data_tmp = {scan: test_x_data[scan][i + 1]}
        print(CSELECTED + "Second model, input:", CRED + str(i + 1) + CEND, " given image:",
              CRED + str(test_x_data[scan][i + 1]) + CEND, "...probability map" + CEND)

        settings['prediction_name'] = 'Second_model_class_' + str(int(i) + 1) + '_probability_map.nii.gz'
        second_model_seg[i] = prediction(model[1],
                              test_x_data_tmp,
                              settings,
                              index=i,
                              save_nifti=True,
                              candidate_mask=Cvoxel)

    print("> INFO:............", "total pipeline time for second  network", round(time.time() - secondnetwork_time),
          "sec")

    # postprocess the output segmentation
    # obtain the orientation from the first scan used for testing
    # scans = list(test_x_data.keys())
    # flair_scans = [test_x_data[s] for s in scans]
    # flair_image = load_nii(flair_scans[0])
    #settings['prediction_name'] = settings['modelname'] + '_hard_seg.nii.gz'

    print('\x1b[6;30;41m' + 'Final segmentation is started ...' + '\x1b[0m')

    settings['prediction_name'] = 'CNN_final_single_class_segmentation.nii.gz'
    for i in range(0, this_size):
             flair_scans = test_x_data[scan][i + 1]
             flair_image = load_nii(flair_scans)

             print(CSELECTED + "Final segmentation, input:", CRED + str(i + 1) + CEND, " given image:",
                   CRED + str(flair_scans) + CEND)
             settings['prediction_name'] = 'CNN_final_' + str(i + 1) + '_segmentation.nii.gz'
             out_segmentation = final_process(second_model_seg[i],
                                                     settings,
                                                     save_nifti=True,
                                                     orientation=flair_image.affine)

    print('')
    print('\x1b[6;30;41m' + 'Prediction is done!' + '\x1b[0m')
    # return out_segmentation
    return out_segmentation


def multithreshold(image, threshold1, threshold2):
    img1 = image != threshold1
    img2 = image > threshold2
    return np.logical_and(img1, img2)


# def dataset_training(train_x_data,
#                        train_y_data,
#                        settings,
#                        model=None,
#                        selected_voxels=None):

def dataset_training(train_x_data,
                           train_y_data,
                           settings,
                           model=None,
                           selected_voxels=None, index=0, value=0):

    '''
    Load training and label samples for all given scans and modalities.

    Inputs:

    train_x_data: a nested dictionary containing training image paths:
        train_x_data['scan_name']['modality'] = path_to_image_modality

    train_y_data: a dictionary containing labels
        train_y_data['scan_name'] = path_to_label

    settings: dictionary containing general hyper-parameters:
        - settings['min_th'] = min threshold to remove voxels for training
        - settings['size'] = tuple containing patch size, either 2D (p1, p2, 1)
                            or 3D (p1, p2, p3)
        - settings['randomize_train'] = randomizes data
       - settings['fully_conv'] = fully_convolutional labels. If false,

    model: CNN model used to select training candidates

    Outputs:
        - X: np.array [num_samples, num_channels, p1, p2, p2]
        - Y: np.array [num_samples, 1, p1, p2, p3] if fully conv,
                      [num_samples, 1] otherwise

    '''

    # get_scan names and number of modalities used
    scans = list(train_x_data.keys())
    # modalities = settings['input_modality'][0]
    # flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    # select voxels for training:
    #  if model is no passed, training samples are extract by discarding CSF
    #  and darker WM in FLAIR, and use all remaining voxels.
    #  if model is passes, use the trained model to extract all voxels
    #  with probability > 0.5
    if model is None:
        flair_scans = [train_x_data[s] for s in scans]
        selected_voxels = select_training_voxels(flair_scans,
                                                 settings['min_th'])
    elif selected_voxels is None:
        # selected_voxels = voxels_from_learned_model(model,
        #                                                     train_x_data,
        #
        #                                                     settings)
        print("selecting voxels from previous model")
        selected_voxels = voxels_from_learned_model(model,
                                                            train_x_data,
                                                            settings, index)
    else:
        pass
    # extract patches and labels for each of the modalities
    data = []

    # for m in modalities:
    x_data = [train_x_data[s] for s in scans]
    y_data = [train_y_data[s] for s in scans]
    if settings['balanced_training'] is True:
        x_patches, y_patches = dataset_train_patches(x_data,
                                                  y_data,
                                                  selected_voxels,
                                                  settings['patch_size'],
                                                  settings['balanced_training'],
                                                  settings['ratio_negative_positive1'])
    else:
        x_patches, y_patches = dataset_train_patches(x_data,
                                                  y_data,
                                                  selected_voxels,
                                                  settings['patch_size'],
                                                  settings['balanced_training'],
                                                  settings['ratio_negative_positive2'])
    data.append(x_patches)

    # stack patches in channels [samples, channels, p1, p2, p3]
    X = np.stack(data, axis=1)
    Y = y_patches

    # apply randomization if selected
    if settings['randomize_train']:

        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        X = np.random.permutation(X.astype(dtype=np.float32))
        np.random.seed(seed)
        Y = np.random.permutation(Y.astype(dtype=np.int32))

    # fully convolutional / voxel labels
    if settings['fully_convolutional']:
        # Y = [ num_samples, 1, p1, p2, p3]
        Y = np.expand_dims(Y, axis=1)
    else:
        # Y = [num_samples,]
        if Y.shape[3] == 1:
            Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, :]
        else:
            Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, Y.shape[3] // 2]
        Y = np.squeeze(Y)

    return X, Y, selected_voxels


def normalize_data(im, datatype=np.float32):
    """
    zero mean / 1 standard deviation image normalization

    """
    im = im.astype(dtype=datatype) - im[np.nonzero(im)].mean()
    im = im / im[np.nonzero(im)].std()

    return im


def select_training_voxels(input_masks, threshold=2, datatype=np.float32):
    """
    Select voxels for training based on a intensity threshold

    Inputs:
        - input_masks: list containing all subject image paths
          for a single modality
        - threshold: minimum threshold to apply (after normalizing images
          with 0 mean and 1 std)

    Output:
        - rois: list where each element contains the subject binary mask for
          selected voxels [len(x), len(y), len(z)]
    """

    # load images and normalize their intensities
    images = [load_nii(image_name).get_data() for image_name in input_masks]
    images_norm = [normalize_data(im) for im in images]

    ###########
    #
    # output_sequence = nib.Nifti1Image(images_norm, affine=images.affine)
    # output_sequence.to_filename('images_norm.nii.gz')



    ###########




    # select voxels with intensity higher than threshold
    rois = [image > -0.5 for image in images_norm]
    # rois = [image != 0 for image in images_norm]

    # rois = [multithreshold(image, 0, -2.5) for image in images_norm]
    # output_sequence_e = nib.Nifti1Image(rois)
    # output_sequence_e.to_filename('roi_images_norm.nii.gz',  affine=images.affine)

    return rois


def dataset_train_patches(x_data,
                       y_data,
                       selected_voxels,
                       patch_size,
                       balanced_training,
                       fraction_negatives,
                       random_state=42,
                       datatype=np.float32):
    """
    Load train patches with size equal to patch_size, given a list of
    selected voxels

    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - y_data: list containing all subject image paths for the labels
       - selected_voxels: list where each element contains the subject binary
         mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)

    Outputs:
       - X: Train X data matrix for the particular channel
       - Y: Train Y labels [num_samples, p1, p2, p3]
    """

    # load images and normalize their intensties
    images = [load_nii(name).get_data() for name in x_data]
    images_norm = [normalize_data(im) for im in images]



    lesion_masks = [load_nii(name).get_data().astype(dtype=np.bool)
                    for name in y_data]

    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain)
                      for lesion, brain in zip(lesion_masks, selected_voxels)]


    for nlm in range(0, len(nolesion_masks)):
        if not nolesion_masks[nlm].any():
            nolesion_masks[nlm] = np.logical_and(np.logical_not(lesion_masks[nlm]), (images_norm[nlm] > -2.5))
            print('\x1b[6;30;41m' + 'Warning:' + '\x1b[0m', 'for the training scan:', x_data[nlm],' after applying probability higher than 0.5, no voxels have been selected as no lesion, using original data instead!' )
            print('')

    # Get all the x,y,z coordinates for each image
    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]



    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]

    # load all positive samples (lesion voxels). If a balanced training is set
    # use the same number of positive and negative samples. On unbalanced
    # training sets, the number of negative samples is multiplied by
    # of random negatives samples

    np.random.seed(random_state)

    #x_pos_patches = [np.array(get_patches(image, centers, patch_size))
    #                 for image, centers in zip(images_norm, lesion_centers)]
    #y_pos_patches = [np.array(get_patches(image, centers, patch_size))
    #                 for image, centers in zip(lesion_masks, lesion_centers)]

    number_lesions = [np.sum(lesion) for lesion in lesion_masks]
    total_lesions = np.sum(number_lesions)
    neg_samples = int((total_lesions * fraction_negatives) / len(number_lesions))
    X, Y = [], []
    st = 1
    for l_centers, nl_centers, image, lesion in zip(lesion_centers,
                                                    nolesion_centers,
                                                    images_norm,
                                                    lesion_masks):

        # balanced training: same number of positive and negative samples
        if balanced_training:
            print("Loading balanced training data of subject:", st)
            if len(l_centers) > 0:
                # positive samples
                x_pos_samples = get_patches(image, l_centers, patch_size)
                y_pos_samples = get_patches(lesion, l_centers, patch_size)
                idx = np.random.permutation(list(range(0, len(nl_centers)))).tolist()[:len(l_centers)]
                nolesion = itemgetter(*idx)(nl_centers)
                if len(idx) == 1:
                    nolesion = (nolesion,)
                x_neg_samples = get_patches(image, nolesion, patch_size)
                y_neg_samples = get_patches(lesion, nolesion, patch_size)
                X.append(np.concatenate([x_pos_samples, x_neg_samples]))
                Y.append(np.concatenate([y_pos_samples, y_neg_samples]))

        # unbalanced dataset: images with only negative samples are allowed
        else:
            print("Loading unbalanced training data of subject:", st)
            if len(l_centers) > 0:
                x_pos_samples = get_patches(image, l_centers, patch_size)
                y_pos_samples = get_patches(lesion, l_centers, patch_size)

            idx = np.random.permutation(list(range(0, len(nl_centers)))).tolist()[:neg_samples]
            nolesion = itemgetter(*idx)(nl_centers)
            if len(idx) == 1:
                nolesion = (nolesion,)
            x_neg_samples = get_patches(image, nolesion, patch_size)
            y_neg_samples = get_patches(lesion, nolesion, patch_size)

            # concatenate positive and negative samples
            if len(l_centers) > 0:
                X.append(np.concatenate([x_pos_samples, x_neg_samples]))
                Y.append(np.concatenate([y_pos_samples, y_neg_samples]))
            else:
                X.append(x_neg_samples)
                Y.append(y_neg_samples)
        st = st + 1
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X, Y


def dataset_test_patches(test_x_data,
                      patch_size,
                      batch_size,
                      voxel_candidates=None,
                      datatype=np.float32):
    """
    Function generator to load test patches with size equal to patch_size,
    given a list of selected voxels. Patches are returned in batches to reduce
    the amount of RAM used

    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - selected_voxels: list where each element contains the subject binary
         mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
       - Voxel candidates: a binary mask containing voxels for testing

    Outputs (in batches):
       - X: Train X data matrix for the each channel [num_samples, p1, p2, p3]
       - voxel_coord: list of tuples with voxel coordinates (x,y,z) of
         selected patches
    """

    # get scan names and number of modalities used
    scans = list(test_x_data.keys())
    # modalities = settings['input_modality'][0]

    # load all image modalities and normalize intensities
    images = []
    # print("raw images:", test_x_data)
    # for m in modalities:
    raw_images = [load_nii(test_x_data[s]).get_data() for s in scans]

    images.append([normalize_data(im) for im in raw_images])

    # select voxels for testing. Discard CSF and darker WM in FLAIR.
    # If voxel_candidates is not selected, using intensity > 0.5 in FLAIR,
    # else use the binary mask to extract candidate voxels
    # print("voxel_candidates", voxel_candidates)
    if voxel_candidates is None:
        print("case: voxel candidates is None!")
        flair_scans = [test_x_data[s] for s in scans]
        selected_voxels = [get_mask_voxels(mask)
                           for mask in select_training_voxels(flair_scans,
                                                              0.5)][0]
        # flair_scans = [test_x_data[s] for s in scans]
        # selected_voxels = select_training_voxels(flair_scans, 0.5)
    else:
        print("case: voxel candidates is known!")
        selected_voxels = get_mask_voxels(voxel_candidates)

    # yield data for testing with size equal to batch_size
    # for i in range(0, len(selected_voxels), batch_size):
    #     c_centers = selected_voxels[i:i+batch_size]
    #     X = []
    #     for m, image_modality in zip(modalities, images):
    #         X.append(get_patches(image_modality[0], c_centers, patch_size))
    #     yield np.stack(X, axis=1), c_centers

    X = []

    for image_modality in images:
        # print(image_modality[0])
        # X.append(get_patches(image_modality[0], selected_voxels, patch_size))
        X.append(get_patches(image_modality[0], selected_voxels, patch_size))
    # x_ = np.empty((9200, 400, 400, 3)
    # Xs = np.zeros_like (X)
    Xs = np.stack(X, axis=1)
    return Xs, selected_voxels

def dataset_test_patches_batch(test_x_data,
                      patch_size,
                      batch_size,
                      voxel_candidates = None,
                      datatype=np.float32):
    """
    Function generator to load test patches with size equal to patch_size,
    given a list of selected voxels. Patches are returned in batches to reduce
    the amount of RAM used
    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - selected_voxels: list where each element contains the subject binary
         mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
       - Voxel candidates: a binary mask containing voxels for testing
    Outputs (in batches):
       - X: Train X data matrix for the each channel [num_samples, p1, p2, p3]
       - voxel_coord: list of tuples with voxel coordinates (x,y,z) of
         selected patches
    """

    # get scan names and number of modalities used
    scans = list(test_x_data.keys())
    # modalities = list(test_x_data[scans[0]].keys())

    # load all image modalities and normalize intensities
    images = []

    # for m in modalities:
    raw_images = [load_nii(test_x_data[s]).get_data() for s in scans]
    images.append([normalize_data(im) for im in raw_images])

    # select voxels for testing. Discard CSF and darker WM in FLAIR.
    # If voxel_candidates is not selected, using intensity > 0.5 in FLAIR,
    # else use the binary mask to extract candidate voxels
    if voxel_candidates is None:
        flair_scans = [test_x_data[s]for s in scans]
        selected_voxels = [get_mask_voxels(mask)
                           for mask in select_training_voxels(flair_scans,
                                                              0.5)][0]
    else:
        selected_voxels = get_mask_voxels(voxel_candidates)

    # yield data for testing with size equal to batch_size
    # for i in range(0, len(selected_voxels), batch_size):
    #     c_centers = selected_voxels[i:i+batch_size]
    #     X = []
    #     for m, image_modality in zip(modalities, images):
    #         X.append(get_patches(image_modality[0], c_centers, patch_size))
    #     yield np.stack(X, axis=1), c_centers
    for i in range(0, len(selected_voxels), batch_size):
            c_centers = selected_voxels[i:i + batch_size]
            X = []
            # for m, image_modality in zip(modalities, images):
            for image_modality in images:
                X.append(get_patches(image_modality[0], c_centers, patch_size))
            yield np.stack(X, axis=1), c_centers




def sc_one_zero(array):
    for x in array.flat:
        if x!=1 and x!=0:
            return True
    return False


def get_mask_voxels(mask):
    """
    Compute x,y,z coordinates of a binary mask

    Input:
       - mask: binary mask

    Output:
       - list of tuples containing the (x,y,z) coordinate for each of the
         input voxels
    """
    # to do what if binary mask got some error (is not real binary!)
    # X = np.array(mask)
    # # im = im.astype(dtype=datatype) - im[np.nonzero(im)].mean()
    # print mask[np.nonzero(mask)].astype(dtype=np.float32)




    # if sc_one_zero(mask[np.nonzero(mask)]):
    #     print("lesion mask is not real binary, please check the inputs and try again!")
    #     time.sleep(1)
    #     os.kill(os.getpid(), signal.SIGTERM)
    indices = np.stack(np.nonzero(mask), axis=1)
    indices = [tuple(idx) for idx in indices]
    return indices


def get_patches(image, centers, patch_size=(15, 15, 15)):
    """
    Get image patches of arbitrary size based on a set of centers
    """
    # If the size has even numbers, the patch will be centered. If not,
    # it will try to create an square almost centered. By doing this we allow
    # pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]

    if list_of_tuples and sizes_match:
        patch_half = tuple([idx//2 for idx in patch_size])
        new_centers = [list(map(add, center, patch_half)) for center in centers]
        padding = tuple((idx, size-idx)
                        for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = [[slice(c_idx-p_idx, c_idx+(s_idx-p_idx))
                   for (c_idx, p_idx, s_idx) in zip(center,
                                                    patch_half,
                                                    patch_size)]
                  for center in new_centers]
        # patches = [new_image[idx] for idx in slices]
        patches = [new_image[tuple(idx)] for idx in slices]

    return patches


def prediction(model,
              test_x_data,
              settings,
              index,
              save_nifti=True,
              candidate_mask=None):


    # get_scan name and create an empty nifti image to store segmentation
    scans = list(test_x_data.keys())
    flair_scans = [test_x_data[s] for s in scans]
    flair_image = load_nii(flair_scans[0])
    seg_image = np.zeros_like(flair_image.get_data().astype('float32'))
    this_size = len(settings['all_mod'])
    if candidate_mask is not None:
        all_voxels = np.sum(candidate_mask[index])
    else:
        all_voxels = np.sum(flair_image.get_data() > 0)

    if settings['debug'] is True:
            print("> DEBUG ", scans[0], "Voxels to classify:", all_voxels)

    # compute lesion segmentation in batches of size settings['batch_size']
    # batch, centers = dataset_test_patches(test_x_data,
    #                                    settings['patch_size'],
    #                                    settings['batch_size'],
    #                                    candidate_mask)
    batch = {}
    centers = {}
    if candidate_mask is None:
        print("candidate_mask is None!")

        for i in range(0, this_size):
            batch[i], centers[i] = dataset_test_patches(test_x_data,
                                                     patch_size=settings['patch_size'],
                                                     batch_size=settings['batch_size'], voxel_candidates=None)


    if candidate_mask is not None:
        print("candidate_mask is known!")
        for i in range(0, this_size):
            batch[i], centers[i] = dataset_test_patches(test_x_data,
                                                     patch_size=settings['patch_size'],
                                                     batch_size=settings['batch_size'],
                                                     voxel_candidates=candidate_mask[i])
    # print ("centers:", centers)
    if settings['debug'] is True:
        print("> DEBUG: testing current_batch[", index + 1, "]:", batch[index].shape, end=' ')

    # if settings['debug'] is True:
    #     print("> DEBUG: testing current_batch:", batch.shape, end=' ')

    batch_array = []
    for ind in range(0, this_size):
        batch_array.append(batch[index])

    print(" \n")
    # print("Prediction or loading learned model started........................> \n")
    w_class = int(index) + 1
    out_class = int(index)
    print("Prediction or loading learned model for input:", '\x1b[6;30;41m' + str(index + 1) + '\x1b[0m',  "of", CRED2 + str(test_x_data) + CEND, "\n")
    # print("Loading data of label:", CRED2 + label_n[i] + CENprint("Prediction or loading learned model started........................> \n")D, ", class:", CGREEN + str(j + 1) + CEND, "for training")
    prediction_time = time.time()
    # batch = [batch, batch, batch, batch, batch]
    # y_pred_all = model['model'].predict([np.squeeze(batch[index]), np.squeeze(batch[index]), np.squeeze(batch[index]), np.squeeze(batch[index]), np.squeeze(batch[index])],
    #                               settings['batch_size'])
    y_pred_all = model['model'].predict(batch_array, settings['batch_size'])

    print("Prediction or loading learned model: ", round(time.time() - prediction_time), "sec")

    # for i in range(0, 5):
    # print("y_pred_all[", i, "].shape[0]", y_pred_all[i].shape[0])
    print("Prediction final stage for input:", '\x1b[6;30;41m' + str(index + 1) + '\x1b[0m', "of", CRED2 + str(test_x_data) + CEND, "\n")
    # y_pred = y_pred_all[index]
    # y_pred = y_pred_all

    y_pred = y_pred_all[out_class]
    [x, y, z] = np.stack(centers[index], axis=1)
    seg_image[x, y, z] = y_pred[:, 1]
    if settings['debug'] is True:
        print("...done!")

    # check if the computed volume is lower than the minimum accuracy given
    # by the error_tolerance parameter
    # if check_error_tolerance(seg_image, settings, flair_image.header.get_zooms()):
    #       if settings['debug']:
    #          print("> DEBUG ", scans[0], "lesion volume below ", settings['error_tolerance'], 'ml')
    #       seg_image = np.zeros_like(flair_image.get_data().astype('float32'))
    #



    if save_nifti:
        out_scan = nib.Nifti1Image(seg_image, affine=flair_image.affine)
        out_scan.to_filename(os.path.join(settings['inference_folder'],
                                          settings['prediction'],
                                          settings['modelname'],
                                          settings['prediction_name']))


    # if save_nifti:
    #     out_scan = nib.Nifti1Image(seg_image, affine=flair_image.affine)
    #     out_scan.to_filename(os.path.join(settings['inference_folder'],
    #                                       settings['prediction'],
    #                                       settings['modelname'], str(index) +
    #                                       settings['prediction_name']))


    return seg_image

def prediction_single(model,
              test_x_data,
              settings,
              save_nifti=True,
              candidate_mask=None):


    # get_scan name and create an empty nifti image to store segmentation
    scans = list(test_x_data.keys())
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0])
    seg_image = np.zeros_like(flair_image.get_data().astype('float32'))

    if candidate_mask is not None:
        all_voxels = np.sum(candidate_mask)
    else:
        all_voxels = np.sum(flair_image.get_data() > 0)

    if settings['debug'] is True:
            print("> DEBUG ", scans[0], "Voxels to classify:", all_voxels)

    if settings['batch_prediction']:
        for batch, centers in dataset_test_patches_batch(test_x_data,
                                                      settings['patch_size'],
                                                      settings['batch_size'],
                                                      candidate_mask):
            if settings['debug'] is True:
                print("> DEBUG: testing current_batch:", batch.shape, end=' ')
            print(" \n")
            print("Batch_Prediction or loading learned model started........................> \n")

            prediction_time = time.time()
            y_pred = model['model'].predict(batch,
                                          settings['batch_size'])
            print("Prediction or loading learned model: ", round(time.time() - prediction_time), "sec")

            [x, y, z] = np.stack(centers, axis=1)
            seg_image[x, y, z] = y_pred[:, 1]
        if settings['debug'] is True:
            print("...done!")

        #  ////////////////
    else:
    # compute lesion segmentation in batches of size settings['batch_size']
        batch, centers = dataset_test_patches(test_x_data,
                                       settings['patch_size'],
                                       settings['batch_size'],
                                       candidate_mask)
        if settings['debug'] is True:
            print("> DEBUG: testing current_batch:", batch.shape, end=' ')
        print (" \n")
        print("Prediction or loading learned model started........................> \n")

        prediction_time = time.time()
    # y_pred = model['model'].predict(np.squeeze(batch),
    #                               settings['batch_size'])
        y_pred = model['model'].predict(batch,
                                  settings['batch_size'])
        print("Prediction or loading learned model: ", round(time.time() - prediction_time), "sec")


        [x, y, z] = np.stack(centers, axis=1)
        seg_image[x, y, z] = y_pred[:, 1]
        if settings['debug'] is True:
            print("...done!")

    # check if the computed volume is lower than the minimum accuracy given
    # by the error_tolerance parameter
    if check_error_tolerance(seg_image, settings, flair_image.header.get_zooms()):
        if settings['debug']:
            print("> DEBUG ", scans[0], "lesion volume below ", \
                settings['error_tolerance'], 'ml')
        seg_image = np.zeros_like(flair_image.get_data().astype('float32'))

    if save_nifti:
        out_scan = nib.Nifti1Image(seg_image, affine=flair_image.affine)
        out_scan.to_filename(os.path.join(settings['inference_folder'],
                                          settings['prediction'],
                                          settings['modelname'],
                                          settings['prediction_name']))

    return seg_image


def check_error_tolerance(input_scan, settings, voxel_size):
    """
    check that the output volume is higher than the minimum accuracy
    given by the
    parameter error_tolerance
    """

    from scipy import ndimage

    threshold = settings['threshold']
    volume_tolerance = settings['volume_tolerance']

    # get voxel size in mm^3
    voxel_size = np.prod(voxel_size) / 1000.0

    # threshold input segmentation
    output_scan = np.zeros_like(input_scan)
    t_segmentation = input_scan >= threshold

    # filter candidates by size and store those > volume_tolerance
    labels, num_labels = ndimage.label(t_segmentation)
    label_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
                                                           labels,
                                                           label_list,
                                                           np.sum,
                                                           float, 0)

    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > volume_tolerance:
            # assign voxels to output
            current_voxels = np.stack(np.where(labels == l), axis=1)
            output_scan[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1

    return (np.sum(output_scan == 1) * voxel_size) < settings['error_tolerance']


# def voxels_from_learned_model(model, train_x_data, settings):
def voxels_from_learned_model(model, train_x_data, settings, index):

    """
    Select training voxels from image segmentation masks

    """

    # get_scan names and number of modalities used
    scans = list(train_x_data.keys())
    print(train_x_data.keys())
    # select voxels for training. Discard CSF and darker WM in FLAIR.
    # flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    # selected_voxels = select_training_voxels(flair_scans, settings['min_th'])

    # evaluate training scans using the learned model and extract voxels with
    # probability higher than 0.5

    seg_masks = []
    for scan, s in zip(list(train_x_data.keys()), list(range(len(scans)))):
        seg_mask = prediction(model,
                             dict(list(train_x_data.items())[s:s+1]),
                             settings, index, save_nifti=False)
        seg_masks.append(seg_mask > 0.5)

    flair_scans = [train_x_data[s] for s in scans]
    images = [load_nii(name).get_data() for name in flair_scans]
    images_norm = [normalize_data(im) for im in images]

    num = 1
    for seg in seg_masks:
        if not seg.any():
            print('\x1b[6;30;41m' + 'Warning:' + '\x1b[0m', 'after evaluating the training scan number:', num,' and applying probability higher than 0.5, no voxels have been selected, list contains empty element!' )
            print('')
            num = num + 1

    # seg_mask = [im > -2.5 if np.sum(seg) == 0 else seg
    #             for im, seg in zip(images_norm, seg_masks)]

    seg_mask = [im > -2.5 if not seg.any() else seg
                 for im, seg in zip(images_norm, seg_masks)]

    return seg_mask


def final_process(input_scan,
                              settings,
                              save_nifti=True,
                              orientation=np.eye(4)):


    from scipy import ndimage

    threshold = settings['threshold']
    volume_tolerance = settings['volume_tolerance']
    output_scan = np.zeros_like(input_scan)

    # threshold input segmentation
    t_segmentation = input_scan >= threshold

    # filter candidates by size and store those > volume_tolerance
    labels, num_labels = ndimage.label(t_segmentation)
    label_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
                                                           labels,
                                                           label_list,
                                                           np.sum,
                                                           float, 0)

    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > volume_tolerance:
            # assign voxels to output
            current_voxels = np.stack(np.where(labels == l), axis=1)
            output_scan[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1

    # save the output segmentation as Nifti1Image
    if save_nifti:
        nifti_out = nib.Nifti1Image(output_scan,
                                    affine=orientation)
        nifti_out.to_filename(os.path.join(settings['inference_folder'],
                                           settings['prediction'],
                                           settings['modelname'],
                                           settings['prediction_name']))

    return output_scan

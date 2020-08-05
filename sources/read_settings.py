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

def load_settings(traintest_config):

    settings = {}

    # experiment name (where trained weights are)
    settings['modelname'] = traintest_config.get('traintestset', 'name')
    settings['training_folder'] = traintest_config.get('traintestset', 'training_folder')
    settings['cross_validation_folder'] = traintest_config.get('traintestset', 'cross_validation_folder')
    settings['inference_folder'] = traintest_config.get('traintestset', 'inference_folder')
    settings['output_folder'] = '/output'
    settings['current_scan'] = 'scan'
    # settings['t1_name'] = traintest_config.get('traintestset', 't1_name')
    # settings['flair_name'] = traintest_config.get('traintestset', 'flair_name')
    settings['InputModality'] = [el.strip() for el in
                             traintest_config.get('traintestset',
                                                'InputModality').split(',')]

    settings['InputLabel'] = [el.strip() for el in
                           traintest_config.get('traintestset',
                                              'InputLabel').split(',')]



    # settings['ROI_name'] = traintest_config.get('traintestset', 'ROI_name')
    settings['debug'] = traintest_config.get('traintestset', 'debug')

    # modalities = [str(settings['InputModality'][0]),
    #               settings['t1_tags'][0],
    #               settings['mod3_tags'][0],
    #               settings['mod4_tags'][0]]
    # names = ['FLAIR', 'T1', 'MOD3', 'MOD4']

    modalities = [str(settings['InputModality'][0])]
    names = ['FLAIR']

    settings['input_modality'] = [n for n, m in
                             zip(names, modalities) if m != 'None']
    settings['image_tags'] = [m for m in modalities if m != 'None']
    settings['x_names'] = [n + '_tmp.nii.gz' for n, m in
                          zip(names, modalities) if m != 'None']

    settings['out_name'] = 'out_seg.nii.gz'

    # preprocessing
    settings['register_modalities'] = (traintest_config.get('traintestset',
                                                         'register_modalities'))


    settings['Longitudinal'] = (traintest_config.get('traintestset',
                                                         'Longitudinal'))

    settings['Cross_Sectional'] = (traintest_config.get('traintestset',
                                                         'Cross_Sectional'))


    settings['Homogeneous'] = (traintest_config.get('traintestset',
                                                         'Homogeneous'))

    settings['Hybrid'] = (traintest_config.get('traintestset',
                                                         'Hybrid'))



    settings['reg_space'] = (traintest_config.get('traintestset',
                                               'reg_space'))

    settings['denoise'] = (traintest_config.get('traintestset',
                                             'denoise'))
    settings['denoise_iter'] = (traintest_config.getint('traintestset',
                                                     'denoise_iter'))
    settings['bias_iter'] = (traintest_config.getint('traintestset',
                                                  'bias_iter'))

    settings['inputmodlengthwise'] = (traintest_config.getint('traintestset',
                                                  'inputmodlengthwise'))




    settings['bias_smooth'] = (traintest_config.getint('traintestset',
                                                    'bias_smooth'))
    settings['bias_type'] = (traintest_config.getint('traintestset',
                                                  'bias_type'))
    settings['bias_choice'] = (traintest_config.get('traintestset',
                                                 'bias_choice'))

    settings['bias_correction'] = (traintest_config.get('traintestset',
                                                     'bias_correction'))

    settings['batch_prediction'] = (traintest_config.get('traintestset',
                                                      'batch_prediction'))

    settings['skull_stripping'] = (traintest_config.get('traintestset',
                                                     'skull_stripping'))
    settings['save_tmp'] = (traintest_config.get('traintestset', 'save_tmp'))

    settings['all_label'] = [el.strip() for el in
                           traintest_config.get('traintestset',
                                              'all_label').split(',')]

    settings['all_mod'] = [el.strip() for el in
                           traintest_config.get('traintestset',
                                              'all_input_mod').split(',')]


    # net settings
    # settings['gpu_mode'] = traintest_config.get('model', 'gpu_mode')
    settings['gpu_number'] = traintest_config.getint('traintestset', 'gpu_number')
    settings['learnedmodel'] = traintest_config.get('traintestset', 'learnedmodel')
    settings['min_th'] = -0.5
    settings['fully_convolutional'] = False
    settings['patch_size'] = (11, 11, 11)
    settings['model_saved_paths'] = None
    settings['train_split'] = traintest_config.getfloat('traintestset', 'train_split')
    settings['max_epochs'] = traintest_config.getint('traintestset', 'max_epochs')
    settings['patience'] = traintest_config.getint('traintestset', 'patience')
    settings['batch_size'] = traintest_config.getint('traintestset', 'batch_size')
    settings['net_verbose'] = traintest_config.getint('traintestset', 'net_verbose')

    settings['tensorboard'] = traintest_config.get('tensorboard', 'tensorboard_folder')
    settings['port'] = traintest_config.getint('tensorboard', 'port')

    # settings['load_weights'] = traintest_config.get('model', 'load_weights')
    settings['load_weights'] = True
    settings['randomize_train'] = True

    # post processing settings
    settings['threshold'] = traintest_config.getfloat('traintestset', 'threshold')
    settings['volume_tolerance'] = traintest_config.getint('traintestset', 'volume_tolerance')
    settings['error_tolerance'] = traintest_config.getfloat('traintestset',
                                                   'error_tolerance')

    # training settings  model_1_train
    settings['full_train'] = (traintest_config.get('traintestset', 'full_train'))
    settings['model_1_train'] = (traintest_config.get('completed', 'model_1_train'))
    settings['model_2_train'] = (traintest_config.get('completed', 'model_2_train'))
    settings['pre_processing'] = (traintest_config.get('completed', 'pre_processing'))
    settings['learnedmodel_model'] = traintest_config.get('traintestset',
                                                     'learnedmodel_model')

    settings['balanced_training'] = traintest_config.get('traintestset',
                                                      'balanced_training')

    settings['ratio_negative_positive1'] = traintest_config.getfloat('traintestset',
                                                                 'fraction_negatives')

    settings['ratio_negative_positive2'] = traintest_config.getfloat('traintestset',
                                                                 'fraction_negatives_CV')
    settings['num_layers'] = None

    keys = list(settings.keys())
    for k in keys:
        value = settings[k]
        if value == 'True':
            settings[k] = True
        if value == 'False':
            settings[k] = False

    return settings

def Train_Test_settings(settings):
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
    print('\x1b[6;30;45m' + 'Train/Test settings' + '\x1b[0m')
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
    print(" ")
    keys = list(settings.keys())
    for key in keys:
        print(CRED + key, ':' + CEND, settings[key])
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
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

import configparser
import argparse
import platform
import subprocess
import os
import signal
import queue
import threading
from __init__ import __version__
from tkinter import Frame, LabelFrame, Label, END, Tk
from tkinter import Entry, Button, Checkbutton, OptionMenu, Toplevel, Text
from tkinter import BooleanVar, StringVar, IntVar, DoubleVar
from tkinter.filedialog import askdirectory
from tkinter.ttk import Notebook
# from tkinter import *
from PIL import Image, ImageTk
import webbrowser
from cnn_main_Longitudinal import train_network, train_test_network, infer_segmentation, overall_config
from cnn_main_cross import train_network_cross, train_test_network_cross, infer_segmentation_cross, overall_config
##################

from tkinter.ttk import Label
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

class AnimatedGIF(Label, object):
    def __init__(self, master, path, forever=True):
        self._master = master
        self._loc = 0
        self._forever = forever

        self._is_running = False

        im = Image.open(path)
        self._frames = []
        i = 0
        try:
            while True:
                # photoframe = ImageTk.PhotoImage(im.copy().convert('RGBA'))
                photoframe = ImageTk.PhotoImage(im)
                self._frames.append(photoframe)

                i += 1
                im.seek(i)
        except EOFError:
            pass

        self._last_index = len(self._frames) - 1

        try:
            self._delay = im.info['duration']
        except:
            self._delay = 1000

        self._callback_id = None

        super(AnimatedGIF, self).__init__(master, image=self._frames[0])

    def start_animation(self, frame=None):
        if self._is_running: return

        if frame is not None:
            self._loc = 0
            self.configure(image=self._frames[frame])

        self._master.after(self._delay, self._animate_GIF)
        self._is_running = True

    def stop_animation(self):
        if not self._is_running: return

        if self._callback_id is not None:
            self.after_cancel(self._callback_id)
            self._callback_id = None

        self._is_running = False

    def _animate_GIF(self):
        self._loc += 1
        self.configure(image=self._frames[self._loc])

        if self._loc == self._last_index:
            if self._forever:
                self._loc = 0
                self._callback_id = self._master.after(self._delay, self._animate_GIF)
            else:
                self._callback_id = None
                self._is_running = False
        else:
            self._callback_id = self._master.after(self._delay, self._animate_GIF)

    def pack(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).pack(**kwargs)

    def grid(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).grid(**kwargs)

    def place(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).place(**kwargs)

    def pack_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).pack_forget(**kwargs)

    def grid_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).grid_forget(**kwargs)

    def place_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).place_forget(**kwargs)


class CNN:

    def __init__(self, master, container):

        self.master = master
        master.title("UCL Department of Medical Physics and Biomedical Engineering")

        # running on a container
        self.container = container

        # gui attributes
        self.path = os.getcwd()
        self.traintest_config = None
        self.user_config = None
        self.current_folder = os.getcwd()
        self.list_train_learnedmodel_nets = []
        self.setting_bias_choice = StringVar()

        self.list_bias = ['All', 'FLAIR', 'T1', 'MOD3', 'MOD4']

        self.list_standard_space_reg_list = ['FlairtoT1', 'T1toFlair', 'avg152T1.nii.gz', 'avg152T1_brain.nii.gz',
                                             'FMRIB58_FA_1mm.nii.gz', 'FMRIB58_FA-skeleton_1mm.nii.gz',
                                             'Fornix_FMRIB_FA1mm.nii.gz', 'LowerCingulum_1mm.nii.gz',
                                             'MNI152lin_T1_1mm.nii.gz', 'MNI152lin_T1_1mm_brain.nii.gz',
                                             'MNI152lin_T1_1mm_subbr_mask.nii.gz',
                                             'MNI152lin_T1_2mm.nii.gz', 'MNI152lin_T1_2mm_brain.nii.gz',
                                             'MNI152lin_T1_2mm_brain_mask.nii.gz',
                                             'MNI152_T1_0.5mm.nii.gz', 'MNI152_T1_1mm.nii.gz',
                                             'MNI152_T1_1mm_brain.nii.gz',
                                             'MNI152_T1_1mm_brain_mask.nii.gz', 'MNI152_T1_1mm_brain_mask_dil.nii.gz',
                                             'MNI152_T1_1mm_first_brain_mask.nii.gz',
                                             'MNI152_T1_1mm_Hipp_mask_dil8.nii.gz', 'MNI152_T1_2mm.nii.gz',
                                             'MNI152_T1_2mm_b0.nii.gz', 'MNI152_T1_2mm_brain.nii.gz',
                                             'MNI152_T1_2mm_brain_mask.nii.gz',
                                             'MNI152_T1_2mm_brain_mask_deweight_eyes.nii.gz',
                                             'MNI152_T1_2mm_brain_mask_dil.nii.gz',
                                             'MNI152_T1_2mm_brain_mask_dil1.nii.gz', 'MNI152_T1_2mm_edges.nii.gz',
                                             'MNI152_T1_2mm_eye_mask.nii.gz',
                                             'MNI152_T1_2mm_LR-masked.nii.gz', 'MNI152_T1_2mm_skull.nii.gz',
                                             'MNI152_T1_2mm_strucseg.nii.gz',
                                             'MNI152_T1_2mm_strucseg_periph.nii.gz',
                                             'MNI152_T1_2mm_VentricleMask.nii.gz']
        self.list_test_nets = []
        self.version = __version__
        self.training_do = None
        self.testing_do = None
        self.test_queue = queue.Queue()
        self.train_queue = queue.Queue()
        self.setting_training_folder = StringVar()
        self.setting_cross_validation = StringVar()
        self.setting_test_folder = StringVar()
        self.setting_tensorboard_folder = StringVar()
        self.setting_port_value = IntVar()
        self.setting_FLAIR_tag = StringVar()
        self.setting_PORT_tag = IntVar()
        self.setting_T1_tag = StringVar()
        self.setting_MOD3_tag = StringVar()
        self.setting_MOD4_tag = StringVar()
        self.setting_mask_tag = StringVar()
        self.setting_model_tag = StringVar()
        self.setting_register_modalities = BooleanVar()
        self.setting_bias_correction = BooleanVar()
        self.setting_batch_prediction = BooleanVar()
        self.setting_register_modalities_Kind = StringVar()
        self.setting_skull_stripping = BooleanVar()
        self.setting_denoise = BooleanVar()
        self.setting_denoise_iter = IntVar()
        self.setting_save_tmp = BooleanVar()
        self.setting_debug = BooleanVar()
        self.setting_inputmodlengthwise = IntVar()
        self.setting_net_folder = os.path.join(self.current_folder, 'models')
        self.setting_use_learnedmodel_model = BooleanVar()
        self.setting_reg_space = StringVar()
        self.setting_learnedmodel_model = StringVar()
        self.setting_inference_model = StringVar()
        self.setting_num_layers = IntVar()
        self.setting_net_name = StringVar()
        self.setting_net_name.set('None')
        self.setting_balanced_dataset = StringVar()
        self.setting_fract_negatives = DoubleVar()
        self.setting_fract_negatives_cv = DoubleVar()
        self.setting_Longitudinal = BooleanVar()
        self.setting_Cross_Sectional = BooleanVar()
        self.setting_Homogeneous = BooleanVar()
        self.setting_Hybrid = BooleanVar()
        self.model_1_train = BooleanVar()
        self.model_2_train = BooleanVar()
        self.pre_processing = BooleanVar()
        self.all_label = StringVar()
        self.all_input_mod = StringVar()
        self.setting_Bias_cor_niter = IntVar()
        self.setting_Bias_cor_smooth = IntVar()
        self.setting_Bias_cor_type = IntVar()
        self.setting_predefiend_reg1 = None
        self.setting_predefiend_reg2 = 'T1toFlair'
        self.setting_learnedmodel = None
        self.setting_min_th = DoubleVar()
        self.setting_patch_size = IntVar()
        self.setting_weight_paths = StringVar()
        self.setting_load_weights = BooleanVar()
        self.setting_train_split = DoubleVar()
        self.setting_max_epochs = IntVar()
        self.setting_patience = IntVar()
        self.setting_batch_size = IntVar()
        self.setting_net_verbose = IntVar()
        self.setting_threshold = DoubleVar()
        self.setting_volume_tolerance = IntVar()
        self.setting_error_tolerance = DoubleVar()
        self.setting_mode = BooleanVar()
        self.setting_gpu_number = IntVar()
        self.load_traintest_configuration()
        self.updated_traintest_configuration()
        self.note = Notebook(self.master)
        self.note.pack()

        os.system('cls' if platform.system() == 'Windows' else 'clear')

        print("##################################################")
        print('\x1b[6;30;45m' + 'Deep multi-task learning framework        ' + '\x1b[0m')
        print('\x1b[6;30;45m' + 'Medical Physics and Biomedical Engineering' + '\x1b[0m')
        print('\x1b[6;30;45m' + 'UCL - 2020                                ' + '\x1b[0m')
        print('\x1b[6;30;45m' + 'Kevin Bronik                              ' + '\x1b[0m')
        print("##################################################")

        self.train_frame = Frame()
        self.note.add(self.train_frame, text="Training and Inference")

        # label frames
        cl_s = 6
        self.tr_frame = LabelFrame(self.train_frame, text="Training images:")
        self.tr_frame.grid(row=0, columnspan=cl_s, sticky='WE',
                           padx=5, pady=5, ipadx=5, ipady=5)
        self.model_frame = LabelFrame(self.train_frame, text="CNN model:")
        self.model_frame.grid(row=5, columnspan=cl_s, sticky='WE',
                              padx=5, pady=5, ipadx=5, ipady=5)
        self.tb_frame = LabelFrame(self.train_frame, text="TensorBoard Option:")
        self.tb_frame.grid(row=6, columnspan=cl_s, sticky='WE',
                              padx=5, pady=5, ipadx=5, ipady=5)

        # training settings
        self.inFolderLbl = Label(self.tr_frame, text="Training folder:")
        self.inFolderLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        self.inFolderTxt = Entry(self.tr_frame)
        self.inFolderTxt.grid(row=0,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.inFileBtn = Button(self.tr_frame, text="Browse ...",
                                command=self.load_training_path)
        self.inFileBtn.grid(row=0,
                            column=5,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)

        self.incvFolderLbl = Label(self.tr_frame, text="Cross-validation folder:")
        self.incvFolderLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.incvFolderTxt = Entry(self.tr_frame)
        self.incvFolderTxt.grid(row=1,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.incvFileBtn = Button(self.tr_frame, text="Browse ...",
                                command=self.load_cross_validation_path)
        self.incvFileBtn.grid(row=1,
                            column=5,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)


        self.test_inFolderLbl = Label(self.tr_frame, text="Inference folder:")
        self.test_inFolderLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.test_inFolderTxt = Entry(self.tr_frame)
        self.test_inFolderTxt.grid(row=2,
                                   column=1,
                                   columnspan=5,
                                   sticky="W",
                                   pady=3)
        self.test_inFileBtn = Button(self.tr_frame, text="Browse ...",
                                     command=self.load_testing_path)
        self.test_inFileBtn.grid(row=2,
                                 column=5,
                                 columnspan=1,
                                 sticky='W',
                                 padx=5,
                                 pady=1)



        self.settingsBtn = Button(self.tr_frame,
                                 text="Settings",
                                 command=self.settings)
        self.settingsBtn.grid(row=0,
                             column=10,
                             columnspan=1,
                             sticky="W",
                             padx=(50, 1),
                             pady=1)

        self.train_aboutBtn = Button(self.tr_frame,
                                     text="About",
                                     command=self.abw)
        self.train_aboutBtn.grid(row=1,
                                 column=10,
                                 columnspan=1,
                                 sticky="W",
                                 padx=(50, 1),
                                 pady=1)

        self.train_helpBtn = Button(self.tr_frame,
                                     text="Help",
                                     command=self.hlp)
        self.train_helpBtn.grid(row=2,
                                 column=10,
                                 columnspan=1,
                                 sticky="W",
                                 padx=(50, 1),
                                 pady=1)



        self.TensorBoard_inFolderLbl = Label(self.tb_frame , text="TensorBoard folder:")
        self.TensorBoard_inFolderLbl.grid(row=6, column=0, sticky='E', padx=5, pady=2)
        self.TensorBoard_inFolderTxt = Entry(self.tb_frame )
        self.TensorBoard_inFolderTxt.grid(row=6,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.TensorBoard_inFileBtn = Button(self.tb_frame, text="Browse ...",
                                command=self.load_tensorBoard_path)
        self.TensorBoard_inFileBtn.grid(row=6,
                            column=9,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)
        self.portTagLbl = Label(self.tb_frame, text="Port:")
        self.portTagLbl.grid(row=7, column=0, sticky='E', padx=5, pady=2)
        self.portTxt = Entry(self.tb_frame,
                              textvariable=self.setting_PORT_tag)
        self.portTxt.grid(row=7, column=1, columnspan=1, sticky="W", pady=1)


        self.TensorBoardBtn = Button(self.tb_frame,
                                  state='disabled',
                                  text="Start TensorBoard",
                                  command=self.start_tensorBoard)
        self.TensorBoardBtn.grid(row=8, column=0, sticky='W', padx=1, pady=1)

        self.flairTagLbl = Label(self.tr_frame, text="Input Modality:")
        self.flairTagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.flairTxt = Entry(self.tr_frame,
                              textvariable=self.setting_FLAIR_tag)
        self.flairTxt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)

        self.maskTagLbl = Label(self.tr_frame, text="Input label:")
        self.maskTagLbl.grid(row=4, column=0,
                             sticky='E', padx=5, pady=2)
        self.maskTxt = Entry(self.tr_frame, textvariable=self.setting_mask_tag)
        self.maskTxt.grid(row=4, column=1, columnspan=1, sticky="W", pady=1)

        self.inputmodlengthwise = Label(self.tr_frame, text="Lengthwise:")
        self.inputmodlengthwise.grid(row=5, column=0,
                             sticky='E', padx=5, pady=2)
        self.inputmodlengthwiseTxt = Entry(self.tr_frame,
                              textvariable=self.setting_inputmodlengthwise)
        self.inputmodlengthwiseTxt.grid(row=5, column=1, columnspan=1, sticky="W", pady=1)

        self.modelTagLbl = Label(self.model_frame, text="Model name:")
        self.modelTagLbl.grid(row=6, column=0,
                              sticky='E', padx=5, pady=2)
        self.modelTxt = Entry(self.model_frame,
                              textvariable=self.setting_net_name)
        self.modelTxt.grid(row=6, column=1, columnspan=1, sticky="W", pady=1)

        self.checkPretrain = Checkbutton(self. model_frame,
                                         text="use learnedmodel",
                                         var=self.setting_use_learnedmodel_model)
        self.checkPretrain.grid(row=6, column=2, padx=5, pady=5)

        self.update_learnedmodel_nets()

        self.pretrainTxt = OptionMenu(self.model_frame,
                                      self.setting_learnedmodel_model,
                                      *self.list_train_learnedmodel_nets)
        self.pretrainTxt.grid(row=6, column=5, sticky='E', padx=5, pady=5)

        self.withLongitudinal = Checkbutton(self.model_frame,
                                    text="First model trained!",
                                    var=self.model_1_train)
        self.withLongitudinal .grid(row=7, column=0, sticky='W', padx=1,
                                  pady=1)


        self.withLongitudinal = Checkbutton(self.model_frame,
                                    text="Second model trained!",
                                    var=self.model_2_train)
        self.withLongitudinal .grid(row=7, column=1, sticky='W', padx=1,
                                  pady=1)

        self.withLongitudinal = Checkbutton(self.model_frame,
                                    text="Preprocessing done!",
                                    var=self.pre_processing)
        self.withLongitudinal .grid(row=7, column=2, sticky='W', padx=1,
                                  pady=1)

        # START button links
        self.trainingBtn = Button(self.train_frame,
                                  state='disabled',
                                  text="Run only training",
                                  command=self.train_net)
        self.trainingBtn.grid(row=7, column=0, sticky='W', padx=1, pady=1)

        self.traininginferenceBtn = Button(self.train_frame,
                                  state='disabled',
                                  text="Run training and inference",
                                  command=self.train_test_net)
        self.traininginferenceBtn.grid(row=8, column=0, sticky='W', padx=1, pady=1)

        self.testingBtn = Button(self.train_frame,
                                  state='disabled',
                                  text="Run only inference",
                                  command=self.test_net)
        self.testingBtn.grid(row=9, column=0, sticky='W', padx=1, pady=1)

        self.withLongitudinal = Checkbutton(self.train_frame,
                                    text="Longitudinal",
                                    var=self.setting_Longitudinal)
        self.withLongitudinal .grid(row=7, column=0, sticky='W', padx=(240, 1),
                                  pady=1)

        self.withCross_Sectional = Checkbutton(self.train_frame,
                                    text="Cross Sectional",
                                    var=self.setting_Cross_Sectional)
        self.withCross_Sectional.grid(row=7, column=0, sticky='W', padx=(360, 1),
                                  pady=1)


        self.Homogeneous = Checkbutton(self.train_frame,
                                    text="Homogeneous",
                                    var=self.setting_Homogeneous)
        self.Homogeneous.grid(row=8, column=0, sticky='W', padx=(240, 1),
                                  pady=1)


        self.Hybrid = Checkbutton(self.train_frame,
                                    text="Hybrid",
                                    var=self.setting_Hybrid)
        self.Hybrid.grid(row=8, column=0, sticky='W', padx=(360, 1),
                                  pady=1)





        img1 = ImageTk.PhotoImage(Image.open('images/U1.jpg'))
        imglabel = Label(self.train_frame, image=img1)
        imglabel.image = img1
        imglabel.grid(row=10, column=0, padx=1, pady=1)



        # train / test ABOUT button


        # Processing state
        self.process_indicator = StringVar()
        self.process_indicator.set(' ')
        self.label_indicator = Label(master,
                                     textvariable=self.process_indicator)
        self.label_indicator.pack(side="left")

        # Closing processing events is implemented via
        # a master protocol
        self.master.protocol("WM_DELETE_WINDOW", self.terminate)

    def settings(self):

        t = Toplevel(self.master)
        t.wm_title("Additional Parameter Settings")

        # data parameters
        t_data = LabelFrame(t, text="Pre/Post-Processing Settings")
        t_data.grid(row=0, sticky="WE")
        checkBias = Checkbutton(t_data,
                                    text="Bias correction",
                                    var=self.setting_bias_correction)
        checkBias.grid(row=0, sticky='W')

        self.biasTxt = OptionMenu(t_data,  self.setting_bias_choice,
                                      *self.list_bias)
        self.biasTxt.grid(row=0, column=1, sticky='E', padx=5, pady=5)




        Bias_par_iter_label = Label(t_data, text="Bias correction iteration number:")
        Bias_par_iter_label.grid(row=1, sticky="W")
        Bias_par_niter = Entry(t_data,
                                textvariable=self.setting_Bias_cor_niter)
        Bias_par_niter.grid(row=1, column=1, sticky="E")



        Bias_par_smooth_label = Label(t_data, text="Bias correction smooth:")
        Bias_par_smooth_label.grid(row=2, sticky="W")
        Bias_par_smooth = Entry(t_data,
                                    textvariable=self.setting_Bias_cor_smooth)
        Bias_par_smooth.grid(row=2, column=1, sticky="E")

        Bias_par_type_label = Label(t_data, text="Bias correction type: 1 = T1w, 2 = T2w, 3 = PD ")
        Bias_par_type_label.grid(row=3, sticky="W")
        Bias_par_type = Entry(t_data,
                                    textvariable=self.setting_Bias_cor_type)
        Bias_par_type.grid(row=3, column=1, sticky="E")





        checkPretrain = Checkbutton(t_data,
                                    text="Registration",
                                    var=self.setting_register_modalities)


        checkPretrain.grid(row=4, sticky='W')

        # register_label = Label(t_data, text="register mod:(FlairtoT1 or T1toFlair)")
        register_label = Label(t_data, text="Register Mod. to T1, Flair or Std. Space:")
        register_label.grid(row=5, sticky="W")
        # register_label_entry = Entry(t_data, textvariable=self.setting_register_modalities_Kind)
        # register_label_entry.grid(row=1, column=1, sticky="E")
        self.regTxt = OptionMenu(t_data,  self.setting_reg_space,
                                      *self.list_standard_space_reg_list)
        self.regTxt.grid(row=5, column=1, sticky='E', padx=5, pady=5)


        checkSkull = Checkbutton(t_data,
                                 text="Skull Striping",
                                 var=self.setting_skull_stripping)
        checkSkull.grid(row=6, sticky="W")
        checkDenoise = Checkbutton(t_data,
                                   text="Denoising",
                                   var=self.setting_denoise)
        checkDenoise.grid(row=7, sticky="W")

        denoise_iter_label = Label(t_data, text="Noise reduction iterations:               ")
        denoise_iter_label.grid(row=8, sticky="W")
        denoise_iter_entry = Entry(t_data, textvariable=self.setting_denoise_iter)
        denoise_iter_entry.grid(row=8, column=1, sticky="E")


        threshold_label = Label(t_data, text="Threshold:      ")
        threshold_label.grid(row=11, sticky="W")
        threshold_entry = Entry(t_data, textvariable=self.setting_threshold)
        threshold_entry.grid(row=11, column=1, sticky="E")

        volume_tolerance_label = Label(t_data, text="Output Volume Tolerance:         ")
        volume_tolerance_label.grid(row=12, sticky="W")
        volume_tolerance_entry = Entry(t_data, textvariable=self.setting_volume_tolerance)
        volume_tolerance_entry.grid(row=12, column=1, sticky="E")

        vovolume_tolerance_label = Label(t_data, text="Error Tolerance:   ")
        vovolume_tolerance_label.grid(row=13, sticky="W")
        vovolume_tolerance_entry = Entry(t_data, textvariable=self.setting_error_tolerance)
        vovolume_tolerance_entry.grid(row=13, column=1, sticky="E")



        # model parameters
        t_model = LabelFrame(t, text="Training:")
        t_model.grid(row=14, sticky="EW")

        maxepochs_label = Label(t_model, text="Max epochs:                  ")
        maxepochs_label.grid(row=15, sticky="W")
        maxepochs_entry = Entry(t_model, textvariable=self.setting_max_epochs)
        maxepochs_entry.grid(row=15, column=1, sticky="E")

        trainsplit_label = Label(t_model, text="Validation %:           ")
        trainsplit_label.grid(row=16, sticky="W")
        trainsplit_entry = Entry(t_model, textvariable=self.setting_train_split)
        trainsplit_entry.grid(row=16, column=1, sticky="E")

        batchsize_label = Label(t_model, text="Test batch size:")
        batchsize_label.grid(row=17, sticky="W")
        batchsize_entry = Entry(t_model, textvariable=self.setting_batch_size)
        batchsize_entry.grid(row=17, column=1, sticky="E")


        mode_label = Label(t_model, text="Verbosity:")
        mode_label.grid(row=18, sticky="W")
        mode_entry = Entry(t_model, textvariable=self.setting_net_verbose)
        mode_entry.grid(row=18, column=1, sticky="E")

        # gpu_mode = Checkbutton(t_model,
        #                         text="GPU:",
        # #                         var=self.setting_mode)
        # #gpu_mode.grid(row=10, sticky="W")

        gpu_number = Label(t_model, text="GPU number:")
        gpu_number.grid(row=19, sticky="W")
        gpu_entry = Entry(t_model, textvariable=self.setting_gpu_number)
        gpu_entry.grid(row=19, column=1, sticky="W")


        # # training parameters
        # tr_model = LabelFrame(t, text="Training:")
        # tr_model.grid(row=18, sticky="EW")

        balanced_label = Label(t_model, text="Balanced dataset:    ")
        balanced_label.grid(row=20, sticky="W")
        balanced_entry = Entry(t_model, textvariable=self.setting_balanced_dataset)
        balanced_entry.grid(row=20, column=1, sticky="E")

        fraction_label = Label(t_model, text="Fraction negative/positives Training: ")
        fraction_label.grid(row=21, sticky="W")
        fraction_entry = Entry(t_model, textvariable=self.setting_fract_negatives)
        fraction_entry.grid(row=21, column=1, sticky="E")

        fraction_label_cv = Label(t_model, text="Fraction negative/positives Cross Validation: ")
        fraction_label_cv.grid(row=22, sticky="W")
        fraction_entry_cv = Entry(t_model, textvariable=self.setting_fract_negatives_cv)
        fraction_entry_cv.grid(row=22, column=1, sticky="E")


    def load_tensorBoard_path(self):

        initialdir = '/tensorboardlogs' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_tensorboard_folder.set(fname)
                self.TensorBoard_inFolderTxt.delete(0, END)
                self.TensorBoard_inFolderTxt.insert(0, self.setting_tensorboard_folder.get())
                self.TensorBoardBtn['state'] = 'normal'
            except:
                pass

    def load_traintest_configuration(self):


        # traintest_config = configparser.SafeConfigParser()
        traintest_config = configparser.ConfigParser()
        traintest_config.read(os.path.join(self.path, 'config', 'configuration.cfg'))

        # dastaset parameters
        self.setting_training_folder.set(traintest_config.get('traintestset',
                                                          'training_folder'))
        self.setting_cross_validation.set(traintest_config.get('traintestset',
                                                          'cross_validation_folder'))


        self.setting_tensorboard_folder.set(traintest_config.get('tensorboard',
                                                          'tensorBoard_folder'))
        self.setting_PORT_tag.set(traintest_config.getint('tensorboard',
                                                     'port'))

        self.setting_test_folder.set(traintest_config.get('traintestset',
                                                      'inference_folder'))
        # self.setting_FLAIR_tag.set(traintest_config.get('traintestset','Shinkei_tags'))
        self.setting_FLAIR_tag.set(traintest_config.get('traintestset', 'InputModality'))

        # self.setting_T1_tag.set(traintest_config.get('traintestset','t1_tags'))
        # self.setting_MOD3_tag.set(traintest_config.get('traintestset','mod3_tags'))
        # self.setting_MOD4_tag.set(traintest_config.get('traintestset','mod4_tags'))
        self.setting_mask_tag.set(traintest_config.get('traintestset','InputLabel'))


        self.setting_Longitudinal.set(traintest_config.get('traintestset', 'Longitudinal'))
        self.setting_Cross_Sectional.set(traintest_config.get('traintestset', 'Cross_Sectional'))


        self.setting_Homogeneous.set(traintest_config.get('traintestset', 'Homogeneous'))
        self.setting_Hybrid.set(traintest_config.get('traintestset', 'Hybrid'))




        self.setting_register_modalities.set(traintest_config.get('traintestset', 'register_modalities'))
        self.setting_bias_correction.set(traintest_config.get('traintestset', 'bias_correction'))
        self.setting_batch_prediction.set(traintest_config.get('traintestset', 'batch_prediction'))
        # self.setting_batch_prediction
        # self.setting_register_modalities_Kind.set(traintest_config.get('traintestset', 'register_modalities_Kind'))self
        self.setting_denoise.set(traintest_config.get('traintestset', 'denoise'))
        self.setting_denoise_iter.set(traintest_config.getint('traintestset', 'denoise_iter'))
        # self.setting_Bias_cor_niter = IntVar()
        # self.setting_Bias_cor_smooth = IntVar()
        # self.setting_Bias_cor_type = IntVar()
        self.setting_Bias_cor_niter.set(traintest_config.getint('traintestset', 'bias_iter'))
        self.setting_inputmodlengthwise.set(traintest_config.getint('traintestset', 'inputmodlengthwise'))

        self.setting_bias_choice.set(traintest_config.get('traintestset', 'bias_choice'))

        self.setting_Bias_cor_smooth.set(traintest_config.getint('traintestset', 'bias_smooth'))
        self.setting_Bias_cor_type.set(traintest_config.getint('traintestset', 'bias_type'))


        self.setting_skull_stripping.set(traintest_config.get('traintestset', 'skull_stripping'))
        self.setting_save_tmp.set(traintest_config.get('traintestset', 'save_tmp'))
        self.setting_debug.set(traintest_config.get('traintestset', 'debug'))

        # train parameters
        self.setting_use_learnedmodel_model.set(traintest_config.get('traintestset', 'full_train'))
        self.setting_learnedmodel_model.set(traintest_config.get('traintestset', 'learnedmodel'))
        self.setting_reg_space.set(traintest_config.get('traintestset', 'reg_space'))
        # ///////
        self.setting_learnedmodel_model.set(traintest_config.get('traintestset', 'learnedmodel'))

        self.setting_inference_model.set("      ")
        self.setting_balanced_dataset.set(traintest_config.get('traintestset', 'balanced_training'))
        self.setting_fract_negatives.set(traintest_config.getfloat('traintestset', 'fraction_negatives'))
        self.setting_fract_negatives_cv.set(traintest_config.getfloat('traintestset', 'fraction_negatives_CV'))

        # model parameters
        self.setting_net_folder = os.path.join(self.current_folder, 'models')
        self.setting_net_name.set(traintest_config.get('traintestset', 'name'))
        self.setting_train_split.set(traintest_config.getfloat('traintestset', 'train_split'))
        self.setting_max_epochs.set(traintest_config.getint('traintestset', 'max_epochs'))
        self.setting_patience.set(traintest_config.getint('traintestset', 'patience'))
        self.setting_batch_size.set(traintest_config.getint('traintestset', 'batch_size'))
        self.setting_net_verbose.set(traintest_config.get('traintestset', 'net_verbose'))
        self.setting_gpu_number.set(traintest_config.getint('traintestset', 'gpu_number'))

        self.setting_volume_tolerance.set(traintest_config.getint('traintestset',
                                                   'volume_tolerance'))
        self.setting_threshold.set(traintest_config.getfloat('traintestset',
                                                     'threshold'))
        self.setting_error_tolerance.set(traintest_config.getfloat('traintestset',
                                                     'error_tolerance'))


        self.model_1_train.set(traintest_config.get('completed', 'model_1_train'))
        self.model_2_train.set(traintest_config.get('completed', 'model_2_train'))
        self.pre_processing.set(traintest_config.get('completed', 'pre_processing'))
        self.all_label.set(traintest_config.get('traintestset', 'all_label'))
        self.all_input_mod.set(traintest_config.get('traintestset', 'all_input_mod'))


    def updated_traintest_configuration(self):


        # traintest_config = configparser.SafeConfigParser()
        traintest_config = configparser.ConfigParser()
        traintest_config.read(os.path.join(self.path, 'config', 'configuration.cfg'))
        self.model_1_train.set(traintest_config.get('completed', 'model_1_train'))
        self.model_2_train.set(traintest_config.get('completed', 'model_2_train'))
        self.pre_processing.set(traintest_config.get('completed', 'pre_processing'))

        # self.setting_net_name.set(traintest_config.get('traintestset', 'name'))



    def write_user_configuration(self):

        user_config = configparser.ConfigParser()
        # dataset parameters
        user_config.add_section('traintestset')
        user_config.set('traintestset', 'Longitudinal', str(self.setting_Longitudinal.get()))
        user_config.set('traintestset', 'Cross_Sectional', str(self.setting_Cross_Sectional.get()))

        user_config.set('traintestset', 'Homogeneous', str(self.setting_Homogeneous.get()))
        user_config.set('traintestset', 'Hybrid', str(self.setting_Hybrid.get()))


        user_config.set('traintestset', 'training_folder', self.setting_training_folder.get())
        user_config.set('traintestset', 'cross_validation_folder', self.setting_cross_validation.get())

        user_config.set('traintestset', 'inference_folder', self.setting_test_folder.get())


        #user_config.set('traintestset', 'Shinkei_tags', self.setting_FLAIR_tag.get())
        user_config.set('traintestset', 'InputModality', self.setting_FLAIR_tag.get())

        # user_config.set('traintestset', 't1_tags', self.setting_T1_tag.get())
        # user_config.set('traintestset', 'mod3_tags', self.setting_MOD3_tag.get())
        # user_config.set('traintestset', 'mod4_tags', self.setting_MOD4_tag.get())
        user_config.set('traintestset', 'InputLabel', self.setting_mask_tag.get())

        user_config.set('traintestset', 'register_modalities', str(self.setting_register_modalities.get()))
        user_config.set('traintestset', 'bias_correction', str(self.setting_bias_correction.get()))
        user_config.set('traintestset', 'batch_prediction', str(self.setting_batch_prediction.get()))
        # self.setting_batch_prediction.set(traintest_config.get('traintestset', 'batch_prediction'))
        # user_config.set('traintestset', 'register_modalities_Kind', str(self.setting_register_modalities_Kind.get()))
        user_config.set('traintestset', 'reg_space', str(self.setting_reg_space.get()))
        user_config.set('traintestset', 'denoise', str(self.setting_denoise.get()))
        user_config.set('traintestset', 'denoise_iter', str(self.setting_denoise_iter.get()))
        user_config.set('traintestset', 'inputmodlengthwise', str(self.setting_inputmodlengthwise.get()))


        # self.setting_Bias_cor_niter.set(traintest_config.getint('traintestset', 'bias_iter'))
        # self.setting_Bias_cor_smooth.set(traintest_config.getint('traintestset', 'bias_smooth'))
        # self.setting_Bias_cor_type.set(traintest_config.getint('traintestset', 'bias_type'))
        user_config.set('traintestset', 'bias_iter', str(self.setting_Bias_cor_niter.get()))
        user_config.set('traintestset', 'bias_smooth', str(self.setting_Bias_cor_smooth.get()))
        user_config.set('traintestset', 'bias_type', str(self.setting_Bias_cor_type.get()))
        user_config.set('traintestset', 'bias_choice', str(self.setting_bias_choice.get()))

        user_config.set('traintestset', 'skull_stripping', str(self.setting_skull_stripping.get()))
        user_config.set('traintestset', 'save_tmp', str(self.setting_save_tmp.get()))
        user_config.set('traintestset', 'debug', str(self.setting_debug.get()))

        user_config.set('traintestset',
                        'full_train',
                        str(not (self.setting_use_learnedmodel_model.get())))
        user_config.set('traintestset',
                        'learnedmodel_model',
                        str(self.setting_learnedmodel_model.get()))
        user_config.set('traintestset',
                        'balanced_training',
                        str(self.setting_balanced_dataset.get()))
        user_config.set('traintestset',
                        'fraction_negatives',
                        str(self.setting_fract_negatives.get()))

        user_config.set('traintestset',
                        'fraction_negatives_CV',
                        str(self.setting_fract_negatives_cv.get()))

        user_config.set('traintestset', 'name', self.setting_net_name.get())
        user_config.set('traintestset', 'learnedmodel', str(self.setting_learnedmodel))
        user_config.set('traintestset', 'train_split', str(self.setting_train_split.get()))
        user_config.set('traintestset', 'max_epochs', str(self.setting_max_epochs.get()))
        user_config.set('traintestset', 'patience', str(self.setting_patience.get()))
        user_config.set('traintestset', 'batch_size', str(self.setting_batch_size.get()))
        user_config.set('traintestset', 'net_verbose', str(self.setting_net_verbose.get()))
        # user_config.set('model', 'gpu_mode', self.setting_mode.get())
        user_config.set('traintestset', 'gpu_number', str(self.setting_gpu_number.get()))
        user_config.set('traintestset', 'threshold', str(self.setting_threshold.get()))
        user_config.set('traintestset', 'volume_tolerance', str(self.setting_volume_tolerance.get()))
        user_config.set('traintestset',
                        'error_tolerance', str(self.setting_error_tolerance.get()))

        user_config.set('traintestset', 'all_label', self.all_label.get())
        user_config.set('traintestset', 'all_input_mod', self.all_input_mod.get())



        user_config.add_section('tensorboard')
        user_config.set('tensorboard', 'port', str(self.setting_PORT_tag.get()))
        # postprocessing parameters
        user_config.set('tensorboard', 'tensorBoard_folder', self.setting_tensorboard_folder.get())


        user_config.add_section('completed')
        user_config.set('completed', 'model_1_train', str(self.model_1_train.get()))
        user_config.set('completed', 'model_2_train', str(self.model_2_train.get()))
        user_config.set('completed', 'pre_processing', str(self.pre_processing.get()))

        # Writing our configuration file to 'example.cfg'
        with open(os.path.join(self.path,
                               'config',
                               'configuration.cfg'), 'w') as configfile:
            user_config.write(configfile)

    def write_user_configuration_inference(self):


        user_config = configparser.ConfigParser()
        # dataset parameters
        user_config.add_section('traintestset')
        user_config.set('traintestset', 'Longitudinal', str(self.setting_Longitudinal.get()))
        user_config.set('traintestset', 'Cross_Sectional', str(self.setting_Cross_Sectional.get()))

        user_config.set('traintestset', 'Homogeneous', str(self.setting_Homogeneous.get()))
        user_config.set('traintestset', 'Hybrid', str(self.setting_Hybrid.get()))


        user_config.set('traintestset', 'training_folder', self.setting_training_folder.get())
        user_config.set('traintestset', 'cross_validation_folder', self.setting_cross_validation.get())

        user_config.set('traintestset', 'inference_folder', self.setting_test_folder.get())


        #user_config.set('traintestset', 'Shinkei_tags', self.setting_FLAIR_tag.get())
        user_config.set('traintestset', 'InputModality', self.setting_FLAIR_tag.get())

        # user_config.set('traintestset', 't1_tags', self.setting_T1_tag.get())
        # user_config.set('traintestset', 'mod3_tags', self.setting_MOD3_tag.get())
        # user_config.set('traintestset', 'mod4_tags', self.setting_MOD4_tag.get())
        user_config.set('traintestset', 'InputLabel', self.setting_mask_tag.get())

        user_config.set('traintestset', 'register_modalities', str(self.setting_register_modalities.get()))
        user_config.set('traintestset', 'bias_correction', str(self.setting_bias_correction.get()))
        user_config.set('traintestset', 'batch_prediction', str(self.setting_batch_prediction.get()))
        # self.setting_batch_prediction.set(traintest_config.get('traintestset', 'batch_prediction'))
        # user_config.set('traintestset', 'register_modalities_Kind', str(self.setting_register_modalities_Kind.get()))
        user_config.set('traintestset', 'reg_space', str(self.setting_reg_space.get()))
        user_config.set('traintestset', 'denoise', str(self.setting_denoise.get()))
        user_config.set('traintestset', 'denoise_iter', str(self.setting_denoise_iter.get()))
        user_config.set('traintestset', 'inputmodlengthwise', str(self.setting_inputmodlengthwise.get()))


        # self.setting_Bias_cor_niter.set(traintest_config.getint('traintestset', 'bias_iter'))
        # self.setting_Bias_cor_smooth.set(traintest_config.getint('traintestset', 'bias_smooth'))
        # self.setting_Bias_cor_type.set(traintest_config.getint('traintestset', 'bias_type'))
        user_config.set('traintestset', 'bias_iter', str(self.setting_Bias_cor_niter.get()))
        user_config.set('traintestset', 'bias_smooth', str(self.setting_Bias_cor_smooth.get()))
        user_config.set('traintestset', 'bias_type', str(self.setting_Bias_cor_type.get()))
        user_config.set('traintestset', 'bias_choice', str(self.setting_bias_choice.get()))

        user_config.set('traintestset', 'skull_stripping', str(self.setting_skull_stripping.get()))
        user_config.set('traintestset', 'save_tmp', str(self.setting_save_tmp.get()))
        user_config.set('traintestset', 'debug', str(self.setting_debug.get()))

        user_config.set('traintestset',
                        'full_train',
                        str(not (self.setting_use_learnedmodel_model.get())))
        user_config.set('traintestset',
                        'learnedmodel_model',
                        str(self.setting_learnedmodel_model.get()))
        user_config.set('traintestset',
                        'balanced_training',
                        str(self.setting_balanced_dataset.get()))
        user_config.set('traintestset',
                        'fraction_negatives',
                        str(self.setting_fract_negatives.get()))

        user_config.set('traintestset',
                        'fraction_negatives_CV',
                        str(self.setting_fract_negatives_cv.get()))

        user_config.set('traintestset', 'name', self.setting_net_name.get())

        user_config.set('traintestset', 'learnedmodel', str(self.setting_learnedmodel))
        user_config.set('traintestset', 'train_split', str(self.setting_train_split.get()))
        user_config.set('traintestset', 'max_epochs', str(self.setting_max_epochs.get()))
        user_config.set('traintestset', 'patience', str(self.setting_patience.get()))
        user_config.set('traintestset', 'batch_size', str(self.setting_batch_size.get()))
        user_config.set('traintestset', 'net_verbose', str(self.setting_net_verbose.get()))
        # user_config.set('model', 'gpu_mode', self.setting_mode.get())
        user_config.set('traintestset', 'gpu_number', str(self.setting_gpu_number.get()))
        user_config.set('traintestset', 'threshold', str(self.setting_threshold.get()))
        user_config.set('traintestset', 'volume_tolerance', str(self.setting_volume_tolerance.get()))
        user_config.set('traintestset',
                        'error_tolerance', str(self.setting_error_tolerance.get()))

        user_config.set('traintestset', 'all_label', self.all_label.get())
        user_config.set('traintestset', 'all_input_mod', self.all_input_mod.get())



        user_config.add_section('tensorboard')
        user_config.set('tensorboard', 'port', str(self.setting_PORT_tag.get()))
        # postprocessing parameters
        user_config.set('tensorboard', 'tensorBoard_folder', self.setting_tensorboard_folder.get())


        user_config.add_section('completed')
        user_config.set('completed', 'model_1_train', str(self.model_1_train.get()))
        user_config.set('completed', 'model_2_train', str(self.model_2_train.get()))
        user_config.set('completed', 'pre_processing', str(self.pre_processing.get()))

        # Writing our configuration file to 'example.cfg'
        with open(os.path.join(self.path,
                               'config',
                               'configuration.cfg'), 'w') as configfile:
            user_config.write(configfile)

    def load_training_path(self):
        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_training_folder.set(fname)
                self.inFolderTxt.delete(0, END)
                self.inFolderTxt.insert(0, self.setting_training_folder.get())

            except:
                pass

    def load_cross_validation_path(self):
        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_cross_validation.set(fname)
                self.incvFolderTxt.delete(0, END)
                self.incvFolderTxt.insert(0, self.setting_cross_validation.get())
                self.trainingBtn['state'] = 'normal'

            except:
                pass

    def load_testing_path(self):
        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_test_folder.set(fname)
                self.test_inFolderTxt.delete(0, END)
                self.test_inFolderTxt.insert(0, self.setting_test_folder.get())
                self.testingBtn['state'] = 'normal'
                if self.trainingBtn['state'] == 'normal':
                    self.traininginferenceBtn['state'] = 'normal'
            except:
                pass

    def update_learnedmodel_nets(self):
        folders = os.listdir(self.setting_net_folder)
        self.list_train_learnedmodel_nets = folders
        self.list_test_nets = folders

    def write_to_console(self, txt):
        self.command_out.insert(END, str(txt))

    def write_to_train_test_console(self, txt):
        self.command_out_tt.insert(END, str(txt))

    def write_to_test_console(self, txt):
        self.test_command_out.insert(END, str(txt))

    def start_tensorBoard(self):

            try:
                if self.setting_PORT_tag.get() == None:
                    print("\n")
            except ValueError:
                print("ERROR: Port number and TensorBoard folder must be defined  before starting...\n")
                return

            self.TensorBoardBtn['state'] = 'disable'

            if self.setting_PORT_tag.get() is not None:
                # self.TensorBoardBtn['state'] = 'normal'
                print("\n-----------------------")
                print("Starting TensorBoard ...")
                print("TensorBoard folder:", self.setting_tensorboard_folder.get(), "\n")
                thispath = self.setting_tensorboard_folder.get()
                thisport = self.setting_PORT_tag.get()
                self.write_user_configuration()
                print("The port for TensorBoard is set to be:", thisport)
                # import appscript
                pp = os.path.join(self.path, 'spider', 'bin')

                THIS_PATHx = os.path.split(os.path.realpath(__file__))[0]
                # tensorboard = THIS_PATHx + '/libs/bin/tensorboard'
                Folder=thispath
                Port=thisport
                os_host = platform.system()
                if os_host == 'Windows':
                    arg1 = ' ' + '--logdir  ' + str(Folder) + ' ' + '  --port  ' + str(Port)
                    os.system("start cmd  /c   'tensorboard   {}'".format(arg1))
                elif os_host == 'Linux':
                    arg1 =str(Folder)+'  ' + str(Port)
                    os.system("dbus-launch gnome-terminal -e 'bash -c \"bash  tensorb.sh   {}; exec bash\"'".format(arg1))

                elif os_host == 'Darwin':
                    import appscript
                    appscript.app('Terminal').do_script(
                        'tensorboard    --logdir=' + str(
                            thispath) + '  --port=' + str(thisport))

                else:
                    print("> ERROR: The OS system", os_host, "is not currently supported.")


    def test_net(self):

        if self.setting_net_name.get() == 'None' or self.setting_net_name.get() == '':

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, define network name before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.testing_do is None:
            self.testingBtn.config(state='disabled')
            self.traininginferenceBtn['state'] = 'disable'
            self.trainingBtn['state'] = 'disable'
            self.traininginferenceBtn.update()
            self.trainingBtn.update()
            self.updated_traintest_configuration
            self.setting_net_name.set(self.setting_net_name.get())
            self.setting_use_learnedmodel_model.set(False)
            self.write_user_configuration_inference()
            self.testing_do = ThreadedTask(self.write_to_test_console,
                                          self.test_queue, mode='testing')
            self.testing_do.start()

            self.master.after(100, self.process_run)
            self.testingBtn['state'] = 'normal'

    def train_test_net(self):
        if self.setting_net_name.get() == 'None' or self.setting_net_name.get() == '':

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, define network name before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.setting_Longitudinal.get() is False and self.setting_Cross_Sectional.get() is False:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + " Please, set training to Longitudinal or Cross_Sectional before starting..." + '\x1b[0m')
            print("\n")
            return

        if self.setting_Longitudinal.get() is  True and self.setting_Cross_Sectional.get() is True:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + " Traninig  with Longitudinal and Cross_Sectional is not possible!,"
                                    " please set correctly training to Longitudinal or Cross_Sectional before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.setting_inputmodlengthwise.get() == 0:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + " Traninig  with zero lengthwise is not possible!, "
                                    " please set correctly number of lengthwise before starting..." + '\x1b[0m')
            print("\n")

            return

        self.traininginferenceBtn['state'] = 'disable'
        self.trainingBtn['state'] = 'disable'
        self.testingBtn['state'] = 'disable'

        if self.training_do is None:
            self.traininginferenceBtn.update()
            self.trainingBtn.update()
            self.testingBtn.update()
            self.write_user_configuration()
            self.training_do = ThreadedTask(self.write_to_train_test_console,
                                           self.test_queue,
                                           mode='trainingandinference')
            self.training_do.start()
            self.master.after(100, self.process_run)

    def train_net(self):

        if self.setting_net_name.get() == 'None' or self.setting_net_name.get() == '':

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, define network name before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.setting_Longitudinal.get() is False and self.setting_Cross_Sectional.get() is False:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + " Please, set training to Longitudinal or Cross_Sectional before starting..." + '\x1b[0m')
            print("\n")
            return

        if self.setting_Longitudinal.get() is  True and self.setting_Cross_Sectional.get() is True:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + " Traninig  with Longitudinal and Cross_Sectional is not possible!,"
                                    " please set correctly training to Longitudinal or Cross_Sectional before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.setting_inputmodlengthwise.get() == 0:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + " Traninig  with zero lengthwise is not possible!, "
                                    " please set correctly number of lengthwise before starting..." + '\x1b[0m')
            print("\n")

            return

        self.traininginferenceBtn['state'] = 'disable'
        self.trainingBtn['state'] = 'disable'
        self.testingBtn['state'] = 'disable'

        if self.training_do is None:
            self.traininginferenceBtn.update()
            self.trainingBtn.update()
            self.testingBtn.update()
            self.write_user_configuration()
            self.training_do = ThreadedTask(self.write_to_console,
                                           self.test_queue,
                                           mode='training')
            self.training_do.start()
            self.master.after(100, self.process_run)

    def abw(self):

        t = Toplevel(self.master, width=500, height=500)
        t.wm_title("Multi task 3D Convolutional Neural Network")
        title = Label(t,
                      text="3D Convolutional Neural Network \n"
                      "Single Modality Multiple Sclerosis Lesions Segmentation \n"
                      "Medical Physics and Biomedical Engineering - UCL \n"
                      "Kevin Bronik - 2020")
        title.grid(row=2, column=1, padx=20, pady=10)
        img = ImageTk.PhotoImage(Image.open('images/gifanim.gif'))
        imglabel = Label(t, image=img)
        imglabel.image = img
        imglabel.grid(row=1, column=1, padx=10, pady=10)
        # self.gif = tk.PhotoImage(file=self.gif_file,
        root = imglabel

        # Add the path to a GIF to make the example working
        l = AnimatedGIF(root, "images/brain_lesion.gif")
        # l = AnimatedGIF(root, "./brain_lesion.png")
        l.pack()
        root.mainloop()

    def hlp(self):
            t = Toplevel(self.master, width=500, height=500)

            img = ImageTk.PhotoImage(Image.open('images/help.png'))
            imglabel = Label(t, image=img)
            imglabel.image = img
            imglabel.grid(row=1, column=1, padx=10, pady=10)
            # self.gif = tk.PhotoImage(file=self.gif_file,
            root = imglabel
            root.mainloop()
    def process_run(self):

        self.process_indicator.set('Training/Testing is Running... please wait')
        try:
            msg = self.test_queue.get(0)
            self.process_indicator.set('Training/Testing completed.')

            self.trainingBtn['state'] = 'normal'
            self.traininginferenceBtn['state'] = 'normal'
            # self.testingBtn['state'] = 'normal'
        except queue.Empty:
            self.master.after(100, self.process_run)

    def terminate(self):

        if self.training_do is not None:
            self.training_do.stop_process()
        if self.testing_do is not None:
            self.testing_do.stop_process()
        os.system('cls' if platform.system == "Windows" else 'clear')
        root.destroy()


class ThreadedTask(threading.Thread):

    def __init__(self, print_func, queue, mode):
        threading.Thread.__init__(self)
        self.queue = queue
        self.mode = mode
        self.print_func = print_func
        self.process = None

    def run(self):

        settings = overall_config()
        if self.mode == 'training':
            if settings['Longitudinal'] is True and settings['inputmodlengthwise'] > 1:
                print('\x1b[6;30;41m' + "                                   " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting  Longitudinal training ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                   " + '\x1b[0m')
                train_network(settings)
            else:
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting  Cross_Sectional  training ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                train_network_cross(settings)

        elif self.mode == 'trainingandinference':
            if settings['Longitudinal'] is True and settings['inputmodlengthwise'] > 1:
                print('\x1b[6;30;41m' + "                                                 " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting  Longitudinal training and inference ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                                 " + '\x1b[0m')
                train_test_network(settings)
            else:
                print('\x1b[6;30;41m' + "                                                   " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting Cross_Sectional training and inference ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                                   " + '\x1b[0m')
                train_test_network_cross(settings)


        else:
            if settings['Longitudinal'] is True and settings['inputmodlengthwise'] > 1:
                print('\x1b[6;30;41m' + "                                   " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting  Longitudinal inference ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                   " + '\x1b[0m')
                infer_segmentation(settings)
            else:
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting  Cross_Sectional  inference ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                infer_segmentation_cross(settings)
        self.queue.put(" ")

    def stop_process(self):
        try:
            if platform.system() == "Windows":
                subprocess.Popen("taskkill /F /T /PID %i" % os.getpid(), shell=True)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except:
            os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--docker',
                        dest='docker',
                        action='store_true')
    parser.set_defaults(docker=False)
    args = parser.parse_args()
    root = Tk()
    root.resizable(width=False, height=False)
    GUI = CNN(root, args.docker)
    root.mainloop()

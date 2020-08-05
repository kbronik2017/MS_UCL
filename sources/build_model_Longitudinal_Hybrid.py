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
import threading
import time
import shutil
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sources.build_Hybird_nets import hybird_network
from keras import optimizers, losses

import tensorflow as tf
import gc
import warnings


from numpy import inf
from keras  import backend as K
# from keras.preprocessing.image import  ImageDataGenerator
from scipy.spatial.distance import directed_hausdorff, chebyshev
# import horovod.keras as hvd

# force data format to "channels first"
keras.backend.set_image_data_format('channels_first')

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

class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting settings,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """
    #
    # def __init__(self, filepath, monitor='val_loss', verbose=0,
    #              save_best_only=False, save_weights_only=False,
    #              mode='auto', period=1):
    #     super(ModelCheckpoint, self).__init__()
    #     self.monitor = monitor
    #     self.verbose = verbose
    #     self.filepath = filepath
    #     self.save_best_only = save_best_only
    #     self.save_weights_only = save_weights_only
    #     self.period = period
    #     self.epochs_since_last_save = 0
    #
    #     if mode not in ['auto', 'min', 'max']:
    #         warnings.warn('ModelCheckpoint mode %s is unknown, '
    #                       'fallback to auto mode.' % (mode),
    #                       RuntimeWarning)
    #         mode = 'auto'
    #
    #     if mode == 'min':
    #         self.monitor_op = np.less
    #         self.best = np.Inf
    #     elif mode == 'max':
    #         self.monitor_op = np.greater
    #         self.best = -np.Inf
    #     else:
    #         if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
    #             self.monitor_op = np.greater
    #             self.best = -np.Inf
    #         else:
    #             self.monitor_op = np.less
    #             self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # logs = logs or {}
        # self.epochs_since_last_save += 1
        # if self.epochs_since_last_save >= self.period:
        #     self.epochs_since_last_save = 0
        #     filepath = self.filepath.format(epoch=epoch + 1, **logs)
        #     if self.save_best_only:
        #         current = logs.get(self.monitor)
        #         if current is None:
        #             warnings.warn('Can save best model only with %s available, '
        #                           'skipping.' % (self.monitor), RuntimeWarning)
        #         else:
        #             if self.monitor_op(current, self.best):
        #                 if self.verbose > 0:
        #                     print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
        #                           ' saving model to %s'
        #                           % (epoch + 1, self.monitor, self.best,
        #                              current, filepath))
        #                 self.best = current
        #                 if self.save_weights_only:
        #                     self.model.save_weights(filepath, overwrite=True)
        #                 else:
        #                     self.model.save(filepath, overwrite=True)
        #             else:
        #                 if self.verbose > 0:
        #                     print('\nEpoch %05d: %s did not improve from %0.5f' %
        #                           (epoch + 1, self.monitor, self.best))
        #     else:
        #         if self.verbose > 0:
        #             print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        #         if self.save_weights_only:
        #             self.model.save_weights(filepath, overwrite=True)
        #         else:
        #             self.model.save(filepath, overwrite=True)
        print(CYELLOW +'Full garbage collection:'+ CEND, 'epoch {}'.format(epoch + 1))
        gc.collect()
        print(gc.get_stats())






def transform(Xb, yb):
    """
    handle class for on-the-fly data augmentation on batches.
    Applying 90,180 and 270 degrees rotations and flipping
    """
    # Flip a given percentage of the images at random:
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs // 2, replace=False)
    x_da = Xb[indices]

    # apply rotation to the input batch
    rotate_90 = x_da[:, :, :, ::-1, :].transpose(0, 1, 2, 4, 3)
    rotate_180 = rotate_90[:, :, :, :: -1, :].transpose(0, 1, 2, 4, 3)
    rotate_270 = rotate_180[:, :, :, :: -1, :].transpose(0, 1, 2, 4, 3)
    # apply flipped versions of rotated patches
    rotate_0_flipped = x_da[:, :, :, :, ::-1]
    rotate_90_flipped = rotate_90[:, :, :, :, ::-1]
    rotate_180_flipped = rotate_180[:, :, :, :, ::-1]
    rotate_270_flipped = rotate_270[:, :, :, :, ::-1]

    augmented_x = np.stack([x_da, rotate_90, rotate_180, rotate_270,
                            rotate_0_flipped,
                            rotate_90_flipped,
                            rotate_180_flipped,
                            rotate_270_flipped],
                            axis=1)

    # select random indices from computed transformations
    r_indices = np.random.randint(0, 3, size=augmented_x.shape[0])

    Xb[indices] = np.stack([augmented_x[i,
                                        r_indices[i], :, :, :, :]
                            for i in range(augmented_x.shape[0])])

    return Xb, yb

#################### make the data generator threadsafe ####################

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def da_generator(x_train, y_train, batch_size):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """
    num_samples = int(x_train.shape[0] / batch_size) * batch_size
    while True:
        for b in range(0, num_samples, batch_size):
            x_ = x_train[b:b+batch_size]
            y_ = y_train[b:b+batch_size]
            x_, y_ = transform(x_, y_)
            yield x_, y_

@threadsafe_generator
def val_generator(x_train, y_train, batch_size):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """
    num_samples = int(x_train.shape[0] / batch_size) * batch_size
    while True:
        for b in range(0, num_samples, batch_size):
            x_ = x_train[b:b+batch_size]
            y_ = y_train[b:b+batch_size]
            x_, y_ = transform(x_, y_)
            yield x_, y_


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def Jaccard_index(y_true, y_pred):
    smooth = 100.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    score = (intersection + smooth) / (union + smooth)
    return score
##################################


def rand_bin_array(K, N):
        arr = np.zeros(N)
        arr[:K] = 1
        # np.random.shuffle(arr)
        return arr


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def true_false_positive_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):
    # y_pred_f = tf.reshape(y_pred, [-1])

    # y_pred_f = tf.reshape(y_pred, [-1])
    # this_size = K.int_shape(y_pred_f)[0]
    # # arr_len = this_size
    # # num_ones = 1
    # # arr = np.zeros(arr_len, dtype=int)
    # # if this_size is not None:
    # #      num_ones= np.int32((this_size * value * 100) / 100)
    # #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
    # #      arr[idx] = 1
    # #      p_labels = arr
    # # else:
    # #      p_labels = np.random.randint(2, size=this_size)
    # if this_size is not None:
    #      p_labels = np.random.binomial(1, value, size=this_size)
    #      p_labels = tf.reshape(p_labels, y_pred.get_shape())
    # else:
    #      p_labels =  y_pred

    # p_lprint ('num_classes .....///', num_classes.shape[0])abels = tf.constant(y_pred_f)
    # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
    # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)

    # num_classes = 50
    # print ('num_classes .....', num_classes)
    # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
    # p_labels = tf.reduce_max(p_labels, 0)
    # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
    # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
    # y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

    C11 = tf.math.multiply(y_true, y_pred)
    c_y_pred = 1 - y_pred
    C12 = tf.math.multiply(y_true, c_y_pred)
    weighted_y_pred_u = tf.math.multiply(K.cast(p_labels, y_pred.dtype), y_pred)
    weighted_y_pred_d = 1 - weighted_y_pred_u
    y_pred = tf.math.add(tf.math.multiply(C11, weighted_y_pred_u), tf.math.multiply(C12, weighted_y_pred_d))

    # y_pred /= tf.reduce_sum(y_pred,
    #                             reduction_indices=len(y_pred.get_shape()) - 1,
    #                             keep_dims=True)
    #     # manual computation of crossentropy
    # _EPSILON = 10e-8
    # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
    #                            reduction_indices=len(y_pred.get_shape()) - 1)

    loss = Jaccard_loss(y_true, y_pred)
    # with tf.GradientTape() as t:
    #     t.watch(y_pred)
    #     dpred = t.gradient(loss, y_pred)

    return loss

    # return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


def false_true_negative_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):
    # arr_len = this_size
    # num_ones = 1
    # arr = np.zeros(arr_len, dtype=int)
    # if this_size is not None:
    #      num_ones= np.int32((this_size * value * 100) / 100)
    #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
    #      arr[idx] = 1
    #      p_labels = arr
    # else:
    #      p_labels = np.random.randint(2, size=this_size)
    # np.random.binomial(1, 0.34, size=10)

    # if this_size is not None:
    #     this_value= np.int32((this_size * value * 100) / 100)
    # else:
    #     this_value =  1

    # p_labels = np.random.randint(2, size=this_size)

    # p_labels = tf.constant(y_pred_f)
    # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
    # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)
    # num_classes = 50
    # print ('num_classes .....', num_classes)
    # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
    # # p_labels = tf.reduce_max(p_labels, 0)
    # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
    # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
    # y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

    c_y_true = 1 - y_true
    c_y_pred = 1 - y_pred
    C21 = tf.math.multiply(c_y_true, y_pred)

    C22 = tf.math.multiply(c_y_true, c_y_pred)
    weighted_y_pred_u = tf.math.multiply(K.cast(p_labels, y_pred.dtype), y_pred)
    weighted_y_pred_d = 1 - weighted_y_pred_u

    y_pred = tf.math.add(tf.math.multiply(C21, weighted_y_pred_u), tf.math.multiply(C22, weighted_y_pred_d))
    # y_pred /= tf.reduce_sum(y_pred,
    #                             reduction_indices=len(y_pred.get_shape()) - 1,
    #                             keep_dims=True)
    #     # manual computation of crossentropy
    # _EPSILON = 10e-8
    # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
    #                            reduction_indices=len(y_pred.get_shape()) - 1)
    # with tf.GradientTape() as t:
    #     t.watch(y_pred)
    #     dpred = t.gradient(loss, y_pred)
    y_true = 1 - y_true
    loss = Jaccard_loss(y_true, y_pred)

    return loss


def penalty_loss_trace_normalized_confusion_matrix(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred

    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    sum1 = tp + fn

    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    sum2 = fp + tn

    tp_n = tp / sum1
    fn_n = fn / sum1
    fp_n = fp / sum2
    tn_n = tn / sum2
    trace = (tf.math.square(tp_n) + tf.math.square(tn_n) + tf.math.square(fn_n) + tf.math.square(fp_n))

    return  1 - trace * 0.5


def p_loss(y_true, y_pred):
    smooth = 100.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    pg1 = (2. * (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)) * tf.reduce_sum(y_true_f)
    pg2 = (2. * intersection + smooth)
    pg3 = K.square(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    pg = (pg1 - pg2) / pg3

    # w = tf.Variable(y_pred_f, trainable=True)
    # with tf.GradientTape() as t:
    #      t.watch(y_pred_f)
    #
    # pg = t.gradient(score, y_pred_f)

    # pg = K.gradients(score, y_pred_f)[0]
    # pg = K.sqrt(sum([K.sum(K.square(g)) for g in pg_t]))

    return score, pg


def constrain(y_true, y_pred):
    loss, g = p_loss(y_true, y_pred)
    return loss


def constrain_loss(y_true, y_pred):
    return 1 - constrain(y_true, y_pred)


# def augmented_Lagrangian_loss(y_true, y_pred, augment_Lagrangian=1):

#     C_value, pgrad = p_loss(y_true, y_pred)
#     Ld, grad1 = loss_down(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
#     Lu, grad2 = loss_up(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
#     ploss = 1 - C_value
#     # adaptive lagrange multiplier
#     _EPSILON = 10e-8
#     if all(v is not None for v in [grad1, grad2, pgrad]):
#          alm = - ((grad1 + grad2) / pgrad + _EPSILON)
#     else:
#          alm =  augment_Lagrangian
#     ploss = ploss * alm
#     total_loss = Ld + Lu + ploss
#     return total_loss
def calculate_gradient(y_true, y_pred):
    constrain_l = constrain_loss(y_true, y_pred)
    this_value = (-1 * constrain_l) + 1
    y_pred_f = tf.reshape(y_pred, [-1])
    this_size = K.int_shape(y_pred_f)[0]
    if this_size is not None:
        #  numpy.random.rand(4)
        #  p_labels = np.random.binomial(1, this_value, size=this_size)
        p_labels_g = rand_bin_array(this_value, this_size)
        p_labels_g = my_func(np.array(p_labels_g, dtype=np.float32))
        # p_labels = tf.dtypes.cast((tf.dtypes.cast(tf.size(p_labels), tf.int32)
        p_labels_g = tf.reshape(p_labels_g, y_pred.get_shape())

    else:
        p_labels_g = 0

    loss1 = true_false_positive_loss(y_true, y_pred, p_labels=p_labels_g, label_smoothing=0, value=this_value)
    loss2 = false_true_negative_loss(y_true, y_pred, p_labels=p_labels_g, label_smoothing=0, value=this_value)

    g_loss1 = K.sum(K.gradients(loss1, y_pred)[0])
    g_loss2 = K.sum(K.gradients(loss2, y_pred)[0])
    # cg_loss = K.sum(K.gradients(losses.categorical_crossentropy(y_true, y_pred), y_pred)[0])
    # pg_loss = K.sum(K.gradients(penalty_loss_trace_normalized_confusion_matrix(y_true, y_pred), y_pred)[0])

    return g_loss1, g_loss2


#
# def calculate_gradient(y_true, y_pred, loss1, loss2):
#
#         # w = tf.Variable(y_pred,  trainable=True)
#         # with tf.GradientTape(persistent=True) as t:
#         #       t.watch(y_pred)
#         #
#         # g_loss1 = t.gradient(loss1, y_pred)
#         # g_loss2 = t.gradient(loss2, y_pred)
#
#         g_loss1 = K.sum(K.gradients(loss1, y_pred)[0])
#         g_loss2 =K.sum(K.gradients(loss2, y_pred)[0])
#         # g_loss1 = K.sqrt(sum([K.sum(K.square(g)) for g in g_loss1_t]))
#         # g_loss2 = K.sqrt(sum([K.sum(K.square(g)) for g in g_loss2_t]))
#         # loss, g_constrain = p_loss (y_true, y_pred)
#         loss, g_constrain = p_loss(y_true, y_pred)
#         return g_loss1, g_loss2, g_constrain

def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg


def Adaptive_Lagrange_Multiplier(y_pred, y_true):
    smooth = 1.

    augment_Lagrangian = 1
    # if all(v is not None for v in [Gloss1, Gloss2, Gloss3]):
    if y_pred is not None:

        loss, g_constrain = p_loss(y_true, y_pred)
        g_loss1, g_loss2 = calculate_gradient(y_true, y_pred)

        res = (g_loss1 + g_loss2 + smooth) / (g_constrain + smooth)
        # res = (g_loss1 + g_loss2 ) / (g_constrain )
        # res = ((K.sum(Gloss1) + K.sum(Gloss2)))

        # losses.categorical_crossentropy(y_true,
        #                              y_pred) + penalty_loss_trace_normalized_confusion_matrix(
        # y_true, y_pred)

        return res * (K.sum(y_true * y_true) / K.sum(y_true * y_true))
    else:
        # print("adaptive_lagrange_multiplier", augment_Lagrangian)
        return K.sum(y_true * y_true) / K.sum(y_true * y_true)


def Individual_loss(y_true, y_pred):
    # C_value = p_loss(y_true, y_pred)
    constrain_l = constrain_loss(y_true, y_pred)
    this_value = (-1 * constrain_l) + 1
    y_pred_f = tf.reshape(y_pred, [-1])
    this_size = K.int_shape(y_pred_f)[0]
    if this_size is not None:
        #  numpy.random.rand(4)
        #  p_labels = np.random.binomial(1, this_value, size=this_size)
        p_labels = rand_bin_array(this_value, this_size)
        p_labels = my_func(np.array(p_labels, dtype=np.float32))
        # p_labels = tf.dtypes.cast((tf.dtypes.cast(tf.size(p_labels), tf.int32)
        p_labels = tf.reshape(p_labels, y_pred.get_shape())

    else:
        p_labels = 0

    loss1 = true_false_positive_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
    loss2 = false_true_negative_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
    # grad1, grad2, pgrad = calculate_gradient(y_true, y_pred, loss1, loss2)

    # ploss = 1 - C_value
    # adaptive lagrange multiplier
    # adaptive_lagrange_multiplier(y_true, y_pred):
    # _EPSILON = 10e-8
    # if all(v is not None for v in [grad1, grad2, pgrad]):
    #     return (((grad1 + grad2) + _EPSILON) / pgrad + _EPSILON)
    # else:
    #     return  augment_Lagrangian
    # ploss = ploss * alm
    lm = Adaptive_Lagrange_Multiplier(y_pred, y_true)

    # outF = open(
    #     os.path.join('/home/kbronik/Desktop/IE_MULTIINPUT/CNN_multiinputs_singleoutput_modified_Keras/utils/this.txt'),
    #     "w")
    # parsed_line = str(lm)
    # outF.writelines(parsed_line)
    # outF.close()

    to_loss = loss1 + loss2 + (lm * constrain_l)
    return to_loss + losses.categorical_crossentropy(y_true,
                                                     y_pred) + penalty_loss_trace_normalized_confusion_matrix(
        y_true, y_pred)


##################################



def Symmetric_Hausdorf_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    # Calculating the forward HD: mean(min(each col))
    left = K.maximum(K.minimum(y_true - y_pred, inf), -inf)

    # Calculating the reverse HD: mean(min(each row))
    right = K.maximum(K.minimum(y_pred - y_true, inf), -inf)
    # Calculating mhd
    res = K.maximum(left, right)
    return K.max(res)

def Jaccard_loss(y_true, y_pred):
    loss = 1 - Jaccard_index(y_true, y_pred)
    return loss

def Dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def Combi_Dist_loss(y_true, y_pred):
    y_truec = K.l2_normalize(y_true, axis=-1)
    y_predc = K.l2_normalize(y_pred, axis=-1)

    #loss1 = K.sum(K.abs(y_pred - y_true), axis=-1) + K.mean(K.square(y_pred - y_true), axis=-1)
    #loss2 = -K.mean(y_true_c * y_pred_c, axis=-1) + 100. * K.mean(diff, axis=-1)
    #loss = K.max(loss1+ loss2)
    return K.maximum(K.maximum(K.sum(K.abs(y_pred - y_true), axis=-1) , K.mean(K.square(y_pred - y_true), axis=-1)), -K.sum(y_truec * y_predc, axis=-1))

def accuracy_loss(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(y_true * y_pred)
    acc = (tp + tn) / (tp + tn + fn + fp)
    return 1.0 - acc


def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    spec = tn / (tn + fp + K.epsilon())
    return spec


def specificity_loss(y_true, y_pred):
        return 1.0 - specificity(y_true, y_pred)


def sensitivity(y_true, y_pred):
    # neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(y_true * y_pred)
    sens = tp / (tp + fn + K.epsilon())
    return sens

def sensitivity_loss(y_true, y_pred):
        return 1.0 - sensitivity(y_true, y_pred)

def precision(y_true, y_pred):
    neg_y_true = 1 - y_true
    # neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tp = K.sum(y_true * y_pred)
    pres = tp / (tp + fp + K.epsilon())
    return pres

def precision_loss(y_true, y_pred):
        return 1.0 - precision(y_true, y_pred)

#  MCC={\frac {{\mathit {TP}}\times {\mathit {TN}}-{\mathit {FP}}\times {\mathit {FN}}}{\sqrt
# {({\mathit {TP}}+{\mathit {FP}})({\mathit {TP}}+{\mathit {FN}})({\mathit {TN}}+{\mathit {FP}})({\mathit {TN}}+{\mathit {FN}})}}}}

def Mattews_loss(y_true, y_pred):

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred

    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)


    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    mult1 = tp * tn
    mult2 = fp * fn

    sum1 = fp + tp
    sum2 = tp + fn
    sum3 = tn + fp
    sum4 = tn + fn

    mcc = (mult1 - mult2) / tf.math.sqrt(sum1 * sum2 * sum3 * sum4)

    return K.abs(1 - mcc)




def concatenated_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + Dice_loss(y_true, y_pred) + Jaccard_loss(y_true, y_pred) + \
           Combi_Dist_loss(y_true, y_pred) + Symmetric_Hausdorf_loss(y_true, y_pred) + \
           specificity_loss(y_true, y_pred) + sensitivity_loss(y_true, y_pred) + precision_loss(y_true, y_pred) \
           + accuracy_loss(y_true, y_pred) + Individual_loss(y_true, y_pred) + Mattews_loss(y_true, y_pred)
    #loss = losses.categorical_crossentropy(y_true, y_pred)
    return loss

def build_and_compile_models_tensor_2(settings):

    if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                       settings['modelname'])):
        os.mkdir(os.path.join(settings['model_saved_paths'],
                              settings['modelname']))
    if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                       settings['modelname'], 'models')):
        os.mkdir(os.path.join(settings['model_saved_paths'],
                              settings['modelname'], 'models'))
    if settings['debug']:
        if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                           settings['modelname'],
                                           '.train')):
            os.mkdir(os.path.join(settings['model_saved_paths'],
                                  settings['modelname'],
                                  '.train'))

    # --------------------------------------------------
    # model 1
    # --------------------------------------------------
    first = 'First'
    second = 'Second'
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        model = hybird_network(settings, first)
        # try:
        #     model = tf.keras.utils.multi_gpu_model(model, cpu_relocation=True)
        #     print("Training first model using multiple GPUs..")
        # except:
        #     print("Training first model using single GPU or CPU..")
        model.compile(loss=concatenated_loss,
                      optimizer='adadelta',
                      metrics=[concatenated_loss, Dice_loss, Jaccard_loss, Combi_Dist_loss,
                               Symmetric_Hausdorf_loss, specificity_loss, sensitivity_loss, precision_loss,
                               accuracy_loss, Individual_loss])
        # if settings['debug']:
        #     model.summary()

        # save weights
        net_model_1 = 'model_1'
        net_weights_1 = os.path.join(settings['model_saved_paths'],
                                     settings['modelname'],
                                     'models', net_model_1 + '.hdf5')

        network1 = {}
        network1['model'] = model
        network1['weights'] = net_weights_1
        network1['history'] = None
        network1['special_name_1'] = net_model_1

        # --------------------------------------------------
        # model 2
        # --------------------------------------------------

        model2 = hybird_network(settings, second)
        # try:
        #     model2 = tf.keras.utils.multi_gpu_model(model2, cpu_relocation=True)
        #     print("Training second model using multiple GPUs..")
        # except:
        #     print("Training second model using single GPU or CPU..")
        model2.compile(loss=concatenated_loss,
                       optimizer='adadelta',
                       metrics=[concatenated_loss, Dice_loss, Jaccard_loss, Combi_Dist_loss,
                                Symmetric_Hausdorf_loss, specificity_loss, sensitivity_loss, precision_loss,
                                accuracy_loss, Individual_loss])
        # if settings['debug']:
        #    model2.summary()

        # save weights
        # save weights
        net_model_2 = 'model_2'
        net_weights_2 = os.path.join(settings['model_saved_paths'],
                                     settings['modelname'],
                                     'models', net_model_2 + '.hdf5')

        network2 = {}
        network2['model'] = model2
        network2['weights'] = net_weights_2
        network2['history'] = None
        network2['special_name_2'] = net_model_2

        # load predefined weights if transfer learning is selected

        if settings['full_train'] is False:

            # load default weights
            print("> CNN: Loading learnedmodel weights from the", \
                  settings['learnedmodel_model'], "configuration")
            learnedmodel_model = os.path.join(settings['model_saved_paths'], settings['learnedmodel_model'], 'models')
            model = os.path.join(settings['model_saved_paths'],
                                 settings['modelname'])
            network1_w_def = os.path.join(model, 'models', 'model_1.hdf5')
            network2_w_def = os.path.join(model, 'models', 'model_2.hdf5')

            if not os.path.exists(model):
                shutil.copy(learnedmodel_model, model)
            else:
                shutil.copyfile(os.path.join(learnedmodel_model,
                                             'model_1.hdf5'),
                                network1_w_def)
                shutil.copyfile(os.path.join(learnedmodel_model,
                                             'model_2.hdf5'),
                                network2_w_def)
            try:
                network1['model'].load_weights(network1_w_def, by_name=True)
                network2['model'].load_weights(network2_w_def, by_name=True)
            except:
                print("> ERROR: The model", \
                      settings['modelname'], \
                      'selected does not contain a valid network model')
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if settings['load_weights'] is True:
            print("> CNN: loading weights from", \
                  settings['modelname'], 'configuration')
            print(net_weights_1)
            print(net_weights_2)

            network1['model'].load_weights(net_weights_1, by_name=True)
            network2['model'].load_weights(net_weights_2, by_name=True)

        return [network1, network2]


def build_and_compile_models_tensor_1(settings):
    """
    3D cascade model using Nolearn and Lasagne

    Inputs:
    - model_settings:
    - weights_path: path to where weights should be saved

    Output:
    - models = list of NeuralNets (CNN1, CNN2)
    """

    # save model to disk to re-use it. Create an experiment folder
    # organize experiment
    if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                       settings['modelname'])):
        os.mkdir(os.path.join(settings['model_saved_paths'],
                              settings['modelname']))
    if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                       settings['modelname'], 'models')):
        os.mkdir(os.path.join(settings['model_saved_paths'],
                              settings['modelname'], 'models'))
    if settings['debug']:
        if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                           settings['modelname'],
                                           '.train')):
            os.mkdir(os.path.join(settings['model_saved_paths'],
                                  settings['modelname'],
                                  '.train'))

    # --------------------------------------------------
    # model 1
    # --------------------------------------------------
    first = 'First'
    second = 'Second'
    model = hybird_network(settings, first)
    try:
        model = tf.keras.utils.multi_gpu_model(model, cpu_relocation=True)
        print("Training first model using multiple GPUs..")
    except:
        print("Training first model using single GPU or CPU..")
    model.compile(loss=concatenated_loss,
                  optimizer='adadelta',
                  metrics=[concatenated_loss, Dice_loss, Jaccard_loss, Combi_Dist_loss,
                           Symmetric_Hausdorf_loss, specificity_loss, sensitivity_loss, precision_loss, accuracy_loss, Individual_loss])
    # if settings['debug']:
    #     model.summary()

    # save weights
    net_model_1 = 'model_1'
    net_weights_1 = os.path.join(settings['model_saved_paths'],
                                settings['modelname'],
                                'models', net_model_1 + '.hdf5')

    network1 = {}
    network1['model'] = model
    network1['weights'] = net_weights_1
    network1['history'] = None
    network1['special_name_1'] = net_model_1

    # --------------------------------------------------
    # model 2
    # --------------------------------------------------

    model2 = hybird_network(settings, second)
    try:
        model2 = tf.keras.utils.multi_gpu_model(model2, cpu_relocation=True)
        print("Training second model using multiple GPUs..")
    except:
        print("Training second model using single GPU or CPU..")
    model2.compile(loss=concatenated_loss,
                   optimizer='adadelta',
                   metrics=[concatenated_loss, Dice_loss, Jaccard_loss, Combi_Dist_loss,
                            Symmetric_Hausdorf_loss, specificity_loss, sensitivity_loss, precision_loss, accuracy_loss, Individual_loss])
    # if settings['debug']:
    #    model2.summary()

    # save weights
    # save weights
    net_model_2 = 'model_2'
    net_weights_2 = os.path.join(settings['model_saved_paths'],
                                 settings['modelname'],
                                 'models', net_model_2 + '.hdf5')

    network2 = {}
    network2['model'] = model2
    network2['weights'] = net_weights_2
    network2['history'] = None
    network2['special_name_2'] = net_model_2

    # load predefined weights if transfer learning is selected

    if settings['full_train'] is False:

        # load default weights
        print("> CNN: Loading learnedmodel weights from the", \
        settings['learnedmodel_model'], "configuration")
        learnedmodel_model = os.path.join(settings['model_saved_paths'], \
                                        settings['learnedmodel_model'],'models')
        model = os.path.join(settings['model_saved_paths'],
                             settings['modelname'])
        network1_w_def = os.path.join(model, 'models', 'model_1.hdf5')
        network2_w_def = os.path.join(model, 'models', 'model_2.hdf5')

        if not os.path.exists(model):
            shutil.copy(learnedmodel_model, model)
        else:
            shutil.copyfile(os.path.join(learnedmodel_model,
                                         'model_1.hdf5'),
                            network1_w_def)
            shutil.copyfile(os.path.join(learnedmodel_model,
                                         'model_2.hdf5'),
                            network2_w_def)
        try:
            network1['model'].load_weights(network1_w_def, by_name=True)
            network2['model'].load_weights(network2_w_def, by_name=True)
        except:
            print("> ERROR: The model", \
                settings['modelname'],  \
                'selected does not contain a valid network model')
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    if settings['load_weights'] is True:
        print("> CNN: loading weights from", \
            settings['modelname'], 'configuration')
        print(net_weights_1)
        print(net_weights_2)

        network1['model'].load_weights(net_weights_1, by_name=True)
        network2['model'].load_weights(net_weights_2, by_name=True)

    return [network1, network2]


def define_training_layers(model, num_layers=1, number_of_samples=None):
    """
    Define the number of layers to train and freeze the rest

    inputs: - model: Neural network object network1 or network2 - number of
    layers to retrain - nunber of training samples

    outputs - updated model
    """
    # use the nunber of samples to choose the number of layers to retrain
    if number_of_samples is not None:
        if number_of_samples < 10000:
            num_layers = 1
        elif number_of_samples < 100000:
            num_layers = 2
        else:
            num_layers = 3

    # all layers are first set to non trainable

    net = model['model']
    for l in net.layers:
         l.trainable = False

    print("> CNN: re-training the last", num_layers, "layers")

    # re-train the FC layers based on the number of retrained
    # layers
    net.get_layer('out').trainable = True

    if num_layers == 1:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
    if num_layers == 2:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
    if num_layers == 3:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True

    #net.compile(loss='categorical_crossentropy',
    #            optimizer='adadelta',
    #            metrics=['accuracy'])

    model['model'] = net
    return model


def train_model(model, x_train, y_train, val_x_train, val_y_train, settings, initial_epoch=0):
    """
    fit the cascaded model.

    """
    num_epochs = settings['max_epochs']
    train_split_perc = settings['train_split']
    batch_size = settings['batch_size']

    # convert labels to categorical
    # y_train = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
    # y_train = keras.utils.to_categorical(y_train == 1,
    #                                      len(np.unique(y_train == 1)))
    y_train = keras.utils.to_categorical(y_train == 1,
                                         len(np.unique(y_train == 1)))
    Y_val = keras.utils.to_categorical(val_y_train == 1,
                                       len(np.unique(val_y_train == 1)))

    # split training and validation
    perm_indices = np.random.permutation(x_train.shape[0])
    train_val = int(len(perm_indices)*train_split_perc)

    x_train_ = x_train[:train_val]
    y_train_ = y_train[:train_val]

    # split training and validation
    perm_indices_val = np.random.permutation(val_x_train.shape[0])
    train_val_extra = int(len(perm_indices_val)*train_split_perc)


    x_val_ = val_x_train[:train_val_extra]
    y_val_ = Y_val [:train_val_extra]


    print("x_train_:", x_train_.shape[0])

    print("y_train_:", y_train_.shape[0])

    print("x_val_:", x_val_.shape[0])

    print("y_val_", y_val_.shape[0])








    history = model['model'].fit_generator(da_generator(
        x_train_, y_train_,
        batch_size=batch_size),
        validation_data=val_generator(x_val_, y_val_, batch_size=batch_size),
        # validation_data = (x_val_, y_val_),
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=int(x_train_.shape[0] / batch_size),
        validation_steps=int(x_val_.shape[0] / batch_size),

        verbose=settings['net_verbose'],
        callbacks=[ModelCheckpoint(model['weights'],
                                   monitor='val_Dice_loss',
                                   save_best_only=True,
                                   save_weights_only=True),
                   EarlyStopping(monitor='val_concatenated_loss',
                                 min_delta=0,
                                 patience=settings['patience'],
                                 verbose=0,
                                 mode='auto'),
                   TensorBoard(log_dir='./tensorboardlogs', histogram_freq=0,
                               write_graph=True,  write_images=True)])

                               # write_graph = True, write_images = True), GarbageCollectionCallback()], use_multiprocessing = True)

    # h = model['model'].fit_generator(da_generator(
    #     x_train_, y_train_,
    #     batch_size=batch_size),
    #     validation_data=val_generator(x_val_, y_val_, batch_size=batch_size),
    #     # validation_data = (x_val_, y_val_),
    #     epochs=num_epochs,
    #     initial_epoch=initial_epoch,
    #     steps_per_epoch=x_train_.shape[0]/batch_size,
    #     validation_steps=x_val_.shape[0]/batch_size,
    #
    #     verbose=settings['net_verbose'],
    #     callbacks=[ModelCheckpoint(model['weights'],
    #                                # monitor=['val_all_loss'],
    #                                save_best_only=True,
    #                                save_weights_only=True),
    #                EarlyStopping(  # monitor='val_loss',
    #                    # min_delta=0,
    #                    patience=settings['patience'])])
    # mode='auto'),
    # TensorBoard(log_dir='./tensorboardlogs',
    # histogram_freq=0,
    # write_graph=True)])


    # h = model['model'].fit_generator(da_generator(
    #     x_train_, y_train_,
    #     batch_size=batch_size),
    #     validation_data=(x_val_, y_val_),
    #     epochs=num_epochs,
    #     initial_epoch=initial_epoch,
    #     steps_per_epoch=x_train_.shape[0]/batch_size,
    #     verbose=settings['net_verbose'],
    #     callbacks=[ModelCheckpoint(model['weights'],
    #                                monitor='val_Dice_loss',
    #                                save_best_only=True,
    #                                save_weights_only=True),
    #                EarlyStopping(monitor='val_concatenated_loss',
    #                              min_delta=0,
    #                              patience=settings['patience'],
    #                              verbose=0,
    #                              mode='auto'),
    #                TensorBoard(log_dir='./tensorboardlogs', histogram_freq=0,
    #                            write_graph=True,  write_images=True
    #                            # histogram_freq=0,
    #                            # #batch_size=32,
    #                            # write_graph=True,
    #                            #write_grads=False,
    #                            # write_images=True,
    #                            #embeddings_freq=0,
    #                            #embeddings_layer_names=None,
    #                            #embeddings_metadata=None,
    #                            #embeddings_data=None,
    #                            #update_freq='epoch'
    #                            #)
    #                            )], use_multiprocessing=True)
    model['history'] = history
    print(history.history.keys())
    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    H_Dice_loss = np.array(history.history['Dice_loss'])
    array_length = len(H_Dice_loss)
    last_element = H_Dice_loss[array_length - 1]

    print(last_element)
    print('\x1b[6;30;42m' + 'Dice_loss:' + '\x1b[0m')
    print(last_element)

    if settings['debug']:
        print("Loading best weights after training")

    model['model'].load_weights(model['weights'])

    return model

def fit_thismodel(model, x_train, y_train, settings, initial_epoch=0):

    model['model'].load_weights(model['weights'])

    return model

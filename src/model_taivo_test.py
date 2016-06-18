import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import csv
import time
import code
import numpy as np
import tensorflow as tf
import math
import scipy

import patch_extraction_module as pem
import data_loading_module as dlm
import constants as const
import postprocessing as pm

import cProfile
import pstats
import io

ROOT_DIR = "../"
PIXEL_DEPTH = 255
NUM_LABELS = 2
NUM_CHANNELS = 3  # RGB images

SEED = None
NP_SEED = int(time.time());

BATCH_SIZE = 32
BALANCE_SIZE_OF_CLASSES = True  # recommended to leave True

RESTORE_MODEL = True
TERMINATE_AFTER_TIME = True
NUM_EPOCHS = 1
MAX_TRAINING_TIME_IN_SEC = 2 * 3600  # NB: 28800 = 8 hours
RECORDING_STEP = 100

BASE_LEARNING_RATE = 0.01
DECAY_RATE = 0.99
DECAY_STEP = 100000
LOSS_WINDOW_SIZE = 10

IMG_PATCHES_RESTORE = False
TRAINING_SIZE = 100

VALIDATION_SIZE = 10000  # Size of the validation set in # of patches
VALIDATE = False
VALIDATION_STEP = 500  # must be multiple of RECORDING_STEP

VISUALIZE_PREDICTION_ON_TRAINING_SET = False
VISUALIZE_NUM = -1  # -1 means visualize all

RUN_ON_TEST_SET = False
TEST_SIZE = 50

tf.app.flags.DEFINE_string("train_dir", ROOT_DIR + "tmp/", """Directory where to write event logs and checkpoint.""")

FLAGS = tf.app.flags.FLAGS


# predictions - Nx2 array for image where width * height = N
#             - each cell contains 2 probabilities for 2 classes
# labels      - array of same size as predicitons, 1 hot assignment
def error_rate(predictions, labels):
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def initialization_check():
    if VALIDATION_STEP % RECORDING_STEP != 0:
        print("Error: Validation step must be divisible by recording step.")
        sys.exit(1)
    if const.IMG_HEIGHT % const.IMG_PATCH_SIZE != 0 or const.IMG_WIDTH % const.IMG_PATCH_SIZE != 0:
        print("Error: Patch size must divide both image height and width.")
        sys.exit(1)
    if const.IMG_CONTEXT_SIZE < const.IMG_PATCH_SIZE:
        print("Error: Patch size be smaller or equal than context size.")
        sys.exit(1)
    if (const.IMG_CONTEXT_SIZE - const.IMG_PATCH_SIZE) % 2 == 1:
        print("Error: Border size not well defined (different between context and patch size must be even).")
        sys.exit(1)


def main(argv=None):  # pylint: disable=unused-argument

    # Start profiling
    pr = cProfile.Profile()
    pr.enable()

    initialization_check()
    ######################
    ### INITIALIZATION ###
    ######################
    print("----------- SETTINGS -----------")
    print("Batch size: " + str(BATCH_SIZE))
    print("Context size: " + str(const.IMG_CONTEXT_SIZE))
    print("Patch size: " + str(const.IMG_PATCH_SIZE))
    print("Time is termination criterion: " + str(TERMINATE_AFTER_TIME))
    print("Train for: " + str(MAX_TRAINING_TIME_IN_SEC) + "s")
    print("--------------------------------\n")
    np.random.seed(NP_SEED)
    num_epochs = NUM_EPOCHS

    train_data_filename = ROOT_DIR + "data/training/images/"
    train_labels_filename = ROOT_DIR + "data/training/groundtruth/"
    test_data_filename = ROOT_DIR + "data/test_set/"

    #######################
    ### LOADING PATCHES ###
    #######################
    prefix = ROOT_DIR + "objects/"
    patches_filename = prefix + "patches_imgs"
    labels_filename = prefix + "patches_labels"
    patches_balanced_filename = prefix + "patches_imgs_balanced"
    labels_balanced_filename = prefix + "patches_labels_balanced"

    if IMG_PATCHES_RESTORE and os.path.isfile(patches_balanced_filename + ".npy"):
        if BALANCE_SIZE_OF_CLASSES:
            train_data = np.load(patches_balanced_filename + ".npy")
            train_labels = np.load(labels_balanced_filename + ".npy")
        else:
            train_data = np.load(patches_filename + ".npy")
            train_labels = np.load(labels_filename + ".npy")
        train_size = train_labels.shape[0]
    else:
        if os.path.isfile(const.PATCHES_MEAN_PATH + ".npy"):
            os.remove(const.PATCHES_MEAN_PATH + ".npy")
        print(const.PATCHES_MEAN_PATH + ".npy" + " removed.")
        train_data = dlm.extract_data(train_data_filename, TRAINING_SIZE, 1)
        train_labels = dlm.extract_labels(train_labels_filename, TRAINING_SIZE, 1)
        np.save(patches_filename, train_data)
        np.save(labels_filename, train_labels)

    print("Shape of patches: " + str(train_data.shape))
    print("Shape of labels:  " + str(train_labels.shape))

    ##############################
    ### BALANCING TRAINING SET ###
    ##############################
    if BALANCE_SIZE_OF_CLASSES:
        ### AUXILIARY FUNCTION ###
        def num_of_datapoints_per_class():
            c0 = 0
            c1 = 0
            for i in range(len(train_labels)):
                if train_labels[i][0] == 1:
                    c0 = c0 + 1
                else:
                    c1 = c1 + 1
            print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
            return (c0, c1)

        ### END OF AUXILIARY FUNCTION ###

        # Computing per class number of data points
        (c0, c1) = num_of_datapoints_per_class();

        # Balancing
        if IMG_PATCHES_RESTORE:
            print("Skipping balancing - balanced data already loaded from the disk.")
        else:
            print("Balancing training data.")
            min_c = min(c0, c1)
            idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
            idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
            new_indices = idx0[0:min_c] + idx1[0:min_c]

            train_data = train_data[new_indices, :, :, :]
            train_labels = train_labels[new_indices]
            train_size = train_labels.shape[0]

            num_of_datapoints_per_class();
            np.save(patches_balanced_filename, train_data)
            np.save(labels_balanced_filename, train_labels)

    ##########################################
    ### SETUP OUT OF SAMPLE VALIDATION SET ###
    ##########################################
    PATCHES_VALIDATION = ROOT_DIR + "objects/validation_patches"
    LABELS_VALIDATION = ROOT_DIR + "objects/validation_labels"
    os.path.isfile(const.PATCHES_MEAN_PATH + ".npy")

    if VALIDATE:
        if RESTORE_MODEL and os.path.isfile(PATCHES_VALIDATION + ".npy") and os.path.isfile(LABELS_VALIDATION + ".npy"):
            msg = "Validation data read from the disk."
            validation_data = np.load(PATCHES_VALIDATION + ".npy")
            validation_labels = np.load(LABELS_VALIDATION + ".npy")
        else:
            msg = "Validation data recreated from training data."
            perm_indices = np.random.permutation(np.arange(0, len(train_data)))

            validation_data = train_data[perm_indices[0:VALIDATION_SIZE]]
            validation_labels = train_labels[perm_indices[0:VALIDATION_SIZE]]

            np.save(PATCHES_VALIDATION, validation_data)
            np.save(LABELS_VALIDATION, validation_labels)

            train_data = train_data[perm_indices[VALIDATION_SIZE:perm_indices.shape[0]]]
            train_labels = train_labels[perm_indices[VALIDATION_SIZE:perm_indices.shape[0]]]
            train_size = train_labels.shape[0]

        print("\n----------- VALIDATION INFO -----------")
        print(msg)
        print("Shape of validation set: " + str(validation_data.shape))
        print("New shape of training set: " + str(train_data.shape))
        print("New shape of labels set: " + str(train_labels.shape))
        print("---------------------------------------\n")

    ####################################
    ### CREATING VARIABLES FOR GRAPH ###
    ####################################
    train_data_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE, const.IMG_CONTEXT_SIZE, const.IMG_CONTEXT_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

    ###############
    ### WEIGHTS ###
    ###############
    num_of_CNN_params_to_learn = 0
    num_of_FC_params_to_learn = 0

    ### CONVOLUTIONAL LAYER 1 ###
    with tf.name_scope('conv1') as scope:
        conv1_dim = 5
        conv1_num_of_maps = 16
        conv1_weights = tf.Variable(
            tf.truncated_normal([conv1_dim, conv1_dim, NUM_CHANNELS, conv1_num_of_maps],
                                stddev=0.1,
                                seed=SEED), name='weights')
        conv1_biases = tf.Variable(tf.zeros([conv1_num_of_maps]), name='biases')
    num_of_CNN_params_to_learn += conv1_dim * conv1_dim * conv1_num_of_maps

    ### CONVOLUTIONAL LAYER 2 ###
    with tf.name_scope('conv2') as scope:
        conv2_dim = 3
        conv2_num_of_maps = 32
        conv2_weights = tf.Variable(
            tf.truncated_normal([conv2_dim, conv2_dim, conv1_num_of_maps, conv2_num_of_maps],
                                stddev=0.1,
                                seed=SEED), name='weights')
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_num_of_maps]), name='biases')
    num_of_CNN_params_to_learn += conv2_dim * conv2_dim * conv2_num_of_maps

    ### CONVOLUTIONAL LAYER 3 ###
    with tf.name_scope('conv3') as scope:
        conv3_dim = 3
        conv3_num_of_maps = 32
        conv3_weights = tf.Variable(
            tf.truncated_normal([conv3_dim, conv3_dim, conv2_num_of_maps, conv3_num_of_maps],
                                stddev=0.1,
                                seed=SEED), name='weights')
        conv3_biases = tf.Variable(tf.constant(0.1, shape=[conv3_num_of_maps]), name='biases')
    num_of_CNN_params_to_learn += conv3_dim * conv3_dim * conv3_num_of_maps

    ### CONVOLUTIONAL LAYER 4 ###
    with tf.name_scope('conv4') as scope:
        conv4_dim = 3
        conv4_num_of_maps = 64
        conv4_weights = tf.Variable(
            tf.truncated_normal([conv4_dim, conv4_dim, conv3_num_of_maps, conv4_num_of_maps],
                                stddev=0.1,
                                seed=SEED), name='weights')
        conv4_biases = tf.Variable(tf.constant(0.1, shape=[conv4_num_of_maps]), name='biases')
    num_of_CNN_params_to_learn += conv4_dim * conv4_dim * conv4_num_of_maps

    ### FULLY CONNECTED LAYER 1 ###
    tmp_neuron_num = int((const.IMG_PATCH_SIZE / 8) * (const.IMG_PATCH_SIZE / 8) * conv4_num_of_maps);
    with tf.name_scope('fc1') as scope:
        fc1_size = 64
        fc1_weights = tf.Variable(
            tf.truncated_normal([tmp_neuron_num, fc1_size],
                                stddev=0.1,
                                seed=SEED), name='weights')
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc1_size]), name='biases')
    num_of_FC_params_to_learn += tmp_neuron_num * fc1_size;

    ### FULLY CONNECTED LAYER 2 ###
    with tf.name_scope('fc1') as scope:
        fc2_weights = tf.Variable(
            tf.truncated_normal([fc1_size, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED), name='weights')
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='biases')
    num_of_FC_params_to_learn += fc1_size * NUM_LABELS

    if not RESTORE_MODEL:
        print("----------- NUM OF PARAMS TO LEARN -----------")
        print("Num of CNN params to learn: " + str(num_of_CNN_params_to_learn));
        print("Num of FC params to learn: " + str(num_of_FC_params_to_learn));
        print("Total num of params to learn: " + str(num_of_CNN_params_to_learn + num_of_FC_params_to_learn))
        print("----------------------------------------------\n")

    # Get prediction for given input image
    def get_prediction(tf_session, img, stride):
        data = pem.zero_center(
            np.asarray(pem.input_img_crop(img, const.IMG_PATCH_SIZE, const.IMG_BORDER_SIZE, stride, 0)))
        data_node = tf.cast(tf.constant(data), tf.float32)
        prediction = tf_session.run(tf.nn.softmax(model(data_node)))

        ### UPSAMPLING ###
        imgheight = img.shape[0]
        imgwidth = img.shape[1]
        prediction_img_per_pixel = np.zeros((imgheight, imgwidth))
        count_per_pixel = np.zeros((imgheight, imgwidth))
        idx = 0
        for i in range(0, imgheight - const.IMG_PATCH_SIZE + 1, stride):
            for j in range(0, imgwidth - const.IMG_PATCH_SIZE + 1, stride):
                prediction_img_per_pixel[j: j + const.IMG_PATCH_SIZE, i: i + const.IMG_PATCH_SIZE] += prediction[idx][1]
                count_per_pixel[j: j + const.IMG_PATCH_SIZE, i: i + const.IMG_PATCH_SIZE] += 1.0
                idx += 1

        prediction = np.zeros((imgheight * imgwidth, 2))
        idx = 0
        for i in range(imgheight):
            for j in range(imgwidth):
                prediction[idx][1] = prediction_img_per_pixel[j][i] / count_per_pixel[j][i]
                prediction[idx][0] = 1.0 - prediction[idx][1]
                idx += 1
        ### END OF UPSAMPLING ###

        return prediction

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(tf_session, input_path, output_path_overlay, output_path_raw,
                                    truth_input_path=None):
        ### AUXILIARY FUNCTION 0 ###
        def label_to_binary_img(imgwidth, imgheight, w, h, labels):
            array_labels = np.zeros([imgwidth, imgheight])
            idx = 0
            for i in range(0, imgheight, h):
                for j in range(0, imgwidth, w):
                    if labels[idx][0] > 0.5:
                        l = 0
                    else:
                        l = 1
                    array_labels[j:j + w, i:i + h] = l
                    idx = idx + 1
            return array_labels

        ### END OF AUXILIARY FUNCTION 0 ###

        ### AUXILIARY FUNCTION 1 ###
        def label_to_img(imgwidth, imgheight, w, h, labels):
            array_labels = np.zeros([imgwidth, imgheight])
            idx = 0
            for i in range(0, imgheight, h):
                for j in range(0, imgwidth, w):
                    array_labels[j:j + w, i:i + h] = labels[idx][1]
                    idx = idx + 1
            return array_labels
            ### END OF AUXILIARY FUNCTION 1 ###

        ### AUXILIARY FUNCTION 2 ###
        def make_img_overlay(img, predicted_img, true_img=None):
            ### AUXILIARY FUNCTION 2.1 ###
            def img_float_to_uint8(img):
                rimg = img - np.min(img)
                rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
                return rimg

            ### END OF AUXILIARY FUNCTION 2.1 ###

            w = img.shape[0]
            h = img.shape[1]
            color_mask = np.zeros((w, h, 3), dtype=np.uint8)
            color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH
            if (true_img is not None):
                color_mask[:, :, 1] = true_img * PIXEL_DEPTH

            img8 = img_float_to_uint8(img)
            background = Image.fromarray(img8, 'RGB').convert("RGBA")
            overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
            new_img = Image.blend(background, overlay, 0.2)
            return new_img

        ### END OF AUXILIARY FUNCTION 2 ###

        def pixels_to_patches(img, round=False, foreground_threshold=0.5, stride=const.IMG_PATCH_SIZE):
            res_img = np.zeros(img.shape)
            for i in range(0, img.shape[0], stride):
                for j in range(0, img.shape[1], stride):
                    tmp = np.zeros((stride, stride))
                    tmp[0: stride, 0: stride] = img[j: j + stride, i: i + stride]
                    tmp[tmp < 0.5] = 0
                    tmp[tmp >= 0.5] = 1
                    res_img[j: j + stride, i: i + stride] = np.mean(tmp)

                    # res_img[j : j + stride, i : i + stride] = np.mean(img[j : j + stride, i : i + stride])
                    if round:
                        if res_img[j, i] >= foreground_threshold:
                            res_img[j: j + stride, i: i + stride] = 1
                        else:
                            res_img[j: j + stride, i: i + stride] = 0
            return res_img

        # Read images from disk
        img = mpimg.imread(input_path)
        img_truth = None
        if truth_input_path != None:
            img_truth = mpimg.imread(truth_input_path)

        # Get prediction
        stride = const.IMG_PATCH_SIZE
        prediction = get_prediction(tf_session, img, stride)
        ### POST PROCESSING ###
        # for i in range(1):
        #     prediction = pm.postprocess_prediction(prediction, int(np.sqrt(prediction.shape[0])), int(np.sqrt(prediction.shape[0])))
        #######################

        # Show per pixel probabilities
        prediction_as_per_pixel_img = label_to_img(img.shape[0], img.shape[1], 1, 1, prediction)

        # Show per patch probabilities
        prediction_as_img = pixels_to_patches(prediction_as_per_pixel_img)

        # Rounded to 0 / 1 - per pixel
        prediction_as_binary_per_pixel_img = label_to_binary_img(img.shape[0], img.shape[1], 1, 1, prediction)

        # Round to 0 / 1 - per patch
        prediction_as_binary_img = pixels_to_patches(prediction_as_per_pixel_img, True)

        # Save output to disk
        # Overlay
        oimg = make_img_overlay(img, prediction_as_binary_per_pixel_img, img_truth)
        oimg.save(output_path_overlay + "_pixels.png")

        oimg2 = make_img_overlay(img, prediction_as_binary_img, img_truth)
        oimg2.save(output_path_overlay + "_patches.png")

        # Raw image
        scipy.misc.imsave(output_path_raw + "_pixels.png", prediction_as_per_pixel_img)
        scipy.misc.imsave(output_path_raw + "_patches.png", prediction_as_img)

        return (prediction, prediction_as_binary_img)

    def validate(validation_model, labels):
        print("\n --- Validation ---")
        prediction = s.run(tf.nn.softmax(validation_model))
        err = error_rate(prediction, labels)
        print("Validation set size: %d" % VALIDATION_SIZE)
        print("Error: %.1f%%" % err)
        print("--------------------")
        return err

    def model(data, train=False):
        # CONV. LAYER 1
        conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        norm1 = tf.nn.lrn(relu1)
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # CONV. LAYER 2
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        norm2 = tf.nn.lrn(relu2)
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # CONV. LAYER 3
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        norm3 = tf.nn.lrn(relu3)
        pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # CONV. LAYER 4
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        norm4 = tf.nn.lrn(relu4)
        pool4 = tf.nn.max_pool(norm4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv_out = pool4

        # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
        pool_shape = conv_out.get_shape().as_list()
        reshape = tf.reshape(
            conv_out,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        ##### DROPOUT #####
        # if train:
        #     hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        ### FINAL ACTIVATION ###
        out = tf.sigmoid(tf.matmul(hidden, fc2_weights) + fc2_biases)

        # Make an image summary for 4d tensor image with index idx
        def get_image_summary(img, idx=0):
            V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
            img_w = img.get_shape().as_list()[1]
            img_h = img.get_shape().as_list()[2]
            min_value = tf.reduce_min(V)
            V = V - min_value
            max_value = tf.reduce_max(V)
            V = V / (max_value * PIXEL_DEPTH)
            V = tf.reshape(V, (img_w, img_h, 1))
            V = tf.transpose(V, (2, 0, 1))
            V = tf.reshape(V, (-1, img_w, img_h, 1))
            return V

        if train == True:
            tf.image_summary('summary_data', get_image_summary(data))
            tf.image_summary('summary_conv1', get_image_summary(conv1))
            tf.image_summary('summary_pool1', get_image_summary(pool1))
            tf.image_summary('summary_conv2', get_image_summary(conv2))
            tf.image_summary('summary_pool2', get_image_summary(pool2))
            tf.image_summary('summary_conv3', get_image_summary(conv3))
            tf.image_summary('summary_pool3', get_image_summary(pool3))
            tf.image_summary('summary_conv4', get_image_summary(conv4))
            tf.image_summary('summary_pool4', get_image_summary(pool4))
            tf.histogram_summary('weights_conv1', conv1_weights)
            tf.histogram_summary('weights_conv2', conv2_weights)
            tf.histogram_summary('weights_conv3', conv3_weights)
            tf.histogram_summary('weights_conv4', conv4_weights)
            tf.histogram_summary('weights_FC1', fc1_weights)
            tf.histogram_summary('weights_FC2', fc2_weights)

        return out

    ### END OF MODEL ###

    ##################
    ### SETUP LOSS ###
    ##################
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))
    tf.scalar_summary('loss', loss)

    cumulative_loss = tf.Variable(1.0)
    loss_window = np.zeros(LOSS_WINDOW_SIZE)
    index_loss_window = 0

    tf.scalar_summary('loss_smoothed', cumulative_loss)
    #########################
    ### L2 REGULARIZATION ###
    #########################
    # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
    #                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # loss = tf.add(loss, 5e-4 * regularizers)

    ### IN SAMPLE ERROR ###
    error_insample_tensor = tf.Variable(0)
    tf.scalar_summary('error_insample_smoothed', error_insample_tensor)

    insample_error_window = np.zeros(LOSS_WINDOW_SIZE)
    index_insample_error_window = 0

    ### VALIDATION ERROR ###
    error_validation_tensor = tf.Variable(0)
    tf.scalar_summary('error_validation', error_validation_tensor)

    # Create the validation model here to prevent recreating large constant nodes in graph later
    if VALIDATE:
        data_node = tf.cast(tf.constant(np.asarray(validation_data)), tf.float32)
        validation_model = model(data_node)

    ### SUMMARY OF WEIGHTS ###
    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, conv3_weights, conv3_biases,
                       fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'conv3_weights',
                        'conv3_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.scalar_summary(all_params_names[i], norm_grad_i)

    #######################
    ### OPTIMIZER SETUP ###
    #######################
    batch = tf.Variable(0)
    learning_rate = tf.Variable(BASE_LEARNING_RATE)
    # tf.train.exponential_decay(
    #     BASE_LEARNING_RATE,  # Base learning rate.
    #     batch * BATCH_SIZE,  # Current index into the dataset.
    #     DECAY_STEP,          # Decay step.
    #     DECAY_RATE,          # Decay rate.
    #     staircase=True)

    tf.scalar_summary('learning_rate', learning_rate)

    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1).minimize(loss, global_step=batch)

    ### Predictions for the minibatch, validation set and test set. ###
    train_prediction = tf.nn.softmax(logits)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #######################
    ### RUNNING SESSION ###
    #######################
    with tf.Session() as s:
        if RESTORE_MODEL:
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("### MODEL RESTORED ###")
        else:
            tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=s.graph)
            print("### MODEL INITIALIZED ###")
            print("### TRAINING STARTED ###")

            training_indices = range(train_size)
            start = time.time()
            run_training = True
            iepoch = 0
            batch_index = 1;
            while run_training:
                perm_indices = np.random.permutation(training_indices)

                for step in range(int(train_size / BATCH_SIZE)):
                    if not run_training:
                        break;

                    offset = (batch_index * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]

                    # This dictionary maps the batch data (as a np array) to the node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    # Run the operations
                    _, l, lr, predictions = s.run(
                        [optimizer, loss, learning_rate, train_prediction],
                        feed_dict=feed_dict)

                    # Update cumulative loss
                    loss_window[index_loss_window] = l
                    index_loss_window = (index_loss_window + 1) % loss_window.shape[0]

                    # Update insample error
                    insample_error_window[index_insample_error_window] = error_rate(predictions, batch_labels)
                    index_insample_error_window = (index_insample_error_window + 1) % insample_error_window.shape[0]

                    if batch_index % RECORDING_STEP == 0 and batch_index > 0:
                        closs = np.mean(loss_window)
                        s.run(cumulative_loss.assign(closs))

                        insample_error = error_rate(predictions, batch_labels)
                        s.run(error_insample_tensor.assign(insample_error))

                        if VALIDATE and batch_index % VALIDATION_STEP == 0:
                            validation_err = validate(validation_model, validation_labels)
                            s.run(error_validation_tensor.assign(validation_err))

                        # Writing to disk
                        summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, batch_index)
                        summary_writer.flush()

                        print("\nEpoch: %d, Batch #: %d" % (iepoch, step))
                        print("Global step:     %d" % (batch_index * BATCH_SIZE))
                        print("Time elapsed:    %.3fs" % (time.time() - start))
                        print("Minibatch loss:  %.6f" % l)
                        print("Cumulative loss: %.6f" % closs)
                        print("Learning rate:   %.6f" % lr)

                        print("Minibatch insample error: %.1f%%" % insample_error)
                        sys.stdout.flush()

                        # Save the variables to disk.
                        save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")

                    batch_index += 1
                    if (TERMINATE_AFTER_TIME and time.time() - start > MAX_TRAINING_TIME_IN_SEC):
                        run_training = False;
                        # Save the variables to disk.
                        save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")

                iepoch += 1
                if (not TERMINATE_AFTER_TIME and iepoch >= NUM_EPOCHS):
                    run_training = False;
                    # Save the variables to disk.
                    save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")

        prefix_results = ROOT_DIR + "results/"

        if VISUALIZE_PREDICTION_ON_TRAINING_SET:
            print("--- Visualizing prediction on training set ---")
            prediction_training_dir = prefix_results + "predictions_training/"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)
            limit = TRAINING_SIZE + 1 if VISUALIZE_NUM == -1 else VISUALIZE_NUM
            for i in range(1, limit):
                print("Image: " + str(i))
                img_name = "satImage_%.3d" % i
                input_path = train_data_filename + img_name + ".png"
                truth_path = train_labels_filename + img_name + ".png"
                output_path_overlay = prediction_training_dir + "overlay_" + img_name
                output_path_raw = prediction_training_dir + "raw_" + img_name

                get_prediction_with_overlay(s, input_path, output_path_overlay, output_path_raw, truth_path)

        if VALIDATE:
            validation_err = validate(validation_model, validation_labels)

        if RUN_ON_TEST_SET:
            print("--- Running prediction on test set ---")
            prediction_test_dir = prefix_results + "predictions_test/"
            if not os.path.isdir(prediction_test_dir):
                os.mkdir(prediction_test_dir)

            with open(prefix_results + "submission.csv", "w") as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['id', 'prediction'])
                for i in range(1, TEST_SIZE + 1):

                    print("Test img: " + str(i))
                    # Visualization
                    img_name = "test_" + str(i)
                    input_path = test_data_filename + img_name + ".png"
                    output_path_overlay = prediction_test_dir + "overlay_" + img_name
                    output_path_raw = prediction_test_dir + "raw_" + img_name

                    (_, prediction_as_img) = get_prediction_with_overlay(s, input_path, output_path_overlay,
                                                                         output_path_raw)
                    prediction_as_img = prediction_as_img.astype(np.int)

                    # Saving to csv file for submission
                    num_rows = prediction_as_img.shape[0]
                    num_cols = prediction_as_img.shape[1]
                    rows_out = np.empty((0, 2))
                    for x in range(0, num_rows, const.IMG_PATCH_SIZE):
                        for y in range(0, num_cols, const.IMG_PATCH_SIZE):
                            id = str(i).zfill(3) + "_" + str(x) + "_" + str(y)
                            next_row = np.array([[id, str(prediction_as_img[y][x])]])
                            rows_out = np.concatenate((rows_out, next_row))
                    writer.writerows(rows_out)
            csvfile.close()

    # End profiling and save stats
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    stream = open('profile.txt', 'w');
    ps = pstats.Stats(pr, stream=stream).sort_stats(sortby)
    ps.print_stats()


if __name__ == '__main__':

    tf.app.run()



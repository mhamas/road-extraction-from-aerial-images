"""
Provides methods to train a CNN for post-processing predictions.
(not an improvement over SVM-based post-processing)

"""

import os
import sys

import time as time
import numpy as np

import patch_extraction_module as pem 
import data_loading_module as dlm
import constants as const

from skimage.transform import resize
import matplotlib.image as mpimg
import scipy

import tensorflow as tf

import postprocessing as postpro


ROOT_DIR = "../"
PIXEL_DEPTH = 255
NUM_LABELS = 2
NUM_CHANNELS = 3 # RGB images

SEED = None
NP_SEED = int(time.time());

BATCH_SIZE = 32 
BALANCE_SIZE_OF_CLASSES = True # recommended to leave True

RESTORE_MODEL = False
TERMINATE_AFTER_TIME = True
NUM_EPOCHS = 1
MAX_TRAINING_TIME_IN_SEC = 2 * 3600 # NB: 28800 = 8 hours
#MAX_TRAINING_TIME_IN_SEC = 900 # NB: 28800 = 8 hours
RECORDING_STEP = 100

BASE_LEARNING_RATE = 0.01
DECAY_RATE = 0.99
DECAY_STEP = 100000
LOSS_WINDOW_SIZE = 10

IMG_PATCHES_RESTORE = False
TRAINING_SIZE = 100

VALIDATION_SIZE = 10000  # Size of the validation set in # of patches
VALIDATE = True
VALIDATION_STEP = 500 # must be multiple of RECORDING_STEP

VISUALIZE_PREDICTION_ON_TRAINING_SET = False
VISUALIZE_NUM = -1 # -1 means visualize all

RUN_ON_TEST_SET = True
TEST_SIZE = 50

TRAIN_DIR = ROOT_DIR + "tmp/"

def error_rate(predictions, labels):
    """ Computes the error rate for a set of predictions and groundtruth labels """
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def train_and_apply_model():
    """ Trains a CNN model to post-process predictions and applies it to the raw CNN output. """
    
    ######################
    ### INITIALIZATION ###
    ######################
    np.random.seed(NP_SEED)
    
    t = time.time()
    
    train_data_filename = "../results/CNN_Output/training/raw/"
    train_labels_filename = "../data/training/groundtruth/"
    
    num_images = 100
    
    # ground truth label images and CNN output
    labelsTrue = dlm.extract_label_images(train_labels_filename, num_images, const.POSTPRO_PATCH_SIZE, const.POSTPRO_PATCH_SIZE)
    labelsCNN  = dlm.read_image_array(train_data_filename, num_images, "raw_satImage_%.3d_patches")
    
    for i in range(0, len(labelsCNN)):
        labelsCNN[i] = resize(labelsCNN[i], (labelsCNN[i].shape[0] // const.POSTPRO_PATCH_SIZE, labelsCNN[i].shape[1] // const.POSTPRO_PATCH_SIZE), order=0, preserve_range=True)        
        
    elapsed = time.time() - t
    print("Loading training data took: " + str(elapsed) + " s")


    # extract patches and corresponding groundtruth center value
    t = time.time()
    patch_size = 1
    border_size = const.POSTPRO_CNN_PATCH_SIZE // 2
    stride = 1
    nTransforms = 5
    
    patches = []
    labels = []
    for i in range(0, num_images):
        patches.extend(pem.img_crop(labelsCNN[i], patch_size, border_size, stride, nTransforms))
        labels.extend(pem.img_crop(labelsTrue[i], 1, 0, stride, nTransforms))

    train_data = pem.zero_center(np.expand_dims(np.asarray([np.squeeze(np.asarray(p)) for p in patches]), axis=3))
    train_labels = np.expand_dims(np.squeeze(np.asarray(labels)), axis=1)
    train_labels = np.hstack((train_labels, 1 - train_labels))
    
    elapsed = time.time() - t
    print("Extracting patches from training data took: " + str(elapsed) + " s")  
    train_size = train_labels.shape[0]

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
            print ("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
            return (c0, c1)
        ### END OF AUXILIARY FUNCTION ###

        # Computing per class number of data points
        (c0, c1) = num_of_datapoints_per_class();

        # Balancing
        print ("Balancing training data.")
        min_c = min(c0, c1)
        idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
        idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
        new_indices = idx0[0:min_c] + idx1[0:min_c]
        
        train_data = train_data[new_indices,:,:,:]
        train_labels = train_labels[new_indices]
        train_size = train_labels.shape[0]

        num_of_datapoints_per_class();

            
    ##########################################
    ### SETUP OUT OF SAMPLE VALIDATION SET ###
    ##########################################
    PATCHES_VALIDATION = ROOT_DIR + "objects/validation_patches"
    LABELS_VALIDATION = ROOT_DIR + "objects/validation_labels"
    os.path.isfile(const.PATCHES_MEAN_PATH + ".npy")
    
    if VALIDATE:
        perm_indices = np.random.permutation(np.arange(0,len(train_data)))
        
        validation_data = train_data[perm_indices[0:VALIDATION_SIZE]]
        validation_labels = train_labels[perm_indices[0:VALIDATION_SIZE]]
        
        if not os.path.isdir(const.OBJECTS_PATH):
            os.makedirs(const.OBJECTS_PATH)
            
        np.save(PATCHES_VALIDATION, validation_data)
        np.save(LABELS_VALIDATION, validation_labels)
        
        train_data = train_data[perm_indices[VALIDATION_SIZE:perm_indices.shape[0]]]
        train_labels = train_labels[perm_indices[VALIDATION_SIZE:perm_indices.shape[0]]]
        train_size = train_labels.shape[0]
        
        print("\n----------- VALIDATION INFO -----------")
        print("Validation data recreated from training data.")
        print("Shape of validation set: " + str(validation_data.shape))
        print("New shape of training set: " + str(train_data.shape))
        print("New shape of labels set: " + str(train_labels.shape))
        print("---------------------------------------\n")

    ####################################
    ### CREATING VARIABLES FOR GRAPH ###
    ####################################
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, const.POSTPRO_CNN_PATCH_SIZE, const.POSTPRO_CNN_PATCH_SIZE, 1))
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
            tf.truncated_normal([conv1_dim, conv1_dim, 1, conv1_num_of_maps],  
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

    ### FULLY CONNECTED LAYER 1 ###
    tmp_neuron_num = int(3 * 3 * conv2_num_of_maps);
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

    # Get prediction overlaid on the original image for given input file  

    def validate(validation_model, labels):
        print("\n --- Validation ---")
        prediction = s.run(tf.nn.softmax(validation_model))
        err = error_rate(prediction, labels)
        print ("Validation set size: %d" % VALIDATION_SIZE)
        print ("Error: %.1f%%" % err)
        print ("--------------------")
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
        conv_out = pool2
        
        # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
        pool_shape = conv_out.get_shape().as_list()
        reshape = tf.reshape(
            conv_out,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        
        # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)    
        
        ##### DROPOUT #####
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        
        ### FINAL ACTIVATION ###
        out = tf.sigmoid(tf.matmul(hidden, fc2_weights) + fc2_biases)
        
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
        data_node = tf.cast(tf.constant(np.asarray(validation_data)),tf.float32)
        validation_model = model(data_node)

    
    #######################
    ### OPTIMIZER SETUP ###
    #######################
    batch = tf.Variable(0)
    learning_rate = tf.Variable(BASE_LEARNING_RATE)
    
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 0.1).minimize(loss, global_step=batch)

    ### Predictions for the minibatch, validation set and test set. ###
    train_prediction = tf.nn.softmax(logits)

    # Add ops to save and restore all the variables.train_data_filename
    saver = tf.train.Saver()

    #######################
    ### RUNNING SESSION ###
    #######################
    with tf.Session() as s:
        if RESTORE_MODEL:
            saver.restore(s, TRAIN_DIR + "postpro_model.ckpt")
            print("### MODEL RESTORED ###")
        else:
            tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            print ("### MODEL INITIALIZED ###")
            print ("### TRAINING STARTED ###")

            training_indices = range(train_size)
            start = time.time()
            run_training = True
            iepoch = 0
            batch_index = 1;
            while run_training:
                perm_indices = np.random.permutation(training_indices)

                for step in range (int(train_size / BATCH_SIZE)):
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

                        print ("\nEpoch: %d, Batch #: %d"  % (iepoch, step))
                        print ("Global step:     %d" % (batch_index * BATCH_SIZE))
                        print ("Time elapsed:    %.3fs" %(time.time() - start))
                        print ("Minibatch loss:  %.6f" % l)
                        print ("Cumulative loss: %.6f" % closs)
                        print ("Learning rate:   %.6f" % lr)

                        print ("Minibatch insample error: %.1f%%" % insample_error)
                        sys.stdout.flush()

                        saver.save(s, TRAIN_DIR + "/postpro_model.ckpt")

                    batch_index += 1
                    if (TERMINATE_AFTER_TIME and time.time() - start > MAX_TRAINING_TIME_IN_SEC):
                        run_training = False;
                        saver.save(s, TRAIN_DIR + "/postpro_model.ckpt")
                    
                iepoch += 1
                if (not TERMINATE_AFTER_TIME and iepoch >= NUM_EPOCHS):
                    run_training = False;
                    saver.save(s, TRAIN_DIR + "/postpro_model.ckpt")

        if VALIDATE:
            validation_err = validate(validation_model, validation_labels)         
            
        def get_prediction(tf_session, img):
            data_orig = np.asarray(pem.img_crop(img, 1, const.POSTPRO_CNN_PATCH_SIZE // 2, 1, 0))
            data = pem.zero_center(np.asarray([np.expand_dims(np.squeeze(p), axis=3) for p in data_orig]))
            data_node = tf.cast(tf.constant(data), tf.float32)
            prediction = tf_session.run(tf.nn.softmax(model(data_node)))
            return np.reshape(prediction[:, 0], img.shape, order=1)

    
        # load image, use get_prediction(tf_session, img)
#        num_test_images = 50
#        test_data_filename = "../results/CNN_Output/test/raw/"
#        testLabels  = dlm.read_image_array(test_data_filename, num_test_images, "raw_test_%d_patches")    
#        for i in range(0, len(testLabels)):
#            testLabels[i] = resize(testLabels[i], (testLabels[i].shape[0] // const.POSTPRO_PATCH_SIZE, testLabels[i].shape[1] // const.POSTPRO_PATCH_SIZE), order=0, preserve_range=True)        
#            output = get_prediction(s, testLabels[i])
            
        postpro_fn = const.RESULTS_PATH + "/postprocessing_output"
        # test set
        prob_fn = "../results/CNN_Output/test/raw/"  
        inputFileName = "raw_test_%d_patches"
        outputDir = postpro_fn + "/test/"
        num_images = 50
    
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)
    
        outputImages = []
        imSizes = []
        for i in range(1, num_images+1):
        #    imageid = "raw_test_%d_patches" % i
            imageid = inputFileName % i
            image_filename = prob_fn + imageid + ".png"
        
            if os.path.isfile(image_filename):
                img = mpimg.imread(image_filename) 
                imSizes.append(img.shape)
                output = get_prediction(s, resize(img, (img.shape[0] // const.POSTPRO_PATCH_SIZE, img.shape[1] // const.POSTPRO_PATCH_SIZE), order=0, preserve_range=True) )
                output = np.round(output)
                outputImages.append(output)
                scipy.misc.imsave(outputDir + ("satImage_%d" % i) + ".png" , resize(outputImages[i - 1],  img.shape, order=0, preserve_range=True))
            else:
                print ('File ' + image_filename + ' does not exist')
                
        postpro.create_submission_file(outputImages)
            
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os
import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, MaxPooling2D, Flatten, Dense, GlobalMaxPooling2D, GlobalAveragePooling2D, Softmax, Concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.backend import set_session
from tensorflow.keras.backend import clear_session
from tensorflow.keras.backend import get_session

from gggfc.grammar import Grammar
from gggfc.policy import  Policy
from gggfc.generator import Generator

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_trials', 100, 'Number of graphs to try')
flags.DEFINE_integer('epochs_per_trial',40,'Number of epochs to train per trial')
flags.DEFINE_integer('n_grow', 15, 'Number of grow steps in generator')
flags.DEFINE_integer('max_depth', 10, 'Maximum depth of graph generator')
flags.DEFINE_string('grammar_path','gggfc/grammar_files','Location of grammar files')
flags.DEFINE_string('ops_map_file','gggfc/grammar_files/ops.keras','Location of operator definitions')
flags.DEFINE_string('policy_file','gggfc/grammar_files/gen2_policy.json','Location of generation policy file')
flags.DEFINE_string('model_path','models','Path to store generated models')
flags.DEFINE_string('graph_path','graphs','Path to store generated model graphs in dot')
flags.DEFINE_string('sequences_path','sequences','Path to store model generation sequences')
flags.DEFINE_string('history_path','history','Path to store model training histories')
flags.DEFINE_integer('min_params', 1000000,'Minimum number of model parameters')
flags.DEFINE_integer('max_params',15000000,'Maximum number of model parameters')
flags.DEFINE_integer('batch_size',256,'Batch size')
flags.DEFINE_integer('train_size',16384,'Number of elements in training populationl remainder in validation')

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))


def _build_output_dirs():
    output_dirs = [FLAGS.model_path, FLAGS.sequences_path, FLAGS.history_path, FLAGS.graph_path]

    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

def _save_model_function(function_name, code_string):
    this_model_file = os.path.join(FLAGS.model_path, function_name + ".py")
    with open(this_model_file,"w") as f:
            f.write(code_string)

def train_augment(x,y) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.resize_images(x,[36,36], align_corners=True, preserve_aspect_ratio=True)
    x = tf.image.random_crop(x,[32,32,3])
    x = tf.image.per_image_standardization(x)
    return x, y

def test_standardize(x,y) -> (tf.Tensor, tf.Tensor):
    x = tf.image.per_image_standardization(x)
    return x, y

def _build_datasets(train_size):
    (X, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #y = keras.utils.to_categorical(y, 10).astype(np.float32)
    #y_test = keras.utils.to_categorical(y_test, 10).astype(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(X, y, 
                                                      train_size=train_size, 
                                                      random_state=42,
                                                      stratify=y)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_dataset = train_dataset.map(train_augment,num_parallel_calls=4)\
       .shuffle(buffer_size=10*FLAGS.batch_size)\
       .batch(FLAGS.batch_size, drop_remainder=True)\
       .repeat()
    
    val_dataset = val_dataset.map(test_standardize,num_parallel_calls=4)\
        .batch(FLAGS.batch_size, drop_remainder=True)\
        .repeat()

    return train_dataset, val_dataset

def schedule_fn(epoch):
    mid_point = float(FLAGS.epochs_per_trial / 2)
    if epoch < mid_point:
        return 1e-5 + 0.05 * epoch / mid_point
    elif epoch < FLAGS.epochs_per_trial:
        return 0.05 - 0.05 * (epoch-mid_point) / mid_point
    else:
        return 1e-5

def _log_bad_run(function_name):
    log_file = os.path.join(FLAGS.history_path, function_name + '.csv')
    with open(log_file, "w") as f:
        f.write("epoch,acc,loss,val_acc,val_loss\n")
        f.write("0.0,0.0,0.0,0.0\n")

def main(argv):
    del argv  # Unused.
    
    # Set up output directories
    _build_output_dirs()

    TRAIN_STEPS_PER_EPOCH = FLAGS.train_size / FLAGS.batch_size
    VAL_STEPS = TRAIN_STEPS_PER_EPOCH / 2


    ## SET UP LOOP HERE
    for trial_num in range(FLAGS.n_trials):
        logging.info("** NEW TRIAL: %d" % trial_num)

        # Set up dataset
        train_dataset, val_dataset = _build_datasets(train_size=FLAGS.train_size)
        logging.info('Successfully built datasets')

        # Set up training functions
        lr_schedule_cb = keras.callbacks.LearningRateScheduler(schedule_fn, verbose=0)
        opt = tf.keras.optimizers.SGD(learning_rate=1e-5, decay=1e-6, clipvalue=0.7, momentum=0.9, nesterov=True)

        # Generate model files
        # Build generator
        grammar = Grammar()
        grammar.build(FLAGS.grammar_path)
        policy = Policy(FLAGS.policy_file)
        gen = Generator(grammar,policy, max_depth=FLAGS.max_depth, grow_n = FLAGS.n_grow)
        new_graph = gen.generate(save_dir=FLAGS.sequences_path)
        function_name, code_string = new_graph.convert_to_keras_builder(ops_map_file=FLAGS.ops_map_file)
        _save_model_function(function_name, code_string)

        new_graph.write_dot(FLAGS.graph_path,function_name)

        exec(code_string)

        model = locals()[function_name](16)
        if model.count_params() > FLAGS.max_params:
            model = locals()[function_name](8)
        elif model.count_params() < FLAGS.min_params:
            model = locals()[function_name](32)

        if model.count_params() > FLAGS.max_params:
            logging.warning('Cannot appropriately size model %s - too big: %d parameters' % (function_name, model.count_params()))
            _log_bad_run(function_name)
            del model
            continue
        elif model.count_params() <FLAGS.min_params:
            logging.warning('Cannot appropriately size model %s - too small: %d parameters' % (function_name, model.count_params()))
            _log_bad_run(function_name)
            del model
            continue 
        else:
            logging.info('Succesfully sized model %s: %d parameters; %d nodes' % (function_name, model.count_params(),len(new_graph.nodes) ))

        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

        csv_log_file = os.path.join(FLAGS.history_path,function_name +'.csv')
        csv_logger_cb = keras.callbacks.CSVLogger(csv_log_file,append=True)

        try:
            model.fit(train_dataset, 
                      steps_per_epoch=TRAIN_STEPS_PER_EPOCH, 
                      epochs=FLAGS.epochs_per_trial, 
                      callbacks=[csv_logger_cb, lr_schedule_cb] , 
                      validation_data=val_dataset, 
                      validation_steps=VAL_STEPS,
                      verbose=2)
        except:
            logging.error('Bad train')
            _log_bad_run(function_name)

        reset_keras()

if __name__ == '__main__':
  app.run(main)

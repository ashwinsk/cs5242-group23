import os

import numpy as np
import wandb
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Input, Bidirectional, LSTM, Flatten
from keras.layers import GlobalMaxPooling1D, Conv1D
from keras.layers import multiply, concatenate
from keras.models import Model
from keras.optimizers import Adam
from wandb.keras import WandbCallback

from dataset_iterator import DataGenerator
from utilities import ROCAUCCallback

np.random.seed(5242)

wandb.init(project="cs5242", entity="cs5242-group-23", name="pafa combined", notes="Using 90% of dataset for train",
           config={"epochs": 10, "batch_size": 150, 'lr': 0.001, 'val_batch_size': 50})

lr_decay = [1.] * 10
lr_decay[4] = 0.5
lr_decay[7] = 0.5

TRAIN_DATA_DIR = '../dataset/train/'
VALID_DATA_DIR = '../dataset/valid/'
LABELS_FILE = '../train_kaggle.csv'


def common_conv_skip_block(filters, kernel, input):
    conv1 = Conv1D(filters=filters, kernel_size=kernel, padding='same', strides=1, trainable=False)(input)
    conv1 = Activation(activation='sigmoid')(conv1)

    conv2 = Conv1D(filters=filters, kernel_size=kernel, padding='same', strides=1, trainable=False)(input)
    return multiply([conv1, conv2])


def get_model(weights_path, input):
    batch_norm = BatchNormalization()(input)
    conv1 = common_conv_skip_block(64, 2, batch_norm)
    conv2 = common_conv_skip_block(64, 3, batch_norm)
    conv3 = common_conv_skip_block(64, 4, batch_norm)
    conv1_4 = common_conv_skip_block(64, 8, batch_norm)
    conv1_5 = common_conv_skip_block(64, 16, batch_norm)

    output_3 = concatenate([conv1, conv2, conv3, conv1_4, conv1_5])
    batch_norm_2 = BatchNormalization()(output_3)
    bilstm = Bidirectional(LSTM(units=50, return_sequences=True, trainable=False), trainable=False)(batch_norm_2)
    gb_maxpool = GlobalMaxPooling1D()(bilstm)

    dense_1 = Dense(units=64, kernel_initializer='uniform', activation='relu')(gb_maxpool)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(dropout_1)
    model = Model(inputs=input, outputs=dense_2)
    model.load_weights(weights_path)

    return bilstm


def skip_path(input):
    bn = BatchNormalization()(input)
    conv4 = Conv1D(filters=64, kernel_size=50, strides=25, padding='same', trainable=True)(bn)
    conv5 = Conv1D(filters=64, kernel_size=5, strides=5, padding='valid', trainable=True)(conv4)
    conv6 = Conv1D(filters=32, kernel_size=2, strides=2, padding='valid', trainable=True)(conv5)
    bn2 = BatchNormalization()(conv6)
    flatten = Flatten()(bn2)
    return flatten


if __name__ == '__main__':
    input = Input(shape=(1000, 102))
    ones_model = get_model('model-ones.hdf5', input)
    zeros_model = get_model('model-zeros.hdf5', input)

    out_conv = concatenate([ones_model, zeros_model])
    skip_ft = skip_path(out_conv)

    dense1 = Dense(units=64, kernel_initializer='uniform', activation='relu')(skip_ft)
    dropout_1 = Dropout(0.5)(dense1)

    dense2 = Dense(units=32, kernel_initializer='uniform', activation='relu')(dropout_1)
    dropout_2 = Dropout(0.5)(dense2)

    dense3 = Dense(units=16, kernel_initializer='uniform', activation='relu')(dropout_2)
    dropout_3 = Dropout(0.3)(dense3)

    dense4 = Dense(units=8, kernel_initializer='uniform', activation='relu')(dropout_3)
    dropout_4 = Dropout(0.2)(dense4)

    dense5 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(dropout_4)

    model = Model(inputs=input, outputs=dense5)

    model.summary()

    adam = Adam(lr=wandb.config.lr, )
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])

    train_batch = DataGenerator(TRAIN_DATA_DIR, LABELS_FILE, wandb.config.batch_size, update_batch_size=False)
    valid_batch = DataGenerator(VALID_DATA_DIR, LABELS_FILE, wandb.config.val_batch_size)

    filepath = os.path.join(wandb.run.dir, "saved-model-{epoch:02d}-{val_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)

    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * lr_decay[epoch], verbose=1)

    history = model.fit_generator(train_batch, epochs=wandb.config.epochs, validation_data=valid_batch, verbose=1,
                                  workers=2, use_multiprocessing=False,
                                  callbacks=[lr_scheduler, TerminateOnNaN(),
                                             ROCAUCCallback(validation_data=valid_batch), WandbCallback(), checkpoint])

import os

import numpy as np
import wandb
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Input, Bidirectional, LSTM
from keras.layers import GlobalMaxPooling1D, Conv1D
from keras.layers import multiply, concatenate
from keras.models import Model
from keras.optimizers import Adam
from wandb.keras import WandbCallback

from dataset_iterator import DataGenerator
from utilities import custom_loss_sig, accuracy_zeros

np.random.seed(5242)

wandb.init(project="cs5242", entity="cs5242-group-23", name="pafa zeros", notes="Using 90% of dataset for train",
           config={"epochs": 8, "batch_size": 80, 'lr': 0.001, 'val_batch_size': 100})

TRAIN_DATA_DIR = '../dataset/train/'
VALID_DATA_DIR = '../dataset/valid/'
LABELS_FILE = '../train_kaggle.csv'


def common_conv_skip_block(filters, kernel, input):
    conv1 = Conv1D(filters=filters, kernel_size=kernel, padding='same', strides=1)(input)
    conv1 = Activation(activation='sigmoid')(conv1)

    conv2 = Conv1D(filters=filters, kernel_size=kernel, padding='same', strides=1)(input)
    return multiply([conv1, conv2])


if __name__ == '__main__':
    input = Input(shape=(1000, 102))
    batch_norm = BatchNormalization()(input)
    conv1 = common_conv_skip_block(64, 2, batch_norm)
    conv2 = common_conv_skip_block(64, 3, batch_norm)
    conv3 = common_conv_skip_block(64, 4, batch_norm)
    conv1_4 = common_conv_skip_block(64, 8, batch_norm)
    conv1_5 = common_conv_skip_block(64, 16, batch_norm)

    output_3 = concatenate([conv1, conv2, conv3, conv1_4, conv1_5])
    batch_norm_2 = BatchNormalization()(output_3)
    bilstm = Bidirectional(LSTM(units=50, return_sequences=True))(batch_norm_2)
    gb_maxpool = GlobalMaxPooling1D()(bilstm)

    dense_1 = Dense(units=64, kernel_initializer='uniform', activation='relu')(gb_maxpool)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(dropout_1)
    model = Model(inputs=input, outputs=dense_2)
    model.summary()

    adam = Adam(lr=wandb.config.lr, clipnorm=1.)
    model.compile(loss=custom_loss_sig(0.75), optimizer=adam, metrics=['accuracy', accuracy_zeros])

    train_batch = DataGenerator(TRAIN_DATA_DIR, LABELS_FILE, wandb.config.batch_size, update_batch_size=False,
                                partition=0)
    valid_batch = DataGenerator(VALID_DATA_DIR, LABELS_FILE, wandb.config.val_batch_size)

    filepath = os.path.join(wandb.run.dir, "saved-model-{epoch:02d}-{val_accuracy_zeros:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy_zeros', verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)

    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.85, verbose=1)

    history = model.fit_generator(train_batch, epochs=wandb.config.epochs, validation_data=valid_batch, verbose=1,
                                  workers=2, use_multiprocessing=False,
                                  callbacks=[lr_scheduler, TerminateOnNaN(), WandbCallback(), checkpoint])

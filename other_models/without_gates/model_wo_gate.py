import os

import wandb
from keras.callbacks.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout, Input, concatenate, \
    BatchNormalization, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from wandb.keras import WandbCallback

from dataset_iterator import DataGenerator

wandb.init(project="cs5242", entity="cs5242-group-23",
           name="3 blocks of conv ks 2", notes="Using 90% of dataset for train",
           config={"epochs": 15, "batch_size": 100, 'lr': 0.001})

from utilities import ROCAUCCallback

TRAIN_DATA_DIR = 'dataset/train/'
VALID_DATA_DIR = 'dataset/valid/'
LABELS_FILE = 'train_kaggle.csv'
TRAIN_FULL = 'train/train'

if __name__ == '__main__':
    input = Input(shape=(1000, 102))
    batch_norm = BatchNormalization()(input)

    conv2_1 = Conv1D(filters=64, kernel_size=2, padding='same', strides=1, activation='relu')(batch_norm)
    conv21_1 = Conv1D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu')(batch_norm)
    conv22_1 = Conv1D(filters=64, kernel_size=4, padding='same', strides=1, activation='relu')(batch_norm)

    op2 = concatenate([conv2_1, conv21_1, conv22_1])
    batch_norm_3 = BatchNormalization()(op2)
    bilstm2 = Bidirectional(LSTM(units=100, return_sequences=True))(batch_norm_3)
    gb_max2 = GlobalMaxPooling1D()(bilstm2)

    conv3_1 = Conv1D(filters=128, kernel_size=2, padding='same', strides=1, activation='relu')(batch_norm)
    conv31_1 = Conv1D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(batch_norm)
    conv32_1 = Conv1D(filters=128, kernel_size=4, padding='same', strides=1, activation='relu')(batch_norm)

    op3 = concatenate([conv3_1, conv31_1, conv32_1])
    batch_norm_4 = BatchNormalization()(op3)
    bilstm3 = Bidirectional(LSTM(units=100, return_sequences=True))(batch_norm_4)
    gb_max3 = GlobalMaxPooling1D()(bilstm3)

    output_4 = concatenate([gb_max2, gb_max3])
    dropout_2 = Dropout(0.3)(output_4)

    dense_1 = Dense(units=128, kernel_initializer='uniform', activation='relu')(dropout_2)
    dropout_3 = Dropout(0.3)(dense_1)
    dense_2 = Dense(units=32, kernel_initializer='uniform', activation='relu')(dropout_3)
    dense_3 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(dense_2)

    model = Model(inputs=input, outputs=dense_3)

    model.summary()

    adam = Adam(learning_rate=0.001)
    # sgd = SGD(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    train_batch = DataGenerator(TRAIN_DATA_DIR, LABELS_FILE,
                                200)
    valid_batch = DataGenerator(VALID_DATA_DIR, LABELS_FILE, 100)

    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, mode='auto', verbose=1)
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch % 2 == 0 else lr, verbose=1)

    filepath = os.path.join(wandb.run.dir, "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                                 mode='auto', period=1)

    history = model.fit_generator(train_batch, epochs=wandb.config.epochs, validation_data=valid_batch, verbose=1,
                                  workers=2, use_multiprocessing=False,
                                  callbacks=[earlystop, lr_scheduler,
                                             ROCAUCCallback(validation_data=valid_batch), WandbCallback(), checkpoint])

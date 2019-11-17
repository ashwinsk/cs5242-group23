from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling2D,LSTM, Conv2D, TimeDistributed, SeparableConv1D, BatchNormalization, Bidirectional, GRU, CuDNNGRU,Dropout, Input, multiply,concatenate, BatchNormalization, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD

from keras.callbacks.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from dataset_iterator import DataGenerator

import wandb
import os
from wandb.keras import WandbCallback
wandb.init(project="cs5242", entity="cs5242-group-23",
           name="3 blocks of conv ks 2", notes="Using 90% of dataset for train",
           config={"epochs": 15, "batch_size": 100, 'lr': 0.001})

from utilities import ROCAUCCallback


from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class roc_callback(Callback):
    def __init__(self, validation_data):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('roc-auc_val: %s' % (str(round(roc_val,4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


TRAIN_DATA_DIR = 'dataset/train/'
VALID_DATA_DIR = 'dataset/valid/'
LABELS_FILE = 'train_kaggle.csv'
TRAIN_FULL = 'train/train'


if __name__ == '__main__':

    input = Input(shape=(1000, 102))
    batch_norm = BatchNormalization()(input)

    conv2_1 = Conv1D(filters=64, kernel_size=2, padding='same', strides=1, activation='relu')(batch_norm)
    conv2_2 =  Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(conv2_1)
    conv2_3 = Conv1D(filters=64, kernel_size=2, padding='same', strides=1, activation='relu')(batch_norm)
    output_21 = multiply([conv2_2, conv2_3])

    conv21_1 = Conv1D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu')(batch_norm)
    conv21_2 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(conv21_1)
    conv21_3 = Conv1D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu')(batch_norm)
    output_22 = multiply([conv21_2, conv21_3])

    conv22_1 = Conv1D(filters=64, kernel_size=4, padding='same', strides=1, activation='relu')(batch_norm)
    conv22_2 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(conv22_1)
    conv22_3 = Conv1D(filters=64, kernel_size=4, padding='same', strides=1, activation='relu')(batch_norm)
    output_23 = multiply([conv22_2, conv22_3])

    op2 = concatenate([output_21, output_22, output_23])
    batch_norm_3 = BatchNormalization()(op2)
    bilstm2 = Bidirectional(LSTM(units=100, return_sequences=True))(batch_norm_3)
    gb_max2 = GlobalMaxPooling1D()(bilstm2)

    conv3_1 = Conv1D(filters=128, kernel_size=2, padding='same', strides=1, activation='relu')(batch_norm)
    conv3_2 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(conv3_1)
    conv3_3 = Conv1D(filters=128, kernel_size=2, padding='same', strides=1, activation='relu')(batch_norm)
    output_31 = multiply([conv3_2, conv3_3])

    conv31_1 = Conv1D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(batch_norm)
    conv31_2 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(conv31_1)
    conv31_3 = Conv1D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(batch_norm)
    output_32 = multiply([conv31_2, conv31_3])

    conv32_1 = Conv1D(filters=128, kernel_size=4, padding='same', strides=1, activation='relu')(batch_norm)
    conv32_2 = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')(conv32_1)
    conv32_3 = Conv1D(filters=128, kernel_size=4, padding='same', strides=1, activation='relu')(batch_norm)
    output_33 = multiply([conv32_2, conv32_3])

    op3 = concatenate([output_31, output_32, output_33])
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
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr*0.9 if epoch % 2 == 0 else lr, verbose=1)

    filepath = os.path.join(wandb.run.dir, "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                                 mode='auto', period=1)

    history = model.fit_generator(train_batch, epochs=wandb.config.epochs, validation_data=valid_batch, verbose=1,
                                  workers=2, use_multiprocessing=False,
                                  callbacks=[earlystop, lr_scheduler,
                                             ROCAUCCallback(validation_data=valid_batch), WandbCallback(), checkpoint])

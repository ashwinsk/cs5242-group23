import keras.backend as K
import tensorflow as tf
import wandb
from keras.callbacks import Callback
from keras.losses import binary_crossentropy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

eps = 1e-12


def accuracy_ones(y_true, y_pred):
    indices_1 = tf.where(tf.greater(y_true, 0.5))
    preds = K.cast(K.greater(K.gather(y_pred, indices_1), 0.5), K.floatx())
    return K.cast(K.equal(K.gather(y_true, indices_1), preds), K.floatx())


def accuracy_zeros(y_true, y_pred):
    indices_0 = tf.where(tf.less(y_true, 0.5))
    preds = K.cast(K.greater(K.gather(y_pred, indices_0), 0.5), K.floatx())
    return K.cast(K.equal(K.gather(y_true, indices_0), preds), K.floatx())


class BatchMetricsCallback(Callback):

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        wandb.log({'batch_loss': logs['loss'], 'batch_acc': logs['accuracy']})
        return


class ROCAUCCallback(Callback):
    def __init__(self, validation_data):
        self.valid_batch = validation_data

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = []
        y_val = []
        for i in tqdm(range(len(self.valid_batch)), desc="Calculating ROC-AUC for Validation Data"):
            y_pred.extend(self.model.predict(self.valid_batch[i][0]).flatten())
            y_val.extend(self.valid_batch[i][1].flatten())
        roc_val = roc_auc_score(y_val, y_pred)
        wandb.log({'roc_0': round(roc_val, 4)})
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def custom_loss():
    def loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)
        y_pred = (y_pred + 1.0) / 2.0
        y_true = (y_true + 1.0) / 2.0
        return K.mean(
            K.binary_crossentropy(y_true, y_pred, from_logits=False), axis=-1)

    return loss


def custom_loss_ce():
    def loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=1e-5)

    return loss


def custom_loss_tanh():
    def loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)
        return -(1.0 / 2.0) * K.mean((1 - y_true) * K.log(1 - y_pred + eps) + (1 + y_true) * K.log(1 + y_pred + eps))

    return loss


def custom_loss_sig(multiplier):
    def loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)
        return -1. * K.mean(
            multiplier * y_true * K.log(y_pred + eps) + (1. / multiplier) * (1 - y_true) * K.log(1 - y_pred + eps))

    return loss

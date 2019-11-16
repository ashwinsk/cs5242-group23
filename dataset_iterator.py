import os

import keras
import numpy as np
import wandb


def read_labels_file(file_path: str) -> np.array:
    """Reads the labels from the csv file and returns it as an array

    :param file_path: path to the csv file containing the labels to the files in the dataset
    :return: labels as a numpy array
    """
    with open(file_path, 'r') as f:
        f.readline()
        vals = f.readlines()

    data = np.empty(shape=(len(vals), 1), dtype=np.int)

    for val in vals:
        val_split = val.split(",")
        data[int(val_split[0])] = int(val_split[1])

    return data


class DataGenerator(keras.utils.Sequence):
    """Iterator can be used to iterate the dataset given for CS5242

    Attributes
    ----------
        max_seq_len : int
            max length of each sample in the dataset
        batch_size : int
            number of samples in each batch of training
        folder : str
            name of the folder containing the files, all the files are assumed to be in .npy format
        shuffle : bool
            this is to enable shuffle of the indices for each epoch
        file_ids : [int]
            array containing the names of all the files in the dataset excluding the .npy extension
        indexes : [int]
            copy of the file_ids array, but used for shuffling the indexes if shuffle enabled and used for slicing the batch
        labels : np.array(np.float32)
            contains the label for each file with indices corresponding to the file id from the file_ids or indexes list

    Usage
    -----
        train = DatasetGenerator('train/train', 'train_kaggle.csv')

    """

    def __init__(self, data_folder='', labels_file='', batch_size=32, max_seq_len=1_000, shuffle=True,
                 update_batch_size=False, partition=None):
        """Constructor to initialize the iterator

        :param data_folder: folder containing the data files; in our case the path to the folder containing the .npy files
        :param labels_file: path to the csv containing the labels to the training samples
        :param batch_size: number of samples in each batch
        :param max_seq_len: max len of each sample
        :param shuffle: whether to shuffle or not on each epoch
        """
        self.batch_size_increment = 15
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.folder = data_folder
        self.shuffle = shuffle
        self.file_ids = np.array([int(file_name.split(".")[0]) for file_name in os.listdir(self.folder)])
        self.indexes = np.copy(self.file_ids)
        self.labels = read_labels_file(labels_file)
        self.epochs = -1
        self.update_batch_size = update_batch_size
        self.on_epoch_end()
        self.partition = partition
        self.len_batch_size = self.__len__()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch

        :return: batches per epoch
        """
        return int(np.floor(len(self.file_ids) / self.batch_size)) if self.partition is None else int(
            np.floor(len(self.file_ids) / self.batch_size)) // 2

    def __getitem__(self, index: int) -> (np.array, np.array):
        """Returns each batch of samples

        :param index: based on the __len__ function this function is called with the index of the batch of samples to be sent
        :return: tuple of the X and the Y for the batch in the dataset
        """
        if self.partition is not None:
            index = int(self.partition * self.len_batch_size) + index

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.empty(shape=(self.batch_size, self.max_seq_len, 102), dtype=float)
        y = np.zeros(shape=(self.batch_size, 1), dtype=np.float)

        for idx, id in enumerate(indexes):
            # get X data for each file
            data_x = np.load(os.path.join(self.folder, str(id) + ".npy"))
            if len(data_x) < self.max_seq_len:
                X[idx] = (np.append(data_x, np.zeros(shape=(self.max_seq_len - len(data_x), 102)), axis=0)).reshape(
                    self.max_seq_len, 102)
            else:
                X[idx] = data_x.reshape(self.max_seq_len, 102)

            # add y for each file
            y[idx] = self.labels[id]

        return X, y

    def on_epoch_end(self):
        """Called at the end of each epoch to shuffle the elements in the next batch"""
        if self.update_batch_size:
            if self.epochs % 2 == 0:
                self.batch_size += self.batch_size_increment
            self.epochs += 1
            wandb.log({'batch_size': self.batch_size})
        if self.shuffle:
            np.random.shuffle(self.indexes)

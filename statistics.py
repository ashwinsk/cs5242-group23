traindata_folder = 'dataset/train/'
TRAIN_FULL = 'train/train'

import os
import numpy as np
from tqdm import tqdm
import csv as csv

if __name__ == '__main__':

    max_seq_len = 1000
    sum_x = np.zeros(102)
    sum_x2 = np.zeros(102)
    mean = np.zeros(102)
    stddev = np.zeros(102)
    length = 0
    files = os.listdir(traindata_folder)
    arr_sum = np.zeros(shape= (1000, 102))
    square_arr_sum = np.zeros(shape= (1000, 102))
    max_arr = np.zeros(shape= (1000, 102))
    for fname in tqdm(files, desc="Calculating mean and standard deviation"):
        data_x = np.load(os.path.join(traindata_folder, fname))
        if len(data_x) < max_seq_len:
            data_x = (np.append(data_x, np.zeros(shape=(max_seq_len - len(data_x), 102)), axis=0))
        length = length + 1
        arr_sum += data_x
        square_arr_sum += np.power(data_x, 2)
        max_arr1 = np.max(data_x, axis=0)
        max_arr = np.maximum(max_arr,max_arr1)
    mean = np.divide(arr_sum, length)
    stddev = np.sqrt(np.divide(square_arr_sum, length) - np.square(mean))
    with open('mean.csv', "w", newline="") as myfile:
        wr = csv.writer(myfile)  # , quoting=csv.QUOTE_ALL)
        wr.writerows(mean)
    with open('stddev.csv', "w", newline="") as myfile:
        wr = csv.writer(myfile)  # , quoting=csv.QUOTE_ALL)
        wr.writerows(stddev)
    with open('max.csv', "w", newline="") as myfile:
        wr = csv.writer(myfile)  # , quoting=csv.QUOTE_ALL)
        wr.writerows(max_arr)
    print("Done")

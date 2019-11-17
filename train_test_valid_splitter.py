import numpy as np
import os
from shutil import copyfile
import math
from tqdm import tqdm

TRAINDATA_FOLDER = 'train/train'

if __name__ == '__main__':
    filenames = np.array(os.listdir(TRAINDATA_FOLDER))
    np.random.shuffle(filenames)

    total = len(filenames)

    # train, test, valid = 0.90, 0.05, 0.05
    train_end_idx = math.floor(0.9*total)
    test_end_idx = math.floor(0.95*total)
    valid_end_idx = total

    os.mkdir('dataset')
    os.mkdir('dataset/train')
    os.mkdir('dataset/test')
    os.mkdir('dataset/valid')

    print("Total number of files:", total)

    # creating train set
    for idx in tqdm(range(0, train_end_idx), desc="Creating train dataset"):
        copyfile(os.path.join(TRAINDATA_FOLDER, filenames[idx]), os.path.join('dataset/train/', filenames[idx]))

    for idx in tqdm(range(train_end_idx, test_end_idx), desc="Creating test dataset"):
        copyfile(os.path.join(TRAINDATA_FOLDER, filenames[idx]), os.path.join('dataset/test/', filenames[idx]))

    for idx in tqdm(range(test_end_idx, total), desc="Creating valid dataset"):
        copyfile(os.path.join(TRAINDATA_FOLDER, filenames[idx]), os.path.join('dataset/valid/', filenames[idx]))

    print("Split training data into train, test and valid.")

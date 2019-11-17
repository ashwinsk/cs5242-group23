import os
import numpy as np

import pandas as pd
from keras.models import load_model

DATA_DIR = 'test/test/'
MODEL_PATH = 'model.h5'
OUTPUT_CSV_FILE = 'out.csv'
MAX_SEQ_LEN = 1000


def get_data(DATA_DIR, file_names):
    X = np.empty(shape=(len(file_names), MAX_SEQ_LEN, 102), dtype=float)

    for idx, id in enumerate(file_names):
        # get X data for each file
        data_x = np.load(os.path.join(DATA_DIR, id))
        if len(data_x) < MAX_SEQ_LEN:
            X[idx] = (np.append(data_x, np.zeros(shape=(MAX_SEQ_LEN - len(data_x), 102)), axis=0)).reshape(
                MAX_SEQ_LEN, 102)
        else:
            X[idx] = data_x.reshape(MAX_SEQ_LEN, 102)
    return X


if __name__ == '__main__':

    model = load_model(MODEL_PATH)

    file_names = os.listdir(DATA_DIR)
    expected_preds = model.predict(get_data(DATA_DIR, file_names))
    df = pd.DataFrame(expected_preds)

    with open(OUTPUT_CSV_FILE, 'w') as f:
        f.write('Id,Predicted\n')
        for idx, file_id in enumerate(file_names):
            f.write(str(file_id.split('.')[0]) + "," + '%.12f' % (df[0][idx]) + "\n")

    print("Done")

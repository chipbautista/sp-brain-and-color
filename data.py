
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.model_selection import train_test_split

from settings import *


class BOLD5000:
    def __init__(self, level, subject=1):
        _df = pd.read_csv(
            EXTRACTED_DATA_DIR + 'data-subj-{}.csv'.format(subject))
        df = _df[~_df[level].isnull()]

        # hardcoding downsampling ugh...
        # red_rows = df[df[level] == 'red'].index.values
        # rows_to_drop = np.random.choice(red_rows, 2200, replace=False)
        # df = df.drop(rows_to_drop)

        self.slice_filenames = df['slice_filename'].values
        self.stimulus_filenames = df['stimulus_filename'].values
        self.labels = LabelEncoder().fit_transform(df[level].values)

        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)

        print('----- BOLD5000 -----')
        print('Found {} valid samples out of {}.'.format(
            len(self.labels), len(_df)))
        print(df[level].value_counts())

    def train_test_split(self, batch_size=32):
        def _get_dataloader(x, y):
            return DataLoader(BOLD5000_Split(x, y), batch_size=batch_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.slice_filenames, self.labels,
            train_size=0.8, stratify=self.labels)

        return (_get_dataloader(X_train, y_train),
                _get_dataloader(X_test, y_test))


class BOLD5000_Split(Dataset):
    def __init__(self, slice_filenames, labels):
        self.slice_filenames = slice_filenames
        self.labels = labels

    def __len__(self):
        return len(self.slice_filenames)

    def __getitem__(self, i):
        # return (
        #     np.load(SLICE_DIR + self.slice_filenames[i] + '.npy'),
        #     self.labels[i]
        # )
        fmri_slice = np.load(SLICE_DIR + self.slice_filenames[i] + '.npy')
        return (
            minmax_scale(fmri_slice.reshape(-1, 1), (-1, 1)
                         ).reshape(fmri_slice.shape),
            self.labels[i]
        )

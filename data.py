
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from settings import *


class BOLD5000:
    def __init__(self, level, subjects=[1], do_downsample=True):
        self.stimulus_filenames = []
        self.slice_filenames = []
        self.labels = []

        print('----- BOLD5000 -----')
        print('Subjects:', subjects)
        print('Downsample:', do_downsample)
        for subj in subjects:
            _df = pd.read_csv(
                EXTRACTED_DATA_DIR + 'data-subj-{}.csv'.format(subj))
            df = _df[~_df[level].isnull()]

            # hardcoding downsampling ugh...
            # if do_downsample:
            #     red_rows = df[df[level] == 'red'].index.values
            #     rows_to_drop = np.random.choice(red_rows, 2200, replace=False)
            #     df = df.drop(rows_to_drop)

            self.slice_filenames.extend('{}/'.format(subj) +
                                        df['slice_filename'].values)
            self.stimulus_filenames.extend(df['stimulus_filename'].values)
            # self.labels.extend(df[level].values)
            self.labels.extend([self.label_scene(filename)
                                for filename in df['stimulus_filename'].values])

        # self.labels = LabelEncoder().fit_transform(self.labels)
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        print('Found {} valid samples'.format(len(self.labels)))

    def fit_normalizer(self, train_data):
        self.normalizer = StandardScaler()
        for i in range(0, len(train_data), 50):
            filenames = train_data[i:i + 50]
            batch = np.array([
                crop(np.load(SLICE_DIR + filename + '.npy'))
                for filename in filenames
            ])
            self.normalizer.partial_fit(batch.reshape(len(filenames), -1))
        print('Normalizer is fit on training data.')

    def train_test_split(self, batch_size=32):
        def _get_dataloader(x, y):
            return DataLoader(BOLD5000_Split(x, y, self.normalizer),
                              batch_size=batch_size)

        X_train, X_test, y_train, y_test = train_test_split(
            self.slice_filenames, self.labels,
            train_size=0.8, stratify=self.labels)

        self.fit_normalizer(X_train)

        return (_get_dataloader(X_train, y_train),
                _get_dataloader(X_test, y_test))

    def label_scene(self, filename):
        if filename.startswith('COCO'):
            return 0
        if filename.startswith('n') and '_' in filename:
            return 1
        return 2


class BOLD5000_Split(Dataset):
    def __init__(self, slice_filenames, labels, normalizer):
        self.slice_filenames = slice_filenames
        self.labels = labels
        self.normalizer = normalizer

    def __len__(self):
        return len(self.slice_filenames)

    def __getitem__(self, i):
        # return (
        #     np.load(SLICE_DIR + self.slice_filenames[i] + '.npy'),
        #     self.labels[i]
        # )
        fmri_slice = crop(np.load(SLICE_DIR +
                                  self.slice_filenames[i] + '.npy'))
        return (
            self.normalizer.transform(fmri_slice.reshape(1, -1)
                                      ).reshape(fmri_slice.shape),
            self.labels[i])


def crop(fmri_volume):
    """
    CSI1 Scans: (71, 89, 72)
    CSI3 Scans: (72, 88, 67)

    => Crop all scans to: (71, 88, 67)
    """
    x, y, z = MIN_3D_SHAPE
    return fmri_volume[:x, :y, :z]

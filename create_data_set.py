"""
Script to create a data set (x, y) pairs
from fMRI slices and stimuli dominant colors.
"""

import pandas as pd
import numpy as np
from nilearn import image

from settings import *


def get_dominant(df, level):
    columns = [c for c in df.columns if level in c]
    values = df[columns].values[0]
    dominant_idx = values.argmax()
    if values[dominant_idx] >= DOMINANCE_THRESHOLD[level]:
        return columns[dominant_idx].split('_')[1]
    return None


color_df = pd.read_csv('data/image_color_percentages.csv')
events_df = pd.read_pickle('data/df_events.p')

for subject in SUBJECTS[:1]:  # 1-4
    data_set = {
        'slice_filename': [],
        'stimulus_filename': [],
        'primary': [],
        'secondary': [],
        'tertiary': []
    }
    for sess in SESSIONS:  # 1 - 15
        for run in RUNS:  # 1 - 30
            try:
                fmri = image.load_img(FMRI_DIR.format(
                    subj=subject, ses=sess, run=run))
            except ValueError:
                print('data for subject:', subject, 'session:', sess, 'run:',
                      run, 'not found.')
                continue

            stimuli_rows = events_df[
                (events_df.Subj == int(subject)) &
                (events_df.Sess == int(sess)) &
                (events_df.Run == int(run))
            ]
            for i, stimuli in enumerate(
                    stimuli_rows[['onset', 'stim_file']].itertuples()):
                fmri_idx = int(round(stimuli.onset) / 2)
                fmri_slice = image.index_img(fmri, fmri_idx)
                # fmri_slices.append(fmri_slice.get_data())

                color_data = color_df[color_df.filename == stimuli.stim_file]

                slice_filename = '-'.join([sess, run, str(i)])
                data_set['slice_filename'].append(slice_filename)
                data_set['slice_filename'].append()
                data_set['stimulus_filename'].append(stimuli.stim_file)

                # get the dominant color for each level (prim., sec., ter.)
                for key in ['primary', 'secondary', 'tertiary']:
                    data_set[key].append(get_dominant(color_data, key))

                np.save(file=SLICE_DIR + slice_filename,
                        arr=fmri_slice.get_data(), allow_pickle=True)

        print('Data from session:', sess, 'extracted.')
    csv_filename = EXTRACTED_DATA_DIR + 'data-subj-' + subject + '.csv'
    pd.DataFrame(data_set).to_csv(csv_filename)
    print('Data set for subject', subject, 'saved to', csv_filename)

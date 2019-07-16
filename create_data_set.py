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

for subject in SUBJECTS[:1]:
    fmri_slices = []
    dominant_colors = {
        'primary': [],
        'secondary': [],
        'tertiary': []
    }
    for sess in SESSIONS:
        for run in RUNS:
            fmri = image.load_img(FMRI_DIR.format(
                subj=subject, ses=sess, run=run))
            stimuli_rows = events_df[
                (events_df.Subj == int(subject)) &
                (events_df.Sess == int(sess)) &
                (events_df.Run == int(run))
            ]
            for stimuli in stimuli_rows[['onset', 'stim_file']].itertuples():
                fmri_idx = int(round(stimuli.onset) / 2)
                fmri_slice = image.index_img(fmri, fmri_idx)
                fmri_slices.append(fmri_slice.get_data())

                color_data = color_df[color_df.filename == stimuli.stim_file]
                for key in dominant_colors:
                    dominant_colors[key].append(get_dominant(color_data, key))

    np.save('../processed_data/fmri_slices-' + subject, fmri_slices)
    np.save('../processed_data/colors-' + subject, dominant_colors)
    print('Data set for subject', subject, 'saved.')

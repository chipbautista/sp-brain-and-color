"""
Script to create a data set (x, y) pairs
from fMRI slices and stimuli dominant colors.
"""
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker

from settings import *


parser = ArgumentParser()
parser.add_argument('--average-tr34', default=True)
parser.add_argument('--extract_slices', default=True)
parser.add_argument('--extract_rois', default=False)
args = parser.parse_args()
print(args)


def get_dominant(df, level):
    columns = [c for c in df.columns if level in c]
    values = df[columns].values[0]
    dominant_idx = values.argmax()
    if values[dominant_idx] >= DOMINANCE_THRESHOLD[level]:
        return columns[dominant_idx].split('_')[1]
    return None

def extract_slice(fmri, args, stimuli):
    """
    Stimuli is shown for 1s starting at 6s,
    and then in 10s intervals.

    Each TR is 2 seconds.
    Stimuli at 6s corresponds to TR4, (indexed at 3)
    Peak response is at 4-6 seconds after onset,
    equivalent to 2-3 TRs later = TR6 and TR7 (indexed at 5 and 6)
    """
    onset_tr_idx = int(round(stimuli.onset) / 2)
    peak_tr_slice = image.index_img(
        fmri, onset_tr_idx + 2).get_data()
    if args.average_tr34:
        tr4 = image.index_img(fmri, onset_tr_idx + 3).get_data()
        peak_tr_slice = (peak_tr_slice + tr4) / 2
    return peak_tr_slice

def extract_roi(subject, fmri, args, stimuli, roi):
    """
    Stimuli is shown for 1s starting at 6s,
    and then in 10s intervals.

    Each TR is 2 seconds.
    Stimuli at 6s corresponds to TR4, (indexed at 3)
    Peak response is at 4-6 seconds after onset,
    equivalent to 2-3 TRs later = TR6 and TR7 (indexed at 5 and 6)
    """
    onset_tr_idx = int(round(stimuli.onset) / 2)

    roi_left = image.load_img('../roi/' + ROI_LEFT_DIR.format(subj=subject, roi=roi))
    roi_right = image.load_img('../roi/' + ROI_RIGHT_DIR.format(subj=subject, roi=roi))
    
    masker_left = NiftiMasker(mask_img=roi_left)
    masker_right = NiftiMasker(mask_img=roi_right)

    peak_tr_slice = image.index_img(
        fmri, onset_tr_idx + 2)

    extract_left_roi = masker_left.fit_transform(peak_tr_slice)
    extract_right_roi = masker_right.fit_transform(peak_tr_slice)

    if args.average_tr34:
        tr4 = image.index_img(fmri, onset_tr_idx + 3)
        extract_left_roi = (extract_left_roi + masker_left.fit_transform(tr4)) / 2
        extract_right_roi = (extract_right_roi + masker_right.fit_transform(tr4)) / 2

    return extract_left_roi[0], extract_right_roi[0]



color_df = pd.read_csv('data/image_color_percentages.csv')
events_df = pd.read_pickle('data/df_events.p')

for subject in SUBJECTS[:1]:  # 1-4
    print("Extracting data for CSI" + subject)
    data_set = {
        'slice_filename': [],
        'stimulus_filename': [],
        'primary': [],
        'secondary': [],
        'tertiary': []
    }

    if args.extract_rois:
        data_set['roi_left_PPA'] = []
        data_set['roi_left_RSC'] = []
        data_set['roi_left_OPA'] = []
        data_set['roi_left_LOC'] = []
        data_set['roi_left_EarlyVis'] = []
        data_set['roi_right_PPA'] = []
        data_set['roi_right_RSC'] = []
        data_set['roi_right_OPA'] = []
        data_set['roi_right_LOC'] = []
        data_set['roi_right_EarlyVis'] = []

    for sess in SESSIONS:  # 1 - 15
        for run in RUNS:  # 1 - 30
            print("    CSI" + subject + " Extracting Session " + sess + ", Run " + run)
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

                color_data = color_df[color_df.filename == stimuli.stim_file]

                slice_filename = '-'.join([subject, sess, run, str(i)])
                data_set['slice_filename'].append(slice_filename)
                data_set['stimulus_filename'].append(stimuli.stim_file)

                # get the dominant color for each level (prim., sec., ter.)
                for key in ['primary', 'secondary', 'tertiary']:
                    data_set[key].append(get_dominant(color_data, key))

                if args.extract_rois:
                    for roi in ['PPA', 'RSC', 'OPA', 'LOC', 'EarlyVis']:
                        extracted_roi_left, extracted_roi_right = extract_roi(subject, fmri, args, stimuli, roi)
                        data_set['roi_left_'+roi].append(extracted_roi_left)
                        data_set['roi_right_'+roi].append(extracted_roi_right)

                if args.extract_slices:
                    peak_tr_slice = extract_slice(fmri, args, stimuli)
                    np.save(file=SLICE_DIR + slice_filename,
                            arr=peak_tr_slice, allow_pickle=True)

        print('Data from session:', sess, 'extracted.')
    if args.extract_slices:
        csv_filename = EXTRACTED_DATA_DIR + 'data-subj-' + subject + '.csv'
        pd.DataFrame(data_set).to_csv(csv_filename)
        print('Data set for subject', subject, 'saved to', csv_filename)
    if args.extract_rois:
        pickle_filename = EXTRACTED_DATA_DIR + 'data-subj-' + subject + '.p'
        pd.DataFrame(data_set).to_pickle(pickle_filename)
        print('Data set for subject', subject, 'saved to', pickle_filename)

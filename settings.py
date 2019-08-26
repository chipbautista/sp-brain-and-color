
BOLD5000_DIR = '../BOLD5000-OpenNEURO/derivatives/'
FMRI_DIR = (BOLD5000_DIR +
            'fmriprep/sub-CSI{subj}/ses-{ses}/' +
            'func/sub-CSI{subj}_ses-{ses}_task-5000scenes_run-{run}' +
            '_bold_space-T1w_preproc.nii.gz')
EXTRACTED_DATA_DIR = '../extracted_data/'
SLICE_DIR = EXTRACTED_DATA_DIR + 'slices/'

SUBJECTS = ['1', '2', '3', '4']
SESSIONS = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15']
RUNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

"""
LR_DECAY
DECAY_EVERY
"""
BATCH_SIZE = 24
NUM_EPOCHS = 100
INITIAL_LR = 5e-4
DROPOUT_PROB = 0.5

DOMINANCE_THRESHOLD = {
    'primary': 60,
    'secondary': 50,
    'tertiary': 40
}

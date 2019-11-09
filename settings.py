
BOLD5000_DIR = '../BOLD5000-OpenNEURO/derivatives/'
FMRI_DIR = (BOLD5000_DIR +
            'fmriprep/sub-CSI{subj}/ses-{ses}/' +
            'func/sub-CSI{subj}_ses-{ses}_task-5000scenes_run-{run}' +
            '_bold_space-T1w_preproc.nii.gz')

ROI_LEFT_DIR = (BOLD5000_DIR +  'roi/sub-CSI{subj}/' 
				+ 'derivatives-spm-sub-CSI{subj}-sub-CSI{subj}_mask-LH{roi}.nii.gz')
ROI_RIGHT_DIR = (BOLD5000_DIR +  'roi/sub-CSI{subj}/' 
				+ 'derivatives-spm-sub-CSI{subj}-sub-CSI{subj}_mask-RH{roi}.nii.gz')

EXTRACTED_DATA_DIR = '../extracted_data/'
SLICE_DIR = EXTRACTED_DATA_DIR + 'slices/'
ROI_DIR = EXTRACTED_DATA_DIR + 'roi/'

SUBJECTS = ['1', '2', '3', '4']
SESSIONS = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15']
RUNS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

"""
LR_DECAY
DECAY_EVERY
"""
BATCH_SIZE = 16
NUM_EPOCHS = 100
INITIAL_LR = 0.0005
DROPOUT_PROB = 0.5

DOMINANCE_THRESHOLD = {
    'primary': 60,
    'secondary': 50,
    'tertiary': 40
}

MIN_3D_SHAPE = (71, 88, 67)

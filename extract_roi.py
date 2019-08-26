import os
import time
from argparse import ArgumentParser

from nilearn.input_data import NiftiMasker
from nilearn import image
import numpy as np
from settings import *

parser = ArgumentParser()
parser.add_argument('--roi', default='PPA',
					help='PPA (parahippocampal place area), RSC (retrosplenial complex), OPA (occipital place area), EV (early visual), LOC (lateral occipital complex)')
parser.add_argument('--hemisphere', default='both',
					help='left, right, or both')
args = parser.parse_args()
roi = args.roi
hemisphere = args.hemisphere

print('\nARGS: ', args)

def extract_roi():
	# for subject in SUBJECTS[:1]:  # 1-4
	# 	for sess in SESSIONS:  # 1 - 15
	# 		for run in RUNS:  # 1 - 30
	# 			try:
	# 				fmri = image.load_img(FMRI_DIR.format(
	# 					subj=subject, ses=sess, run=run))
	# 			except ValueError:
	# 				print('data for subject:', subject, 'session:', sess, 'run:',
	# 					  run, 'not found.')
	# 				continue

	fmri = image.load_img('../CS1-ses01/sub-CSI1_ses-01_task-5000scenes_run-01_bold_space-T1w_preproc.nii.gz')

	if hemisphere == 'left' or hemisphere == 'both':
		roi_left = image.load_img('../roi/' + ROI_LEFT_DIR.format(subj='1', roi=roi))
		masker_left = NiftiMasker(mask_img=roi_left)
		voxel_values_left = masker_left.fit_transform(fmri)
		if hemisphere == 'left':
			np.save("../CS{subj}_{roi}_{hemisphere}H_ses{sess}_run{run}.npy".format(subj='1', roi=roi, hemisphere=hemisphere, sess='1', run='1'), voxel_values_left)

	if hemisphere == 'right' or hemisphere == 'both':
		roi_right = image.load_img('../roi/' + ROI_RIGHT_DIR.format(subj='1', roi=roi))
		masker_right = NiftiMasker(mask_img=roi_right)
		voxel_values_right = masker_right.fit_transform(fmri)
		if hemisphere == 'right':
			return voxel_values_right

	if hemisphere == 'both':
		voxel_values_both = np.concatenate([voxel_values_left, voxel_values_right], axis=1)
		return voxel_values_both

extract_roi()
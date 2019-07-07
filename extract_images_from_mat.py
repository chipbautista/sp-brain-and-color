import scipy.io as scio
from imageio import imwrite

from settings import *

img_mat = scio.loadmat(IMG_MAT_DIR)
for i, img in enumerate(img_mat['all_imgs'][0]):
    imwrite(IMG_DIR.format(i), img)
    if (i + 1) % 500 == 0:
        print('Extracted', i + 1, 'images.')

print('Done. Images are saved in', IMG_DIR)

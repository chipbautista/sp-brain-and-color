import os
import time
from argparse import ArgumentParser

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import colormap as cm
import webcolors

STIMULI_DIR = '../BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/'
parser = ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--num-clusters', default=10)
parser.add_argument('--save-df', default=True)
parser.add_argument('--secondary', default=False)
parser.add_argument('--tertiary', default=False)
args = parser.parse_args()
print('\nARGS: ', args)


class ClusterExtractor:
    """
    Code adapted from https://github.com/Bashkeel/Pentachromacy/blob/master/A%20Pigment%20Of%20Your%20Imagination.ipynb

    Alternative method would be to use the software for Windows/Linux found here:
    http://mkweb.bcgsc.ca/color-summarizer/?home
    """
    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]


    def get_colour_name(self, requested_colour):
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = self.closest_colour(requested_colour)
            actual_name = None
        return actual_name, closest_name

    def rgb_2_hsv(self, rgb_list):
        r = rgb_list[0] / 255
        g = rgb_list[1] / 255
        b = rgb_list[2] / 255

        minRGB = min(r, g, b)
        maxRGB = max(r, g, b)
        RGBrange = maxRGB - minRGB

        # Black/White/Any Shade of Grey
        if (minRGB == maxRGB):
            Value = minRGB
            return [0, 0, Value]
        else:
            Saturation = RGBrange / maxRGB
            Value = maxRGB

            if (maxRGB == r): Hue = ((g - b) / RGBrange) % 6
            elif (maxRGB == g): Hue = ((b - r) / RGBrange) + 2
            elif (maxRGB == b): Hue = ((r - g) / RGBrange) + 4

            return [Hue * 60, Saturation, Value]

    def process_image(self, pixel_values):
        def modified_rgb2hex(RGB):
            hexs.append(cm.rgb2hex(int(RGB[0]), int(RGB[1]), int(RGB[2])))
            names.append(self.get_colour_name((int(RGB[0]), int(RGB[1]), int(RGB[2])))[1])
            # hsv.append(self.rgb_2_hsv([int(RGB[0]), int(RGB[1]), int(RGB[2])]))

        # K-means clustering
        test_K = KMeans(n_clusters=len(clusters), random_state=0).fit(pixel_values)
        labels = test_K.labels_  # label for each pixel in pixel_values

        # Extract the Cluster Centers Info and Initialize the Final DF
        Kmeans_df = test_K.cluster_centers_
        Kmeans_df = [[int(x) for x in colors] for colors in Kmeans_df]
        Kmeans_df = pd.DataFrame(Kmeans_df, columns=("R", "G", "B"))
        Kmeans_df['RGB'] = Kmeans_df.iloc[:,0:3].values.tolist()
        Kmeans_df.drop(Kmeans_df.columns[0:3], axis=1, inplace=True)

        hexs = []
        names = []
        # hsv = []
        list(map(modified_rgb2hex, Kmeans_df['RGB']))

        # make_percent = lambda x: x*100
        # for v in hsv:
        #     v[0] = int(v[0])
        #     v[1] = round((make_percent(v[1])),1)
        #     v[2] = round((make_percent(v[2])),1)

        Kmeans_df['Hex'] = hexs
        Kmeans_df['Color Name'] = names
        # Kmeans_df['HSV'] = hsv

        # COMMENTING THIS OUT BECAUSE IT TAKES TOO LONG!
        """
        # We don't need the tags anyway
        # Get the color tags (all color names in each cluster)
        tags = list(map(lambda x: self.get_colour_name(((int(x[0]), int(x[1]), int(x[2])))), pixel_values))
        tags = [colors[1] for colors in tags]

        color_tags = pd.DataFrame(pixel_values, columns=("R","G","B"))
        color_tags['Cluster Number'] = labels
        color_tags['Tags'] = tags
        Kmeans_df['Tags'] = color_tags.groupby(by=['Cluster Number']).apply(lambda x: x['Tags'].unique())
        """

            # Cluster Percentages
        Kmeans_df['Percent'] = np.unique(labels, return_counts=True)[1]
        Kmeans_df['Percent'] = round((Kmeans_df['Percent']/Kmeans_df['Percent'].sum())*100, 2)
        Kmeans_df.sort_values('Percent', ascending=False, axis=0, inplace=True)
        Kmeans_df['Percent'] = [str(x) + "%" for x in Kmeans_df['Percent']]
        # Kmeans_df = Kmeans_df[['Percent','Color Name', 'Hex', 'RGB', 'HSV', 'Tags']]
        Kmeans_df = Kmeans_df[['Percent','Color Name', 'Hex', 'RGB']]
        Kmeans_df.reset_index(drop=True, inplace=True)

        return Kmeans_df


class ClusterCategorizer:
    """
    Reference for RGB values:
    https://cdn.sparkfun.com/assets/learn_tutorials/7/1/0/TertiaryColorWheel_Chart.png
    https://www.indezine.com/products/powerpoint/learn/color/color-rgb.html
    """
    def __init__(self):
        self.primary = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }
        self.secondary = {
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255)
        }
        self.tertiary = {
            'orange': (255, 128, 0),
            'yellow-green': (128, 255, 0),  # chartreuse
            'cyan-green': (0, 255, 128),  # spring
            'cyan-blue': (0, 128, 255),  # azure
            'blue-magenta': (128, 0, 255),  # violet
            'red-magenta': (255, 0, 128)  # rose
        }

    def categorize(self, mode='secondary'):
        pass


if not args.dataset:
    args.dataset = ['COCO', 'ImageNet', 'Scene']
    print('\nNo specified data set. Will extract from:', args.dataset, '\n')
else:
    args.dataset = [args.dataset]

clusters = range(args.num_clusters)
extractor = ClusterExtractor()

print(time.strftime('%H:%M'),
      ': Now extracting color clusters. This will take A LONG WHILE. Please wait.')
for dataset in args.dataset:
    dataset_dir = STIMULI_DIR + dataset + '/'
    filenames = os.listdir(dataset_dir)
    print('Found', len(filenames), 'images in', dataset, 'folder')

    for i, filename in enumerate(filenames):
        im = Image.open(dataset_dir + filename)
        width, height = im.size
        im_a = np.array(im)
        im_mode = 'RGBA' if im_a.shape[2] == 4 else 'RGB'
        df = extractor.process_image(pixel_values=list(im.getdata()))
        if i == 0:
            master_df = df
        else:
            master_df = master_df.append(df)

        if (i + 1) % 100 == 0:
            print(time.strftime('%H:%M'), ': Finished', i + 1, 'images.')

    if args.save_df is True:
        txt_filename = dataset + '-filenames.txt'
        csv_filename = '{}-clusters.csv'.format(dataset)

        with open(txt_filename, 'w+') as f:
            for filename in filenames:
                f.write(filename + '\n')

        master_df.to_csv(csv_filename)
        print('Color cluster DataFrames of all images from',
              dataset, 'saved to', csv_filename)
        print('Filenames in the same order are saved to', txt_filename)

categorizer = ClusterCategorizer()
if args.secondary is not False:
    print('Creating data set for secondary colors...')

if args.secondary is not False:
    print('Creating data set for tertiary colors...')

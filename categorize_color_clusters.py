
import pandas as pd
from scipy.spatial.distance import euclidean


class ClusterCategorizer:
    """
    Reference for RGB values:
    https://cdn.sparkfun.com/assets/learn_tutorials/7/1/0/TertiaryColorWheel_Chart.png
    https://www.indezine.com/products/powerpoint/learn/color/color-rgb.html
    """
    def __init__(self):
        self.colors = {
            'primary': {
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                # 'black': (0, 0, 0),
                # 'white': (255, 255, 255)
            },
            'secondary': {
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                # 'black': (0, 0, 0),
                # 'white': (255, 255, 255),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255)
            },
            'tertiary': {
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                # 'black': (0, 0, 0),
                # 'white': (255, 255, 255),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255),
                'orange': (255, 128, 0),
                'yellow-green': (128, 255, 0),  # chartreuse
                'cyan-green': (0, 255, 128),  # spring
                'cyan-blue': (0, 128, 255),  # azure
                'blue-magenta': (128, 0, 255),  # violet
                'red-magenta': (255, 0, 128)  # rose
            }
        }

    def get_nearest_color(self, cluster_rgb, level):
        return min([(color, euclidean(cluster_rgb, color_rgb))
                    for (color, color_rgb) in self.colors[level].items()],
                   key=lambda x: x[1])[0]

    def summarize_image_colors(self, df, filenames):
        def get_percentage(color, level):
            return values[values[level] == color]['Percent'].apply(
                lambda x: x[:-1]).astype(float).sum()

        columns = ['filename']
        columns.extend(['primary_' + c for c in self.colors['primary']])
        columns.extend(['secondary_' + c for c in self.colors['secondary']])
        columns.extend(['tertiary_' + c for c in self.colors['tertiary']])
        percentage_df = pd.DataFrame(columns=columns)

        for i, filename in zip(range(0, len(df), 10), filenames):
            values = df.iloc[i: i + 10]
            row = [filename]
            row.extend([get_percentage(c, 'Primary Color')
                        for c in self.colors['primary']])
            row.extend([get_percentage(c, 'Secondary Color')
                        for c in self.colors['secondary']])
            row.extend([get_percentage(c, 'Tertiary Color')
                        for c in self.colors['tertiary']])
            percentage_df.loc[len(percentage_df)] = row

        percentage_df.to_csv('data/image_color_percentages.csv')
        print('Saved final data set to data/image_color_percentages.csv')


categorizer = ClusterCategorizer()
for dataset in ['COCO', 'ImageNet', 'Scene']:
    with open('data/' + dataset + '-filenames.txt', 'r') as f:
        filenames = f.read().split('\n')[:-1]
    df = pd.read_csv('data/' + dataset + '-clusters.csv')
    df['RGB'] = df['RGB'].apply(eval)
    df['Primary Color'] = df['RGB'].apply(
        categorizer.get_nearest_color, args=('primary',))
    df['Secondary Color'] = df['RGB'].apply(
        categorizer.get_nearest_color, args=('secondary',))
    df['Tertiary Color'] = df['RGB'].apply(
        categorizer.get_nearest_color, args=('tertiary',))
    if dataset == 'COCO':
        master_df = df.copy()
        master_filenames = filenames.copy()
    else:
        master_df = master_df.append(df)
        master_filenames.extend(filenames)

master_df.to_csv('data/cluster_categories.csv')
print('Saved cluster categories to data/cluster_categories.csv')

categorizer.summarize_image_colors(master_df, master_filenames)

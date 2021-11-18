import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import constraints
matplotlib.use("Agg")

TARGET = constraints.TARGET

def plot_map(X, Y, Z, individ):
    """
    for swan input need 1450 - y1, 1450 - y2

    """

    def custom_div_cmap(numcolors=2, name='custom_div_cmap',
                        mincol='black', midcol='white', maxcol='red'):

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(name=name,
                                                 colors=[mincol, maxcol],
                                                 N=numcolors)
        return cmap

    lines_X = [points[::2] for points in individ]
    lines_Y = [points[1:][::2] for points in individ]

    custom_map = custom_div_cmap(250, mincol='white', midcol='0', maxcol='black')
    plt.pcolormesh(X, Y, Z, cmap=custom_map)
    plt.colorbar()
    plt.scatter(X[TARGET[0], TARGET[1]], Y[TARGET[0], TARGET[1]], marker='s', s=20, color='black', label=Z[TARGET[0], TARGET[1]])
    left, bottom, width, height = (580, 200, 900, 900)
    rect = mpatches.Rectangle((left, bottom), width, height,
                              fill=False,
                              color="black",
                              linewidth=2,
                              label='considered domain')

    plt.gca().add_patch(rect)
    label = True
    for line_X, line_Y in zip(lines_X, lines_Y):
        if label:
            plt.plot(line_X,
                     line_Y,
                     color='black',
                     linewidth=2,
                     label='breakwater',
                     marker='o')
        else:
            plt.plot(line_X,
                     line_Y,
                     color='black',
                     linewidth=2,
                     marker='o')
        label = False

    x = np.linspace(578, 1480, 20)
    y = np.linspace(800, 200, 20)
    y1 = np.linspace(900, 354, 20)
    plt.fill_between(x, y, y1, label='prohibited area', alpha=0.3, color='red')

    plt.legend(fontsize=9)
    plt.show()


def plot_binary_mask(individ):

    lines_X = [points[::2] for points in individ]
    lines_Y = [points[1:][::2] for points in individ]

    for line_X, line_Y in zip(lines_X, lines_Y):
        plt.plot(line_X,
                 line_Y,
                 color='black',
                 linewidth=3.5,
                 marker='o')

    plt.xlim(560, 1500)
    plt.ylim(180, 1120)
    plt.axis('off')

    plt.savefig('temp_images/0.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

    image = Image.open('temp_images/0.png').convert('L')
    image = image.resize((224, 224))
    data = np.asarray(image) / 255.0
    image.close()

    return data
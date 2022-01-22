import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import constraints

matplotlib.use("Agg")
TARGET = constraints.TARGET
x = constraints.x
y = constraints.y
X, Y = np.meshgrid(x, y)
Z = constraints.Z


def plot_map(X, Y, Z, individ, type, j, i):
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

    Z_new = []
    for z in Z:
        z_new = []
        for k in z:
            if k <= 0:
                z_new.append(0)
            else:
                z_new.append(k)
        Z_new.append(z_new)
    Z_new = np.array(Z_new)

    plt.pcolormesh(X, Y, Z_new, cmap=custom_map, shading='auto')
    plt.colorbar()
    plt.scatter(X[TARGET[0][0], TARGET[0][1]], Y[TARGET[0][0], TARGET[0][1]], marker='s', s=20, color='green')
    plt.scatter(X[TARGET[1][0], TARGET[1][1]], Y[TARGET[1][0], TARGET[1][1]], marker='s', s=20, color='green',
                label='WH=' + str(
                    round((Z_new[TARGET[0][0], TARGET[0][1]] + Z_new[TARGET[1][0], TARGET[1][1]]) / 2, 3)))

    plt.plot([1000, 700, 800], [100, 600, 800], color='black', linewidth=4)
    plt.plot([1900, 1750], [540, 1000], color='black', linewidth=4)

    label = True
    for line_X, line_Y in zip(lines_X, lines_Y):
        if label:
            plt.plot(line_X,
                     line_Y,
                     color='blue',
                     linewidth=2,
                     label='breakwater',
                     marker='o')
        else:
            plt.plot(line_X,
                     line_Y,
                     color='blue',
                     linewidth=2,
                     marker='o')
        label = False

    plt.axis('off')

    plt.xlim(0, 2075)
    plt.ylim(0, 1450)

    plt.legend(fontsize=9)

    if type == 'cnn':
        plt.savefig('exp_results/HV/3k_data_30/CNN_images/' + str(j + 1) + '/' + str(i) + '.pdf', bbox_inches='tight',
                    pad_inches=0)
        plt.close('all')
    elif type == 'swan':
        plt.savefig('exp_results/HV/3k_data_30/SWAN_images/' + str(j + 1) + '/' + str(i) + '.pdf', bbox_inches='tight',
                    pad_inches=0)
        plt.close('all')


def example_create(Z, individ, i, label=False):
    fig = plt.figure(figsize=(4, 4))

    def custom_div_cmap(numcolors=2, name='custom_div_cmap',
                        mincol='white', maxcol='black'):
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(name=name,
                                                 colors=[mincol, maxcol],
                                                 N=numcolors)
        return cmap

    lines_X = [points[::2] for points in individ]
    lines_Y = [points[1:][::2] for points in individ]

    if label:
        custom_map = custom_div_cmap(250)
        plt.pcolormesh(X, Y, Z, cmap=custom_map, shading='auto')
    else:
        Z_max = 1.2
        Z_train = []
        for z_x in Z:
            z_new = []
            for z_y in z_x:
                if z_y == 0:
                    z_new.append(0)
                else:
                    z_new.append(np.random.normal(loc=Z_max / 2, scale=0.1))
            Z_train.append(z_new)
        Z_train = np.array(Z_train)
        custom_map = custom_div_cmap(250)
        plt.pcolormesh(X, Y, Z_train, cmap=custom_map, shading='auto')

    plt.plot([1000, 700, 800], [100, 600, 800], color='black', linewidth=4)
    plt.plot([1900, 1750], [540, 1000], color='black', linewidth=4)

    plt.axis('off')

    plt.xlim(0, 2075)
    plt.ylim(0, 1450)

    if label:
        for lines_X, lines_Y in zip(lines_X, lines_Y):
            plt.plot(lines_X,
                     lines_Y,
                     color='black',
                     linewidth=3.5,
                     marker='o')
        plt.savefig('dataset/labels/' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        for lines_X, lines_Y in zip(lines_X, lines_Y):
            plt.plot(lines_X,
                     lines_Y,
                     color='black',
                     linewidth=3.5,
                     marker='o')
        plt.savefig('dataset/targets/' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_binary_mask(individ):
    fig = plt.figure(figsize=(4, 4))

    def custom_div_cmap(numcolors=2, name='custom_div_cmap',
                        mincol='white', maxcol='black'):
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(name=name,
                                                 colors=[mincol, maxcol],
                                                 N=numcolors)
        return cmap

    lines_X = [points[::2] for points in individ]
    lines_Y = [points[1:][::2] for points in individ]

    for line_X, line_Y in zip(lines_X, lines_Y):
        plt.plot(line_X,
                 line_Y,
                 color='black',
                 linewidth=3.5,
                 marker='o')

    Z_max = 1.2
    Z_train = []
    Z = np.loadtxt('swan/r/hs47dd8b1c0d4447478fec6f956c7e32d9.d')
    for z_x in Z:
        z_new = []
        for z_y in z_x:
            if z_y <= 0:
                z_new.append(0)
            else:
                z_new.append(np.random.normal(loc=Z_max / 2, scale=0.1))
        Z_train.append(z_new)
    Z_train = np.array(Z_train)
    custom_map = custom_div_cmap(250)
    plt.pcolormesh(X, Y, Z_train, cmap=custom_map, shading='auto')

    plt.plot([1000, 700, 800], [100, 600, 800], color='black', linewidth=4)
    plt.plot([1900, 1750], [540, 1000], color='black', linewidth=4)

    plt.axis('off')

    plt.xlim(0, 2075)
    plt.ylim(0, 1450)

    plt.axis('off')

    plt.savefig('models/temp_images/0.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

    image = Image.open('models/temp_images/0.png').convert('L')
    image = image.resize((224, 224))
    data = np.asarray(image) / 255.0
    image.close()

    return data

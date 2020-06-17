import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from scipy import signal

h = 1  # assume that the distance between two pixels is 1


def image_histogram(image_matrix, density=False):
    """
    Parameters
    ----------
    image_matrix
    density

    Returns
    -------

    """
    flat = np.ndarray.flatten(image_matrix)
    hist = np.histogram(flat, bins=256, density=density)

    return hist


def gradient_histogram(image_matrix, density=False):
    """
    We apply the convolutions with the respective filters,
    and then - calculate the norm of the discrete gradient.
    That gives the result as in the lecture notes.

    Parameters
    ----------
    image_matrix
    density

    Returns
    -------

    """
    filter_x = np.array([[-1, 1], [0, 0]])
    filter_y = np.array([[-1, 0], [1, 0]])
    xx = signal.convolve2d(image_matrix, filter_x, boundary='symm', mode='same')
    yy = signal.convolve2d(image_matrix, filter_y, boundary='symm', mode='same')
    xx_, yy_ = xx[:-1, 1:] / h, yy[1:, :-1] / h
    v_norm = np.sqrt(xx_ ** 2 + yy_ ** 2)

    flat = np.ndarray.flatten(v_norm)
    counts, edges = np.histogram(flat, bins=300, density=density)
    edges = np.array(list(filter(lambda x: x <= 20, edges)))
    counts = counts[:len(edges)-1]

    return counts, edges


def hess_histogram(image_matrix, density=False):
    """
    We apply the convolution with the respective filter
    to calculate the discrete Laplacian.
    Parameters
    ----------
    image_matrix
    density

    Returns
    -------

    """
    filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    xx = signal.convolve2d(image_matrix, filter, boundary='symm')
    v_norm = np.absolute(xx) / (h**2)
    flat = np.ndarray.flatten(v_norm)
    hist = np.histogram(flat, bins=300, density=density)

    return hist


def images_plot(images, histogram_t=''):

    for image in images:
        imshow(image['matrix'])
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        bin_edges, counts = image[histogram_t + 'histogram'][1], image[histogram_t + 'histogram'][0]
        plt.plot(0.5 * (bin_edges[1:] + bin_edges[:-1]), counts)
        plt.subplot(122)
        bin_edges, counts = image[histogram_t + 'probability'][1], image[histogram_t + 'probability'][0]
        plt.plot(0.5 * (bin_edges[1:] + bin_edges[:-1]), counts)
        plt.suptitle(image['title'])
        plt.subplots_adjust(wspace=0.25)
        plt.show()


def caruana_algorithm(xx, yy):

    n = xx.size
    a = np.array([[n, xx.sum(), (xx**2).sum()],
                  [xx.sum(), (xx**2).sum(), (xx**3).sum()],
                  [(xx**2).sum(), (xx**3).sum(), (xx**4).sum()]])

    b = np.array([np.log(yy).sum(),
                  (xx*np.log(yy)).sum(),
                  ((xx**2)*np.log(yy)).sum()])

    aa, bb, cc = np.linalg.solve(a, b)
    #print(aa, bb, cc)
    a_ = np.exp(aa - bb**2 / (4*cc))
    mu = -bb / (2*cc)
    sigma = np.sqrt(-1 / (2*cc)) if cc < 0 else np.sqrt(1 / (2*cc))
    #print('out ', a_, mu, sigma)
    return lambda x: a_ * np.exp(-(x - mu)**2 / (2*sigma**2))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = io.imread(image_path)

    ### YOUR CODE HERE
    # Use skimage io.imread
    
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = 0.5*np.square(image)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = np.sum(image, axis = 2)/3
    ### END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified
    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    ### YOUR CODE HERE
    color_d = {'r': 0, 'g': 1, 'b': 2}
    out = image.copy()
    out[:,:,color_d[channel.lower()]] = 0
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    color_d = {'l': 0, 'a': 1, 'b': 2}
    lab = color.rgb2lab(image)
    lab = (lab + np.abs(np.min(lab)))
    lab = lab / np.max(lab)
    lab[:,:,color_d[channel.lower()]] =  0
    out = lab

    return out

def hsv_decomposition(image, channel):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    color_d = {'h': 0, 's': 1, 'v': 2}
    hsv[:, :, color_d[channel.lower()]] = 0
    out = hsv
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    image1 = rgb_decomposition(image1, channel1)
    image2 = rgb_decomposition(image2, channel2)
    out = np.zeros_like(image1)
    middle = image1.shape[1]//2
    print(middle)
    out[:, 0:middle,:] = image1[:, 0:middle,:]
    out[:, middle:,:] = image2[:, middle / 2:,:]


    return out

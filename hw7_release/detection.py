import numpy as np
from skimage import feature,data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math

def hog_feature(image, pixel_per_cell = 8):
    ''' 
    Compute hog feature for a given image.
    
    Hint: use the hog function provided by skimage
    
    Args:
        image: an image with object that we want to detect
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor
        
    Returns:
        score: a vector of hog representation
        hogImage: an image representation of hog provided by skimage
    '''
    ### YOUR CODE HERE
    hogFeature, hogImage = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell), visualize=True)
    ### END YOUR CODE
    return (hogFeature, hogImage)

def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):
    ''' A sliding window that checks each different location in the image, 
        and finds which location has the highest hog score. The hog score is computed
        as the dot product between hog feature of the sliding window and the hog feature
        of the template. It generates a response map where each location of the
        response map is a corresponding score. And you will need to resize the response map
        so that it has the same shape as the image.
    
    Args:
        image: an np array of size (h,w)
        base_score: hog representation of the object you want to find, an array of size (m,)
        stepSize: an int of the step size to move the window
        windowSize: a pair of ints that is the height and width of the window
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        response_map: an np array of size (h,w)
    '''
    # slide a window across the image
    (max_score, maxr, maxc) = (0,0,0)
    winH, winW = windowSize
    H,W = image.shape
    pad_image = np.lib.pad(image, ((winH//2,winH-winH//2),(winW//2, winW-winW//2)), mode='constant')
    response_map = np.zeros((H//stepSize+1, W//stepSize+1))
#     print(pad_image.shape,windowSize)
    ### YOUR CODE HERE
    for i in range(0, H+1, stepSize):
        for j in range(0, W+1, stepSize):
            
#             print(i,j,winH,winW,H,W,pad_image[i:i+winH, j:j+winW].shape,pad_image.shape)
            hogFeature = feature.hog(pad_image[i:i+winH, j:j+winW], \
                                                 pixels_per_cell = (pixel_per_cell,pixel_per_cell))
            
            score = hogFeature.dot(base_score)
            response_map[i//stepSize,j//stepSize] = score
            
            if score > max_score:
                
                max_score, maxr, maxc = score, i-winH//2, j-winW//2
#                 print(i,j,maxr,maxc)

    ### END YOUR CODE
    
    
    return (max_score, maxr, maxc, response_map)


def pyramid(image, scale=0.9, minSize=(200, 100)):
    '''
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until one of the height or
    width reaches the minimum limit. In the ith iteration, 
    the image is resized to scale^i of the original image.
    
    Args:
        image: np array of (h,w), an image to scale
        scale: float of how much to rescale the image each time
        minSize: pair of ints showing the minimum height and width
        
    Returns:
        images: a list containing pair of 
            (the current scale of the image, resized image)
    '''
    # yield the original image
    images = []
    current_scale = 1.0
    images.append((current_scale, image))
    
    # keep looping over the pyramid
    ### YOUR CODE HERE
    currH, currW = image.shape
    current_scale *= scale
    while currH > minSize[0] and currW >minSize[1]:
        image = rescale(image,current_scale)
        images.append((current_scale,image))
        current_scale *= scale
        currH, currW = image.shape
#         print(current_scale,currH, currW)
    ### END YOUR CODE
    return images

def pyramid_score(image,base_score, shape, stepSize=20, scale = 0.9, pixel_per_cell = 8):
    '''
    Calculate the maximum score found in the image pyramid using sliding window.
    
    Args:
        image: np array of (h,w)
        base_score: the hog representation of the object you want to detect
        shape: shape of window you want to use for the sliding_window
        
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        max_scale: float of scale when the max_score is found
        max_response_map: np array of the response map when max_score is found
    '''
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map =np.zeros(image.shape)
    images = pyramid(image, scale)
    ### YOUR CODE HERE
    for scale, image in images:
        score, r, c, response_map = sliding_window(image, base_score, \
                                                               stepSize, shape, pixel_per_cell=pixel_per_cell)
        if score > max_score:
            max_score, maxr, maxc, max_response_map = score, r, c, response_map
            max_scale = scale
    ### END YOUR CODE
    return max_score, maxr, maxc, max_scale, max_response_map


def compute_displacement(part_centers, face_shape):
    ''' Calculate the mu and sigma for each part. d is the array 
        where each row is the main center (face center) minus the 
        part center. Since in our dataset, the face is the full
        image, face center could be computed by finding the center
        of the image. Vector mu is computed by taking an average from
        the rows of d. And sigma is the standard deviation among 
        among the rows. Note that the heatmap pixels will be shifted 
        by an int, so mu is an int vector.
    
    Args:
        part_centers: np array of shape (n,2) containing centers 
            of one part in each image
        face_shape: (h,w) that indicates the shape of a face
    Returns:
        mu: (1,2) vector
        sigma: (1,2) vector
        
    '''
    d = np.zeros((part_centers.shape[0],2))
    ### YOUR CODE HERE
    face_center = np.array([(face_shape[0]-1)/2, (face_shape[1]-1)/2])

    d = face_center - part_centers
    mu = np.mean(d, axis=0)
    mu = mu.astype('int64')
    sigma = np.std(d, axis=0)
        
    ### END YOUR CODE
    return mu, sigma
        
def shift_heatmap(heatmap, mu):
    '''First normalize the heatmap to make sure that all the values 
        are not larger than 1.
        Then shift the heatmap based on the vector mu.

        Args:
            heatmap: np array of (h,w)
            mu: vector array of (1,2)
        Returns:
            new_heatmap: np array of (h,w)
    '''
    ### YOUR CODE HERE
    H,W = heatmap.shape
    newHeatmap = heatmap/np.sqrt(np.sum(heatmap**2))
    y_shift = int(mu[0])
    x_shift = int(mu[1])
    newHeatmap = np.roll(newHeatmap,y_shift,axis=0)
    new_heatmap = np.roll(newHeatmap,x_shift,axis=1)
    ### END YOUR CODE
    return new_heatmap
    

def gaussian_heatmap(heatmap_face, heatmaps, sigmas):
    '''
    Apply gaussian filter with the given sigmas to the corresponding heatmap.
    Then add the filtered heatmaps together with the face heatmap.
    Find the index where the maximum value in the heatmap is found. 
    
    Hint: use gaussian function provided by skimage
    
    Args:
        image: np array of (h,w)
        sigma: sigma for the gaussian filter
    Return:
        new_image: an image np array of (h,w) after gaussian convoluted
    '''
    ### YOUR CODE HERE
    heatmap = heatmap_face.copy()
    for i,heatMap in enumerate(heatmaps):
        gaussion_map = gaussian(heatmap,sigma=sigmas[i])
        heatmap+=gaussion_map

    index=np.argmax(heatmap)
    r, c = np.unravel_index(index,heatmap_face.shape)
    ### END YOUR CODE
    return heatmap, r , c
            
      
def detect_multiple(image, response_map):
    '''
    Extra credit
    '''
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return detected_faces

            

    
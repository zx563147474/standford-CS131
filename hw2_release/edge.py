import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    kernel = kernel[::-1,::-1]
    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = (kernel*padded[i:i+Hk, j:j+Wk]).sum()
    ### YOUR CODE HERE
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))
    k = (size - 1)//2

    ### YOUR CODE HERE
    for i in range(size):
        for j in range(size):
            kernel[i,j] = 1/(2*np.pi*sigma**2)*np.exp(-((i-k)**2+(j-k)**2)/(2*sigma**2))
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = 0.5*np.array([[1, 0, -1]])
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = 0.5*np.array([[1, 0, -1]]).T
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    Gx, Gy = partial_x(img), partial_y(img)
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.rad2deg(np.arctan2(Gy,Gx))
    theta[theta == 360] = 0
    theta[theta < 0] = theta[theta <0] + 360
    ### YOUR CODE HERE
    
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))
    # Round the gradient direction to the nearest 45 degrees
    theta = (theta + 22.5) // 45 *45
    pad_width = ((1,1), (1,1))
#     theta = theta % 360.0
    G1 = np.pad(G, pad_width, mode='constant')

    ### BEGIN YOUR CODE
    for i in range(H):
        for j in range(W):
            
            #method 1
            alpha = np.deg2rad(theta[i, j])
            p1 = G1[i+1-int(np.round(np.sin(alpha))), j+1-int(np.round(np.cos(alpha)))]
            p2 = G1[i+1+int(np.round(np.sin(alpha))), j+1+int(np.round(np.cos(alpha)))]
            if G[i, j] >= p1 and G[i, j] >= p2:
                out[i, j] = G[i, j]
            
            #method2
#             if theta[i,j] == 0 or theta[i,j] == 180:
#                 neighbors = G1[i+1, j],G1[i+1, j+2]
# #                 print(theta[i,j],neighbors, [i,j])
#             elif theta[i,j] == 45 or theta[i,j] == 225:
#                 neighbors = G1[i, j],G1[i+2, j+2]
# #                 print(theta[i-1,j-1],neighbors,[i,j])
#             elif theta[i,j] == 90 or theta[i,j] == 270:
#                 neighbors = G1[i, j+1],G1[i+2, j+1]
# #                 print(theta[i-1,j-1],neighbors,[i,j])
#             elif theta[i,j] == 135 or theta[i,j] == 315:
#                 neighbors = G1[i+2, j-2],G1[i, j]
# #                 print(theta[i-1,j-1],neighbors,[i,j])
#             else:
#                 raise RuntimeError('argument theta should be in range [0, 360)')
#             if G[i,j]>= max(neighbors):
#                 out[i,j] = G[i,j]
    ### END YOUR CODE
    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    strong_edges = img>high
    weak_edges = (img>low) & (img<high)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    for i,j in indices:
        neighbors = get_neighbors(i, j, H, W)
        edges[i,j] = 1
        viewed = set()
        while neighbors:
            y, x = neighbors.pop(0)
            if (y,x) not in viewed and weak_edges[y,x] == 1:
                edges[y,x] = 1
                neighbors.extend(get_neighbors(y, x, H, W))
            viewed.add((y,x))
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)
    img_filtered = conv(img,kernel)
    img_gradient, theta = gradient(img_filtered)
    img_NMS = non_maximum_suppression(img_gradient, theta)
    strong_edges, weak_edges = double_thresholding(img_NMS, high, low)
    edge = link_edges(strong_edges, weak_edges)
    
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90, 90))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i, j in zip(ys, xs):
        for angle in range(num_thetas):
            r = j*cos_t[angle] + i*sin_t[angle]
             # r has value -Rmax to Rmax, but need map r to its idx 0 : 2*max
            accumulator[int(r + diag_len), angle] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas

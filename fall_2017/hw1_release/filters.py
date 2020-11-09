import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    kernel = kernel[::-1,::-1]
    padW = Wk //2
    padH = Hk //2
    img_new = np.zeros((Hi+2*padH, Wi+2*padW))
    img_new[padH:Hi+padH, padW:Wi+padW] = image
    for i in range(Hi):
        for j in range(Wi):
            value = 0.0
            for p in range(i,i+Hk):
                for q in range(j,j+Wk):
                    value += img_new[p,q] * kernel[p-i,q-j]
            out[i,j] = value
    ### END YOUR CODE
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    
    out = np.zeros((H+2*pad_height,W+2*pad_width))
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image
    ### YOUR CODE HERE
    
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

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
    padW = Wk //2
    padH = Hk //2
    ### YOUR CODE HERE
    img_new = zero_pad(image, padH, padW)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = (kernel*img_new[i:i+Hk, j:j+Wk]).sum()
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
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
    padW = Wk //2
    padH = Hk //2
    ### YOUR CODE HERE
    img_new = zero_pad(image, padH, padW)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = (kernel*img_new[i:i+Hk, j:j+Wk]).sum()
    ### YOUR CODE HERE
    
    
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    
    out = None
    ### YOUR CODE HERE
    g = g[::-1,::-1]
    out = conv_fast(f,g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = g[::-1,::-1]
    newG = g-g.mean()
    out = conv_fast(f,newG)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    newF = (f-f.mean())/f.std()
    newG = (g-g.mean())/g.std()
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    padh = Hk // 2
    padw = Wk // 2
    img_new = zero_pad(f, padh, padw)
    for i in range(Hi):
        for j in range(Wi):
            img_tmp = img_new[i:i+Hk, j:j+Wk]
            img_norm = (img_tmp - img_tmp.mean()) / img_tmp.std()
            out[i, j] = (img_norm * newG).sum()
#     ### YOUR CODE HERE

#     out = conv_fast(newF,newG)
#     ### END YOUR CODE

    return out

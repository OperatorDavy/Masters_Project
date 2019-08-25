'''
Functions used during data processing.
Note that complex data was used and in some instances phase data was 
faked from magnitude images by setting the imaginary parts to zero.
'''

def complex_to_real(x, axis=1):
    """
    Converts complex data to a data with 2 channels with real data
    x: input
    axis: representst the real and complex channel.
    """
    shape = x.shape
    if x.dtype == np.complex64:
        dtype = np.float32
    else:
        dtype = np.float64
    
    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape+(2,))
    n = x.ndim
    if axis < 0:
        axis = n + axis
    if axis < n:
        x = x.transpose(tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)]))
    return x


def to_tensor_format(x, fake_imaginary=False):
    """"
    Takes the data as of shape (N [,T], x, y), T is the sequence times
    reshapse to (n, N_channels, x, t, T)
    """
    if x.ndim == 4:
        x = np.transpose(x, (0,2,3,1))

    if fake_imaginary:
        # Add zero as the imaginary parts
        x = x*(1+0j)

    x = complex_to_real(x)
    return x

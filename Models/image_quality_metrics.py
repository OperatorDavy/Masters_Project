import numpy as np

def mse(x, y):
    """
    Mean Squared Error (MSE)
    """
    return np.mean(np.abs(x - y)**2)

def psnr(x, y):
    """
    Peak signal to noise ratio (PSNR)
    """

    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    psnr =  20 * np.log10(max_intensity) - 10 * np.log10(mse)
    return psnr

# For SSIM I used tensorflow: tf.SSIM

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.color import rgb2lab, deltaE_ciede2000

class ImageMetrics:
    def __init__(self):
        pass
    
    def calculate_metrics(self, gt, r, transpose=True):

        if gt.shape != r.shape:
            raise ValueError("Input images must have the same dimensions.")
        
        if transpose:
            gt = np.transpose(gt, (1, 2, 0))
            r = np.transpose(r, (1, 2, 0))
        
        lab1 = rgb2lab(gt)
        lab2 = rgb2lab(r)
        
        psnr = peak_signal_noise_ratio(gt, r)
        ssim = structural_similarity(gt, r, data_range=1, multichannel=True, channel_axis=-1,
                                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        mse = mean_squared_error(gt, r)
        
        ciede2000 = np.mean(deltaE_ciede2000(lab1, lab2))
        
        return {
            'PSNR': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'CIEDE2000': ciede2000
        }


if __name__ == "__main__":

    gt = np.random.rand(3, 256, 256)
    r = np.random.rand(3, 256, 256)
    
    img_metrics = ImageMetrics()
    metrics = img_metrics.calculate_metrics(gt, r)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
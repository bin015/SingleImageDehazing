import os
import cv2
import numpy as np
from metrics import ImageMetrics

img_metrics = ImageMetrics()

def read_images_from_folders(folder1, folder2):
    
    mses, psnrs, ssims, ciede2000s = [], [], [], []

    for filename in os.listdir(folder1):
        if filename.endswith('.png'):
            base_name = filename.split('_')[0]
            corresponding_filename = f"{base_name}.png"
            
            # Read image from folder1
            img1_path = os.path.join(folder1, filename)
            img1 = cv2.imread(img1_path)
            print(img1.shape)
            
            # Read corresponding image from folder2
            img2_path = os.path.join(folder2, corresponding_filename)
            img2 = cv2.imread(img2_path)
            print(img2.shape)

            metrics = img_metrics.calculate_metrics(img1, img2, transpose=False)

            mses.append(metrics['MSE'])
            psnrs.append(metrics['PSNR'])
            ssims.append(metrics['SSIM'])
            ciede2000s.append(metrics['CIEDE2000'])
            print('predicting for {}  MSE {:.4f}, PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}'
                    .format(base_name, metrics['MSE'], metrics['PSNR'], metrics['SSIM'], metrics['CIEDE2000']))
            
    log = f"MSE: {np.mean(mses):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}"
    print(log)
                

# Example usage
folder2 = '/remote-home/lijuncheng/project/course/DIP/SingleImageDehazing/data/RESIDE/SOTS/nyuhaze500/gt'
folder1 = '/remote-home/lijuncheng/project/course/DIP/GCANet/SOTS_indoor_results/'

read_images_from_folders(folder1, folder2)

# # Convert images to numpy arrays if necessary (though they are already numpy arrays by default with OpenCV)
# images1_np = {k: np.array(v) for k, v in images1.items()}
# images2_np = {k: np.array(v) for k, v in images2.items()}

# # Print the keys (filenames) to verify
# print("Images in folder1:", images1_np.keys())
# print("Images in folder2:", images2_np.keys())

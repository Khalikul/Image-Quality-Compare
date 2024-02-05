import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load images
img1 = cv2.imread('photo-baseline.png')
img2 = cv2.imread('photo-level-1.png')

# Convert images to grayscale (if needed)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calculate PSNR
psnr_value = psnr(img1, img2)

# Check for divide by zero
if np.isinf(psnr_value):
    print("PSNR cannot be calculated as the images are identical.")
else:
    print(f'PSNR: {psnr_value:.2f}')

# Calculate SSIM
ssim_value, _ = ssim(img1_gray, img2_gray, full=True)

print(f'SSIM: {ssim_value:.4f}')

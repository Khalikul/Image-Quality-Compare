# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# import matplotlib.pyplot as plt
#
#
# def calculate_psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
#     return psnr_value
#
#
# def calculate_ssim(img1, img2):
#     ssim_value, ssim_map = ssim(img1, img2, full=True)
#     return ssim_value, ssim_map
#
#
# def generate_heatmap(image1, image2, metric):
#     if metric == 'psnr':
#         # Calculate PSNR for each pixel
#         difference_map = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
#         heatmap = 20 * np.log10(255.0 / np.sqrt(difference_map + 1e-10))  # Small epsilon to avoid log(0)
#     elif metric == 'ssim':
#         # Calculate SSIM
#         _, heatmap = calculate_ssim(image1, image2)
#     else:
#         raise ValueError("Unsupported metric. Choose 'psnr' or 'ssim'.")
#
#     return heatmap
#
#
# def main(image1_path, image2_path, metric):
#     # Load images
#     image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
#
#     if image1 is None or image2 is None:
#         print("Error: Unable to load one or both images.")
#         return
#
#     if image1.shape != image2.shape:
#         print("Error: Images must have the same dimensions.")
#         return
#
#     heatmap = generate_heatmap(image1, image2, metric)
#
#     # Display the heatmap
#     plt.figure(figsize=(10, 8))
#     plt.title(f"Heatmap based on {metric.upper()}")
#     plt.imshow(heatmap, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.show()
#
#
# # User inputs
# image1_path = 'fullscreen_pdfword.png'  # Replace with the path to your first image
# image2_path = 'citrix_pdfword3.png'  # Replace with the path to your second image
# metric = 'psnr'  # Choose 'psnr' or 'ssim'
#
# main(image1_path, image2_path, metric)
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))

    return psnr_value


def calculate_ssim(img1, img2):
    ssim_value, ssim_map = ssim(img1, img2, full=True)
    return ssim_value, ssim_map


def normalize_heatmap(heatmap):
    norm_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    return norm_heatmap


def generate_heatmap(image1, image2, metric):
    if metric == 'psnr':
        # Calculate PSNR for each pixel
        difference_map = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
        heatmap = 20 * np.log10(255.0 / np.sqrt(difference_map + 1e-10))  # Small epsilon to avoid log(0)
        heatmap = normalize_heatmap(heatmap)
    elif metric == 'ssim':
        # Calculate SSIM
        _, heatmap = calculate_ssim(image1, image2)
        heatmap = (heatmap * 255).astype(np.uint8)  # Scale SSIM map to 0-255
    else:
        raise ValueError("Unsupported metric. Choose 'psnr' or 'ssim'.")

    return heatmap


def HeatMap(image1_path, image2_path, metric, color_map='hot'):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: Unable to load one or both images.")
        return

    if image1.shape != image2.shape:
        print("Error: Images must have the same dimensions.")
        return

    heatmap = generate_heatmap(image1, image2, metric)

    # Display the heatmap
    plt.figure(figsize=(10, 8))
    # plt.title(f"Heatmap based on {metric.upper()} {"Baseline"}  {"VS"}  {"HZE Lossless"}")
    plt.imshow(heatmap, cmap=color_map, interpolation='nearest')
    # plt.colorbar()
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


# User inputs
image1_path = 'fullscreen_Xray2.png'
image2_path = 'HZE_XRAY2_11.png'
metric = 'psnr'  # Choose 'psnr' or 'ssim'
color_map = 'hot'  # Choose a color map: 'hot', 'viridis', 'plasma', etc.

HeatMap(image1_path, image2_path, metric, color_map)

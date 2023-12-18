import numpy as np
from skimage import filters, measure, io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from math import sqrt, pi
from skimage.transform import hough_circle

# Load image
image = io.imread('image.jpg', as_gray=True)
image = np.fliplr(image)

# Remove noise
blurred_image = filters.gaussian(image, sigma=5)

# Subtract background 
background_subtracted = image - blurred_image

# Convert to binary 
threshold_value = filters.threshold_otsu(background_subtracted)
binary_image = background_subtracted > threshold_value

#Adaptive tresholding

binary_image1 = filters.threshold_local(background_subtracted, block_size=15, method='gaussian', offset=20)
binary_image2 = ndimage.median_filter(binary_image, size=(3,3))
binary_image3 = ndimage.binary_erosion(binary_image)
binary_image4 = ndimage.binary_dilation(binary_image)

# Label connected components in the binary image
labeled_image = measure.label(binary_image)

# Coin size Excel
coin_size = pd.read_excel('Euro.xlsx')

# Calculate pixel radiuses for each coin
pixel_count = {}
for index, row in coin_size.iterrows():
    radius_min = round(sqrt(row['pixels -200']/pi ))
    radius_max = round(sqrt(row['pixels +200']/pi ))
    if (row['cents'] == 0.2):
        pixel_count[row['cents']] = np.arange(radius_min, radius_max)
    else:
        pixel_count[row['cents']] = np.arange(radius_min, radius_max+1)

total_value = 0
coin_count = 0

for i in pixel_count:
    print(i, " eurs " ,len(hough_circle(binary_image4,pixel_count[i])))
    total_value += float(i)*len(pixel_count[i])
    coin_count += len(pixel_count[i])

print("Total Value: €", total_value)
print("Coin Count:", coin_count)

# Display images 
plt.subplot(2, 3, 1)
plt.imshow(binary_image4, cmap='gray' )
plt.title('dilation')

plt.subplot(2, 3, 2)
plt.imshow(binary_image3, cmap='gray')
plt.title('erosion')

plt.subplot(2, 3, 3)
plt.imshow(binary_image2, cmap='gray')
plt.title('median')

plt.subplot(2, 3, 4)
plt.imshow(binary_image1, cmap='gray')
plt.title('adaptive tresholding')

plt.subplot(2, 3, 5)
plt.imshow(binary_image, cmap='gray')
plt.title('binary')

plt.subplot(2, 3, 6)
plt.imshow(labeled_image, cmap='gray')
plt.title('labeled')

plt.tight_layout()
plt.show()

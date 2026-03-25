from PIL import Image
import numpy as np

# Create a 512x512 RGB image where all pixel values are 255
width, height = 512, 512
all_ones = np.full((height, width, 3), 255, dtype=np.uint8)
img = Image.fromarray(all_ones)
img.save('all_ones_512x512.png')
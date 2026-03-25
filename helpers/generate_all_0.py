from PIL import Image
import numpy as np

# Create a 512x512 RGB image where all pixel values are 0
width, height = 512, 512
all_zeros = np.zeros((height, width, 3), dtype=np.uint8)
img = Image.fromarray(all_zeros)
img.save('all_zeros_512x512.png')
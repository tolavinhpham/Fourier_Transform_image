import numpy as np
from skimage.transform import radon

# Create a black image with a white circle in the middle
image = np.zeros((512, 512))
rr, cc = np.ogrid[:512, :512]
circle = (rr - 256) ** 2 + (cc - 256) ** 2 < 100 ** 2
image[circle] = 1

# Perform the Radon transform
theta = np.linspace(0., 360., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

# Display the original image and the sinogram
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
ax1.imshow(image, cmap=plt.cm.Greys_r)
ax1.set_title('Original')
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 360, 0, sinogram.shape[0]), aspect='auto')
ax2.set_title('Radon transform\n(Sinogram)')
plt.show()
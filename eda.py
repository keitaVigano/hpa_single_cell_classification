import numpy as np
from PIL import Image

path = "kaggle/input/competitions/hpa-single-cell-image-classification "
red   = np.array(Image.open(f"{path}/test/ID_red.png"))
green = np.array(Image.open(f"{path}/test/ID_green.png"))
blue  = np.array(Image.open(f"{path}/test/ID_blue.png"))

# Normalizza da 16bit (0-65535) a 8bit (0-255)
def norm(channel):
    return (channel / 65535 * 255).astype(np.uint8)

rgb = np.stack([norm(red), norm(green), norm(blue)], axis=-1)
# shape: (2000, 2000, 3)

plt.imshow(rgb)
plt.show()
import os
import imageio.v3 as imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# baca gambar (selalu cari di folder yang sama dengan script)
folder = os.path.dirname(os.path.abspath(__file__))
img = imageio.imread(os.path.join(folder, "kucing.jpg"))

# ubah ke grayscale
gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

# ekualisasi histogram
def hist_equal(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normal = (cdf / cdf.max()) * 255
    result = np.interp(image.flatten(), bins[:-1], cdf_normal)
    return result.reshape(image.shape).astype(np.uint8)

equalized = hist_equal(gray)

# filter batas (8 tetangga) pakai scipy.ndimage
fp = np.ones((3, 3), dtype=bool)
fp[1, 1] = False
mn = ndimage.minimum_filter(gray, footprint=fp)
mx = ndimage.maximum_filter(gray, footprint=fp)
filtered = np.clip(gray, mn, mx)

# tampilkan hasil
plt.figure(figsize=(14, 9))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Gambar Asli")

plt.subplot(2, 3, 2)
plt.imshow(equalized, cmap="gray", vmin=0, vmax=255)
plt.title("Ekualisasi Histogram")

plt.subplot(2, 3, 3)
plt.imshow(filtered, cmap="gray", vmin=0, vmax=255)
plt.title("Filter Batas")

plt.subplot(2, 3, 4)
plt.hist(gray.flatten(), 256, [0, 256])
plt.yscale("log")
plt.title("Histogram Asli")

plt.subplot(2, 3, 5)
plt.hist(equalized.flatten(), 256, [0, 256])
plt.yscale("log")
plt.title("Histogram Hasil")

plt.tight_layout(h_pad=3)
plt.savefig(os.path.join(folder, "hasil.png"), dpi=150)
plt.show()
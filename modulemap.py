import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
img = mpimg.imread('module_map.jpg')

# Plot it
plt.figure(figsize=(8, 7))
plt.imshow(img)
plt.axis('on')  # or 'off' to hide axes
plt.title("Module Map")
plt.tight_layout()
plt.show()

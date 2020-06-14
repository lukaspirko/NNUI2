from matplotlib import pyplot as plt
from skimage import feature

img = plt.imread('..\\..\\DataSet\\grapes\\TrainingSet\\Positive\\P0012.png')
plt.imshow(img)
plt.show()

(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=True)
plt.imshow(hogImage)
plt.show()

print(H.shape)

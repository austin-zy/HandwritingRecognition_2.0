# Author: Goh Zhao Yang

# Standard scientific Python imports
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset (1. training 2. testing 3.original image)
digits = datasets.load_digits()

test_digit = datasets.load_digits()

real_digit = datasets.load_digits()

# Param for de-skewing
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
SZ=20
# Param for HOG extraction
bin_n = 16

# De-skew operation
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02'])< 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew],[0,1,0]])
    img = cv2.warpAffine(img,M,(SZ,SZ), flags=affine_flags)
    return img

# Calculate HOG feature vector
def hog(img):
    gx = cv2.Sobel(img, 6, 1, 0)
    gy = cv2.Sobel(img, 6, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist

# Count number of images in dataset
n_samples = len(digits.images)
data = []

################### TRAINING STARTS ############################
for i in range(n_samples/2):
    deskewed = deskew(digits.images[i])
    hogdata=hog(deskewed)
    data.append(hogdata)

# Predicted as Empty tuple - keep predicted value as tuple
predicted = []
# Initialize SVM with gamma of 0.00005
classifier = svm.SVC(gamma=0.00005)
# Train data with Support Vectors (training and target)
classifier.fit(data[:n_samples/2], digits.target[:n_samples/2])
################### TRAINING ENDS ##############################
# Expected value in a tuple for comparison
expected = digits.target[:n_samples]
# Perform for loop for testing
for k in range(n_samples):
    # Classify data
    predicted.append(classifier.predict(hog(deskew(test_digit.images[k]))))

## Print Confusion Matrix
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# USER INTERFACE
# ASK USER FOR INPUT (WHAT IMAGE TO TEST)
indexFile = int(raw_input('Training is done, test image number?'))
# PLOT IMAGE FOR VISUALIZATION
plt.imshow(real_digit.images[indexFile], cmap=plt.cm.gray_r, interpolation='nearest')
# SHOW PREDICTION RESULT
plt.title("Predicted : %i" % classifier.predict(hog(deskew(test_digit.images[indexFile]))))
plt.axis('off')
plt.show()






!pip install --upgrade pip
!pip install imutils
!pip install opencv-python
!pip install --upgrade scikit-learn==0.23.0



def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
  
from numpy.random.mtrand import shuffle
import cv2
import numpy as np
from imutils import paths
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# Define a function to extract the histogram of an image
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

Train_Dir = 'drive/MyDrive/MO_DATA/MO_STEP_2/train'

# Load the image paths and sort them alphabetically
imagePaths = sorted(list(paths.list_images(Train_Dir)))
data = []
labels = []

test_data = []
test_labels = []

# Iterate over the image paths and extract the histogram of each image
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    hist = extract_histogram(image)
    data.append(hist)
    label = imagePath.split('/')[-1].split('.')[0]
    if label == 'cat':
        label = 1
    else:
        label = 0
    labels.append(label)

imagePaths = sorted(list(paths.list_images('drive/MyDrive/MO_DATA/MO_STEP_2/test')))

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    hist = extract_histogram(image)
    test_data.append(hist)
    label = imagePath.split('/')[-1].split('.')[0]
    if label == 'cat':
        label = 1
    else:
        label = 0
    test_labels.append(label)

# Split the data into training and testing sets
#trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.0, shuffle='null')

#trainData = np.array(data).reshape(-1, 1)
trainData = data
trainLabels = labels

# Train the base classifiers
svm = LinearSVC(C=1.55, random_state=69)
dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_leaf_nodes=20, random_state=69)
bagging = BaggingClassifier(base_estimator=dt, n_estimators=20, random_state=69)
rf = RandomForestClassifier(n_estimators=20, criterion='entropy', min_samples_leaf=10, max_leaf_nodes=20, random_state=69)

# Initialize the meta-classifier
lr = LogisticRegression(solver='lbfgs', random_state=69)

# Train the stacking model using 2-fold cross validation
#estimators = [('svm', svm), ('bagging', bagging), ('rf', rf)]
base_estimators = [('SVM', svm), ('Bagging DT', bagging), ('DecisionForest', rf)]
stacking = StackingClassifier(estimators=estimators, final_estimator=lr, cv=2)
stacking.fit(trainData, trainLabels)

# Evaluate the accuracy on the training set
accuracy = cross_val_score(stacking, trainData, trainLabels, cv=2)
print("Accuracy on training set:", np.mean(accuracy))

# Evaluate the accuracy on the training set
accuracy = cross_val_score(stacking, test_data, test_labels, cv=2)
print("Accuracy on test set:", np.mean(accuracy))


# Another variant
# Get predictions for the training data
y_train_pred = stacking.predict(trainData)
# Calculate accuracy score
accuracy = accuracy_score(trainLabels, y_train_pred)
# Print the accuracy rounded to 2 decimal places
print(f"Training accuracy: {accuracy:.2f}")

# Predict the probabilities of class 1 for the given images
testImages = ['cat.1024.jpg', 'cat.1003.jpg', 'dog.1006.jpg', 'dog.1022.jpg']
testData = []
for testImage in testImages:
    image = cv2.imread(testImage)
    hist = extract_histogram(image)
    testData.append(hist)
probabilities = stacking.predict_proba(testData)
for i in range(len(testImages)):
    print(f"Probability of {testImages[i]} being a cat:", probabilities[i][1])

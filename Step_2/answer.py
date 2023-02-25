import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Function to extract histogram from image
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

Train_Dir = 'drive/MyDrive/MO_DATA/MO_STEP_2/train'
# Load dataset and sort filenames alphabetically
imagePaths = sorted(list(paths.list_images(Train_Dir)))

# Extract features and labels from images
data = []
labels = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    features = extract_histogram(image)
    label = 0 if 'cat' in imagePath else 1
    data.append(features)
    labels.append(label)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=16)

# Train Linear SVM classifier
clf = LinearSVC(C=0.72, random_state=16)
clf.fit(X_train, y_train)


# Print coefficients of the hyperplane
print("19th coefficient:", clf.coef_[0][18])
print("371st coefficient:", clf.coef_[0][370])
print("344th coefficient:", clf.coef_[0][343])


# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Compute confusion matrix, precision, recall, and f1 scores
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
precision = precision_score(y_test, y_pred)
print("Precision:", precision)
recall = recall_score(y_test, y_pred)
print("Recall:", recall)
f1 = f1_score(y_test, y_pred)
print("F1 score:", f1)

# Compute macro-averaged F1 score
macro_f1 = f1_score(y_test, y_pred, average='macro')
print("Macro-averaged F1 score:", macro_f1)

# Function to predict the class of an input image
def predict_image_class(image_path):
    # Load the image and extract features
    image = cv2.imread(image_path)
    hist = extract_histogram(image)
    # Make a prediction using the trained classifier
    pred = clf.predict([hist])[0]
    if pred == 0:
        return "cat"
    else:
        return "dog"

# Example usage of the predict_image_class function
paths_for_test = ['cat.1007.jpg', 'dog.1021.jpg', 'cat.1042.jpg', 'cat.1047.jpg']
for image_path in paths_for_test:
    prediction = predict_image_class(image_path)
    print("Prediction for", image_path, "is", prediction)

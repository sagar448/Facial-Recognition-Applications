import sys
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages")
from scipy.ndimage import zoom
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import os
os.chdir("/Users/SagarJaiswal/Desktop/progammingShit")
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import matplotlib.image as mpimg

def pathOfImages(directory):
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".jpg"):
                file_path = os.path.join(root, file_name)
                img_feature = turnToHisto(file_path)
                training.append(img_feature)
            else:
                print(str(file_name)+" is not an image")
    return training

def turnToHisto(file_path):
    image = mpimg.imread(file_path)
    histogram = (cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])).flatten()
    return histogram

def train_SVM(pathA, pathB):
    training_a = pathOfImages(pathA)
    training_b = pathOfImages(pathB)
    data = training_a + training_b
    target = [1] * len(training_a) + [0] * len(training_b)
    x_train, x_test, y_train, y_test = train_test_split(data,
                target, test_size=0.20, random_state=0)
    global svc_1
    svc_1 = SVC(kernel="linear")
    try:
        svc_1.fit(x_train, y_train)
    except:
        print("SVM could not be trained")

def detect_face(frame):
    faceCascade = cv2.CascadeClassifier(cascPath)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_faces = faceCascade.detectMultiScale(
            RGB,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return RGB, detected_faces

def predict_face(extracted_face):
    extracted_faceAr = np.reshape(extracted_face, (1, -1))
    return svc_1.predict(extracted_faceAr)[0]

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
train_SVM("Me", "notMe")

while True:
    ret, frame = video_capture.read()
    RGB, detected_faces = detect_face(frame)

    face_index = 0

    for face in detected_faces:
        (x, y, w, h) = face
        if w > 100:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hist = cv2.calcHist([frame], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            histq = hist.flatten()
            prediction_result = predict_face(histq)
            print(prediction_result)
            if prediction_result == 1:
                cv2.putText(frame, "Me",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
            else:
                cv2.putText(frame, "notMe",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

            face_index += 1
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

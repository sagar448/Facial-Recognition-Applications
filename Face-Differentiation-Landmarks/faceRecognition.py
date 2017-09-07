import sys
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages")
sys.path.append("/usr/local/lib/python3.6/site-packages")
#sys.path.append("/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages")
#Some credit to jrosebr1 as I have used some of his function and his
#facealigner script but the altogher the compilation is mine
from collections import OrderedDict
import numpy as np
import dlib
import cv2
from PIL import Image
import os
from facealigner import FaceAligner
import face_recognition
from sklearn.svm import SVC

#This function is used to convert the images in the two folders into
#normalised images by using the facealigner python file
def AlignAndTrainer():
    countYs = 0
    countXs = 0
    X_train = []
    Y_train = []
    directory = []
    for file in os.listdir("/Users/SagarJaiswal/Desktop/facial-landmarks"):
        if file.endswith(".py") or file.endswith("_") or file.endswith(".dat") or file == "images" or file == ".DS_Store":
            print("not a file")
        else:
            directory.append(file)
    #Initialise the frontal face detector
    #Initialise the pre-trained dlib facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #We initialise the FaceAligner class by inputting the predictor
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    for file in directory:
        print(file)
        for root, _, files in os.walk(file):
            for file_name in files:
                if file_name.endswith(".jpg"):
                    file_path = os.path.join(root, file_name)
                    image = cv2.imread(file_path, 1)
                    image = imutils.resize(image, width=800)
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    rects = detector(gray, 2)
                    faceAligned = fa.align(image, gray, rects[0])
                    os.remove(file_path)
                    saveImage = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
                    saveImage = Image.fromarray(saveImage, "RGB")
                    saveImage.save("/Users/SagarJaiswal/Desktop/facial-landmarks/"+file+"/"+file_name, "JPEG")
                    #This face encoding library is a pre trained algorithm that can find best
                    #measurement between a face, to tell it apart from others
                    #The encoding that is produced we will use that to train the SVM
                    X_train.append(face_recognition.face_encodings(faceAligned))
                    Y_train.append(file)
                    countXs += 1
                    countYs += 1
                else:
                    print(str(file_name)+" is not an image")
    X_train = np.array(X_train)
    #X_train = X_train.transpose(0, 2, 1)
    X_train = X_train.reshape(countXs, 128)
    #X_train = X_train.transpose(2,1, 0)
    Y = np.array(Y_train)
    Y = Y.reshape(countYs,)
    global classifier
    classifier = SVC(kernel = "linear")
    classifier.fit(X_train, Y)
##    predicted = classifier.predict(X_train)
##    print(predicted)

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
AlignAndTrainer()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video_capture = cv2.VideoCapture(0)


while True:
    frame = video_capture.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # loop over the face detections
    #for (i, rect) in enumerate(rects):
    for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            faAligned = fa.align(frame, gray, rect)
            enc = face_recognition.face_encodings(faAligned)
            print(np.array(enc).shape)
            predicted = classifier.predict(enc)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
     
            # show the face number
            cv2.putText(frame, predicted[0]+" #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
     
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()


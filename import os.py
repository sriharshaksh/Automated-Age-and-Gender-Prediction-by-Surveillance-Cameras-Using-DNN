import os

# Get the current working directory
current_dir = os.getcwd()
print("Current working directory:", current_dir)

# Change the current working directory
new_dir = '/Users/brund/Downloads/Age-and-Gender-Recognition-main/Age-and-Gender-Recognition-main/models'
os.chdir(new_dir)

# Verify the change
updated_dir = os.getcwd()
print("Updated working directory:", updated_dir)
      

import cv2

def detectFace(net, frame, confidence_threshold=0.7):
    frameOpencvDNN = frame.copy()
    print(frameOpencvDNN.shape)
    frameHeight = frameOpencvDNN.shape[0]
    frameWidth = frameOpencvDNN.shape[1]
    
    blob = cv2.dnn.blobFromImage(frameOpencvDNN, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDNN, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    
    return frameOpencvDNN, faceBoxes

# File paths corrected
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

genderList = ['Male', 'Female']  # corrected list initialization
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']  # corrected list initialization

# Reading networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)  # Corrected indentation here

video = cv2.VideoCapture(0)
adding = 20  # Corrected variable name from 'addinq' to 'adding'

while True:  # Corrected loop structure
    ret, frame = video.read()
    if not ret:
        break
    
    frameOpencvDNN, faceBoxes = detectFace(faceNet, frame)
    
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - adding):min(faceBox[3] + adding, frame.shape[0] - 1),
                     max(0, faceBox[0] - adding):min(faceBox[2] + adding, frame.shape[1] - 1)]
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        cv2.putText(frameOpencvDNN, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
    
    cv2.imshow("Detecting age and Gender", frameOpencvDNN)
    
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
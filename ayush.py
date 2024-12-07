import cv2

def faceBox(faceNet ,frame):
    frameHeight=frame.shape[0]
    frameWidth = frame.shape[1]
    blob =cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence =detection[0,0,i,2]
        if confidence>0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)     
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
        
        return frame,bboxs
    #print(detection.shape)
    


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"



 
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES=(78.4263377603,87.7689143744,114.895847746)
ageList =['(0-2)','(3-6)','(7-12)','(15-20)','(21-25)','(26-30)','(31-35)','(36-40)','(41-47)','(48-53)','(54-57)','(58-66)','(67-73)','(74-80)']
genderList=['male','female','other']



# Load the pre-trained Haar Cascade for face detection
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Open a connection to the webcam
#video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

#f not video_capture.isOpened():
  #  print("Error: Could not access the camera.")
   # exit()

#print("Press 'q' to exit.")
video =cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, frame = video.read()
    frameNet,bboxs =faceBox(faceNet,frame)
    for bbox in bboxs:
        face = frameNet[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        genderNet.setInput(blob)
        gernderPred = genderNet.forward()
        gender = genderList[gernderPred[0].argmax()]


        ageNet.setInput(blob)
        agePred= ageNet.forward()
        age = ageList[agePred[0].argmax()]


        lable="{},{}".format(gender,age)
        cv2.putText(frame,lable,(bbox[0],bbox[1]-10,),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)




    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()








#    ***if not ret:
 #       print("Error: Failed to capture frame.")
  #      break

    # Convert the frame to grayscale (Haar cascades work with grayscale images)
   # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    #faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    #for (x, y, w, h) in faces:
     #   cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with detected faces
    #cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

# Release the webcam and close all OpenCV windows
#video_capture.release()
#cv2.destroyAllWindows()

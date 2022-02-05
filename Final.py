import cv2
import time

def detectFace(net,frame,confidence_threshold=0.7):
    frameOpencvDNN=frame.copy()
    print(frameOpencvDNN.shape)
    frameHeight=frameOpencvDNN.shape[0]
    frameWidth=frameOpencvDNN.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>confidence_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)
    return frameOpencvDNN,faceBoxes
        
    
faceProto='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
ageProto='age_deploy.prototxt'
ageModel='age_net.caffemodel'
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'

genderList=['Male','Female']
ageList=['(0-2)','(4-6)','(8-12)','(15-22)','(25-35)','(38-45)','(48-53)','(60-100)']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


def cam():
    video=cv2.VideoCapture(0)
    padding=20

    while video.isOpened():
        hasFrame,frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            break
            
        resultImg,faceBoxes=detectFace(faceNet,frame)
        
        if not faceBoxes:
            print("No face detected")
        
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            cv2.putText(resultImg,f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow("Detecting Age & Gender || Press 'Q' to Quit",resultImg)
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()            
    cv2.destroyAllWindows()

OutputGender = ''
OutputAge = ''

def age_gender_detector(frame):
    # Read frame
    padding=20
    t = time.time()
    resultImg,faceBoxes=detectFace(faceNet,frame)
        
    if not faceBoxes:
        return 
        
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]

        global OutputGender
        OutputGender = gender
            
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]

        global OutputAge
        OutputAge = age

        cv2.putText(resultImg,f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
    return resultImg

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def images(path):
    input = cv2.imread(path)
    output = age_gender_detector(input)
    if output is None:
        return
    else:
        scale_percent = 40 # percent of original size
        width = int(output.shape[1] * scale_percent / 100)
        height = int(output.shape[0] * scale_percent / 100)
        dim = (width, height)
  
        # resize image
        resized = ResizeWithAspectRatio(output, height=600)
        # resized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow(f'Output Window || Result: {OutputGender},{OutputAge}', resized)
        return 1
         
    


import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dst
from scipy.spatial import ConvexHull

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()

JAWLINE_POINTS = list(range(0,17))
RIGHT_EYE_POINTS = list(range(36, 42))
def face_size(face):
    faceWidth = dst.euclidean(face[0],face[16])
    hull = ConvexHull(face)
    faceCenter = np.mean(face[hull.vertices,:],axis=0)
    faceCenter = faceCenter.astype(int)
    return int(faceWidth),faceCenter





def place_face(frame,faceCenter,faceSize):
    faceSize = int(faceSize * 1.5)
    x1 = int(faceCenter[0,0] - (faceSize/2.9))
    x2 = int(faceCenter[0,0] + (faceSize/3.0))
    y1 = int(faceCenter[0,1] - (faceSize/1.6))
    y2 = int(faceCenter[0,1] + (faceSize/3.6))

    h,w = frame.shape[:2]

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 =0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    faceOverlayWidth = x2 - x1
    faceOverlayHeight = y2 - y1

    faceOverlay = cv2.resize(imgFace,(faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask_face,(faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_face,(faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)

    roi = frame[y1:y2,x1:x2]

    roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
    roi_fg = cv2.bitwise_and(faceOverlay,faceOverlay,mask=mask)

    dst = cv2.add(roi_bg,roi_fg)
    frame[y1:y2,x1:x2] = dst

# MOUTH_OUTLINE_POINTS = list(range(48,61))

def get_landmarks(im):
    rects = detector(im, 1)

    # if len(rects) > 1:
    #     return "error"
    # if len(rects) == 0:
    #     return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    # if landmarks == "error":
    #     return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
# yawns = 0
yawn_status = False 
# imgTongue=cv2.imread('tongue.png',-1)
#small_res=cv2.resize(small,(0,0),fx=0.6,fy=0.6)
# orig_mask = imgTongue[:,:,3]

# orig_mask_inv = cv2.bitwise_not(orig_mask)

# imgTongue = imgTongue[:,:,0:3]

# origTongueHeight, origTongueWidth = imgTongue.shape[:2]

# def lip_size(lip):
#     lipWidth = dst.euclidean(lip[0],lip[6])
#     hull = ConvexHull(lip)
#     lipCenter = np.mean(lip[hull.vertices,:],axis=0)
#     lipCenter = lipCenter.astype(int)
#     return int(lipWidth), lipCenter

# def place_lip(frame,lipCenter,lipSize):
#     lipSize = int(lipSize*1.5)
#     x1 = int(lipCenter[0,0] - (lipSize/3))
#     x2 = int(lipCenter[0,0] + (lipSize/2))
#     y1 = int(lipCenter[0,1] - (lipSize)/3)
#     y2 = int(lipCenter[0,1] + (lipSize)/4)

#     h, w = frame.shape[:2]
#     if x1<0:
#         x1=0
#     if y1<0:
#         y1=0
#     if x2>w:
#         x2=w
#     if y2>h:
#         y2=h

#     lipOverlayWidth = x2-x1
#     lipOverlayHeight = (y2-y1)

#     lipOverlay = cv2.resize(imgTongue,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)
#     mask = cv2.resize(orig_mask,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)
#     mask_inv = cv2.resize(orig_mask_inv,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)

#     roi = frame[y1:y2,x1:x2]

#     roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
#     roi_fg = cv2.bitwise_and(lipOverlay,lipOverlay,mask=mask)
#     dst = cv2.add(roi_bg,roi_fg)
#     frame[y1:y2,x1:x2] = dst

# ton=cv2.resize(imgTongue,(100,50))
while True:
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    rects=detector(gray,0)
    for rect in rects:

        image_landmarks, lip_distance = mouth_open(frame)
        landmarks=get_landmarks(frame)
        # lip = landmarks[MOUTH_OUTLINE_POINTS]
        # prev_yawn_status = yawn_status  
        # lipsize,lipcenter = lip_size(lip)
        value=0


        # smiles starts



        face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        smile_classifier=cv2.CascadeClassifier("haarcascade_smile.xml")
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)
        if faces is ():
            print("No faces found")
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(127,0,255),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]
            smiles = smile_classifier.detectMultiScale(
                roi_gray,
                scaleFactor= 1.7,
                minNeighbors=50,
                minSize=(25, 25),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                )
            # for (ex,ey,ew,eh) in eyes:
            #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)      
            #cv2.imshow("Eye",img)
            #cv2.waitKey(0)
            # if smiles is ():
            #   output="Not Smiling"
            #   cv2.putText(image,output,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)
            # else:
            #   output="Smiling"
            #   cv2.putText(image,output,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)
            for (x, y, w, h) in smiles:
                value=1
                print "Found", len(smiles), "smiles!"
                output="Smiling"
                imgFace = cv2.imread("Smile.png",-1)
                orig_mask_face = imgFace[:,:,3]
                orig_mask_inv_face = cv2.bitwise_not(orig_mask_face)
                imgFace = imgFace[:,:,0:3]
                faceHeight, faceWidth = imgFace.shape[:2]
                cv2.putText(frame,output,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
                face = landmarks[JAWLINE_POINTS]

                faceSize, faceCenter = face_size(face)
                place_face(frame,faceCenter,faceSize)
            if value == 0:    
               if lip_distance > 25:
            
                 
                imgFace = cv2.imread("Suprised.png",-1)
                orig_mask_face = imgFace[:,:,3]
                orig_mask_inv_face = cv2.bitwise_not(orig_mask_face)
                imgFace = imgFace[:,:,0:3]
                faceHeight, faceWidth = imgFace.shape[:2]
                face = landmarks[JAWLINE_POINTS]
                faceSize,faceCenter = face_size(face)
                place_face(frame,faceCenter,faceSize)
                cv2.putText(frame, "Subject is Surprised", (50,450), 
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
            

                output_text = " Surprised: "

                cv2.putText(frame, output_text, (50,50),
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
               elif (landmarks[RIGHT_EYE_POINTS][4][0,1]-landmarks[RIGHT_EYE_POINTS][2][0,1])<8:
                output_text = "Disgusted"
                
                cv2.putText(frame,output_text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)
                imgFace = cv2.imread("Disgusted.png",-1)
                orig_mask_face = imgFace[:,:,3]
                orig_mask_inv_face = cv2.bitwise_not(orig_mask_face)
                imgFace = imgFace[:,:,0:3]
                faceHeight, faceWidth = imgFace.shape[:2]
                face = landmarks[JAWLINE_POINTS]
                faceSize,faceCenter = face_size(face)
                place_face(frame,faceCenter,faceSize)
               else:
                output_text="Neutral"
                cv2.putText(frame,output_text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)   
                imgFace = cv2.imread("Neutral.png",-1)
                orig_mask_face = imgFace[:,:,3]
                orig_mask_inv_face = cv2.bitwise_not(orig_mask_face)
                imgFace = imgFace[:,:,0:3]
                faceHeight, faceWidth = imgFace.shape[:2]
                face = landmarks[JAWLINE_POINTS]
                faceSize,faceCenter = face_size(face)
                place_face(frame,faceCenter,faceSize)
                
                
            
            
            
            
                 
             
            
                # yawns += 1
                #else:
                #     output_text="Neutral"
                #     cv2.putText(frame,output_text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)    
            print landmarks[RIGHT_EYE_POINTS][4][0,1]-landmarks[RIGHT_EYE_POINTS][2][0,1]
        cv2.imshow('Live Landmarks', image_landmarks )
        cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break























    # if lip_distance > 25:
        
    #     yawn_status = True 
        
    #     cv2.putText(frame, "Subject is Surprised", (50,450), 
    #                 cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

    #     output_text = " Surprised: "

    #     cv2.putText(frame, output_text, (50,50),
    #                 cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
    #     point=landmarks[48]
    #     x=point[0,0]
    #     y=point[0,1]
    #     #frame[y:y+ton.shape[0], x:x+ton.shape[1]] = ton
    #     # place_lip(frame,lipcenter,lipsize)
        
    # else:
    #     yawn_status = False 
         
    # if prev_yawn_status == True and yawn_status == False:
    #     yawns += 1

    # cv2.imshow('Live Landmarks', image_landmarks )
    # cv2.imshow('Yawn Detection', frame )
    
    # if cv2.waitKey(1) == 13: #13 is the Enter Key
    #     break
        
cap.release()
cv2.destroyAllWindows() 
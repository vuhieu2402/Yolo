from ultralytics import YOLO
import cv2
import cvzone
import  math

cap = cv2.VideoCapture(0)



model = YOLO('best.pt')
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

myColor = (0,0,255)

while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            #bouding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #draw with opencv
            # print(x1, y1, x2, y2)
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            
            #draw with cvzone
            w, h = x2-x1, y2-y1
            # cvzone.cornerRect(frame, (x1,y1,w,h))
            # cv2.rectangle(frame, (x1,y1),(x2,y2), myColor,3)

            #confident
            conf = math.ceil((box.conf[0]*100))/100


            #classname
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if conf > 0.5:
                if currentClass=='Hardhat' or currentClass=='Mask' or currentClass=='Safety Vest':
                    myColor = (0,255,0)
                elif currentClass=='NO-Hardhat' or currentClass=='NO-Mask' or currentClass=='NO-Safety Vest':
                    myColor = (0,0,255)
                else:
                    myColor = (255,0,0)
                    
                    
                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(0, y1)), scale=0.5,thickness=1,colorB=myColor,colorT=myColor,colorR=myColor)
                cv2.rectangle(frame, (x1,y1),(x2,y2), myColor,3)
            
    cv2.imshow('cam', frame)
    phim_bam = cv2.waitKey(1)
    if phim_bam == ord('q'):
        break
    
    

cv2.destroyAllWindows()
cap.release()
    

from ultralytics import YOLO
import cv2
import cvzone
import  math

cap = cv2.VideoCapture("../video/1.mp4")



model = YOLO('../yolo-weights/yolov8n.pt')
classNames = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
              "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
              "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
              "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
              "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
              "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
              "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
              "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
              "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
              "teddy bear","hair drier","toothbrush"]

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
            cvzone.cornerRect(frame, (x1,y1,w,h))

            #confident
            conf = math.ceil((box.conf[0]*100))/100


            #classname
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == 'car' and conf > 0.3:
                cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(0, y1)), scale=0.5,thickness=1)
            
    cv2.imshow('cam', frame)
    phim_bam = cv2.waitKey(1)
    if phim_bam == ord('q'):
        break
    
    

cv2.destroyAllWindows()
cap.release()
    

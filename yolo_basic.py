import cv2
from ultralytics import YOLO

model = YOLO("../yolo-weights/yolov8l.pt")
result = model("img/1.png", show = True)
cv2.waitKey(0)
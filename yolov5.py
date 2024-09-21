import torch
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Check if frame is not None before passing it to the model
    if frame is not None:
        result = model(frame)
        # Convert the result to RGB color and then display it
        result_rgb = cv2.cvtColor(np.squeeze(result.render()), cv2.COLOR_BGR2RGB)
        cv2.imshow('Live Video Feed', result_rgb)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Unable to read frame from video capture")

cap.release()
cv2.destroyAllWindows()

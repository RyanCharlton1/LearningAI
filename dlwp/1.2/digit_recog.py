import cv2 as cv
import numpy as np
import os

from keras import models

#events = [i for i in dir(cv) if 'EVENT' in i]
#print(events)

# Change when left msb down and up
held = False
def mouse_call(event, x, y, flags, param):
    global held
    if event == cv.EVENT_LBUTTONDOWN:
        held = True
    if event == cv.EVENT_LBUTTONUP:
        held = False
    # if held draw a circle around the cursor
    if held :
        cv.circle(img, (x,y), 7, 1, -1)

    # Evaluate the picture
    if event == cv.EVENT_RBUTTONDOWN:
        # Scale image down to 28x28(network training size)
        scaled = cv.resize(img, (28,28), 
                    interpolation=cv.INTER_LINEAR)
        scaled = scaled.reshape(28*28)

        # Calculate probabilities the img is each digit
        probs = network.predict(scaled[np.newaxis, :])
        print(probs.argmax())
    
# Intialize canvas and window 
img = np.zeros((140, 140, 1), np.float32)
cv.namedWindow('number')
cv.setMouseCallback('number', mouse_call)

path = os.path.join(os.getcwd(), 'digit_recog.keras')
network = models.load_model(path)

# Until escape is pressed draw canvas
while(1):
    cv.imshow('number', img)
    if cv.waitKey(20) & 0xFF == 27:
        break

# Clean up windows
cv.destroyAllWindows()
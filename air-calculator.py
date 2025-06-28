import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


model = load_model(r'C:\Users\hp\OneDrive\Desktop\computer vision\finger-calculator\mnist_cnn_model.h5')


# To store finger positions
points = []
canvas = np.zeros((125, 125), dtype=np.uint8)
numbers = []

last_seen = time.time()
prediction_done = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the image  else 2 might look like the mirrored one
    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(gray_scale_frame)

        
    if result.multi_hand_landmarks:

        prediction_done = False
        last_seen = time.time()

        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip coordinates
            x = int(hand_landmarks.landmark[8].x * 125)
            y = int(hand_landmarks.landmark[8].y * 125)

            # Draw on canvas
            points.append((x, y))
            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], (255), 4)
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    else:
        if (time.time() - last_seen > 2) and not prediction_done:
            if len(points) > 10:
            
                resized_canvas = cv2.resize(canvas, (28,28))
                resized_canvas = resized_canvas.astype('float32') / 255.
                input_number = resized_canvas.reshape(1,28,28,1)
                prob_outof_all = model.predict(input_number)

                y_pred = np.argmax(prob_outof_all)

                numbers.append(y_pred.item())

            points = []
            canvas = np.zeros((125, 125), dtype=np.uint8)
            prediction_done = True

    cv2.putText(frame, f"The sum is {(np.sum(numbers)).item()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if numbers:
        cv2.putText(frame, f"The number is {numbers[len(numbers)-1]}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

    cv2.imshow("Air Writing", frame)
    cv2.imshow("Canvas", canvas)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

cap.release()
cv2.destroyAllWindows()

print(f"Numbers detected: {numbers}")
print(f"The sum is {(np.sum(numbers)).item()}")
import cv2


def rectangleMA(prev_rects):
    x_array, y_array, w_array, h_array = ([], [], [], [])
    for (x, y, w, h) in prev_rects:
        x_array.append(x)
        y_array.append(y)
        w_array.append(w)
        h_array.append(h)
    x = int(sum(x_array)/len(x_array))
    y = int(sum(y_array)/len(y_array))
    w = int(sum(w_array)/len(w_array))
    h = int(sum(h_array)/len(h_array))
    return (x, y, w, h)        

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Capture video from the built-in webcam (usually camera index 0)
cap = cv2.VideoCapture(0)

# number of previous rectangles to consider in moving average 
n_prev = 6
# list of previous rectangles
prev_rectangles = [(0, 0, 0, 0) for _ in range(n_prev)]
# last valid central point's position read from the frame
last_valid_center = (0, 0)
# last valid rectangle position read from the frame
last_valid_rect = (0, 0, 0, 0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

    if len(faces) > 0:
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # center of the detected rectangle
            # center = (x+w//2, y+h//2)
            prev_rectangles.pop(0)
            prev_rectangles.append((x, y, w, h))
            smoother_rect = rectangleMA(prev_rectangles)
            x, y, w, h = smoother_rect
            
            
            # prev_centers.pop(0)
            # prev_centers.append(center)
            # smooth the central point position using moving average filter
            smoother_center = (x+w//2, y+h//2)
            # prev_centers[-1] = smoother_center

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.circle(frame, smoother_center, 1, (0, 0, 255), 2)
            
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.circle(frame, smoother_center, 1, (0, 255, 0), 2)
            
            # save last valid position of the central point
            last_valid_center = smoother_center
            last_valid_rect = smoother_rect
    else:
        # rectangle stays on the screen in the last valid position
        x, y, w, h = last_valid_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.circle(frame, last_valid_center, 1, (0, 0, 255), 2)
        

    # Display the frame with face tracking
    cv2.imshow('Face Tracking', frame)

    # Break the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

    print(faces)

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
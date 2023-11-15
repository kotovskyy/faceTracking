import cv2

def pointMA(prev_points):
    x_array = [x for x, _ in prev_points]
    y_array = [y for _, y in prev_points]
    x = int(sum(x_array)/len(x_array))
    y = int(sum(y_array)/len(y_array))
    return (x, y)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Capture video from the built-in webcam (usually camera index 0)
cap = cv2.VideoCapture(0)

# number of previous points to consider in moving average for the central point
n_prev = 3
# list of previous central points
prev_centers = [(0, 0) for _ in range(n_prev)]
# last valid central point's position read from the frame
last_valid_center = (0, 0)

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
            center = (x+w//2, y+h//2)
            
            prev_centers.pop(0)
            prev_centers.append(center)
            # smooth the central point position using moving average filter
            smoother_center = pointMA(prev_centers)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.circle(frame, smoother_center, 1, (0, 0, 255), 2)
            
            # save last valid position of the central point
            last_valid_center = smoother_center
    else:
        # central point stays on the screen in the last valid position
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
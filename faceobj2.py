import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet(r"C:\Users\SWAGATA DEY\Downloads\yolov3.weights", r"C:\Users\SWAGATA DEY\Downloads\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class labels
with open(r"C:\Users\SWAGATA DEY\Downloads\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the Haar cascades for face, body, and eye detection
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
upper_body_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
eye_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
profileface_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# Start video capture
cap = cv2.VideoCapture(0)  # '0' is the default webcam on your laptop

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Set frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Process each output layer
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert frame to grayscale for Haar cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces, body, upper body, eyes, and profile faces using Haar cascades
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    bodies = body_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    upper_bodies = upper_body_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    eyes = eye_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    profile_faces = profileface_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Draw rectangles for detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for face

    # Draw rectangles for detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for full body

    # Draw rectangles for detected upper bodies
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan for upper body

    # Draw rectangles for detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for eyes

    # Draw rectangles for detected profile faces
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Magenta for profile faces

    # Display the resulting frame
    cv2.imshow("Object Detection and Face Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()

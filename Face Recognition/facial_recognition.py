import face_recognition
import cv2
import numpy as np
import time
import pickle

# Load encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Start USB camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Lower resolution for faster recognition
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv_scaler = 4

face_locations = []
face_encodings = []
face_names = []

frame_count = 0
start_time = time.time()
fps = 0


def process_frame(frame):

    global face_locations, face_encodings, face_names

    # Resize frame for speed
    resized_frame = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)

    # Convert BGR (OpenCV) → RGB (face_recognition)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_resized_frame)

    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame,
        face_locations,
        model="small"
    )

    face_names = []

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        name = "Unknown"

        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return frame


def draw_results(frame):

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        # Draw label background
        cv2.rectangle(frame, (left, top-30), (right, top), (0,255,0), cv2.FILLED)

        # Draw name
        cv2.putText(
            frame,
            name,
            (left+5, top-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,0),
            2
        )

    return frame


def calculate_fps():

    global frame_count, start_time, fps

    frame_count += 1

    elapsed_time = time.time() - start_time

    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    return fps


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    processed_frame = process_frame(frame)

    display_frame = draw_results(processed_frame)

    current_fps = calculate_fps()

    cv2.putText(
        display_frame,
        f"FPS: {current_fps:.1f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.imshow("Face Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
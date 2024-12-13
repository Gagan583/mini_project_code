import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from utils import (
    apply_effects,
    extract_license_text,
    get_license_info,
    save_unauthorized_plates,
    check_authorized_plate,
    display_license_info
)
from config import VEHICLES, VIDEO_PATH, COCO_MODEL_PATH, LICENSE_PLATE_MODEL_PATH

# Initialize tracker and models
abj_tracker = Sort()
coco_model = YOLO(COCO_MODEL_PATH)  # Replace with correct path
license_plate_detector_model = YOLO(LICENSE_PLATE_MODEL_PATH)  # Replace with correct path

# Initialize the list for unauthorized plates
unauthorized_plates = []

# Real-time video capture
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read from video source.")
        break

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in VEHICLES:  # Vehicle detection
            detections_.append([x1, y1, x2, y2, score])

    # Process vehicle detections and track them
    if len(detections_) > 0:
        detections_np = np.asarray(detections_)
        track_ids = abj_tracker.update(detections_np)
    else:
        track_ids = []

    # Detect license plates
    license_plates = license_plate_detector_model(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Crop license plate region
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Apply effects and extract license number
        processed_plate = apply_effects(license_plate_crop)
        license_number = extract_license_text(processed_plate)

        if license_number:
            # Check if the license plate is authorized
            if check_authorized_plate(license_number):
                authorized_plates = True
                plate_color = (0, 255, 0)  # Green for authorized plates
            else:
                authorized_plates = False
                plate_color = (0, 0, 255)  # Red for unauthorized plates
                save_unauthorized_plates(license_plate_crop, license_number)

            # Draw bounding box for license plate with the proper color
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), plate_color, 2)

            # Display the license plate info inside the window
            display_license_info(frame, license_number, int(x1), int(y1), plate_color)

    # Draw vehicle tracking boxes
    for track in track_ids:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the processed frame
    cv2.imshow("Real-Time Vehicle and License Plate Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

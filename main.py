import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utilis import YOLO_Detection, drawPolygons, label_detection

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load YOLO model and move it to the appropriate device
model = YOLO("yolov8n.pt")
model.to(device)
names = model.names
model.nms = 0.8

# Define working and counter areas
working_area = [[(243, 90), (249, 272), (267, 387), (588, 293), (599, 166), (580, 76)]]
counter_area = [[(602, 84), (665, 85), (665, 280), (603, 296)]]

source_video = "input_video/input.mp4"
cap = cv2.VideoCapture(source_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


counted = 0
counted_objects = {}  # Dictionary to store tracking IDs and their count status

while True:
    detection_points = []
    ret, frame = cap.read()
    new_frame = frame.copy()
    if not ret:
        break

    # Get YOLO detections
    boxes, classes, names, confidences = YOLO_Detection(model, frame, conf=0.05, mode = "pred")
    occupied_pol = -1

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        detected_class = cls
        name = names[int(cls)]

        # Calculate the center point of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_point = (int(center_x), int(center_y))

        # Collect detection points
        detection_points = [(int((x1 + x2 - 10) / 2), int((y1 + y2 - 10) / 2)) for x1, y1, x2, y2 in boxes]

        # Check if the detection is inside the working area
        for pos in working_area:
            matching_result = cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False)
            if matching_result >= 0:
                occupied_pol += 1

        # Draw the polygons and detection labels
        frame, occupied_count = drawPolygons(frame, working_area, detection_points=detection_points,
                                             occupied_polygons=occupied_pol)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Display fill status
        cv2.rectangle(frame, (int((width / 2) - 200), 10), (int((width / 2) - 80), 35), (0, 175, 255), -1)
        cv2.putText(frame, f"Filling: {occupied_count}", (int((width / 2) - 195), 28), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.95, (0, 0, 0), 1, cv2.LINE_AA)

        # Display the count of filled bottles
        cv2.rectangle(frame, (int((width / 2) + 120), 10), (int((width / 2)), 35), (50, 205, 154), -1)
        cv2.putText(frame, f"Filled: {counted}", (int((width / 2) + 10), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95,
                    (0, 0, 0), 1, cv2.LINE_AA)

        # Determine the color of the bounding box based on detection location
        detection_in_polygon = False
        for pos in working_area:
            matching_result = cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False)
            if matching_result >= 0:
                detection_in_polygon = True
                break

        if detection_in_polygon:
            label_detection(frame=frame, text=str("Bottle"), tbox_color=(0, 69, 255), left=x1, top=y1, bottom=x2,
                            right=y2)
        # else:
        #     label_detection(frame=frame, text=str("Bottle"), tbox_color=(100, 25, 50), left=x1, top=y1, bottom=x2,
        #                     right=y2, fontFace=1, fontScale=0.5)


    boxs, classs, nams, confidencs, ids = YOLO_Detection(model, new_frame, conf=0.05, mode="track")

    for box, cls, obj_id in zip(boxs, classs, ids):  # Include obj_id in the loop
        x1, y1, x2, y2 = box
        detected_class = cls
        name = names[int(cls)]

        # Calculate the center point of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_point = (int(center_x), int(center_y))

        # Track detections passing through the counter area
        for pos in counter_area:
            counter_res = cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False)
            if counter_res >= 0:
                # Check if this object has been counted
                if obj_id not in counted_objects:
                    counted += 1
                    counted_objects[obj_id] = True  # Mark this object as counted

    # Display the frame
    cv2.imshow("Frame", cv2.resize(frame, (1024, 820)))
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

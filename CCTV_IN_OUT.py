from datetime import datetime
import sqlite3
import json
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(lambda: [])
model = YOLO("YOLOv8n-seg.pt")

# Global Variables
SERIAL_ID = None
main_roi_points = []
line_points = []
direction_points = []
roi_defined = False
line_defined = False
direction_defined = False
id_defined = False
in_count = 0
out_count = 0


# Load Configuration
def load_configuration():
    global SERIAL_ID, main_roi_points, line_points, direction_points
    global roi_defined, line_defined, direction_defined, id_defined, rtsp_link

    with open('configurations.json') as json_file:
        data = json.load(json_file)
        if data["UNIQUE_NUMBER"] is not None:
            id_defined = True
            SERIAL_ID = data["UNIQUE_NUMBER"]
        if data["MAIN_ROI"] is not None:
            roi_defined = True
            main_roi_points = [(int(point[0]), int(point[1])) for point in data["MAIN_ROI"]]
        if data["LINE_ROI"] is not None:
            line_defined = True
            line_points = [(int(point[0]), int(point[1])) for point in data["LINE_ROI"]]
        if data["DIRECTION"] is not None:
            direction_defined = True
            direction_points = [(int(point[0]), int(point[1])) for point in data["DIRECTION"]]
        if data["RTSP_LINK"] is not None:
            rtsp_link = data["RTSP_LINK"]


# Insert Data into Database
def insert_data(id, in_count, out_count, created_at):
    conn = sqlite3.connect('cctv_in_out.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO in_out (id, `in`, `out`, created_at) VALUES (?, ?, ?, ?)",
                   (id, in_count, out_count, created_at))
    conn.commit()
    cursor.close()
    conn.close()


# Filter Detections to Only Include Persons
def filter_persons(detections, class_id):
    return [(box, cls_id, track_id) for box, cls_id, track_id in detections if cls_id == class_id]


# Check if a Person has Crossed the Line
def has_crossed_line(track_id, current_point, previous_point, line_points):
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]

    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    def point_sign(x, y):
        return a * x + b * y + c

    previous_sign = point_sign(previous_point[0], previous_point[1])
    current_sign = point_sign(current_point[0], current_point[1])

    if previous_sign * current_sign < 0:
        return 'crossed'
    return None


# Determine Direction Based on User Input
def determine_direction(current_point, previous_point, direction_points):
    dir_x1, dir_y1 = direction_points[0]
    dir_x2, dir_y2 = direction_points[1]

    direction_vector = (dir_x2 - dir_x1, dir_y2 - dir_y1)
    movement_vector = (current_point[0] - previous_point[0], current_point[1] - previous_point[1])

    dot_product = direction_vector[0] * movement_vector[0] + direction_vector[1] * movement_vector[1]
    direction_magnitude = (direction_vector[0] ** 2 + direction_vector[1] ** 2) ** 0.5
    movement_magnitude = (movement_vector[0] ** 2 + movement_vector[1] ** 2) ** 0.5

    if direction_magnitude == 0 or movement_magnitude == 0:
        return None

    cos_angle = dot_product / (direction_magnitude * movement_magnitude)
    alignment_threshold = 0.5

    return 'in' if cos_angle > alignment_threshold else 'out'


# ROI Selection Callback
def select_roi(event, x, y, flags, param):
    global main_roi_points, roi_defined, line_points, line_defined, direction_points, direction_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_defined:
            main_roi_points.append((x, y))
            if len(main_roi_points) == 4:
                roi_defined = True
                update_configuration('MAIN_ROI', main_roi_points)
        elif roi_defined and not line_defined:
            line_points.append((x, y))
            if len(line_points) == 2:
                line_defined = True
                update_configuration('LINE_ROI', line_points)
        elif line_defined and not direction_defined:
            direction_points.append((x, y))
            if len(direction_points) == 2:
                direction_defined = True
                update_configuration('DIRECTION', direction_points)


# Update Configuration File
def update_configuration(key, value):
    with open('configurations.json', 'r') as file:
        data = json.load(file)
    data[key] = value
    with open('configurations.json', 'w') as file:
        json.dump(data, file, indent=4)


# Display Text with Background
def box_text(frame, text_position, text, font_scale=1, font_thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)
    box_padding = 10

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    box_width = text_width + 2 * box_padding
    box_height = text_height + 2 * box_padding

    box_top_left = (text_position[0], text_position[1] - text_height - box_padding)
    box_bottom_right = (text_position[0] + box_width, text_position[1] + box_height)

    cv2.rectangle(frame, box_top_left, box_bottom_right, box_color, cv2.FILLED)
    cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness)


# Get User ID from Input
def get_user_id(frame):
    user_id = ""
    input_prompt = "Please enter your ID:"

    while True:
        input_box = frame.copy()
        box_text(input_box, (100, 100), input_prompt)
        box_text(input_box, (100, 150), user_id)
        cv2.imshow("Video", input_box)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:
            break
        elif key in [8, 127]:
            user_id = user_id[:-1]
        elif key != 255:
            user_id += chr(key)

    return user_id


# Main Processing Loop
def main():
    global SERIAL_ID, id_defined, roi_defined, line_defined, direction_defined
    global in_count, out_count, track_history, model, cap, rtsp_link, w, h, fps
    try:
        if rtsp_link is None:
            print("Please Fill the rtsp link")
            exit()
    except:
        print("Please Fill the rtsp link")
        exit()
    cap = cv2.VideoCapture(rtsp_link)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', select_roi)
    frame_count = 0
    skip_frames = 2
    load_configuration()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            cap = cv2.VideoCapture(0)
            continue
        display_frame = frame.copy()
        if roi_defined:
            cv2.polylines(display_frame, [np.array(main_roi_points)], isClosed=True, color=(0, 255, 0), thickness=2)
        if line_defined:
            cv2.line(display_frame, line_points[0], line_points[1], (0, 0, 255), 2)
        if direction_defined:
            cv2.arrowedLine(display_frame, direction_points[0], direction_points[1], (255, 0, 0), 2)
        if frame_count % skip_frames == 0:
            if not id_defined:
                SERIAL_ID = get_user_id(display_frame)
                id_defined = True
                update_configuration('UNIQUE_NUMBER', SERIAL_ID)
            elif not roi_defined:
                box_text(display_frame, (120, 100), "Please select the four points of the polygon to study the area.")
                draw_points(display_frame, main_roi_points, (0, 255, 0))
                print("Test 1")
            elif roi_defined and not line_defined:
                box_text(display_frame, (120, 100), "Please select the two points to define the line.")
                draw_points(display_frame, line_points, (0, 0, 255))
            elif line_defined and not direction_defined:
                box_text(display_frame, (120, 100),
                         "Please select the two points to define the direction (start and end).")
                draw_points(display_frame, direction_points, (255, 0, 0))
            else:
                process_frame(frame, display_frame)
            cv2.imshow("Video", display_frame)

        frame_count += 1
        key = cv2.waitKey(1)
        if key in (ord("q"), ord('Q')):
            break
        elif key in (ord("r"), ord('R')):
            reset_configuration()

    cap.release()
    cv2.destroyAllWindows()


# Draw Points on Frame
def draw_points(frame, points, color):
    for i, point in enumerate(points):
        cv2.circle(frame, point, 5, color, 4)
        if i > 0:
            cv2.line(frame, points[i - 1], point, color, 2)


# Process Frame
def process_frame(frame, display_frame):
    global in_count, out_count, track_history

    annotator = Annotator(display_frame, line_width=2)
    roi_x1, roi_y1 = np.min(main_roi_points, axis=0)
    roi_x2, roi_y2 = np.max(main_roi_points, axis=0)
    margin = 100
    larger_roi_x1 = max(0, roi_x1 - margin)
    larger_roi_y1 = max(0, roi_y1 - margin)
    larger_roi_x2 = min(w, roi_x2 + margin)
    larger_roi_y2 = min(h, roi_y2 + margin)
    larger_roi = frame[larger_roi_y1:larger_roi_y2, larger_roi_x1:larger_roi_x2]

    results = model.track(larger_roi, persist=True)
    if results[0].boxes is not None and results[0].masks is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [None] * len(
            class_ids)

        person_class_id = 0
        person_detections = filter_persons(zip(boxes, class_ids, track_ids), person_class_id)

        for box, _, track_id in person_detections:
            if track_id is None:
                continue

            mask_x1, mask_y1, mask_x2, mask_y2 = map(int, box)
            current_point = ((mask_x1 + mask_x2) // 2, (mask_y1 + mask_y2) // 2)

            mask_x1 += larger_roi_x1
            mask_y1 += larger_roi_y1
            mask_x2 += larger_roi_x1
            mask_y2 += larger_roi_y1
            current_point = (current_point[0] + larger_roi_x1, current_point[1] + larger_roi_y1)

            if track_id in track_history:
                previous_point = track_history[track_id][-1]
                crossing = has_crossed_line(track_id, current_point, previous_point, line_points)
                if crossing == 'crossed':
                    direction = determine_direction(current_point, previous_point, direction_points)
                    if direction == 'out':
                        out_count += 1
                        insert_data(SERIAL_ID, 0, 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    elif direction == 'in':
                        in_count += 1
                        insert_data(SERIAL_ID, 1, 0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            track_history[track_id].append(current_point)

            annotator.box_label((mask_x1, mask_y1, mask_x2, mask_y2), str(track_id), color=colors(track_id, True))

    box_text(display_frame, (1700, 60), f"In: {in_count}", 1, 3)
    box_text(display_frame, (1700, 160), f"Out: {out_count}", 1, 3)


# Reset Configuration
def reset_configuration():
    global id_defined, roi_defined, line_defined, direction_defined, main_roi_points, line_points, direction_points, SERIAL_ID, in_count, out_count, track_history, model

    id_defined = False
    roi_defined = False
    line_defined = False
    direction_defined = False
    main_roi_points = []
    line_points = []
    direction_points = []
    SERIAL_ID = None
    data = {"UNIQUE_NUMBER": None, 'MAIN_ROI': None, 'LINE_ROI': None, 'DIRECTION': None}
    with open('configurations.json', 'w') as file:
        json.dump(data, file, indent=4)
    in_count = 0
    out_count = 0
    track_history.clear()
    model = YOLO("yolov8n-seg.pt")


# Load configuration and start main processing loop
load_configuration()
main()

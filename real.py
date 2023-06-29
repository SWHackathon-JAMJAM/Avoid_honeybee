from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import random
import time

app = Flask(__name__)

# Initialize MediaPipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Load the ant image
ant_image = cv2.imread('ant.jpg')

# Randomly generate initial point coordinates
point_x = random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - ant_image.shape[1])
point_y = 0

# Variables for moving the point
move_speed_x = 1
move_speed_y = 1

points = []

def remove_touching_points(rect_x, rect_y, rect_width, rect_height):
    global points
    points_to_remove = []
    for point in points:
        point_x, point_y, point_color = point
        if rect_x <= point_x <= rect_x + rect_width and rect_y <= point_y <= rect_y + rect_height:
            points_to_remove.append(point)
    for point in points_to_remove:
        points.remove(point)

def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def generate_frames():
    global point_x, point_y, move_speed_x, move_speed_y
    distance = 1
    prev_finger_bend = [False, False, False, False, False]  # Initialize previous finger bend status
    finger_bend_count = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
    hand_landmark_points = []
    fingertip_index = 4
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmark_points = []
                for landmark in hand_landmarks.landmark:
                    hand_landmark_points.append(landmark)

                x_min = int(min(hand_landmark_points, key=lambda x: x.x).x * frame.shape[1])
                x_max = int(max(hand_landmark_points, key=lambda x: x.x).x * frame.shape[1])
                y_min = int(min(hand_landmark_points, key=lambda y: y.y).y * frame.shape[0])
                y_max = int(max(hand_landmark_points, key=lambda y: y.y).y * frame.shape[0])

                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                fingertip_x = int(hand_landmark_points[fingertip_index].x * frame.shape[1])
                fingertip_y = int(hand_landmark_points[fingertip_index].y * frame.shape[0])
                rect_width = x_max - x_min
                rect_height = y_max - y_min
                purple_box_width = rect_width // 8
                purple_box_height = rect_height // 8
                purple_box_x = fingertip_x - purple_box_width // 2
                purple_box_y = fingertip_y - purple_box_height // 2
                cv2.rectangle(frame, (purple_box_x, purple_box_y), (purple_box_x + purple_box_width, purple_box_y + purple_box_height), (128, 0, 128), 2)
                cv2.circle(frame, (fingertip_x, fingertip_y), 10, (0, 255, 0), -1)

                if (purple_box_x <= point_x <= purple_box_x + purple_box_width) and (purple_box_y<= point_y <= purple_box_y + purple_box_height):
                    green_ball_x = random.randint(0, frame.shape[1] - 1)
                    green_ball_y = random.randint(0, frame.shape[0] - 1)
                    move_speed_x = (green_ball_x - point_x) / 100
                    move_speed_y = (green_ball_y - point_y) / 100
                    end_time = time.time()
                    count, total, time_total = finger_bend_count[save_number]
                    finger_bend_count[save_number] = (count, total + 1, (time_total + (end_time-start_time)/(total+1)))
                    point_y = 0  # Move the image to y=0

                    fingertip_index = random.choice([4, 8, 12, 16, 20])
                    continue
                    # if distance < rect_width * 2 + 3:
                    # finger_bend_count[save_number] = (finger_bend_count[save_number][0], finger_bend_count[save_number][1] + 1)

                    # finger_bend_count_json = json.dumps(finger_bend_count)


                distance = calculate_distance(point_x, point_y, fingertip_x, fingertip_y)
                if distance <= rect_width * 2:
                    move_speed_x = (fingertip_x - point_x) / 10
                    move_speed_y = (fingertip_y - point_y) / 10


            # Finger bend detection
                thumb_bend = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_TIP].y
                index_bend = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_bend = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_bend = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.RING_FINGER_PIP].y
                pinky_bend = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_PIP].y

                # Finger bend status
                finger_bend = [thumb_bend, index_bend, middle_bend, ring_bend, pinky_bend]

                # for i, bend in enumerate(finger_bend):
                # if bend and not prev_finger_bend[i]:
                    # print(prev_finger_bend)
                    # # if i == finger_bend[i]
                    # finger_bend_count[i] += 1

                # Increase move speed if finger bend count reaches 10
                if all(acc >= 10 for acc, total, time in finger_bend_count):
                    move_speed_y += 10
                    finger_bend_count = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]


                save_number = fingertip_index//4 - 1
                for i, bend in enumerate(finger_bend):
                    if bend and not prev_finger_bend[i] and i == save_number:
                        end_time = time.time()

                        count, total, time_total = finger_bend_count[save_number]
                        # print(count, total, time_total, end_time, start_time, start_time-end_time)
                        finger_bend_count[save_number] = (count+1, total + 1, (time_total + (end_time-start_time)/(total+1)))
                        point_y = 0  # Move the image to y=0
                        fingertip_index = random.choice([4, 8, 12, 16, 20])

                        # start_time = time.time()
                        # while time.time() - start_time < 1:  # 1초 동안 대기
                        #     pass


            prev_finger_bend = finger_bend

        ant_resized = cv2.resize(ant_image, (46, 50))
        frame[int(point_y):int(point_y) + ant_resized.shape[0], int(point_x)-10:int(point_x) -10 + ant_resized.shape[1]] = ant_resized[:frame.shape[0]-int(point_y), :frame.shape[1]-int(point_x)]
        point_x += move_speed_x
        point_y += move_speed_y

        if point_x < 0 or point_x >= frame.shape[1]:
            move_speed_x *= -1
            point_x = max(0, min(point_x, frame.shape[1] - 1))
        if point_y < 0 or point_y >= frame.shape[0]:
            move_speed_y *= -1
            point_y = max(0, min(point_y, frame.shape[0] - 1))

        if point_y == frame.shape[0] - 1:  # Reset point position at the bottom
            point_x = random.randint(0, frame.shape[1] - 1)
            point_y = 0

            # Display finger bend count
        cv2.putText(frame, f"Thumb: {finger_bend_count[0][0]} / {finger_bend_count[0][1]} / time {finger_bend_count[0][2]/(finger_bend_count[0][1]+1):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Index: {finger_bend_count[1][0]} / {finger_bend_count[1][1]} / time {finger_bend_count[1][2]/(finger_bend_count[1][1]+1):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Middle: {finger_bend_count[2][0]} / {finger_bend_count[2][1]} / time {finger_bend_count[2][2]/(finger_bend_count[2][1]+1):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Ring: {finger_bend_count[3][0]} / {finger_bend_count[3][1]} / time {finger_bend_count[3][2]/(finger_bend_count[3][1]+1):.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Pinky: {finger_bend_count[4][0]} / {finger_bend_count[4][1]} / time {finger_bend_count[4][2]/(finger_bend_count[4][1]+1):.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv2.circle(frame, (int(point_x), int(point_y)), 10, (255, 0, 0), -1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

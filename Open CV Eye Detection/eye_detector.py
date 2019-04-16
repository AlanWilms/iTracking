#!/usr/bin/env python

import sys
import cv2
import numpy as np
import math
import threading
import time
import random
from scipy import stats


class EyeTracker:
    # define several class variables
    def __init__(self, webcam_index, calibration_sleep_time, measure_accuracy):
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
        except Exception as e:
            print("Could not load face detector:", e)

        try:
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
        except Exception as e:
            print("Could not load eye detector:", e)

        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            print("Webcam not detected.")

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        print("Video stats: width = {0:.0f} px | height = {1:.0f} px | FPS = {2:.0f} frames per second.".format(
            self.frame_width, self.frame_height, self.frame_rate))

        self.sleep_time = calibration_sleep_time
        self.message = ""
        self.calibration_stage = 0
        self.calibration_stage1_data = []
        self.calibration_stage2_data = []
        self.calibration_stage3_data = []
        self.calibration_stage4_data = []
        self.top = 0
        self.bottom = 0
        self.left = 0
        self.right = 0
        self.measure_accuracy = True if measure_accuracy.lower() == "true" or measure_accuracy.lower() == "1" else False
        self.accuracy_x = self.frame_width // 2
        self.accuracy_y = self.frame_height // 2

        # start calicbration thread
        calibration = threading.Thread(target=self.start_calibration, args=())
        calibration.daemon = True
        calibration.start()

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not frame.data:
                break

            frame = cv2.flip(frame, 1)
            frame = self.process_frames(frame)

            # Display the resulting frame
            cv2.imshow('Webcam', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # takes 30 frames per second. if the user presses any button, it stops from showing the webcam
            if cv2.waitKey(30) >= 0:
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def get_left_eye(self, eyes):
        # safety check
        assert (len(eyes) >= 1), "no eyes detected for tracking!"
        leftmost_x = eyes[0][0]  # x
        leftmost_index = 0
        # print("eye", leftmost_x)
        for i in range(1, len(eyes)):
            # print("eye", eyes[i][0])
            if (eyes[i][0] < leftmost_x):
                leftmost_x = eyes[i][0]
                leftmost_index = i
        # print(leftmost_x)
        return eyes[leftmost_index]

    def process_frames(self, frame):
        grayscale = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # convert image to grayscale and increase contrast
        faces = self.face_cascade.detectMultiScale(
            grayscale, scaleFactor=1.1, minNeighbors=2, flags=0 | cv2.CASCADE_SCALE_IMAGE, minSize=(150, 150))

        # check cannot be "not faces" because using NumPy arrays
        if len(faces) == 0:
            print("no faces detected")
            return self.add_text(frame)

        # print(faces)
        # face = grayscale[faces[0]]

        # detect closest face
        closest_face = None
        closest_size = None
        for face in faces:
            size = face[2] * face[3]  # w * h
            if closest_face is None or size > closest_size:
                closest_face = face
                closest_size = size

        x, y, w, h = closest_face
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Face', (x + w, y + h), font,
                    0.5, (0, 0, 139), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 139), 2)
        # color_face = frame[y:y + h, x:x + w]
        grayscale_face = grayscale[y:y + h, x:x + w]
        eyes = self.eye_cascade.detectMultiScale(
            grayscale_face, scaleFactor=1.1, minNeighbors=2, flags=0 | cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))
        if len(eyes) != 2:
            print("no eyes detected")
            return self.add_text(frame)

        left_eye = self.get_left_eye(eyes)
        for eye in eyes:
            ex, ey, ew, eh = eye
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                          (0, 140, 255) if eye[0] == left_eye[0] else (0, 0, 139), 2)

        lex, ley, lew, leh = left_eye
        left_eye = grayscale_face[ley:ley + leh, lex:lex + lew]
        # increase contrast
        left_eye = cv2.equalizeHist(left_eye)
        # left_eye = cv2.equalizeHist(cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY))

        thres = cv2.inRange(left_eye, 0, 20)
        kernel = np.ones((3, 3), np.uint8)

        # processing to remove small noise
        # originally set to 2 and 3, respectively
        dilation = cv2.dilate(thres, kernel, iterations=2)
        erosion = cv2.erode(dilation, kernel, iterations=3)

        # find contours
        contours, hierarchy = cv2.findContours(
            erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # algorithm to find pupil and reduce noise of other contours by selecting the closest to center
        closest_cx = None
        closest_cy = None
        closest_distance = None
        center = (ley + leh//2, lex + lew//2)
        if len(contours) >= 1:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                for contour in contours:
                    # distance between center and potential pupil
                    distance = math.sqrt(
                        (cy - center[0])**2 + (cx - center[1])**2)
                    if closest_distance is None or distance < closest_distance:
                        closest_cx = cx
                        closest_cy = cy
                        closest_distance = distance

        if closest_cx is not None and closest_cy is not None:
            # base size of pupil to size of eye
            cv2.circle(frame, (x + lex + closest_cx, y + ley + closest_cy),
                       lew//12, (0, 140, 255), -1)
            # process calibration
            percentage_x = closest_cx / lew
            percentage_y = closest_cy / leh
            if self.calibration_stage > 0 and self.calibration_stage < 5:
                # print(self.calibration_stage)
                # print(percentage_x)
                # self.add_calibration_data(
                #     x + lex + closest_cx, y + ley + closest_cy)
                # self.add_calibration_data(closest_cx, closest_cy)
                self.add_calibration_data(percentage_x, percentage_y)
            if self.calibration_stage >= 5:
                radius = 10

                gaze_x = ((percentage_x - self.left) /
                          (self.right - self.left)) * self.frame_width
                gaze_y = ((percentage_y - self.top) /
                          (self.bottom - self.top)) * self.frame_height

                if percentage_x < self.left:
                    gaze_x = 0
                elif percentage_x > self.right:
                    gaze_x = self.frame_width

                if percentage_y < self.top:
                    gaze_y = 0
                elif percentage_y > self.bottom:
                    gaze_y = self.frame_height

                # adjusting from the edges
                if gaze_x <= radius:
                    gaze_x += radius
                elif gaze_x + radius >= self.frame_width:
                    gaze_x -= radius

                if gaze_y <= radius:
                    gaze_y += radius
                elif gaze_y + radius >= self.frame_height:
                    gaze_y -= radius

                cv2.circle(frame, (int(round(gaze_x)), int(
                    round(gaze_y))), radius, (0, 0, 139), -1)

                # add overlay
                frame = self.add_overlay(frame, gaze_x, gaze_y)

        return self.add_text(frame)

    def add_overlay(self, frame, gaze_x, gaze_y):
        overlay = frame.copy()
        mid_x = self.frame_width // 2
        mid_y = self.frame_height // 2
        alpha = 0.3  # Transparency factor.

        cv2.rectangle(overlay, (0 if gaze_x <= mid_x else mid_x, 0 if gaze_y <= mid_y else mid_y), (mid_x if gaze_x <= mid_x else self.frame_width, mid_y if gaze_y <= mid_y else self.frame_height), (0, 200, 0), -1)
        # Overlays transparent rectangle over the image
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def add_text(self, frame):
        if (self.message != ""):
            cv2.putText(frame, self.message,
                        (50, self.frame_height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)

        if (self.calibration_stage > 0 and self.calibration_stage <= 6):
            radius = random.randint(12, 13)
            color = (0, 0, 139)
            offset = 20

            if self.calibration_stage == 1:
                cv2.circle(frame, (offset, offset), radius, color, -1)
            elif self.calibration_stage == 2:
                cv2.circle(frame, (self.frame_width -
                                   offset, offset), radius, color, -1)
            elif self.calibration_stage == 3:
                cv2.circle(frame, (self.frame_width - offset,
                                   self.frame_height - offset), radius, color, -1)
            elif self.calibration_stage == 4:
                cv2.circle(frame, (offset, self.frame_height -
                                   offset), radius, color, -1)
            elif self.calibration_stage == 6:
                cv2.circle(frame, (self.accuracy_x, self.accuracy_y), radius, (139, 0, 0), -1)
                cv2.circle(frame, (self.accuracy_x, self.accuracy_y), radius + 7, (139, 0, 0), 3)
        return frame

    def add_calibration_data(self, x, y):
        if (self.calibration_stage > 0):
            if self.calibration_stage == 1:
                self.calibration_stage1_data.append([x, y])
            elif self.calibration_stage == 2:
                self.calibration_stage2_data.append([x, y])
            elif self.calibration_stage == 3:
                self.calibration_stage3_data.append([x, y])
            else:
                # calibration = 4
                self.calibration_stage4_data.append([x, y])

    def start_calibration(self):
        self.message = "Welcome to the iTracker!"
        time.sleep(self.sleep_time * 1.5)
        self.message = "Starting calibration..."
        time.sleep(self.sleep_time)
        self.message = "Please look at the sequences of four red dots as they appear."
        time.sleep(self.sleep_time)
        for _ in range(4):
            self.calibration_stage = self.calibration_stage + 1
            time.sleep(self.sleep_time)

        # end of calibration
        self.calibration_stage = 5
        self.message = ""
        self.process_calibration_data(True)
        
        if self.measure_accuracy:
            time.sleep(self.sleep_time)
            self.message = "Starting accuracy test..."
            time.sleep(self.sleep_time)
            self.message = "Please follow the moving blue dot on screen."
            self.calibration_stage = 6
            movement_distance = 10 # px
            # time_end = time.time() + 60 # 1 minute
            # previous_direction = None
            # while time.time() < time_end:
            #     direction = random.randint(1, 8)
            #     movement_distance = 10 # px
            #     # print("adding point...")
            #     if previous_direction is None or (direction != previous_direction + 4 and direction + 4 != previous_direction):
            #         if direction == 1:
            #             self.move_accuracy_point_y(self.accuracy_y + movement_distance)
            #         elif direction == 2:
            #             self.move_accuracy_point_y(self.accuracy_y + movement_distance)
            #             self.move_accuracy_point_x(self.accuracy_x + movement_distance)
            #         elif direction == 3:
            #             self.move_accuracy_point_x(self.accuracy_x + movement_distance)
            #         elif direction == 4:
            #             self.move_accuracy_point_y(self.accuracy_y - movement_distance)
            #             self.move_accuracy_point_x(self.accuracy_x + movement_distance)
            #         elif direction == 5:
            #             self.move_accuracy_point_y(self.accuracy_y - movement_distance)
            #         elif direction == 6:
            #             self.move_accuracy_point_y(self.accuracy_y - movement_distance)
            #             self.move_accuracy_point_x(self.accuracy_x - movement_distance)
            #         elif direction == 7:
            #             self.move_accuracy_point_x(self.accuracy_x - movement_distance)
            #         elif direction == 8:
            #             self.move_accuracy_point_y(self.accuracy_y + movement_distance)
            #             self.move_accuracy_point_x(self.accuracy_x - movement_distance)
            #         previous_direction = direction
            #         time.sleep(0.1)
            time_end = time.time() + 3.5
            while time.time() < time_end:
                self.move_accuracy_point_y(self.accuracy_y - movement_distance)
                self.move_accuracy_point_x(self.accuracy_x - movement_distance)
                time.sleep(0.1)
            time_end = time.time() + 6
            while time.time() < time_end:
                self.move_accuracy_point_x(self.accuracy_x + movement_distance)
                time.sleep(0.1)
            time_end = time.time() + 5
            while time.time() < time_end:
                self.move_accuracy_point_y(self.accuracy_y + movement_distance)
                time.sleep(0.1)
            time_end = time.time() + 7
            while time.time() < time_end:
                self.move_accuracy_point_x(self.accuracy_x - movement_distance)
                time.sleep(0.1)
            time_end = time.time() + 3
            while time.time() < time_end:
                self.move_accuracy_point_y(self.accuracy_y - movement_distance)
                self.move_accuracy_point_x(self.accuracy_x + movement_distance)
                time.sleep(0.1)
                
        self.calibration_stage = 7 # done with everything

    def move_accuracy_point_x(self, new_value):
        offset = 20
        self.accuracy_x = new_value if new_value > offset and new_value < self.frame_width - offset else self.accuracy_x

    def move_accuracy_point_y(self, new_value):
        offset = 20
        self.accuracy_y = new_value if new_value > offset and new_value < self.frame_height - offset else self.accuracy_y

    def process_calibration_data(self, remove_outliers=True):
        top_left_x = 0
        top_left_y = 0
        top_right_x = 0
        top_right_y = 0
        bottom_left_x = 0
        bottom_left_y = 0
        bottom_right_x = 0
        bottom_right_y = 0

        # removing outliers will account for times when the iTracking fails and thinks the pupil is on the eye bounding box border  
        if not remove_outliers:
            top_left_x = sum(i[0] for i in self.calibration_stage1_data[len(self.calibration_stage1_data) // 3: len(
                self.calibration_stage1_data) - 1]) / (len(self.calibration_stage1_data) - (len(self.calibration_stage1_data) // 3) - 1)
            top_left_y = sum(i[1] for i in self.calibration_stage1_data[len(self.calibration_stage1_data) // 3: len(
                self.calibration_stage1_data) - 1]) / (len(self.calibration_stage1_data) - (len(self.calibration_stage1_data) // 3) - 1)
            top_right_x = sum(i[0] for i in self.calibration_stage2_data[len(self.calibration_stage2_data) // 3: len(
                self.calibration_stage2_data) - 1]) / (len(self.calibration_stage2_data) - (len(self.calibration_stage2_data) // 3) - 1)
            top_right_y = sum(i[1] for i in self.calibration_stage2_data[len(self.calibration_stage2_data) // 3: len(
                self.calibration_stage2_data) - 1]) / (len(self.calibration_stage2_data) - (len(self.calibration_stage2_data) // 3) - 1)
            bottom_right_x = sum(i[0] for i in self.calibration_stage3_data[len(self.calibration_stage3_data) // 3: len(
                self.calibration_stage3_data) - 1]) / (len(self.calibration_stage3_data) - (len(self.calibration_stage3_data) // 3) - 1)
            bottom_right_y = sum(i[1] for i in self.calibration_stage3_data[len(self.calibration_stage3_data) // 3: len(
                self.calibration_stage3_data) - 1]) / (len(self.calibration_stage3_data) - (len(self.calibration_stage3_data) // 3) - 1)
            bottom_left_x = sum(i[0] for i in self.calibration_stage4_data[len(self.calibration_stage4_data) // 3: len(
                self.calibration_stage4_data) - 1]) / (len(self.calibration_stage4_data) - (len(self.calibration_stage4_data) // 3) - 1)
            bottom_left_y = sum(i[1] for i in self.calibration_stage4_data[len(self.calibration_stage4_data) // 3: len(
                self.calibration_stage4_data) - 1]) / (len(self.calibration_stage4_data) - (len(self.calibration_stage4_data) // 3) - 1)
        else:
            proportion_to_cut = 0.10
            top_left_x, top_left_y = stats.trim_mean(np.array(self.calibration_stage1_data[len(
                self.calibration_stage1_data) // 3: len(self.calibration_stage1_data) - 1]), proportion_to_cut, axis=0)
            top_right_x, top_right_y = stats.trim_mean(np.array(self.calibration_stage2_data[len(
                self.calibration_stage2_data) // 3: len(self.calibration_stage2_data) - 1]), proportion_to_cut, axis=0)
            bottom_right_x, bottom_right_y = stats.trim_mean(np.array(self.calibration_stage3_data[len(
                self.calibration_stage3_data) // 3: len(self.calibration_stage3_data) - 1]), proportion_to_cut, axis=0)
            bottom_left_x, bottom_left_y = stats.trim_mean(np.array(self.calibration_stage4_data[len(
                self.calibration_stage4_data) // 3: len(self.calibration_stage4_data) - 1]), proportion_to_cut, axis=0)

        self.left = (bottom_left_x + top_left_x) / 2
        self.right = (bottom_right_x + top_right_x) / 2
        self.top = (top_left_y + top_right_y) / 2
        self.bottom = (bottom_left_y + bottom_right_y) / 2
        if self.right <= self.left or self.bottom <= self.top:
            raise Exception('Calibration process failed. Please try again')
        print("Left: %f" % self.left)
        print("Right: %f" % self.right)
        print("Top: %f" % self.top)
        print("Bottom: %f" % self.bottom)

# def getEyeball(eye, circles):
if __name__ == "__main__":
    # print('\n'.join(sys.path))
    # looking for one argument
    if len(sys.argv) == 4:
        EyeTracker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    else:
        exit("Missing Webcam index. Run 'python3 eye_detector.py 0 5 true'.")

# RUN via "python3 eye_detector.py 0"

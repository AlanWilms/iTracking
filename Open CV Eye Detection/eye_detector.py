#!/usr/bin/env python

import sys
import cv2
import numpy as np
import math

class EyeTracker:
    # define several class variables
    def __init__(self, webcam_index):
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

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not frame.data:
                break

            frame=cv2.flip(frame,1)
            frame = self.processFrames(frame)

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

    def getLeftEye(self, eyes):
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
    def processFrames(self, frame):
        grayscale = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # convert image to grayscale and increase contrast
        faces = self.face_cascade.detectMultiScale(
            grayscale, scaleFactor=1.1, minNeighbors=2, flags=0 | cv2.CASCADE_SCALE_IMAGE, minSize=(150, 150))
        if len(faces) == 0:
            print("no faces detected")
            return frame

        # print(faces)
        # face = grayscale[faces[0]]

        # detect closest face
        closest_face = None
        closest_size = None
        for face in faces:
            size = face[2] * face[3] # w * h
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
            return frame

        left_eye = self.getLeftEye(eyes)
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
                    distance = math.sqrt((cy - center[0])**2 + (cx - center[1])**2)
                    if closest_distance == None or distance < closest_distance:
                        closest_cx = cx
                        closest_cy = cy
                        closest_distance = distance

        if closest_cx is not None and closest_cy is not None:
            # base size of pupil to size of eye
            cv2.line(frame, (x + lex + closest_cx, y + ley + closest_cy), (x + lex + closest_cx, y + ley + closest_cy), (0, 140, 255), lew//7)

        return frame
        # return grayscale


# def getEyeball(eye, circles):
if __name__ == "__main__":
    # print('\n'.join(sys.path))
    # looking for one argument
    if len(sys.argv) == 2:
        EyeTracker(int(sys.argv[1]))
    else:
        exit("Missing Webcam index. Run 'python3 eye_detector.py 0'.")

# RUN via "python3 eye_detector.py 0"

# iTracking
Senior Design Project

original path of repo/folder: ~/AIY-projects-python/src/examples/vision

Instructions:
```
sudo systemctl stop joy_detection_demo
cd ~/AIY-projects-python/src/examples/vision
./closest_face_detection_camera.py
```

Currently, this will only highlight the closest face to the camera and dictate which direction for the camera to move (from the perspective of the camera looking at the subject in front of it) in order for the face to be centered (with a 10% buffer).

Example output:
```
Iteration #98: num_faces=1
Don't move camera
Iteration #99: num_faces=0
Iteration #100: num_faces=1
Move camera down
Iteration #101: num_faces=1
Move camera down
Iteration #102: num_faces=1
Move camera down
```

## Stepper Mottors
Instructions:
```
sudo systemctl stop joy_detection_demo
cd ~/AIY-projects-python/src/examples/vision/gpiozero
sudo python3 teset_stepper2.py
```

## Resources:
* [General procedure](http://stanford.edu/class/ee267/Spring2018/report_griffin_ramirez.pdf)
* [Eye classification](https://arxiv.org/pdf/1605.05258.pdf)
* [Servo example](https://github.com/google/aiyprojects-raspbian/blob/aiyprojects/src/examples/gpiozero/servo_example.py)
* [GPIO expansion pins](https://aiyprojects.withgoogle.com/vision/#makers-guide--gpio-expansion-pins)
* [PiCamera start_preview()](https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.start_preview)

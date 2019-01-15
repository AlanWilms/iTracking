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

## Resources:
https://pinout.xyz/pinout/aiy_vision_bonnet

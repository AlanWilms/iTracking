#!/usr/bin/env python3
"""Demonstrates simultaneous control of two servos on the hat.

One servo uses the simple default configuration, the other servo is tuned to
ensure the full range is reachable.
"""

from time import sleep
from gpiozero import Motor
from aiy.vision.pins import (PIN_A, PIN_B, PIN_C, PIN_D)

motor1 = Motor(PIN_A, PIN_C)
motor2 = Motor(PIN_B, PIN_D)

# Move the Servos back and forth until the user terminates the example.
# while True:
# 	motor1.forward()
#	sleep(10)
#	motor1.backward()
#	sleep(10)
#	motor2.forward()
#	sleep(1)
#	motor2.backward()

halfstep_seq = [
	[1,0,0,0],
	[1,1,0,0],
	[0,1,0,0],
	[0,1,1,0],
	[0,0,1,0],
	[0,0,1,1],
	[0,0,0,1],
	[1,0,0,1]
]

for i in range(512):
	for halfstep in range(8):
		for pin in range(4):
			if pin == 0:
				if halfstep_seq[halfstep][pin] == 1:
					motor1.forward()
				else:
					motor1.stop()
			if pin == 1:
				if halfstep_seq[halfstep][pin] == 1:
					motor2.forward()
				else:
					motor2.stop()

			if pin == 2:
				if halfstep_seq[halfstep][pin] == 1:
					motor1.backward()
				else:
					motor1.stop()
			
			if pin == 3:
				if halfstep_seq[halfstep][pin] == 1:
					motor2.backward()
				else:
					motor2.stop()	
			sleep(0.01)
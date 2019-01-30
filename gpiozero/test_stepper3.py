#!/usr/bin/env python3
"""Demonstrates simultaneous control of two servos on the hat.

One servo uses the simple default configuration, the other servo is tuned to
ensure the full range is reachable.
"""

from time import sleep
from gpiozero import Motor
from gpiozero import LED
from aiy.vision.pins import (PIN_A, PIN_B, PIN_C, PIN_D)

LED1 = LED(PIN_A)
LED2 = LED(PIN_C)
LED3 = LED(PIN_B)
LED4 = LED(PIN_D)

# motor1 = Motor(PIN_A, PIN_C)
# motor2 = Motor(PIN_B, PIN_D)

# motor1 = Motor(PIN_A, PIN_B)
# motor2 = Motor(PIN_C, PIN_D)

# Move the Servos back and forth until the user terminates the example.
# while True:
# 	motor1.forward()
#	sleep(10)
#	motor1.backward()
#	sleep(10)
#	motor2.forward()
#	sleep(1)
#       motor2.backward()

# doing the one-phase-on driving because the two-phase is noticably slower

halfstep_seq = [
	[1,0,0,0],
#	[1,1,0,0],
	[0,1,0,0],
#	[0,1,1,0],
	[0,0,1,0],
#	[0,0,1,1],
	[0,0,0,1],
#	[1,0,0,1]
]

while True:
#    	for halfstep in range(8):
	for halfstep in range(4):
		for pin in range(4):
			if pin == 0:
				if halfstep_seq[halfstep][pin] == 1:
					LED1.on()
				else:
					LED1.off()
			elif pin == 1:
				if halfstep_seq[halfstep][pin] == 1:
					LED3.on()
				else:
					LED3.off()

			elif pin == 2:
				if halfstep_seq[halfstep][pin] == 1:
					LED2.on()
				else:
					LED2.off()
			
			elif pin == 3:
				if halfstep_seq[halfstep][pin] == 1:
					LED4.on()
				else:
					LED4.off()	
			# sleep(0.0000001)

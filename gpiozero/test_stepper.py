#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time
#from aiy.vision.pins import PIN_A
from aiy.vision.pins import (PIN_A, PIN_B, PIN_C, PIN_D)

GPIO.setmode(GPIO.BOARD)

control_pins = [PIN_A,PIN_B,PIN_C,PIN_D]

for pin in control_pins:
	GPIO.setup(pin, GPIO.OUT)
	GPIO.output(pin,0)

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
			GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
		time.sleep(0.001)

GPIO.cleanup()



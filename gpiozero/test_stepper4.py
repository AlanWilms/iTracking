#!/usr/bin/env python3
"""
Author: Alan Wilms
2/5/2019
Script to run the servo motors
Inspiration from: https://www.hackster.io/mjrobot/playing-with-electronics-rpi-gpio-zero-library-tutorial-f984c9
"""

import time
import sys
from gpiozero import OutputDevice as Stepper
from aiy.vision.pins import (PIN_A, PIN_B, PIN_C, PIN_D)

IN1 = Stepper(PIN_A)
IN2 = Stepper(PIN_B)
IN3 = Stepper(PIN_C)
IN4 = Stepper(PIN_D)

pins = [IN1, IN2, IN3, IN4] 					# Motor GPIO pins</p><p>
dir = -1        								# Set to 1 for clockwise
                       							# Set to -1 for anti-clockwise
high_speed = True          						# mode = 1: Low Speed ==> Higher Power
                           						# mode = 0: High Speed ==> Lower Power
if not high_speed:              						# Low Speed ==> High Power
  seq = [[1,0,0,1],
             [1,0,0,0],
             [1,1,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,0,1,0],
             [0,0,1,1],
             [0,0,0,1]]
else:                    						# High Speed ==> Low Power
  seq = [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]]

seq_len = len(seq)
if len(sys.argv)>1: 							# [Optional] read wait time from command line
  wait_time = int(sys.argv[1])/float(1000)
else:
  wait_time = 0.002    							# Manually optimized via tests

step_counter = 0

while True:
  for pin in range(0, 4):
    current_pin = pins[pin]          			# Get GPIO
    if seq[step_counter][pin]!= 0:
      current_pin.on()
    else:
      current_pin.off()

  step_counter = (step_counter + dir) % seq_len
  time.sleep(waitTime)     						# Wait before moving on

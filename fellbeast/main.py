from djitellopy import Tello
import time

tello = Tello()
tello.connect()

tello.for_back_velocity = 0
tello.left_right_velocity = 0
tello.up_down_velocity = 0
tello.yaw_velocity = 0
tello.speed = 0

print(tello.get_battery())

tello.takeoff()
time.sleep(8)
tello.rotate_clockwise(90)
time.sleep(3)
tello.move_left(35)
time.sleep(3)
tello.land()

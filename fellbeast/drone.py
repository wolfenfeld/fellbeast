import time

from djitellopy import Tello

from fellbeast.camera import Camera


class Drone(object):
    def __init__(self):
        self.tello = Tello()
        self.camera = Camera()

    def reset_speed(self):
        self.tello.for_back_velocity = 0
        self.tello.left_right_velocity = 0
        self.tello.up_down_velocity = 0
        self.tello.yaw_velocity = 0
        self.tello.speed = 0

    def connect(self, camera=False):
        self.tello.connect()
        print(f'Battery level: {self.battery_level}')
        if camera:
            self.tello.streamoff()
            self.tello.streamon()

    @property
    def battery_level(self):
        return self.tello.get_battery()

    @staticmethod
    def wait(t):
        time.sleep(t)

    def takeoff(self):
        self.tello.takeoff()

    def rotate_clockwise(self, angle):
        self.tello.rotate_clockwise(angle)

    def move_left(self, x):
        self.tello.move_left(x)

    def land(self):
        self.tello.land()
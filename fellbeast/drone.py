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
    def wait(seconds):
        time.sleep(seconds)

    def takeoff(self):
        self.tello.takeoff()

    def land(self):
        self.tello.land()

    def rotate_clockwise(self, angle):
        self.tello.rotate_clockwise(angle)

    def rotate_counter_clockwise(self, angle):
        self.tello.rotate_counter_clockwise(angle)

    def move_left(self, centimeters):
        self.tello.move_left(centimeters)

    def move_right(self, centimeters):
        self.tello.move_right(centimeters)

    def move_back(self, centimeters):
        self.tello.move_back(centimeters)

    def move_forward(self, centimeters):
        self.tello.move_forward(centimeters)

    def move_up(self, centimeters):
        self.tello.move_up(centimeters)

    def move_down(self, centimeters):
        self.tello.move_down(centimeters)

    def flip_left(self):
        self.tello.flip_left()
import time

from djitellopy import Tello

from fellbeast.camera import Camera
from fellbeast.controllers import PID


class Drone(object):
    camera_screen_width = 320
    camera_screen_height = 240
    relevant_object_target_size = 20

    def __init__(self):
        self.tello = Tello()
        self.camera = Camera(self.tello)
        self.yaw_controller = PID(0.4, 0, 0.4)
        self.up_down_controller = PID(0.4, 0, 0.4)
        self.forward_backward_controller = PID(0.4, 0, 0.4)

    def reset_speed(self):
        self.tello.forward_backward_velocity = 0
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

    def get_control_velocity_action(self, object_location, object_size, previous_error):
        yaw_error = (object_location['x'] - self.camera_screen_width) / 2
        up_down_error = (object_location['y'] - self.camera_screen_height) / 2
        forward_backward_error = object_size - self.relevant_object_target_size

        yaw_control = self.yaw_controller.get_action(input_p=yaw_error,
                                                     input_d=yaw_error - previous_error['yaw'],
                                                     input_i=0)

        up_down_control = self.up_down_controller.get_action(input_p=up_down_error,
                                                             input_d=up_down_error - previous_error['up_down'],
                                                             input_i=0)

        forward_backward_control = self.forward_backward_controller.get_action(
            input_p=forward_backward_error,
            input_d=forward_backward_error - previous_error['forward_backward'],
            input_i=0)

        control_action = {'yaw': yaw_control, 'up_down': up_down_control, 'forward_backward': forward_backward_control}
        error = {'yaw': yaw_error, 'up_down': up_down_error, 'forward_backward': forward_backward_error}
        return control_action, error

    def update_speed(self, control_action: dict):

        left_right_velocity = self.tello.left_right_velocity
        forward_backward_velocity = control_action.get('forward_backward', self.tello.forward_backward_velocity)
        up_down_velocity = control_action.get('up_down', self.tello.up_down_velocity)
        yaw_velocity = control_action.get('yaw', self.tello.yaw_velocity)

        self.tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)

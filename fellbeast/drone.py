import time
import numpy as np
import logging


from djitellopy import Tello

from fellbeast.camera import Camera
from fellbeast.controllers import PID

RESET_FREQUENCY = 50
CHECK_FOR_NEW_FACE_FREQUENCY = 25

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)


class Drone(object):
    camera_screen_width = 640
    camera_screen_height = 480
    relevant_object_target_size = 40000
    control_clipping_value = 100

    def __init__(self, known_face_path, mode='DEV'):
        self.logger = logging.getLogger('Drone Logger')
        self.logger.setLevel(logging.DEBUG)
        self.mode = mode
        self.logger.info(f'Initializing drone on {self.mode} mode')
        self.tello = Tello()
        self.logger.info('Initializing drone camera')
        self.camera = Camera(self.tello,
                             width=self.camera_screen_width,
                             height=self.camera_screen_height,
                             known_face_path=known_face_path)

        self.logger.info('Initializing drone controllers')
        self.yaw_controller = PID(0.3, 0, 0.3)
        self.up_down_controller = PID(0.2, 0, 0.2)
        self.forward_backward_controller = PID(0.2, 0, 0.2)

    def reset_speed(self):
        self.logger.info('Resetting speeds')

        self.tello.forward_backward_velocity = 0
        self.tello.left_right_velocity = 0
        self.tello.up_down_velocity = 0
        self.tello.yaw_velocity = 0
        self.tello.speed = 0

    def connect(self, camera=True):
        self.logger.info('Connecting to drone')
        self.tello.connect()
        self.logger.info(f'Battery level: {self.battery_level}')
        if camera:
            self.logger.info('Closing and opening stream')
            self.tello.streamoff()
            self.tello.streamon()

    @property
    def battery_level(self):
        return self.tello.get_battery()

    @staticmethod
    def wait(seconds):
        time.sleep(seconds)

    def takeoff(self):
        tries = 5
        current_try = 0
        while current_try < tries:
            result = self.tello.takeoff()
            if result:
                self.logger.info('Liftoff successful')
                return True
            else:
                self.logger.info(f'Failed liftoff attempt ({current_try})')
                current_try += 1
                self.wait(2)
        return False

    def land(self):
        self.logger.info('Landing drone')
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

    def get_control_velocity_action(self, object_location, object_size):
        self.logger.debug('Getting control actions')
        yaw_error = object_location['x'] - self.camera_screen_width / 2
        up_down_error = (object_location['y'] - self.camera_screen_height) / 2
        forward_backward_error = -(object_size - self.relevant_object_target_size)/\
                                 np.sqrt(self.camera_screen_height * self.camera_screen_width)

        yaw_control = self.yaw_controller.get_action(error=yaw_error)
        yaw_control = int(np.clip(yaw_control, -self.control_clipping_value, self.control_clipping_value))

        up_down_control = self.up_down_controller.get_action(error=up_down_error)
        up_down_control = int(np.clip(up_down_control, -self.control_clipping_value, self.control_clipping_value))

        forward_backward_control = self.forward_backward_controller.get_action(error=forward_backward_error)
        forward_backward_control = int(np.clip(forward_backward_control,
                                               -self.control_clipping_value, self.control_clipping_value))

        control_action = {'yaw': yaw_control, 'up_down': up_down_control, 'forward_backward': forward_backward_control}
        return control_action

    def update_speed(self, control_action: dict):
        if self.mode == 'DEBUG':
            return
        self.logger.info('Updating drone speed')
        left_right_velocity = self.tello.left_right_velocity
        forward_backward_velocity = control_action.get('forward_backward', self.tello.forward_backward_velocity)
        up_down_velocity = control_action.get('up_down', self.tello.up_down_velocity)
        yaw_velocity = control_action.get('yaw', self.tello.yaw_velocity)

        self.tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)

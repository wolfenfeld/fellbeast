from fellbeast.drone import Drone

drone = Drone()
drone.connect(camera=False)
drone.reset_speed()

drone.takeoff()
drone.wait(3)
drone.rotate_clockwise(90)
drone.wait(3)
drone.move_left(35)
drone.wait(3)
drone.flip_left()
drone.wait(3)
drone.land()

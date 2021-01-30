
import cv2

from fellbeast.drone import Drone
from fellbeast.routines import follow_person

import queue
import threading


q = queue.Queue()


def display():
    print("Start Displaying")

    while True:
        if not q.empty():
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    drone = Drone(known_face_path='../face_db/')
    drone.connect(camera=True)
    drone.wait(7)
    for i in range(20):
        print(i)
        test_image = drone.camera.read()
        if len(test_image) > 0:
            break
        else:
            drone.wait(3)
    drone.reset_speed()
    drone_took_off = drone.takeoff()

    if drone_took_off:
        follow_person('Amit', drone, q)


if __name__ == '__main__':

    p1 = threading.Thread(target=main)
    p1.start()
    display()

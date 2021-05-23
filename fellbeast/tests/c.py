import cv2
from queue import Queue
import threading
from fellbeast.drone import Drone
from fellbeast.routines import follow_person

q = Queue()


def display():
    print("Start Displaying")

    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    drone = Drone(known_face_path='../face_db/')
    drone.connect(camera=True)
    drone.reset_speed()
    follow_person('Amit', drone, q)

    drone.camera.drone_camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cv2.namedWindow("detected_face")

    p1 = threading.Thread(target=main)
    p1.start()
    print("Start Displaying")

    display()
import cv2
from queue import Queue
import threading

from fellbeast.camera import Camera
from fellbeast.drone import Drone
from fellbeast.routines import follow_person

q = Queue()


def display():
    print("Start Displaying")
    cv2.namedWindow("detected_face")
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main():

    drone = Drone(known_face_path='../face_db/', mode='DEBUG')
    video_input = './data/videos/test6.mov'

    drone.camera = Camera(drone_camera=cv2.VideoCapture(video_input), known_face_path='../face_db')

    follow_person('Tal', drone, frame_q=q)
    drone.camera.drone_camera.cap.release()


if __name__ == '__main__':

    p1 = threading.Thread(target=main)
    p1.start()
    print("Start Displaying")
    display()

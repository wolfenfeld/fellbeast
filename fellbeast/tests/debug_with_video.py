import cv2
from queue import Queue
import threading

from fellbeast.camera import Camera
from fellbeast.drone import Drone
from fellbeast.routines import follow_person

q = Queue()

from multiprocessing import Process, Pipe


def display_pipe(p_output):
    print("Start Displaying")
    cv2.namedWindow("detected_face")
    while True:
        frame = p_output.recv()

        cv2.imshow("detected_face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def display():
    print("Start Displaying")
    cv2.namedWindow("detected_face")
    while True:
        if not q.empty():
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main(p_input):

    drone = Drone(known_face_path='../face_db/', mode='DEBUG')
    video_input = './data/videos/test6.mov'
    video_input = 2
    drone.camera = Camera(drone_camera=cv2.VideoCapture(video_input), known_face_path='../face_db')

    follow_person('Amit', drone, frame_q=None, p_input=p_input)
    drone.camera.drone_camera.cap.release()


if __name__ == '__main__':

    p_output, p_input = Pipe()

    main_process = Process(target=main, args=(p_input,))
    main_process.daemon = True
    main_process.start()

    display_pipe(p_output)

    # p1 = threading.Thread(target=main)
    # p1.start()
    # print("Start Displaying")
    # display()

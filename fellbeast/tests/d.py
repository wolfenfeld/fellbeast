import cv2
import queue
import threading

from fellbeast.camera import Camera
from fellbeast.drone import Drone
from fellbeast.routines import follow_person, track_multiple_faces

cv2.namedWindow("detected_face")

drone = Drone(known_face_path='../face_db/')
video_input = './data/videos/test6.mov'

drone.camera = Camera(drone_camera=cv2.VideoCapture(video_input), known_face_path='../face_db')

track_multiple_faces(drone.camera)
drone.camera.drone_camera.cap.release()
# cv2.destroyAllWindows()


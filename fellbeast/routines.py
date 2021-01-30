import cv2
from queue import Queue

from fellbeast.drone import Drone
from fellbeast.object_tracker import MultipleObjectTracker
from fellbeast.utils import add_text_to_frame, annotate_frame, add_rectangle_to_frame, \
    add_center_of_bounding_box_to_frame

STEP_SIZE = 60


def track_multiple_faces(drone: Drone):
    # cv2.namedWindow("detected_face")
    frame_number = 0

    faces_tracker = MultipleObjectTracker(logger=drone.logger)
    while True:
        frame = drone.camera.read()
        if frame is None:
            break
        faces_data = faces_tracker.track_faces(frame=frame, frame_number=frame_number, camera=drone.camera)

        frame_number += 1

        # plotting the bounding boxes
        for face_data in faces_data.values():
            frame = add_rectangle_to_frame(frame, face_data['bounding_box'])
            frame = add_center_of_bounding_box_to_frame(frame, face_data['bounding_box'])

            text = f"{face_data['name']} \n"
            left_top = face_data['bounding_box'].left, face_data['bounding_box'].top
            add_text_to_frame(frame, text=text, position=left_top)

        cv2.imshow('detected_face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def follow_person(person: str, drone: Drone, frame_q: Queue):
    drone.logger.info('Initializing Follow Person Routine')
    if person not in drone.camera.face_recognition.encoded_known_faces.keys():
        raise ValueError('Unknown person')

    frame_number = 0

    faces_tracker = MultipleObjectTracker(logger=drone.logger)
    while True:
        frame = drone.camera.read()
        frame_number += 1
        if frame is None:
            break

        faces_data = faces_tracker.track_faces(frame=frame, frame_number=frame_number, camera=drone.camera)
        person_data = [face_data for face_data in faces_data.values()
                       if face_data['name'] == person]
        if len(person_data) == 0:
            continue
            # add scan room

        else:
            person_data = person_data[0]
        object_location = {'x': person_data['bounding_box'].bounding_box_center[1],
                           'y': person_data['bounding_box'].bounding_box_center[0]}

        object_size = person_data['bounding_box'].bounding_box_area

        control_action = drone.get_control_velocity_action(object_location, object_size)

        drone.update_speed(control_action)
        frame = annotate_frame(frame, faces_data, control_action)

        frame_q.put(frame)


def circular_scan_for_person(person: str, drone: Drone, frame_q: Queue, number_of_cycles: int):
    drone.logger.info('Initializing Scan Person Routine')
    for cycle in range(number_of_cycles):
        number_of_steps = 360 // STEP_SIZE + 1
        for step in range(number_of_steps):
            frame = drone.camera.read()
            bounding_boxes = drone.camera.face_detector.detect(frame, method='deepface')
            for bounding_box in bounding_boxes:
                detected_person = drone.camera.face_recognition.find_face_in_encodings(frame, bounding_box)
                if detected_person == person:
                    frame_q.put(frame)
                    return bounding_box
                else:
                    drone.rotate_counter_clockwise(angle=STEP_SIZE)
                    frame_q.put(frame)

    return False

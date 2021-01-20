import cv2
from fellbeast.drone import Drone
from fellbeast.object_tracker import MultipleObjectTracker


def track_multiple_faces(camera):
    cv2.namedWindow("detected_face")
    frame_number = 0

    faces_tracker = MultipleObjectTracker()
    while True:
        frame = camera.read()
        if frame is None:
            break
        faces_bounding_boxes, names = faces_tracker.track_faces(frame=frame, frame_number=frame_number,
                                                                camera=camera)
        frame_number += 1

        # plotting the bounding boxes
        for face_bounding_box in faces_bounding_boxes:
            top_left, bottom_right = face_bounding_box.rectangle_coordinates
            frame = cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            frame = cv2.putText(img=frame, text=str(names[face_bounding_box.bounding_box_center]),
                                org=top_left,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 0, 0), thickness=2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('detected_face', frame)


def follow_person(person, drone: Drone):
    if person not in drone.camera.face_recognition.encoded_known_faces.keys():
        raise ValueError('Unknown person')

    previous_error = {'yaw': 0, 'up_down': 0, 'forward_backward': 0}
    cv2.namedWindow("detected_face")
    frame_number = 0

    faces_tracker = MultipleObjectTracker()
    while True:
        frame = drone.camera.read()
        if frame is None:
            break
        faces_data = faces_tracker.track_faces(frame=frame, frame_number=frame_number, camera=drone.camera)
        frame_number += 1
        person_data = [face_data for face_data in faces_data.values()
                       if face_data['name'] == person]
        if len(person_data) == 0:
            continue
            # add scan room

        else:
            person_data = person_data[0]
        object_location = {'x':person_data['bounding_box'].bounding_box_center[1],
                           'y':person_data['bounding_box'].bounding_box_center[0]}
        object_size = person_data['bounding_box'].bounding_box_area

        previous_error['up_down'] = 0
        # previous_error['forward_backward'] = 0

        control_action, previous_error = drone.get_control_velocity_action(object_location, object_size, previous_error)
        # drone.update_speed(control_action)

        # plotting the bounding boxes
        for face_data in faces_data.values():
            top_left, bottom_right = face_data['bounding_box'].rectangle_coordinates
            frame = cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            dot_center = face_data['bounding_box'].bounding_box_center
            frame = cv2.circle(frame,
                               center=(dot_center[1], dot_center[0]),
                               radius=5,
                               color=(0, 255, 0),thickness= -1)
            text = f"{face_data['name']} /n " \
                   f"yaw: {control_action['yaw']} /n " \
                   f"forward_backward: {control_action['forward_backward']}"
            frame = cv2.putText(img=frame,
                                text=text,
                                org=top_left,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 0, 0), thickness=2)

        if cv2.waitKey(1) == 27:
            drone.land()
            break
        cv2.imshow('detected_face', frame)

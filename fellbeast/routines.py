import cv2
from fellbeast.configurations import CHECK_FOR_NEW_FACE_FREQUENCY
from fellbeast.object_tracker import MultipleObjectTracker
from fellbeast.utils import get_closest_coordinate


def track_multiple_faces(camera):
    cv2.namedWindow("detected_face")
    frame_number = 0
    lost_tracking = True

    while True:
        frame = camera.read()
        if frame is None:
            break

        frame_number += 1

        scan_for_new_faces = frame_number % CHECK_FOR_NEW_FACE_FREQUENCY == 0

        reset_detection = lost_tracking
        # Initial face detection
        if reset_detection:
            # Getting all faces and trying to recognise them
            faces_bounding_boxes = camera.face_detector.detect(frame, method='deepface')

            # If no faces where detected continuing to the next frame
            if len(faces_bounding_boxes) == 0:
                continue

            names = {face_bounding_box.bounding_box_center:
                     camera.face_recognition.find_face_in_encodings(image=frame, face_bounding_box=face_bounding_box)
                     for face_bounding_box in faces_bounding_boxes}

            faces_tracker = MultipleObjectTracker()
            faces_tracker.init(frame, bounding_boxes=faces_bounding_boxes)

            lost_tracking = False

        # Periodic scanning for new faces
        elif scan_for_new_faces:
            new_faces_bounding_box = camera.face_detector.detect(frame, method='deepface')

            # If there are new faces setting the lost_tracking indicator to True
            if len(new_faces_bounding_box) > len(faces_bounding_boxes):
                lost_tracking = True
                continue

        # Updating tracker with new frame
        else:
            (success, faces_bounding_boxes) = faces_tracker.update_tracker(frame)

            faces_coordinates = [face_bounding_box.bounding_box_center for face_bounding_box in faces_bounding_boxes]
            old_coordinates = list(names.keys())
            updated_names = {face_coordinates: names[get_closest_coordinate(face_coordinates, old_coordinates)]
                             for face_coordinates in faces_coordinates}

            names = updated_names

            if not success:
                lost_tracking = True
                continue
            else:
                lost_tracking = False

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

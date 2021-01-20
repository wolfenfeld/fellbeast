from typing import List

import cv2

from fellbeast.bounding_box import BoundingBox
from fellbeast.configurations import CHECK_FOR_NEW_FACE_FREQUENCY
from fellbeast.utils import get_closest_coordinate


class BaseObjectTracker(object):
    bounding_box = None

    def init(self, _, bounding_box):
        raise NotImplementedError

    def update_tracker(self, _):
        raise NotImplementedError


class ObjectTracker(BaseObjectTracker):
    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()

    def init(self, frame, bounding_box):
        self.tracker.init(frame, bounding_box.tracker_format)

    def update_tracker(self, frame):
        success, bounding_box = self.tracker.update(frame)
        return success, BoundingBox.from_tracker(*bounding_box.astype(int))


class MultipleObjectTracker(BaseObjectTracker):
    def __init__(self):
        self.tracker = None
        self.lost_tracking = True
        self.bounding_boxes = list()
        self.objects_data = dict()

    def init(self, frame, bounding_boxes: List[BoundingBox]):
        self.tracker = cv2.MultiTracker_create()
        self.bounding_boxes = bounding_boxes
        for bounding_box in bounding_boxes:
            self.tracker.add(cv2.TrackerCSRT_create(), frame, bounding_box.tracker_format)

    def update_tracker(self, frame):
        success, tracker_bounding_boxes = self.tracker.update(frame)
        bounding_boxes = [BoundingBox.from_tracker(*bounding_box.astype(int))
                          for bounding_box in tracker_bounding_boxes]
        self.bounding_boxes = bounding_boxes

        return success, bounding_boxes

    def track_faces(self, frame, frame_number, camera):

        scan_for_new_faces = frame_number % CHECK_FOR_NEW_FACE_FREQUENCY == 0

        # Initial face detection
        if self.lost_tracking:
            # Getting all faces and trying to recognise them
            self.bounding_boxes = camera.face_detector.detect(frame, method='deepface')

            # If faces where detected they are recognized
            if len(self.bounding_boxes) > 0:
                self.objects_data = {face_bounding_box.bounding_box_center: {
                    'name': camera.face_recognition.find_face_in_encodings(image=frame,
                                                                           face_bounding_box=face_bounding_box),
                    'bounding_box': face_bounding_box}
                    for face_bounding_box in self.bounding_boxes}

                self.init(frame, bounding_boxes=self.bounding_boxes)

                self.lost_tracking = False

        # Periodic scanning for new faces
        elif scan_for_new_faces:
            new_faces_bounding_box = camera.face_detector.detect(frame, method='deepface')

            # If there are new faces setting the lost_tracking indicator to True
            if len(new_faces_bounding_box) > len(self.bounding_boxes):
                self.lost_tracking = True

        # Updating tracker with new frame
        else:
            (success, faces_bounding_boxes) = self.update_tracker(frame)

            old_coordinates = list(self.objects_data.keys())
            updated_objects_data = {face_bounding_box.bounding_box_center: {
                'name': self.objects_data[get_closest_coordinate(face_bounding_box.bounding_box_center,
                                                                 old_coordinates)]['name'],
                'bounding_box': face_bounding_box}
                             for face_bounding_box in faces_bounding_boxes}

            self.objects_data = updated_objects_data
            self.lost_tracking = not success

        return self.objects_data

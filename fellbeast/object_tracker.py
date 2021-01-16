from typing import List

import cv2

from fellbeast.bounding_box import BoundingBox


class BaseObjectTracker(object):
    bounding_box = None

    def init(self, _, bounding_box):
        self.bounding_box = bounding_box

    def update_tracker(self, _):
        return True


class ObjectTracker(object):
    tracker = cv2.TrackerCSRT_create()

    def init(self, frame, bounding_box):
        self.tracker.init(frame, bounding_box.tracker_format)

    def update_tracker(self, frame):
        success, bounding_box = self.tracker.update(frame)
        return success, BoundingBox.from_tracker(*bounding_box.astype(int))


class MultipleObjectTracker(object):
    def __init__(self):
        self.tracker = cv2.MultiTracker_create()

    def init(self, frame, bounding_boxes: List[BoundingBox]):
        for face_bounding_box in bounding_boxes:
            self.tracker.add(cv2.TrackerCSRT_create(), frame, face_bounding_box.tracker_format)

    def update_tracker(self, frame):
        success, faces_bounding_box_from_tracker = self.tracker.update(frame)
        faces_bounding_boxes = [BoundingBox.from_tracker(*bounding_box.astype(int))
                                for bounding_box in faces_bounding_box_from_tracker]
        return success, faces_bounding_boxes

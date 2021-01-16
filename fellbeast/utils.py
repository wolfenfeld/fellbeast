import cv2


def get_closest_coordinate(face_coordinates, old_coordinates):
    dist = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    return min(old_coordinates, key=lambda co: dist(co, face_coordinates))


def add_faces_bounding_boxes(frame, faces_bounding_box, names: dict):
    for bounding_box in faces_bounding_box:
        if bounding_box is not None:
            (x, y, w, h) = [int(v) for v in bounding_box]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            frame = cv2.putText(img=frame, text=str(names[tuple(bounding_box[0:2])]),
                                org=(x, y),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 0, 0), thickness=2)
    return frame


def convert_bounding_box_to_borders(x, y, w, h):
    return y, y + h, x, x + w


def convert_borders_to_bounding_box(top, bottom, left, right):
    return left, top, right - left, bottom - top

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


def add_text_to_frame(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    color = (255, 0, 0)
    thickness = 1
    line_type = cv2.LINE_AA

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    line_height = text_size[1] + 5
    x, y0 = position
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * line_height
        cv2.putText(frame,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    color,
                    thickness,
                    line_type)

    return frame


def annotate_frame(frame, faces_data, control_action):
    for face_data in faces_data.values():
        top_left, bottom_right = face_data['bounding_box'].rectangle_coordinates
        frame = cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        dot_center = face_data['bounding_box'].bounding_box_center
        frame = cv2.circle(frame,
                           center=(dot_center[1], dot_center[0]),
                           radius=5,
                           color=(0, 255, 0),
                           thickness=-1)

        text = f"{face_data['name']} \n" \
               f"Yaw: {control_action['yaw']} \n" \
               f"forward_backward: {control_action['forward_backward']} \n" \
               f"Area: {face_data['bounding_box'].bounding_box_area} \n" \
               f"center x,y: {dot_center[1], dot_center[0]}"
        frame = add_text_to_frame(frame, text=text, position=top_left)
        return frame

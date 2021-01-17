class BoundingBox(object):
    def __init__(self, top, left, bottom=None, right=None, width=None, height=None):
        self.top = top
        self.left = left
        if bottom is None or right is None:
            if width is None or height is None:
                raise ValueError('Both width, height and bottom, right cannot be None')
            else:
                self.width = width
                self.height = height
                self.right = self.left+width
                self.bottom = self.top+height
        else:
            self.bottom = bottom
            self.right = right
            self.width = right-left
            self.height = bottom-top

    @classmethod
    def from_tracker(cls, left, top, width, height):
        return cls(left=left, top=top, width=width, height=height)

    @property
    def tracker_format(self):
        return self.left, self.top, self.width, self.height

    @property
    def css(self):
        return self.top, self.right, self.bottom, self.left

    @property
    def bounding_box_center(self):
        return int(self.top+(self.height/2)), int(self.left+(self.width/2))

    @property
    def rectangle_coordinates(self):
        return (self.left, self.top), (self.right, self.bottom)

    def crop_image(self, image):
        return image[self.top:self.bottom, self.left:self.right]

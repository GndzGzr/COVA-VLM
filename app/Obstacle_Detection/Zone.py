import cv2


class Zone:
    def __init__(self, settings, frame_width, frame_height):
        self.green_roi = None
        self.yellow_roi = None
        self.red_roi = None

        self.width = frame_width
        self.height = frame_height

        self.calculate_zones(settings)

    def calculate_zones(self, settings):
        green_coordinates = settings["green_roi"]
        yellow_coordinates = settings["yellow_roi"]
        red_coordinates = settings["red_roi"]

        multipliers = (self.width, self.height, self.width, self.height)
        green_base = (
            green_coordinates["x1"], green_coordinates["y1"], green_coordinates["x2"], green_coordinates["y2"])
        yellow_base = (
            yellow_coordinates["x1"], yellow_coordinates["y1"], yellow_coordinates["x2"], yellow_coordinates["y2"])
        red_base = (
            red_coordinates["x1"], red_coordinates["y1"], red_coordinates["x2"], red_coordinates["y2"])

        # Convert to actual pixel coordinates using tuples instead of generators
        self.green_roi = tuple(coordinate * multiplier for coordinate, multiplier in zip(green_base, multipliers))
        self.yellow_roi = tuple(coordinate * multiplier for coordinate, multiplier in zip(yellow_base, multipliers))
        self.red_roi = tuple(coordinate * multiplier for coordinate, multiplier in zip(red_base, multipliers))

    def get_bbox_color(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box

        # Red ROI
        rx, ry, rw, rh = self.red_roi
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            return (0, 0, 255)  # Red

        # Yellow ROI
        yx, yy, yw, yh = self.yellow_roi
        if yx <= cx <= yx + yw and yy <= cy <= yy + yh:
            return (0, 255, 255)  # Yellow

        # Green ROI
        gx, gy, gw, gh = self.green_roi
        if gx <= cx <= gx + gw and gy <= cy <= gy + gh:
            return (0, 255, 0)  # Green

        return None  # Outside all ROIs

    def draw_rectangle(self, frame):
        # Draw the ROIs
        cv2.rectangle(frame, (int(self.green_roi[0]), int(self.green_roi[1])),
                     (int(self.green_roi[0] + self.green_roi[2]), int(self.green_roi[1] + self.green_roi[3])), (0, 255, 0), 2)
        cv2.rectangle(frame, (int(self.yellow_roi[0]), int(self.yellow_roi[1])),
                     (int(self.yellow_roi[0] + self.yellow_roi[2]), int(self.yellow_roi[1] + self.yellow_roi[3])), (0, 255, 255), 2)
        cv2.rectangle(frame, (int(self.red_roi[0]), int(self.red_roi[1])),
                     (int(self.red_roi[0] + self.red_roi[2]), int(self.red_roi[1] + self.red_roi[3])), (0, 0, 255), 2)

        return frame

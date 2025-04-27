class DistanceAlgorithm:

    def __init__(self, settings):
        self.class_dimensions_data = settings["class_dimensions"]

        self.class_dimensions = {
            'bench': {'dimension': 'height', 'real_size': self.class_dimensions_data["bench"]},
            'bicycle': {'dimension': 'height', 'real_size': self.class_dimensions_data["bicycle"]},
            'bin': {'dimension': 'height', 'real_size': self.class_dimensions_data["bin"]},
            'bus': {'dimension': 'height', 'real_size': self.class_dimensions_data["bus"]},
            'car': {'dimension': 'height', 'real_size': self.class_dimensions_data["car"]},
            'cone': {'dimension': 'height', 'real_size': self.class_dimensions_data["cone"]},
            'crosswalk': {'dimension': 'width', 'real_size': self.class_dimensions_data["crosswalk"]},
            'door': {'dimension': 'height', 'real_size': self.class_dimensions_data["door"]},
            'fire hydrant': {'dimension': 'height', 'real_size': self.class_dimensions_data["fire-hydrant"]},
            'motorbike': {'dimension': 'height', 'real_size': self.class_dimensions_data["motorbike"]},
            'person': {'dimension': 'height', 'real_size': self.class_dimensions_data["person"]},
            'pole': {'dimension': 'width', 'real_size': self.class_dimensions_data["pole"]},
            'ramp': {'dimension': 'height', 'real_size': self.class_dimensions_data["ramp"]},
            'stairs': {'dimension': 'height', 'real_size': self.class_dimensions_data["stairs"]},
            'stop sign': {'dimension': 'width', 'real_size': self.class_dimensions_data["stop-sign"]},
            'tree': {'dimension': 'height', 'real_size': self.class_dimensions_data["tree"]},
            'tree body': {'dimension': 'width', 'real_size': self.class_dimensions_data["tree-body"]},
            'truck': {'dimension': 'height', 'real_size': self.class_dimensions_data["truck"]}
        }

    def calculate(self, det, class_name):
        class_info = self.class_dimensions.get(class_name, {'dimension': 'width',
                                                            'real_size': 50})
        real_size = class_info['real_size']

        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Convert to integers
        pixel_size = 0
        if class_info['dimension'] == 'width':
            pixel_size = x2 - x1
        elif class_info['dimension'] == 'height':
            pixel_size = y2 - y1

        focal_length = 300
        return (focal_length * real_size) / pixel_size

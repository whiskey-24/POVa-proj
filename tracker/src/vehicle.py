class Vehicle_Detection_gt:
    def __init__(self, id, type):
        self.id = id
        self.type = type
        self.list_of_bboxes = {}

    def add_position_at_time(self, bbox, time):
        self.list_of_bboxes[time] = bbox

class Vehicle_Track:
    def __init__(self, id):
        self.id = id
        self.type = None
        self.trajectory = {}
        self.angles = []
        self.original_ltwh = None
    
    def decode_type(self, bbox):
        x, y, w, h = bbox
        area = w * h
        if area < 200:
            self.type = 'Motorcycle'
        elif area < 1000:
            self.type = 'Car'
        elif area < 2000:
            self.type = 'Taxi'
        elif area < 5000:
            self.type = 'Medium Vehicle'
        elif area < 10000:
            self.type = 'Heavy Vehicle'
        else:
            self.type = 'Bus'

    def add_position_at_time(self, bbox, time):
        self.trajectory[time] = bbox

    def add_angle(self, angle):
        self.angles.append(angle)

    def add_original_ltwh(self, ltwh):
        self.original_ltwh = ltwh
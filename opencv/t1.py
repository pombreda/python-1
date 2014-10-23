import numpy as np
import cv2

class object_t:
    xs,ys = [],[]

    def __init__(self, name):
        self.name = name

class triangle_t(object_t):
    def __init__(self):
        self.name = 'triangle'

class circle_t(object_t):
    def __init__(self):
        self.name = 'circle'

class box_t(object_t):
    def __init__(self):
        self.name = 'box'

def nothing(*arg):
    pass

def process_video():
    triangles = []

    cap = cv2.VideoCapture('heider_simmet.mp4')

    cv2.namedWindow('edge')
    cv2.createTrackbar('thresh1', 'edge', 2000, 5000, nothing)
    cv2.createTrackbar('thresh2', 'edge', 2000, 5000, nothing)

    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        thresh1 = cv2.getTrackbarPos('thresh1', 'edge')
        thresh2 = cv2.getTrackbarPos('thresh2', 'edge')
        edge = cv2.Canny(gray, thresh1, thresh2, apertureSize=5)

        vis = frame.copy()
        vis /= 2
        vis[edge != 0] = (0, 255, 0)

        cv2.imshow('edge', vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_video()

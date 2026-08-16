import cv2
import numpy as np


class BEVTransform:

    def __init__(self, homography_matrix):
        self.H = np.array(homography_matrix, dtype=np.float32)


    @staticmethod
    def bbox_to_ground_point(bbox):
        x1, y1, x2, y2 = bbox

        return ((x1 + x2) / 2, y2)

    def to_bev(self, bbox):
        px, py = self.bbox_to_ground_point(bbox)
        pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), self.H)
        x_lateral, z_forward = pt[0][0]
        return float(x_lateral), float(z_forward)


    @staticmethod
    def calibrate(image_points, word_points):
        """
        One-time Calibration, not per frame due to risk of low computational loss,will fix in future
        """

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(word_points, dtype=np.float32)

        H, _ = cv2.findHomography(src, dst)
        return H


def estimate_placeholder_homography(frame_width, frame_height):
    """
    NOT a real calibration for now, mostly guessing and testing for the future
    """

    bottom = frame_height
    horizon = int(frame_height * 0.55)

    image_points = [
        (frame_width * 0.2, bottom), (frame_width * 0.8, bottom),
        (frame_width * 0.4, horizon), (frame_width * 0.6, horizon),
    ]


    world_points = [
        (-1.75, 1), (1.75, 1),
        (-1.75, 30), (1.75, 30),
    ]

    return BEVTransform.calibrate(image_points=image_points, word_points=world_points)
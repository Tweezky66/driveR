"""JUST DEV FILE FOR NOW, DONT TOUCH"""

import sys
 
import cv2
import numpy as np
 
from perception.BEVTransform import BEVTransform
 
image_points = []
world_points = []
 
 
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        image_points.append((x, y))
        x_lat = float(input(f"  Point at pixel ({x},{y}) -> real x_lateral in meters (+ is right): "))
        z_fwd = float(input(f"  Point at pixel ({x},{y}) -> real z_forward in meters (distance ahead): "))
        world_points.append((x_lat, z_fwd))
        print(f"  recorded {len(image_points)} point(s)\n")
 
 
def main():
    if len(sys.argv) < 2:
        print("Usage: python calibration_tool.py path/to/calibration_frame.jpg")
        return
 
    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f"Could not read image: {sys.argv[1]}")
        return
 
    print(
        "Click at least 4 points on the ground plane whose real-world "
        "position you've measured (lane markings, chalk marks, cone "
        "bases). Press 'q' when done.\n"
    )
 
    cv2.namedWindow("Calibration frame")
    cv2.setMouseCallback("Calibration frame", on_click)
 
    while True:
        display = frame.copy()
        for pt in image_points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        cv2.imshow("Calibration frame", display)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
 
    cv2.destroyAllWindows()
 
    if len(image_points) < 4:
        print(f"Only got {len(image_points)} points - need at least 4. Run again.")
        return
 
    H = BEVTransform.calibrate(image_points, world_points)
    np.save("homography.npy", H)
 
    print("\nSaved homography.npy")
    print("image_points =", image_points)
    print("world_points =", world_points)
    print("\nIn main.py, replace estimate_placeholder_homography(...) with:")
    print("    H = np.load('homography.npy')")
 
 
if __name__ == "__main__":
    main()
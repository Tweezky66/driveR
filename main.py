import cv2
import os
import argparse
import numpy as np
 
from perception.Detector import Detector, DEFAULT_CLASSES
from perception.BEVTransform import BEVTransform, estimate_placeholder_homography
from sim.frame_sources import get_source
from visualization.HUD import HUD
from perception.tracker import Tracker

TEST_VIDEO_PATH = "Datasets/test.mov"
HOMOGRAPHY_PATH = "homography.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["video", "webcam", "carla", "picamera"], default="video",
    help="video (default, uses TEST_VIDEO_PATH), webcam, picamera (Pi only), or carla (needs a running server)",
    )
    args = parser.parse_args()
    

    detector = Detector()
    source = get_source(args.source, video_path=TEST_VIDEO_PATH)
    tracker = Tracker()
    

    if not source.is_opened():
        print(f"Error: could not open video source: {TEST_VIDEO_PATH}")
        return

    ok, first_frame = source.read()

    if not ok:
        print("Error: could not open the first frame")
        return
    frame_h, frame_w = first_frame.shape[:2]

    if args.source == "video":
        source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(
        "Using a PLACEHOLDER homography - distances are rough guesses, not "
        "calibrated. Run BEVTransform.calibrate() with your own measured "
        "points before trusting any distance this prints or draws"
        "Fix in future"
    )

    if os.path.exists(HOMOGRAPHY_PATH):
        H = np.load(HOMOGRAPHY_PATH)
        print(f"Loaded calibrated homography from {HOMOGRAPHY_PATH}")
    else:
        print(
            f"No {HOMOGRAPHY_PATH} found - using an uncalibrated PLACEHOLDER "
            "homography. Distances are rough guesses. Run "
            "'python calibration_tool.py <calibration_frame.jpg>' once, using "
            "a frame from this exact camera mounting, to get a real one."
        )

        H = estimate_placeholder_homography(frame_w, frame_h)

    bev = BEVTransform(H)
    hud = HUD(bev)

    running = True

    while running:
        ok, frame = source.read()

        if not ok:
            print("End of video stream")
            break

        result = detector.predict(frame, classes=DEFAULT_CLASSES)

        detections = detector.extract_detections(result)
        tracked = tracker.update(detections, frame)

        annotated = result.plot()
        cv2.imshow("Raw detections", annotated)

        running = hud.handle_events()
        hud.render(tracked)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    source.close()
    cv2.destroyAllWindows()
    hud.quit()


if __name__ == "__main__":
    main()

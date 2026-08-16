import cv2
 
from perception.Detector import Detector, DEFAULT_CLASSES
from perception.BEVTransform import BEVTransform, estimate_placeholder_homography
from visualization.HUD import HUD

TEST_VIDEO_PATH = "Datasets/test.mov"

def main():
    detector = Detector()
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: could not open video source: {TEST_VIDEO_PATH}")
        return

    ok, first_frame = cap.read()

    if not ok:
        print("Error: could not open the first frame")
        return
    frame_h, frame_w = first_frame.shape[:2]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(
        "Using a PLACEHOLDER homography - distances are rough guesses, not "
        "calibrated. Run BEVTransform.calibrate() with your own measured "
        "points before trusting any distance this prints or draws"
        "Fix in future"
    )

    H = estimate_placeholder_homography(frame_w, frame_h)

    bev = BEVTransform(H)
    hud = HUD(bev)

    running = True

    while running:
        ok, frame = cap.read()

        if not ok:
            print("End of video stream")
            break

        result = detector.predict(frame, classes=DEFAULT_CLASSES)

        detections = detector.extract_detections(result)

        annotated = result.plot()
        cv2.imshow("Raw detections", annotated)

        running = hud.handle_events()
        hud.render(detections)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hud.quit()


if __name__ == "__main__":
    main()

import carla
import queue
import numpy as np
import cv2
from picamera2 import Picamera2


class VideoFileSource:

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        image = self.cap.read()
        return image

    def close(self):
        return self.cap.release()


class WebCamSource:

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        image = self.cap.read()
        return image

    def close(self):
        self.cap.release()


class PiCameraSource:

    def __init__(self, size=(1280, 720)):
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(main={"size": size, "format": "RGB888"})
        self.cam.configure(config)
        self.cam.start()

    def is_opened(self):
        return True


    def read(self):
        image = self.cam.capture_array()
        return True, image

    def close(self):
        self.cam.stop()


class CarlaFrameSource:

    def __init__(self, host="localhost", port=2000, width=1280, height=720):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        bp_lib = self.world.get_blueprint_library()

        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)

        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        self._queue = queue.Queue()
        self.camera.listen(self._queue.put())

    def read(self):
        image = self._queue.get()
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        return arr[:, :, :3]

    def close(self):
        self.camera.stop()
        self.camera.destroy()
        self.vehicle.destroy()

def get_source(name, video_path=None):
    match name:
        case "video":
            return VideoFileSource(video_path)
        case "webcam":
            return WebCamSource(0)
        case "picamera":
            return PiCameraSource()
        case "carla":
            return CarlaFrameSource()
        case _:
            raise ValueError(f"Uknown source {name}")
            
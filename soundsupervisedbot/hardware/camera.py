import numpy as np
import cv2
import pyrealsense2 as rs

class Camera:
    def __init__(self, width = 640, height = 480, connect = False):
        self._width = width
        self._height = height
        if connect:
            self._connect_cam()

    def __enter__(self):
        self._connect_cam()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def _resize(self, img, width = 160, height = 120):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _connect_cam(self):
        self.pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()

        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, 30)

        # Start streaming and align frames
        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        for _ in range(100): # Skip th first few frames - dark/blurry frames
            frames = self.pipeline.wait_for_frames()

        self.hole_filling = rs.hole_filling_filter()
        

    def get_frame(self):

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()  
        aligned_color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not aligned_color_frame:
            raise Exception("ERROR: no new images receieved")
        
        return np.asarray(aligned_color_frame.get_data()), np.asarray(aligned_depth_frame.get_data())

    def release(self):
        self.pipeline.stop()
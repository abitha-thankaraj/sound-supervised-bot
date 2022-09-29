import pickle
import time
import yaml
import cv2
from soundsupervisedbot.hardware.camera import Camera
import os
import datetime
# import imageio
import subprocess


def save_pkl_file(obj, filename, save_dir=None):
    if save_dir:
        filename = os.path.join(save_dir, filename)
    
    with open('{}.pkl'.format(filename), 'wb') as f:
        pickle.dump(obj, f)
    print("Saved {}".format(filename))

def load_pkl_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_from_yaml_file(filename, mode = 'r'):
    with open(filename, mode) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def get_epochs():
    return str(int(time.time()))

def get_datetime():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S-%f")

def stream_video(save=False, filename = None):
    imgs = []
    deps = []

    with Camera(connect=True) as cam:
        while True:
            rgb, dep = cam.get_frame()
            cv2.imshow('RealSense', rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    if save:
        save_pkl_file(imgs, filename + "_raw_imgs")
        save_pkl_file(deps, filename + "_dep")

    return imgs

    

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

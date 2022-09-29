from soundsupervisedbot.hardware.robot import Robot
from soundsupervisedbot.hardware.camera import Camera
from soundsupervisedbot.utils.utils import *
from soundsupervisedbot.hardware.recorders import ImageRecorder, MicRecorder
from soundsupervisedbot.tasks.utils.evaluation import EvaluationHelper
import os
from easydict import EasyDict
import torch


class Task:
    def __init__(self, cfg_file, **kwargs) -> None:
        self.cfg = EasyDict(load_from_yaml_file(cfg_file))
        self._set_action_limits()
        
        kwargs.setdefault('test', False) # Used for testing logic
        kwargs.setdefault('model_dir', None)
        
        if not kwargs['test']:
            self.bot = Robot()
            self.cam = Camera(connect=True)
        
        if kwargs['model_dir']: #Eval mode
            self.evaluator = EvaluationHelper(kwargs['model_dir'])
        
        self.execute_trajectory_sleep_time = 4
        self.reset_sleep_time = 3
            
        
    def sample_primitive(self):
        """ Used for train set. Sample from distribution to collect data
        """
        raise NotImplementedError
    
    def predict_primitive(self, audio):
        data = self.evaluator._apply_transforms(audio)
        
        action = self.evaluator.model(data.unsqueeze(0)) # Batch size of 1

        if self.evaluator.cfg.norm_action:
            action = self.evaluator.action_denormalizer(action)

        print("Predicted action - before clamping {}".format(action))
        action = torch.clamp(action, min = self.action_min, max = self.action_max)
        action = action.squeeze(0).detach().numpy()
        primitive = self._detorchify(action)
        return primitive
    
    def _torchify_primitive(self, primitive):
        raise NotImplementedError
    
    def _detorchify(self, t_params):
        raise NotImplementedError
    
    def _set_action_limits(self):
        raise NotImplementedError
    
    def _generate_trajectory(self, primitive):
        raise NotImplementedError
    
    def execute_trajectory(self, primitive):
        # Do not forget to add sleep here
        raise NotImplementedError
    
    def reset(self):
        # Do not forget to add sleep here
        raise NotImplementedError
    
    def close(self):
        self.bot.close()
        self.cam.release()


def record_task(save_dir, task, primitive, record = True, n_mics =3, sample_rate = 44100, record_force = False):
    
    save_dir = os.path.join(save_dir, get_datetime()+ '_'+ get_git_revision_hash()) #Each data point is saved with time and git hash.

    #Reset bot
    image_recorder = ImageRecorder(cam = task.cam, save_dir = save_dir)
    mic = MicRecorder(output_path = save_dir, channels = n_mics, rate = sample_rate)

    # Reset to start state
    task.reset()
    
    # Start streaming
    mic._start_stream()
    image_recorder.start()

    #Execute trajectory from sampled primitive
    task.execute_trajectory(primitive)
     
    image_recorder.stop()
    mic.stop_recording()

    if record:
        os.makedirs(save_dir, exist_ok=True)
        
        print('Saving trajectory details')
        torch.save(task._torchify_primitive(primitive), os.path.join(save_dir, 'params.pt')) #Actions
        
        mic.write_audio_file() #Audio
        image_recorder.save_video()
        image_recorder.clear()  #Images

        print('Saved trajectory in {}'.format(save_dir))

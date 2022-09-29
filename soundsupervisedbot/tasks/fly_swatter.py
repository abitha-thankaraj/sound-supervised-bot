from collections import namedtuple
import time
import numpy as np
import random
from soundsupervisedbot.tasks.sound_task import Task 
import torch

FlySwatterParams = namedtuple('FlySwatterParams', ['j_vel', 'a'])

class FlySwatterTask(Task):
    def __init__(self, cfg_file = '/home/abitha/projects/sound-supervised-bot/soundsupervisedbot/configs/fly_swatter.yaml', **kwargs) -> None:
        super().__init__(cfg_file, **kwargs)
        # Override default values
        self.execute_trajectory_sleep_time = 2
        self.reset_sleep_time = 1
        
    def sample_primitive(self):
        j_vel = np.array([ 
            random.uniform(self.cfg.joint_params.base[0], self.cfg.joint_params.base[1]),
            random.uniform(self.cfg.joint_params.shoulder[0], self.cfg.joint_params.shoulder[1]),
            0., 0., 0., 0.])
        a = random.uniform(self.cfg.acc[0], self.cfg.acc[1])
        return FlySwatterParams(j_vel, a)
    
    def _torchify_primitive(self, primitive):
        # base, shoulder, acc
        return torch.tensor(np.array([primitive.j_vel[0], primitive.j_vel[1], primitive.a]))
    
    def _detorchify(self, t_params):
        return FlySwatterParams(
            j_vel = np.array([t_params[0], t_params[1], 0., 0., 0., 0.]),
            a = t_params[2] 
        )
    
    def _set_action_limits(self):
        self.action_min = torch.Tensor([self.cfg.joint_params.base[0], self.cfg.joint_params.shoulder[0], self.cfg.acc[0]]) 
        self.action_max = torch.Tensor([self.cfg.joint_params.base[1], self.cfg.joint_params.shoulder[1], self.cfg.acc[1]])
    
        
    def _generate_trajectory(self, primitive):
        traj = np.stack([primitive.j_vel], axis = 0)
        return traj
    
    
    def execute_trajectory(self, primitive):
        traj = self._generate_trajectory(primitive)
        msg = self.bot.prepare_msg_speedj(traj, primitive.a,  None)
        self.bot.send_program(msg)
        time.sleep(self.execute_trajectory_sleep_time)
    
    def reset(self):
        self.bot.movej(self.cfg.RESET_JS, acc=2, vel=2)
        self.bot.movej(self.cfg.HOME, acc=2, vel=2)
        time.sleep(self.reset_sleep_time)

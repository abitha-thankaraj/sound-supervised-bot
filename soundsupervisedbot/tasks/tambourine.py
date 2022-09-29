from soundsupervisedbot.tasks.sound_task import Task
import random
import time
from collections import namedtuple
import numpy as np
import torch
from soundsupervisedbot.tasks.utils.evaluation import round_fn

TambourinePrimitive = namedtuple('TambourinePrimitive', ['elbow_vel', 'acc', 'n'])

class TambourineTask(Task):
    def __init__(self, cfg_file = '/home/abitha/projects/sound-supervised-bot/soundsupervisedbot/configs/tambourine.yaml', **kwargs) -> None:
        super().__init__(cfg_file, **kwargs)
        
    def sample_primitive(self):
        elbow_vel =  random.uniform(self.cfg.joint_params.elbow[0], self.cfg.joint_params.elbow[1])
        acc = random.uniform(self.cfg.acc[0], self.cfg.acc[1])
        n = random.randint(self.cfg.n[0], self.cfg.n[1])
        return TambourinePrimitive(elbow_vel, acc, n)
    
    def _torchify_primitive(self, primitive: TambourinePrimitive):
        return torch.tensor([primitive.elbow_vel, primitive.acc, primitive.n])
    
    def _detorchify(self, t_params):
        return TambourinePrimitive(t_params[0], t_params[1], int(round_fn(t_params[2])))
    
    def _set_action_limits(self):
        self.action_min = torch.Tensor([self.cfg.joint_params.elbow[0],
                                        self.cfg.acc[0],
                                        self.cfg.n[0]])
        self.action_max = torch.Tensor([self.cfg.joint_params.elbow[1],
                                self.cfg.acc[1],
                                self.cfg.n[1]])        
    
    def _generate_trajectory(self, primitive: TambourinePrimitive):
        vels = np.array([0., 0., primitive.elbow_vel, 0., 0., 0.]) # [base, shoulder, elbow, wrist1, wrist2, wrist3]
        traj = np.stack([vels] * primitive.n) # stack n times
        traj[1::2] *= -1. # Alternate b/w  +v and -v
        return traj
    
    def execute_trajectory(self, primitive):
        traj = self._generate_trajectory(primitive)
        msg = self.bot.prepare_msg_speedj(traj, primitive.acc, None)
        self.bot.send_program(msg)
        time.sleep(self.execute_trajectory_sleep_time)
    
    def reset(self):
        self.bot.movej(self.cfg.HOME_JS, acc=2, vel=2)
        time.sleep(self.reset_sleep_time)

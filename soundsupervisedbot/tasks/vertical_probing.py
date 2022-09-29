import numpy as np
import random
import time
from collections import namedtuple
import torch
from soundsupervisedbot.tasks.sound_task import Task

VerticalProbingPrimitive = namedtuple('VerticalProbingPrimitive', ['vels0', 'vels1', 'acc'])


class VerticalProbingTask(Task):
    def __init__(self, cfg_file = '/home/abitha/projects/sound-supervised-bot/soundsupervisedbot/configs/vertical_probing.yaml', **kwargs) -> None:
        super().__init__(cfg_file, **kwargs)

    def sample_primitive(self):
        # Uniform sampling between low, high for specific joint velocities
        vels0 = np.array([random.uniform(self.cfg.joint_params.base_0[0], self.cfg.joint_params.base_0[1]), 
            0.0,
            random.uniform(self.cfg.joint_params.elbow_0[0], self.cfg.joint_params.elbow_0[1]), #TODO-Rename to elbow
            random.uniform(self.cfg.joint_params.wrist1_0[0], self.cfg.joint_params.wrist1_0[1]), 
            0.0,
            0.0,
            ])
        vels1 = np.array([0.0, 
            random.uniform(self.cfg.joint_params.shoulder[0], self.cfg.joint_params.shoulder[1]),
            random.uniform(self.cfg.joint_params.elbow[0], self.cfg.joint_params.elbow[1]),
            random.uniform(self.cfg.joint_params.wrist1[0], self.cfg.joint_params.wrist1[1]),
            0.0,
            0.0,
            ])
        acc = random.uniform(self.cfg.acc[0], self.cfg.acc[1])

        return VerticalProbingPrimitive(vels0, vels1, acc)
    
    def _torchify_primitive(self, primitive):
        t_params = torch.tensor([
            primitive.vels0[0], # Vel0 : base - Controls left/right side of box 
            primitive.vels0[2], # Vel0 : elbow - Controls momentum
            primitive.vels0[3], # Vel0 : wrist - Controls momentum
            
            primitive.vels1[1], # Vel1 : shoulder
            primitive.vels1[2], # Vel1 : elbow
            primitive.vels1[3], # Vel1 : wrist

            primitive.acc # Acceleration
            ])

        return t_params 
    
    def _detorchify(self, t_params):
        return VerticalProbingPrimitive(
            vels0 = np.array([t_params[0], 0., t_params[1], t_params[2], 0., 0.]),
            vels1 = np.array([0., t_params[3], t_params[4], t_params[5], 0., 0.]),
            acc = t_params[6]
        )
    
    def _set_action_limits(self):
        self.action_min = torch.Tensor([self.cfg.joint_params.base_0[0], 
                                        self.cfg.joint_params.elbow_0[0], 
                                        self.cfg.joint_params.wrist1_0[0],
                                        self.cfg.joint_params.shoulder[0], 
                                        self.cfg.joint_params.elbow[0], 
                                        self.cfg.joint_params.wrist1[0],
                                        self.cfg.acc[0]
                                        ]) 
        self.action_max = torch.Tensor([self.cfg.joint_params.base_0[1], 
                                        self.cfg.joint_params.elbow_0[1], 
                                        self.cfg.joint_params.wrist1_0[1],
                                        self.cfg.joint_params.shoulder[1], 
                                        self.cfg.joint_params.elbow[1], 
                                        self.cfg.joint_params.wrist1[1],
                                        self.cfg.acc[1]
                                        ])
        
    def _generate_trajectory(self, primitive):
        traj = np.stack([primitive.vels0, -1. * primitive.vels1], axis = 0)
        return traj
    
    def execute_trajectory(self, primitive):
        traj = self._generate_trajectory(primitive)
        msg = self.bot.prepare_msg_speedj(traj, primitive.acc, None)
        self.bot.send_program(msg)
        time.sleep(self.execute_trajectory_sleep_time)
    
    def reset(self):
        self.bot.movej(self.cfg.RESET_JS_1, acc=2, vel=2)
        self.bot.movej(self.cfg.RESET_JS_2, acc=2, vel=2)
        self.bot.movej(self.cfg.HOME,acc=0.05, vel=0.05)
        time.sleep(self.reset_sleep_time)

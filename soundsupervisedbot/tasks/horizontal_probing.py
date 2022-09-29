from soundsupervisedbot.tasks.sound_task import Task
import numpy as np
import torch
from collections import namedtuple
import random
import time
from soundsupervisedbot.tasks.utils.evaluation import round_fn


HorizontalProbingPrimitive = namedtuple('HorizontalProbingPrimitive', ['j_vel', 'a', 's', 'e', 'w'])

class HorizontalProbingTask(Task):
    def __init__(self, cfg_file = '/home/abitha/projects/sound-supervised-bot/soundsupervisedbot/configs/horizontal_probing.yaml', **kwargs) -> None:
        super().__init__(cfg_file, **kwargs)
            
    def sample_primitive(self):
        j_vel = np.array([0.0, 
            random.uniform(self.cfg.joint_params.shoulder[0], self.cfg.joint_params.shoulder[1]),
            random.uniform(self.cfg.joint_params.elbow[0], self.cfg.joint_params.elbow[1]),
            random.uniform(self.cfg.joint_params.wrist1[0], self.cfg.joint_params.wrist1[1]),
            0.0,
            0.0,
            ])
        a = random.uniform(self.cfg.acc[0], self.cfg.acc[1])
        s = random.randint(self.cfg.nsteps_limits[0], self.cfg.nsteps_limits[1]) # Number of steps for shoulder
        w = random.randint(self.cfg.nsteps_limits[0], self.cfg.nsteps_limits[1]) # Number of steps for wrist
        e = random.randint(min(s,w)-1, max(s,w)-1) # Number of steps for elbow - constrained to prevent collision

        return HorizontalProbingPrimitive(j_vel, a, s, e, w)
    
    def _torchify_primitive(self, primitive):
        return torch.tensor(np.append(primitive.j_vel[1:4],[primitive.a, primitive.s, primitive.e, primitive.w]))
    
    def _detorchify(self, t_params):
        return HorizontalProbingPrimitive(
            j_vel = np.array([0., t_params[0], t_params[1], t_params[2], 0., 0.]),
            a = t_params[3],
            s = round_fn(t_params[4]),
            e = round_fn(t_params[5]),
            w = round_fn(t_params[6])
        )

    def _set_action_limits(self):
        self.action_min = torch.Tensor([self.cfg.joint_params.shoulder[0], 
                                        self.cfg.joint_params.elbow[0], 
                                        self.cfg.joint_params.wrist1[0],
                                        self.cfg.acc[0],
                                        self.cfg.nsteps_limits[0],
                                        self.cfg.nsteps_limits[0],
                                        self.cfg.nsteps_limits[0]
                                        ]) 
        self.action_max = torch.Tensor([self.cfg.joint_params.shoulder[1], 
                                        self.cfg.joint_params.elbow[1], 
                                        self.cfg.joint_params.wrist1[1],
                                        self.cfg.acc[1],
                                        self.cfg.nsteps_limits[1],
                                        self.cfg.nsteps_limits[1],
                                        self.cfg.nsteps_limits[1]
                                        ])
 
    def _generate_trajectory(self, primitive):
        shoulder_nsteps, elbow_nsteps, wrist1_nsteps, vels = primitive.s, primitive.e, primitive.w, primitive.j_vel

        shoulder_step_size, elbow_step_size, wrist1_step_size = vels[1]/ shoulder_nsteps, vels[2]/ elbow_nsteps, vels[3]/ wrist1_nsteps
        nsteps = max(shoulder_nsteps, elbow_nsteps, wrist1_nsteps)
        shoulder_vels, elbow_vels, wrist1_vels = [0.0], [0.0], [0.0]
        
        for _ in range(shoulder_nsteps):
            shoulder_vels.append(shoulder_vels[-1] + shoulder_step_size)
        for _ in range(elbow_nsteps):
            elbow_vels.append(elbow_vels[-1] + elbow_step_size)
        for _ in range(wrist1_nsteps):
            wrist1_vels.append(wrist1_vels[-1] + wrist1_step_size)
        
        shoulder_vels[0] = -1 * vels[1]
        elbow_vels[0] = -1 * vels[2]
        wrist1_vels[0] = -1 * vels[3]
            
        while len(shoulder_vels) <= nsteps:
            shoulder_vels.append(0.0)
        while len(elbow_vels) <= nsteps:
            elbow_vels.append(0.0)
        while len(wrist1_vels) <= nsteps:
            wrist1_vels.append(0.0)
        traj = np.stack([np.array([0.0]*(nsteps+1)), 
                        np.array(shoulder_vels),
                        np.array(elbow_vels),
                        np.array(wrist1_vels),
                        np.array([0.0]*(nsteps+1)),
                        np.array([0.0]*(nsteps+1))
                        ],
                        axis = 1)
        
        return traj
    
    def execute_trajectory(self, primitive):
        # Eval mode - prevent collision -> constraint
        if primitive.e >= (max(primitive.s, primitive.w)):
            primitive = primitive._replace(e = max(primitive.s, primitive.w) - 1)

        traj = self._generate_trajectory(primitive)
        msg = self.bot.prepare_msg_speedj(traj, primitive.a, None)
        self.bot.send_program(msg)
        time.sleep(self.execute_trajectory_sleep_time)
        
    def reset(self):
        self.bot.movej(self.cfg.RESET_JS, acc=2, vel=2)
        self.bot.movej(self.cfg.HOME, acc=0.05, vel=0.05)
        time.sleep(self.reset_sleep_time)

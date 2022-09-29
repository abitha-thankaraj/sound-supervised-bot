import torch
from soundsupervisedbot.utils.audio_transforms import to_spectrogram, to_mel_spectrogram, to_log_mel_spectrogram
from soundsupervisedbot.utils.utils import load_pkl_file
import torchaudio.transforms as TA
import torchvision.transforms as TV
import torch.nn as nn
import os
import math

TRANSFORMS = {
    'mel': to_mel_spectrogram,
    'lms' : to_log_mel_spectrogram,
    'spec' : to_spectrogram,
}

class EvaluationHelper:
    def __init__(self, dir_name) -> None:
        self.dir_name = dir_name
        self._load_model(dir_name) #This should not be private
        self._load_cfgs(dir_name)
        self._init_normalizers()    
        
    def _load_cfgs(self, dir_name):
        self.cfg = load_pkl_file(os.path.join(dir_name, 'run_cfgs.pkl'))
        self.audio_norm_stats, self.action_min_max_norm_stats = None, None
        if self.cfg.norm_audio_data:
            self.audio_norm_stats = load_pkl_file(os.path.join(dir_name, 'audio_norm_stats.pkl'))
        if self.cfg.norm_action:
            self.action_min_max_norm_stats = load_pkl_file(os.path.join(dir_name,'action_min_max_norm_stats.pkl'))
        
    def _init_normalizers(self):
        if self.cfg.norm_audio_data:
            self.audio_data_normalizer = nn.Sequential(
                TV.Normalize(mean = self.audio_norm_stats.mean, std = self.audio_norm_stats.std)
            )
        self.action_normalizer = lambda actions : (actions - self.action_min_max_norm_stats.min) / (self.action_min_max_norm_stats.max - self.action_min_max_norm_stats.min)
        self.action_denormalizer  = lambda normalized_actions : (normalized_actions * (self.action_min_max_norm_stats.max - self.action_min_max_norm_stats.min)) + self.action_min_max_norm_stats.min
    
    def _load_model(self, dir_name, model_fname = 'model.pth'):
        self.model = torch.load(os.path.join(dir_name, model_fname), map_location = 'cpu')
    
    def _apply_transforms(self, audio):
        
        # TODO - nn.Sequential for Audio tfs

        # Only read relevant channels + 
        audio = audio[:self.cfg.dataset.num_mics,:self.cfg.dataset.max_num_frames]
        #Resample
        resampled_audio = TA.Resample(self.cfg.dataset.sample_rate, self.cfg.dataset.resample_rate)(audio)
        # Convert to mel spec - TODO - Pick better way according to model
        data = TRANSFORMS[self.cfg.dataset.name](self.cfg.dataset, resampled_audio.float())

        if self.cfg.norm_audio_data:
            data = self.audio_data_normalizer(data)
        
        return data

def round_fn(num):
    if num - math.floor(num) >= 0.5:
        return math.ceil(num)
    else:
        return math.floor(num)

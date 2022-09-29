import torchaudio
import torch
import numpy as np

def to_spectrogram(cfg, waveform):
    to_spec = torchaudio.transforms.Spectrogram(n_fft=cfg.n_fft, normalized = cfg.normalize)
    return to_spec(waveform)
    
def to_mel_spectrogram(cfg, waveform):
    # Instantiate mel spectrogram converter 
    to_mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
        hop_length=cfg.hop_length, normalized= cfg.normalize)
    return to_mel_spec(waveform)

def to_log_mel_spectrogram(cfg, waveform):
    # Add epsilon to prevent exploding gradients
    to_mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
        hop_length=cfg.hop_length, normalized= cfg.normalize)    
    return (to_mel_spec(waveform) + torch.finfo().eps).log() 
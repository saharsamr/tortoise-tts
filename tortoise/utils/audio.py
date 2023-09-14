import os
from glob import glob

import librosa
import torch
import torchaudio
import numpy as np
from scipy.io.wavfile import read

from tortoise.utils.stft import STFT


BUILTIN_VOICES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../voices')


def load_wav_to_torch(full_path):
    """
    This function loads .wav audio files.
    """
    sampling_rate, data = read(full_path)
    # norm_fix is the normalization factor for the audio data.
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.
    else:
        raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def load_audio(audiopath, sampling_rate):
    """
    This function loads audio files with specified sampling rate which are either .wav or .mp3 format.
    """
    if audiopath[-4:] == '.wav':
        # loads the .wav audio file
        audio, lsr = load_wav_to_torch(audiopath)
    elif audiopath[-4:] == '.mp3':
        # loads the .mp3 audio file
        audio, lsr = librosa.load(audiopath, sr=sampling_rate)
        audio = torch.FloatTensor(audio)
    else:
        # if the audio file is not .wav or .mp3, an assertion will be raised.
        assert False, f"Unsupported audio format provided: {audiopath[-4:]}"

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    # if the sampling rate of the audio file is not equal to the desired sampling rate, it will be resampled here.
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Original Comment: Check some assumptions about audio range. This should be automatically fixed in
    # load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)


TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254


def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN


def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def get_voices(extra_voice_dirs=[]):
    """
    This function returns a dictionary of voices. the keys are the names of the voices in the voices directories
    , and the values are the paths to the audio files of the voices.
    """
    dirs = [BUILTIN_VOICES_DIR] + extra_voice_dirs
    voices = {}
    for d in dirs:
        subs = os.listdir(d)
        for sub in subs:
            subj = os.path.join(d, sub)
            if os.path.isdir(subj):
                voices[sub] = list(glob(f'{subj}/*.wav')) + list(glob(f'{subj}/*.mp3')) + list(glob(f'{subj}/*.pth'))
    return voices


def load_voice(voice, extra_voice_dirs=[]):
    """
    This function loads the voice for generating the audio for the selected_voice. it handles a single given voice.
    """
    # if the voice is set to random, then we will return None, None, which will be handled by the caller.
    if voice == 'random':
        return None, None

    # following function returns a dictionary, containing voice-names as keys, and paths to the audio files as values.
    voices = get_voices(extra_voice_dirs)
    # the path to the current desired voice is here extracted from the voices dictionary
    paths = voices[voice]
    # if the is only one path there, and it ends with .pth, then it is a latent, so the conditionals will be None
    # and the latent will be loaded and returned.
    if len(paths) == 1 and paths[0].endswith('.pth'):
        return None, torch.load(paths[0])
    else:
        # otherwise, it is an audio voice, so the latent will be None, and the conditionals will be loaded and returned.
        conds = []
        # conditions can be in several files, for each of them:
        for cond_path in paths:
            # the audio file will be loaded with proper sampling rate and appended to the conds list.
            c = load_audio(cond_path, 22050)
            conds.append(c)
        return conds, None


def load_voices(voices, extra_voice_dirs=[]):
    """
    This function loads the voice(s) for generating the audio for the selected_voice. you can combine audio voices,
    or latent voices. if you want to combine an audio voice with a latent voice, you should do this manually.
    voice_samples are list of 2 or more reference clips, while conditioning_latents are tuples of
    (autoregressive_conditioning_latent, diffusion_conditioning_latent) which can be used instead of voice_samples.
    However, the second one will be ignored if the voice_samples is not none.
    """
    latents = []
    clips = []
    # for every voice given in the voices argument:
    for voice in voices:
        # if the voice is set to random, then we will return None, None, which will be handled by the caller.
        # also, if there are more than one element in voices argument while one of them are random, an assertion
        # will be raised and announce that a random-voice cannot be combined with a non-random voice.
        if voice == 'random':
            if len(voices) > 1:
                print("Cannot combine a random voice with a non-random voice. Just using a random voice.")
            return None, None
        # load the voice, which can be either a latent or an audio clip.
        clip, latent = load_voice(voice, extra_voice_dirs)
        # the above function (load_voice) returns a tuple of (clip, latent), so if the clip is not None, then it is an
        # audio clip, and if the latent is not None, then it is a latent.
        # the following section of code checks if the voice is an audio clip while the latents list is not empty, or
        # if the voice is a latent while the clips list is not empty, then an assertion will be raised and announce
        # that a latent voice cannot be combined with an audio voice. if everythin is ok, then the clip or the latent
        # will be appended to the clips or latents list, respectively.
        if latent is None:
            assert len(latents) == 0, "Can only combine raw audio voices or latent voices, not both. Do it yourself if you want this."
            clips.extend(clip)
        elif clip is None:
            assert len(clips) == 0, "Can only combine raw audio voices or latent voices, not both. Do it yourself if you want this."
            latents.append(latent)
    # if the clips list is empty, then the clips will be None, and the autoregressive_conditioning_latent and
    # diffusion_conditioning_latent will be the average of the latents list and returned as a tuple.
    # otherwise, the latents will be None, and the clips will be the average of the clips will be returned.
    if len(latents) == 0:
        return clips, None
    else:
        # autoregressive_conditioning_latent average
        latents_0 = torch.stack([l[0] for l in latents], dim=0).mean(dim=0)
        # diffusion_conditioning_latent average
        latents_1 = torch.stack([l[1] for l in latents], dim=0).mean(dim=0)
        # final tuple
        latents = (latents_0,latents_1)
        return None, latents


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        from librosa.filters import mel as librosa_mel_fn
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -10)
        assert(torch.max(y.data) <= 10)
        y = torch.clip(y, min=-1, max=1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


def wav_to_univnet_mel(wav, do_normalization=False, device='cuda' if not torch.backends.mps.is_available() else 'mps'):
    stft = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000)
    stft = stft.to(device)
    mel = stft.mel_spectrogram(wav)
    if do_normalization:
        mel = normalize_tacotron_mel(mel)
    return mel

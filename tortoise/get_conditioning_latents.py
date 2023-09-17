import argparse
import os
import torch

from api import TextToSpeech
from tortoise.utils.audio import load_audio, get_voices

"""
Dumps the conditioning latents for the specified voice to disk. These are expressive latents which can be used for
other ML models, or can be augmented manually and fed back into Tortoise to affect vocal qualities.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # it takes of a voice, which should be the name of a directory in the data/voices directory that contains several
    # samples of a voice to create conditioning latents for that voice.
    parser.add_argument('--voice', type=str, help='Selects the voice to convert to conditioning latents', default='pat2')
    # the path in which you wish to save the latents.
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='../results/conditioning_latents')
    args = parser.parse_args()
    # it creates the output directory if it doesn't exist.
    os.makedirs(args.output_path, exist_ok=True)

    # full documentation of TextToSpeech class is available in api.py
    tts = TextToSpeech()
    # the get_voices() function returns a dictionary of voices, where the key is the name of the voice, and the value
    # is a list of paths to the audio files of the voice.
    voices = get_voices()
    # you can specify multiple voices in the voice argument, and they should be separated by comma.
    # if you do so, the latents will be dumped for each of them separately.
    selected_voices = args.voice.split(',')
    # for each of the voice specified in the input --voice argument:
    for voice in selected_voices:
        # the paths to the audio files of the voice are extracted from the voices dictionary.
        cond_paths = voices[voice]
        # the latents will be stored in this list.
        conds = []
        # each directory of voices may contain several sample, for each of them:
        for cond_path in cond_paths:
            # the audio file will be loaded with proper sampling rate and appended to the conds list.
            c = load_audio(cond_path, 22050)
            conds.append(c)
        # Transforms one or more voice_samples into a tuple
        # (autoregressive_conditioning_latent, diffusion_conditioning_latent). Full documentation of this function is
        # available in api.py, class TextToSpeech.
        conditioning_latents = tts.get_conditioning_latents(conds)
        # after getting the latents, they will be saved in the output directory with the name of the voice.
        torch.save(conditioning_latents, os.path.join(args.output_path, f'{voice}.pth'))


import argparse
import os

import torchaudio

from api import TextToSpeech
from tortoise.utils.audio import load_audio


"""
This File evaluates the result of running tortoise on a TSV file.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # the path to the TSV file. The TSV file should have two columns, the first column is the text, and the second
    # column is the path to the real audio file.
    parser.add_argument('--eval_path', type=str, help='Path to TSV test file', default="D:\\tmp\\tortoise-tts-eval\\test.tsv")
    # the path in which you wish to save the generated audio files.
    parser.add_argument('--output_path', type=str, help='Where to put results', default="D:\\tmp\\tortoise-tts-eval\\baseline")
    # the preset you want to use for running the model. The default is standard. you can read more about these presets
    # in the documentation of the TextToSpeech class in api.py, function tts_with_preset.
    parser.add_argument('--preset', type=str, help='Rendering preset.', default="standard")
    args = parser.parse_args()
    # it creates the output directory if it doesn't exist.
    os.makedirs(args.output_path, exist_ok=True)

    # full documentation of TextToSpeech class is available in api.py
    tts = TextToSpeech()

    # the TSV file is read line by line.
    with open(args.eval_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # for each line in the TSV file:
    for line in lines:
        # the text and the path to the real audio file are extracted.
        text, real = line.strip().split('\t')
        # the audio file is loaded with proper sampling rate.
        conds = [load_audio(real, 22050)]
        # the audio file is generated using the text and the real audio file as input voice_samples. full
        # documentation of this function is available in api.py, class TextToSpeech.
        gen = tts.tts_with_preset(text, voice_samples=conds, conditioning_latents=None, preset=args.preset)
        # the generated audio file is saved in the output directory with the name of the real audio file.
        torchaudio.save(os.path.join(args.output_path, os.path.basename(real)), gen.squeeze(0).cpu(), 24000)


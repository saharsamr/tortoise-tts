import argparse

from api import classify_audio_clip
from tortoise.utils.audio import load_audio


"""
This file takes an audio clip and classifies it as either being generated from Tortoise or not.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # the path to the audio clip.
    parser.add_argument('--clip', type=str, help='Path to an audio clip to classify.', default="../examples/favorite_riding_hood.mp3")
    args = parser.parse_args()

    # the audio clip is loaded with proper sampling rate. The documentation of this function is available in
    # tortoise/utils/audio.py
    clip = load_audio(args.clip, 24000)
    # the audio clip is trimmed to nearly 10 seconds.
    clip = clip[:, :220000]
    # the audio clip is classified as either being generated from Tortoise or not. The documentation of this function is
    # available in api.py.
    prob = classify_audio_clip(clip)
    print(f"This classifier thinks there is a {prob*100}% chance that this clip was generated from Tortoise.")
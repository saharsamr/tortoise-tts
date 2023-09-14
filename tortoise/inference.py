import argparse
import os
from time import time

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_audio, load_voices
from utils.text import split_and_recombine_text


if __name__ == '__main__':
    # input arguments for tuning the model.
    parser = argparse.ArgumentParser()
    # relative path to the text file you want to read from must come after the --textfile flag.
    parser.add_argument('--textfile', type=str, help='A file containing the text to read.', default="tortoise/data/riding_hood.txt")
    # the voice you want to use for generation, must come after the --voice flag: 
    # - if you separate voices with a comma, it will run the inference for each of the provided voices. 
    # - if between the commas, there are voices separated by &, the model will joint two voices to generate the audio.
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='pat')
    # here you should specify the path in which you want to save the output voices.
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/longform/')
    # you can specify the name of the output file after --output_name argument, by default it is called combined.wav.
    parser.add_argument('--output_name', type=str, help='How to name the output file', default='combined.wav')
    # there are four options for --preset argument: 
    # 1- ultra_fast: this is the fastest option, but the quality is not very good.
    # 2- fast: the second fastest option, and the quality is better than the first option. it is recommended for mass
    # generation.
    # 3- standard: this is the default option, the quality is really good and is recommended for most cases.
    # 4- high_quality: the absolute best quality! however, the computation time may not worth it.
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='standard')
    # the input text will be split to be fed to the model, so you'll have some clips of audio (the ones that the final 
    # audio will be made of). as the model can mess up in some parts, you can specify which clips you want to
    # re-generate by passing the clip numbers to --regenerate argument. clip numbers should be separated by commas.
    parser.add_argument('--regenerate', type=str, help='Comma-separated list of clip numbers to re-generate, or nothing.', default=None)
    # you can ask the model to generate more than one candidate for each voice. the final output would be made using the 
    # first candidate, however, if you prefer some over the others, you can manually select them and create the final.
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice. Only the first candidate is actually used in the final product, the others can be used manually.', default=1)
    # the path to the pre-trained model you want to be used for inference. 
    # you can use different pre-trained versions using this argument.
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
    # with specifying the seed, you can reproduce the same results. this is the number that is used for generating
    # random numbers. for example, if you set the seed to 42, it doesn't matter how many times you run the program, 
    # you will always get the same result using the same arguments.
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    # by setting this parameter to True or False, you can ask the model to produce a debug_state.pth file which can be
    # used to reproduce the results for debugging purposes.
    parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
    # you can enable deepseed that can boost the speed of very large models. it still doesn't work in macOS.
    parser.add_argument('--use_deepspeed', type=bool, help='Use deepspeed for speed bump.', default=False)
    # It caches keys and values, so to speed up the process. disabling this can significantly reduce the speed.
    parser.add_argument('--kv_cache', type=bool, help='If you disable this please wait for a long a time to get the output', default=True)
    # uses float16 instead of float32. it is faster, and takes less memory. However, you will loose precision.
    parser.add_argument('--half', type=bool, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True)

    args = parser.parse_args()
    
    # If the code is running on a M1/M2 chip, it will automatically disable deepspeed, as it is not supported yet.
    # you can change this part, if the support is added in the future.
    if torch.backends.mps.is_available():
        args.use_deepspeed = False
    
    # the tts object is the main instance to run the model. you can also set the batch-size of the autoregressive model
    # through the autoregressive_batch_size argument. you can also set the enable_redaction (which by default is True!)
    # so that the embraced text in the bracket are redacted from the spoken output, which is useful in some cases like
    # prompt-engineering. other arguments which are used in the line below, has been explained earlier.
    tts = TextToSpeech(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half)

    outpath = args.output_path
    outname = args.output_name
    
    # the voices which are selected for generation, have been specified through the --voice argument. as explained 
    # earlier, you can specify multiple voices separated by commas. selected_voices contain the chosen voices (each of
    # the elements in selected_voices list, can contain multiple voices separated by & character, which means that the
    # model will joint those voices together to generate the audio).
    selected_voices = args.voice.split(',')
    
    # if tortoise messes up in some parts of the text, you can re-generate those parts by specifying the clip numbers. 
    # here, we convert the string of clip numbers to a list of integers. which will be used later in order to just
    # regenerating the parts that are messed up, instead of re-generating the whole text.
    regenerate = args.regenerate
    if regenerate is not None:
        regenerate = [int(e) for e in regenerate.split(',')]
    # reading the text-file. the text variable is a string containing the whole text.
    with open(args.textfile, 'r', encoding='utf-8') as f:
        text = ' '.join([l for l in f.readlines()])
    
    # text should be splitted to desired length in order to generate the audio by the model. you can either split it 
    # using | sign, or the split_and_recombine_text function do this for you and tries to split the text in a way that
    # the model can generate the audio without messing up.
    if '|' in text:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        texts = text.split('|')
    else:
        texts = split_and_recombine_text(text)

    # if the seed is not specified, it will be set to the current time.
    seed = int(time()) if args.seed is None else args.seed
    # the process of generating the audio is done in the following for loop. for each of the selected voices, which are
    # stored in selected_voices list, the model will generate the audio and the audio will be saved.
    for selected_voice in selected_voices:
        # the path to save each final audio is specified in voice_outpath variable, using the outpath, and the voice
        # name. then if the directory doesn't exist, it will be created.
        voice_outpath = os.path.join(outpath, selected_voice)
        os.makedirs(voice_outpath, exist_ok=True)
        
        # each voice in the selected_voices can contain multiple voices separated by & character. if this is the case,
        # the voices will be joined together to generate the audio. the voice_sel variable contains the list of voices
        # that should be joined together.
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        
        # This function loads the voice(s) for generating the audio for the selected_voice. you can combine audio voices
        # or latent voices. if you want to combine an audio voice with a latent voice, you should do this manually. 
        # voice_samples are list of 2 or more reference clips, while conditioning_latents are tuples of 
        # (autoregressive_conditioning_latent, diffusion_conditioning_latent) which can be used instead of voice_samples
        # However, the second one will be ignored if the voice_samples is not none.
        voice_samples, conditioning_latents = load_voices(voice_sel)
        # sub-audios will be added to this list
        all_parts = []
        
        # input text was split to different segments. in this loop, the audio of each related segment is going to be 
        # generated. the audio of each part will be appended to all_parts variable to be further used to generate 
        # the final audio.
        
        for j, text in enumerate(texts):
            
            # if we are running this script to regenerate the parts that was messed up in previous renders, the
            # regererate variable will be a list of messed up segment numbers. if so, there is no need to regenerate
            # the sections which are not listed after the --regenerate argument. so the following if checks whether we
            # want to do regeneration, the if we are in a good section (which is not listed in input arguments) it
            # simply loads the previously-generated audio to be used in this round, then it goes for the next part of
            # text in texts variable. if we are regenerating, and j IS in regenerate, the audio will be again produced.
            if regenerate is not None and j not in regenerate:
                # load_audio function loads audio files with specified sampling rate which are either .wav or .mp3
                # format.
                all_parts.append(load_audio(os.path.join(voice_outpath, f'{j}.wav'), 24000))
                continue
            
            # the following function generates the related audio clip in this step of for loop. the following arguments
            # can be passed to this function:
            # 
            # - text: this is the essential input.
            # - voice_samples: List of 2 or more ~10 second reference clips.
            # - conditioning_latents: A tuple of (autoregressive_conditioning_latent, diffusion_conditioning_latent), 
            #     which can be provided in lieu of voice_samples. This is ignored unless voice_samples=None.
            # - k: The k most likely (as determined by Tortoises' CLVP model) clips are returned. it can be specified 
            # through the --candidates argument when running this script.
            # - verbose: Whether or not to print log messages indicating the progress of creating a clip. Default=true.
            # 
            # AUTO-REGRESSIVE GENERATION PARAMETERS:
            # - num_autoregressive_samples: Number of samples taken from the autoregressive model, all of which are 
            #     filtered using CLVP. As Tortoise is a probabilistic model, more samples means a higher probability of 
            #     creating something "great".
            # - temperature: The softmax temperature of the autoregressive model.
            # - length_penalty: A length penalty applied to the autoregressive decoder. Higher settings causes the model 
            #     to produce more terse outputs.
            # - repetition_penalty: A penalty that prevents the autoregressive decoder from repeating itself during 
            #     decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc.
            # - top_p: P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely" 
            #     (aka boring) outputs.
            # - max_mel_tokens: Restricts the output length. (0,600] integer. Each unit is 1/20 of a second.
            # - typical_sampling: Turns typical sampling on or off. This sampling mode is discussed in this paper: 
            #     https://arxiv.org/abs/2202.00666. This is off by default, but could use some tuning.
            # - typical_mass: The typical_mass parameter from the typical_sampling algorithm.
            # 
            # CVVP Parameters:
            # cvvp_amount: Controls the influence of the CVVP model in selecting the best output from the autoregressive
            #     model. [0,1]. Values closer to 1 mean the CVVP model is more important, 0 disables the CVVP model.
            #     
            # DIFFUSION GENERATION PARAMETERS:
            # diffusion_iterations: Number of diffusion steps to perform. [0,4000]. More steps means the network has
            #     more chances to iteratively refine the output, which should theoretically mean a higher quality output
            #     Generally a value above 250 is not noticeably better, however.
            # cond_free: Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs
            #     two forward passes for each diffusion step: one with the outputs of the autoregressive model and one 
            #     with no conditioning priors. The output of the two is blended according to the cond_free_k value below
            #     Conditioning-free diffusion is the real deal, and dramatically improves realism.
            # cond_free_k: Knob that determines how to balance the conditioning free signal with the
            #     conditioning-present signal. [0,inf]. As cond_free_k increases, the output becomes dominated by the 
            #     conditioning-free signal. Formula is: 
            #     output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k
            # diffusion_temperature: Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
            #     are the "mean" prediction of the diffusion network and will sound bland and smeared.
            #     
            # OTHER:
            # hf_generate_kwargs: The huggingface Transformers generate API is used for the autoregressive transformer.
            #     Extra keyword args fed to this function get forwarded directly to that API. Documentation here: 
            #     https://huggingface.co/docs/transformers/internal/generation_utils
            # preset: preset is specified in the --preset arguments.
            # use_deterministic_seed: it uses the value set in the seed variable.
            # 
            # RETURNS:
            # Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                      preset=args.preset, k=args.candidates, use_deterministic_seed=seed)
            
            # if the number of candidates (k in the tts_with_preset function) is set to 1, then we only get the most
            # probable output, which will be saved as a sub-audio and append to all_parts for generating the final audio
            # However, if it is set to a larger number, the gen variable will help k generated audios. the else section, 
            # saves each of these candidates. the important point here is that it only appended the first candidate 
            # (the most probable one) to all_parts variable. so the final audio will be generated using the first 
            # candidate, and if you want to replace some parts, you have to merge them manually using the sub-audios
            # which are store in meanwhile.
            if args.candidates == 1:
                audio_ = gen.squeeze(0).cpu()
                torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), audio_, 24000)
            else:
                candidate_dir = os.path.join(voice_outpath, str(j))
                os.makedirs(candidate_dir, exist_ok=True)
                for k, g in enumerate(gen):
                    torchaudio.save(os.path.join(candidate_dir, f'{k}.wav'), g.squeeze(0).cpu(), 24000)
                audio_ = gen[0].squeeze(0).cpu()
            all_parts.append(audio_)

        # sub-audios are concatenated in the following lines (for --candidates 1) and the final audio is saved on disk.
        if args.candidates == 1:
            full_audio = torch.cat(all_parts, dim=-1)
            torchaudio.save(os.path.join(voice_outpath, f"{outname}.wav"), full_audio, 24000)

        # if the --produce_debug_state is set to True, seed, texts, voice_samples, and conditioning_latents are stored
        # in the debug_states directory for debugging or reproducing the results.
        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            dbg_state = (seed, texts, voice_samples, conditioning_latents)
            torch.save(dbg_state, f'debug_states/read_debug_{selected_voice}.pth')

        # if the --candidate argument is set to a larger value (2, 3, ...), this section will combine the 1st audios for 
        # each part of the texts variables and generate a new clip using the 1st candidates of each parts; then it 
        # combines the 2nd candidate for each part and generates a new one, and so on.
        if args.candidates > 1:
            audio_clips = []
            for candidate in range(args.candidates):
                for line in range(len(texts)):
                    wav_file = os.path.join(voice_outpath, str(line), f"{candidate}.wav")
                    audio_clips.append(load_audio(wav_file, 24000))
                audio_clips = torch.cat(audio_clips, dim=-1)
                torchaudio.save(os.path.join(voice_outpath, f"{outname}_{candidate:02d}.wav"), audio_clips, 24000)
                audio_clips = []

import os
import random
import uuid
from time import time
from urllib import request

import torch
import torch.nn.functional as F
import progressbar
import torchaudio

from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.models.autoregressive import UnifiedVoice
from tqdm import tqdm
from tortoise.models.arch_util import TorchMelSpectrogram
from tortoise.models.clvp import CLVP
from tortoise.models.cvvp import CVVP
from tortoise.models.random_latent_generator import RandomLatentConverter
from tortoise.models.vocoder import UnivNetGenerator
from tortoise.utils.audio import wav_to_univnet_mel, denormalize_tacotron_mel
from tortoise.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from tortoise.utils.tokenizer import VoiceBpeTokenizer
from tortoise.utils.wav2vec_alignment import Wav2VecAlignment
from contextlib import contextmanager
pbar = None

DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'tortoise', 'models')
MODELS_DIR = os.environ.get('TORTOISE_MODELS_DIR', DEFAULT_MODELS_DIR)
MODELS = {
    'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth',
    'classifier.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth',
    'clvp2.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth',
    'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth',
    'diffusion_decoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth',
    'vocoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth',
    'rlg_auto.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth',
    'rlg_diffuser.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth',
}

def download_models(specific_models=None):
    """
    Call to download all the models that Tortoise uses.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None
    for model_name, url in MODELS.items():
        if specific_models is not None and model_name not in specific_models:
            continue
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            continue
        print(f'Downloading {model_name} from {url}...')
        request.urlretrieve(url, model_path, show_progress)
        print('Done.')


def get_model_path(model_name, models_dir=MODELS_DIR):
    """
    Get path to given model, download it if it doesn't exist.
    """
    if model_name not in MODELS:
        raise ValueError(f'Model {model_name} not found in available models.')
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path) and models_dir == MODELS_DIR:
        download_models([model_name])
    return model_path


def pad_or_truncate(t, length):
    """
    Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
    """
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length-t.shape[-1]))
    else:
        return t[..., :length]


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, cond_free=True, cond_free_k=1):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    # details are in SpacedDiffusion class
    return SpacedDiffusion(
        use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
        model_mean_type='epsilon',  model_var_type='learned_range', loss_type='mse',
        betas=get_named_beta_schedule('linear', trained_diffusion_steps),
        conditioning_free=cond_free, conditioning_free_k=cond_free_k)


def format_conditioning(clip, cond_length=132300, device="cuda" if not torch.backends.mps.is_available() else 'mps'):
    """
    Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
    """
    gap = clip.shape[-1] - cond_length
    # if clip is shorter, pad it to reach the desired length. otherwise, get a random number between 0 and gap, use it
    # as the starting point and clip the clip to reach the desired length.
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).to(device)


def fix_autoregressive_output(codes, stop_token, complain=True):
    """
    Original Comment: This function performs some padding on coded audio that fixes a mismatch issue between what the
    diffusion model was trained on and what the autoregressive code generator creates (which has no padding or end).
    This is highly specific to the DVAE being used, so this particular coding will not necessarily work if used with
    a different DVAE. This can be inferred by feeding a audio clip padded with lots of zeros on the end through the DVAE
    and copying out the last few codes.

    Failing to do this padding will produce speech with a harsh end that sounds like "BLAH" or similar.
    """
    # Strip off the autoregressive stop token and add padding.
    stop_token_indices = (codes == stop_token).nonzero()
    # if no stop token is found, the following error is raised.
    if len(stop_token_indices) == 0:
        if complain:
            print("No stop tokens found in one of the generated voice clips. This typically means the spoken audio is "
                  "too long. In some cases, the output will still be good, though. Listen to it and if it is missing words, "
                  "try breaking up your input text.")
        return codes
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    # 83 is the calm_token, but others are not explained.
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248

    return codes


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_seq_len = latents.shape[1] * 4 * 24000 // 22050
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, conditioning_latents, output_seq_len, False)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                      model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                     progress=verbose)
        return denormalize_tacotron_mel(mel)[:,:,:output_seq_len]


def classify_audio_clip(clip):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    classifier.load_state_dict(torch.load(get_model_path('classifier.pth'), map_location=torch.device('cpu')))
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]


def pick_best_batch_size_for_gpu():
    """
    Tries to pick a batch size that will fit in your GPU. These sizes aren't guaranteed to work, but they should give
    you a good shot.
    """
    if torch.cuda.is_available():
        _, available = torch.cuda.mem_get_info()
        availableGb = available / (1024 ** 3)
        if availableGb > 14:
            return 16
        elif availableGb > 10:
            return 8
        elif availableGb > 7:
            return 4
    if torch.backends.mps.is_available():
        import psutil
        available = psutil.virtual_memory().total
        availableGb = available / (1024 ** 3)
        if availableGb > 14:
            return 16
        elif availableGb > 10:
            return 8
        elif availableGb > 7:
            return 4
    return 1

class TextToSpeech:
    """
    Main entry point into Tortoise.
    """

    def __init__(self, autoregressive_batch_size=None, models_dir=MODELS_DIR, 
                 enable_redaction=True, kv_cache=False, use_deepspeed=False, half=False, device=None):
        """
        Constructor
        """
        # Where model weights are stored. This should only be specified if you are providing your own
        # models, otherwise use the defaults.
        self.models_dir = models_dir
        # Specifies how many samples to generate per batch. Lower this if you are seeing
        # GPU OOM errors. Larger numbers generates slightly faster.
        self.autoregressive_batch_size = pick_best_batch_size_for_gpu() if autoregressive_batch_size is None else autoregressive_batch_size
        # When true, text enclosed in brackets are automatically redacted from the spoken output
        # (but are still rendered by the model). This can be used for prompt engineering. Default is true.
        # If you are going to fine-tune the model, for example adding anger or anxiety to the text, you can embed
        # your prompt in the brackets and they will not being spoken in the final output.
        self.enable_redaction = enable_redaction
        # Device to use when running the model. If omitted, the device will be automatically chosen.
        self.device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
        # This is for M1/M2 chips.
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        # Wav2VecAlignment is used to align audio and text. If the enable_redaction is set to true, this will be
        # used to remove the audio generated for the prompts in the bracket.
        if self.enable_redaction:
            self.aligner = Wav2VecAlignment()
        # The Tokenizer
        self.tokenizer = VoiceBpeTokenizer()
        # Whether or not to use half precision (fp16) for the models.
        self.half = half

        # if there is a pre-trained model that you are going to use, here is where it is loaded.
        if os.path.exists(f'{models_dir}/autoregressive.ptt'):
            # The Autoregressive and Diffusion parts are loaded from the specified path.
            self.autoregressive = torch.jit.load(f'{models_dir}/autoregressive.ptt')
            self.diffusion = torch.jit.load(f'{models_dir}/diffusion_decoder.ptt')
        else:
            # If no path is specified using the models_dir argument, the default models are loaded here.
            # Each of them has the documentation in their respective classes. Please check them out before
            # applying any changes.
            self.autoregressive = UnifiedVoice(
                max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
                model_dim=1024, heads=16, number_text_tokens=255, start_text_token=255,
                checkpointing=False, train_solo_embeddings=False
            ).cpu().eval()
            # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.
            # the get_model_path function gets path to the given model, and downloads it if it doesn't exist.
            self.autoregressive.load_state_dict(
                torch.load(get_model_path('autoregressive.pth', models_dir)), strict=False)
            # Initializing GPT-2
            self.autoregressive.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=self.half)
            # Initializing DiffusionTts
            self.diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                          in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False, num_heads=16,
                                          layer_drop=0, unconditioned_percentage=0).cpu().eval()
            # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.
            # the get_model_path function gets path to the given model, and downloads it if it doesn't exist.
            self.diffusion.load_state_dict(torch.load(get_model_path('diffusion_decoder.pth', models_dir)))

        # Initializing CLVP
        self.clvp = CLVP(dim_text=768, dim_speech=768, dim_latent=768, num_text_tokens=256, text_enc_depth=20,
                         text_seq_len=350, text_heads=12,
                         num_speech_tokens=8192, speech_enc_depth=20, speech_heads=12, speech_seq_len=430,
                         use_xformers=True).cpu().eval()
        # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.
        # the get_model_path function gets path to the given model, and downloads it if it doesn't exist.
        self.clvp.load_state_dict(torch.load(get_model_path('clvp2.pth', models_dir)))
        # CVVP model is only loaded if used.
        self.cvvp = None
        # Initializing Vocoder
        self.vocoder = UnivNetGenerator().cpu()
        # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.
        # the get_model_path function gets path to the given model, and downloads it if it doesn't exist.
        self.vocoder.load_state_dict(torch.load(get_model_path('vocoder.pth', models_dir), map_location=torch.device('cpu'))['model_g'])
        self.vocoder.eval(inference=True)

        # Random latent generators (RLGs) are loaded lazily.
        self.rlg_auto = None
        self.rlg_diffusion = None

    # This function temporarily moves a model to gpu, and yields it for the use of the user, after that, it load it
    # back in the cpu to free gpu space.
    @contextmanager
    def temporary_cuda(self, model):
        m = model.to(self.device)
        yield m
        m = model.cpu()

    
    def load_cvvp(self):
        """Load CVVP model."""
        # More details about the arguments can be found in the class CVVP itself.
        self.cvvp = CVVP(model_dim=512, transformer_heads=8, dropout=0, mel_codes=8192, conditioning_enc_depth=8, cond_mask_percentage=0,
                         speech_enc_depth=8, speech_mask_percentage=0, latent_multiplier=1).cpu().eval()
        self.cvvp.load_state_dict(torch.load(get_model_path('cvvp.pth', self.models_dir)))

    def get_conditioning_latents(self, voice_samples, return_mels=False):
        """
        Transforms one or more voice_samples into a tuple (autoregressive_conditioning_latent, diffusion_conditioning_latent).
        These are expressive learned latents that encode aspects of the provided clips like voice, intonation, and acoustic
        properties.
        :param voice_samples: List of 2 or more ~10 second reference clips, which should be torch tensors containing 22.05kHz waveform data.
        """
        with torch.no_grad():
            # voices are moved into the specified device for further processing.
            voice_samples = [v.to(self.device) for v in voice_samples]
            # autoregressive conditioning are stored in this variable.
            auto_conds = []
            # if only one voice sample is given, it is converted into a list first.
            if not isinstance(voice_samples, list):
                voice_samples = [voice_samples]
            # for every voice in voice_samples argument, the format_conditioning function is called to convert the voice
            # into a mel spectrogram as expected by the models, and then it is appended to the auto_conds list.
            for vs in voice_samples:
                auto_conds.append(format_conditioning(vs, device=self.device))
            # stack the auto_conds list into a tensor and move it to the specified device.
            auto_conds = torch.stack(auto_conds, dim=1)
            self.autoregressive = self.autoregressive.to(self.device)
            # get the conditioning latent from the autoregressive model.
            auto_latent = self.autoregressive.get_conditioning(auto_conds)
            # move the autoregressive model back to cpu.
            self.autoregressive = self.autoregressive.cpu()

            # The following section is for the diffusion conditioning using given voice samples.
            diffusion_conds = []
            for sample in voice_samples:
                # The diffuser operates at a sample rate of 24000 (except for the latent inputs)
                sample = torchaudio.functional.resample(sample, 22050, 24000)
                # the sample should have a length of 102400, so it should be truncated or pad to reach this length
                sample = pad_or_truncate(sample, 102400)
                # change the sample into a mel spectrogram as expected by the models and appending the result.
                # in the diffusion_conds list.
                cond_mel = wav_to_univnet_mel(sample.to(self.device), do_normalization=False, device=self.device)
                diffusion_conds.append(cond_mel)
            # stack the diffusion_conds list into a tensor and move it to the specified device.
            diffusion_conds = torch.stack(diffusion_conds, dim=1)
            self.diffusion = self.diffusion.to(self.device)
            # get the conditioning latent from the diffusion model.
            diffusion_latent = self.diffusion.get_conditioning(diffusion_conds)
            # move the diffusion model back to cpu.
            self.diffusion = self.diffusion.cpu()

        # if return_mels is true, the mel spectrograms are returned as well.
        # otherwise, only the conditioning latents are returned.
        if return_mels:
            return auto_latent, diffusion_latent, auto_conds, diffusion_conds
        else:
            return auto_latent, diffusion_latent

    def get_random_conditioning_latents(self):
        """
        generating random latent using random latent generator (rlg)
        """
        if self.rlg_auto is None:
            # Generating random latents for autoregressive and diffusion models. more details are in the corresponding
            # functions.
            self.rlg_auto = RandomLatentConverter(1024).eval()
            self.rlg_auto.load_state_dict(torch.load(get_model_path('rlg_auto.pth', self.models_dir), map_location=torch.device('cpu')))
            self.rlg_diffusion = RandomLatentConverter(2048).eval()
            self.rlg_diffusion.load_state_dict(torch.load(get_model_path('rlg_diffuser.pth', self.models_dir), map_location=torch.device('cpu')))
        with torch.no_grad():
            return self.rlg_auto(torch.tensor([0.0])), self.rlg_diffusion(torch.tensor([0.0]))

    def tts_with_preset(self, text, preset='fast', **kwargs):
        """
        Calls TTS with one of a set of preset generation parameters. Options:
            'ultra_fast': Produces speech at a speed which belies the name of this repo. (Not really, but it's definitely fastest).
            'fast': Decent quality speech at a decent inference rate. A good choice for mass inference.
            'standard': Very good quality. This is generally about as good as you are going to get.
            'high_quality': Use if you want the absolute best. This is not really worth the compute, though.
        """
        # Use generally found best tuning knobs for generation. the parameters specified in the setting dictionary
        # are declared in the parameters of tts function in this class.
        settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                    'top_p': .8,
                    'cond_free_k': 2.0, 'diffusion_temperature': 1.0}
        # Presets are defined here.
        presets = {
            'ultra_fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
            'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
            'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
            'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
        }
        # settings dictionary will be update by the selected preset. you can also override the settings by
        # passing them as kwargs. (they should be in the list of parameters of tts function in this class)
        settings.update(presets[preset])
        settings.update(kwargs) # allow overriding of preset settings with kwargs
        return self.tts(text, **settings)

    def tts(self, text, voice_samples=None, conditioning_latents=None, k=1, verbose=True, use_deterministic_seed=None,
            return_deterministic_state=False,
            # autoregressive generation parameters follow
            num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8, max_mel_tokens=500,
            # CVVP parameters follow
            cvvp_amount=.0,
            # diffusion generation parameters follow
            diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0,
            **hf_generate_kwargs):
        """
        it is the main function and produces an audio clip of the given text being spoken with the given
        reference voice(s).
        :param text: Text to be spoken.
        :param voice_samples: List of 2 or more ~10 second reference clips which should be torch tensors containing 
        22.05kHz waveform data.
        :param conditioning_latents: A tuple of (autoregressive_conditioning_latent, diffusion_conditioning_latent), 
            which can be provided in lieu of voice_samples. This is ignored unless voice_samples=None.
            Conditioning latents can be retrieved via get_conditioning_latents().
        :param k: The number of returned clips. The most likely (as determined by Tortoises' CLVP model) clips 
            are returned.
        :param verbose: Whether or not to print log messages indicating the progress of creating a clip. Default=true.
        
        ~~AUTOREGRESSIVE KNOBS~~
        :param num_autoregressive_samples: Number of samples taken from the autoregressive model, all of which are 
            filtered using CLVP. As Tortoise is a probabilistic model, more samples means a higher probability of 
            creating something "great".
        :param temperature: The softmax temperature of the autoregressive model.
        :param length_penalty: A length penalty applied to the autoregressive decoder. Higher settings causes the 
            model to produce more terse outputs.
        :param repetition_penalty: A penalty that prevents the autoregressive decoder from repeating itself during 
            decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc.
        :param top_p: P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely" 
            (aka boring) outputs.
        :param max_mel_tokens: Restricts the output length. (0,600] integer. Each unit is 1/20 of a second.
        :param typical_sampling: Turns typical sampling on or off. This sampling mode is discussed in this paper: 
            https://arxiv.org/abs/2202.00666 I was interested in the premise, but the results were not as good as 
            I was hoping. This is off by default, but could use some tuning.
        :param typical_mass: The typical_mass parameter from the typical_sampling algorithm.
        
        ~~CLVP-CVVP KNOBS~~
        :param cvvp_amount: Controls the influence of the CVVP model in selecting the best output from the 
            autoregressive model. [0,1]. Values closer to 1 mean the CVVP model is more important, 0 disables 
            the CVVP model.
            
        ~~DIFFUSION KNOBS~~
        :param diffusion_iterations: Number of diffusion steps to perform. [0,4000]. More steps means the network 
            has more chances to iteratively refine the output, which should theoretically mean a higher quality output. 
            Generally a value above 250 is not noticeably better, however.
        :param cond_free: Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs 
            two forward passes for each diffusion step: one with the outputs of the autoregressive model and one with 
            no conditioning priors. The output of the two is blended according to the cond_free_k value below. 
            Conditioning-free diffusion is the real deal, and dramatically improves realism.
        :param cond_free_k: Knob that determines how to balance the conditioning free signal with the 
            conditioning-present signal. [0,inf]. As cond_free_k increases, the output becomes dominated by the 
            conditioning-free signal. Formula is: 
                output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k
        :param diffusion_temperature: Controls the variance of the noise fed into the diffusion model. [0,1]. 
            Values at 0 are the "mean" prediction of the diffusion network and will sound bland and smeared.
            
        ~~OTHER STUFF~~
        :param hf_generate_kwargs: The huggingface Transformers generate API is used for the autoregressive transformer.
            Extra keyword args fed to this function get forwarded directly to that API. Documentation here: 
                https://huggingface.co/docs/transformers/internal/generation_utils
        :return: Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
                Sample rate is 24kHz.
        """

        # sets the seed to the current time if use_deterministic_seed is false. also returns the seed value for
        # reproducing results in future.
        deterministic_seed = self.deterministic_state(seed=use_deterministic_seed)

        # tokenizes the text and converts it into a Inttensor. then the tensor is moved to the specified device.
        # then the tensor is padded to reach the maximum length.
        text_tokens = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
        # if number of tokens is more than 400, an error is raised.
        assert text_tokens.shape[-1] < 400, 'Too much text provided. Break the text up into separate segments and re-try inference.'

        auto_conds = None
        # if voice_samples are given in the input argument, the get_conditioning_latents function is called which its
        # full documentation is available in the body of the function itself.
        # if conditioning_latents are given in the input argument, then the user is using a previously stored one so
        # we assign auto_conditioning and diffusion_conditioning respectively without calling get_conditioning_latents
        # function. if neither of them are given, then we generate random conditioning latents using
        # get_random_conditioning_latents
        if voice_samples is not None:
            auto_conditioning, diffusion_conditioning, auto_conds, _ = self.get_conditioning_latents(voice_samples, return_mels=True)
        elif conditioning_latents is not None:
            auto_conditioning, diffusion_conditioning = conditioning_latents
        else:
            auto_conditioning, diffusion_conditioning = self.get_random_conditioning_latents()
        # autoregressive and diffusion conditioning are moved to the specified device.
        auto_conditioning = auto_conditioning.to(self.device)
        diffusion_conditioning = diffusion_conditioning.to(self.device)
        # Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free, cond_free_k=cond_free_k)

        # as we are not training models, we do not need gradients here.
        with torch.no_grad():
            samples = []
            # number of batches is calculated here
            num_batches = num_autoregressive_samples // self.autoregressive_batch_size
            # the stop mel token
            stop_mel_token = self.autoregressive.stop_mel_token
            # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output
            calm_token = 83
            if verbose:
                print("Generating autoregressive samples..")
            # if the code is NOT running on M1/M2 chips, then the following code is executed, otherwise, the else
            # section both of them are related to the autoregression part.
            if not torch.backends.mps.is_available():
                # models are moved to cuda and autocast is enabled if the half precision is enabled.
                with self.temporary_cuda(self.autoregressive
                ) as autoregressive, torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
                    # here for each batch of data:
                    for b in tqdm(range(num_batches), disable=not verbose):
                        # the inference_speech function is called to generate a mel spectrogram from the autoregressive
                        # model. parameters are defined inside inference_speech.
                        codes = autoregressive.inference_speech(auto_conditioning, text_tokens,
                                                                    do_sample=True,
                                                                    top_p=top_p,
                                                                    temperature=temperature,
                                                                    num_return_sequences=self.autoregressive_batch_size,
                                                                    length_penalty=length_penalty,
                                                                    repetition_penalty=repetition_penalty,
                                                                    max_generate_length=max_mel_tokens,
                                                                    **hf_generate_kwargs)
                        # if the generated mel spectrogram is not padded, then it is padded to reach the maximum length.
                        padding_needed = max_mel_tokens - codes.shape[1]
                        codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                        # finally codes are appended to the samples list.
                        samples.append(codes)
            else:
                # half precision is not supported in M1/M2 chips, so we do not need to enable autocast.
                with self.temporary_cuda(self.autoregressive) as autoregressive:
                    for b in tqdm(range(num_batches), disable=not verbose):
                        # rest of the codes are exactly same as the above section
                        codes = autoregressive.inference_speech(auto_conditioning, text_tokens,
                                                                    do_sample=True,
                                                                    top_p=top_p,
                                                                    temperature=temperature,
                                                                    num_return_sequences=self.autoregressive_batch_size,
                                                                    length_penalty=length_penalty,
                                                                    repetition_penalty=repetition_penalty,
                                                                    max_generate_length=max_mel_tokens,
                                                                    **hf_generate_kwargs)
                        padding_needed = max_mel_tokens - codes.shape[1]
                        codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                        samples.append(codes)

            clip_results = []
            # if the code is NOT running on M1/M2 chips, then the following code is executed, otherwise, the else
            # section both of them are related to the CLVP-CVVP part.
            if not torch.backends.mps.is_available():
                # models are moved to cuda and autocast is enabled if the half precision is enabled.
                with self.temporary_cuda(self.clvp) as clvp, torch.autocast(
                    device_type="cuda" if not torch.backends.mps.is_available() else 'mps', dtype=torch.float16, enabled=self.half
                ):
                    # if the cvvp_amount is not 0, then it has impact on selecting the best output from the
                    # autoregressive model and should be loaded. so, if it has not been already loaded (meaning that
                    # the self.cvvp is still none, then it is loaded using load_cvvp function.
                    if cvvp_amount > 0:
                        if self.cvvp is None:
                            self.load_cvvp()
                        self.cvvp = self.cvvp.to(self.device)
                    if verbose:
                        if self.cvvp is None:
                            print("Computing best candidates using CLVP")
                        else:
                            print(f"Computing best candidates using CLVP {((1-cvvp_amount) * 100):2.0f}% and CVVP {(cvvp_amount * 100):2.0f}%")
                    # for each batch in the data:
                    for batch in tqdm(samples, disable=not verbose):
                        for i in range(batch.shape[0]):
                            # This function performs some padding on coded audio that fixes a mismatch issue between
                            # what the diffusion model was trained on and what the autoregressive code generator
                            # creates (which has no padding or end).
                            batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                        # if the cvvp_amount is 1, then the clvp will not be used. otherwise, it will be used.
                        if cvvp_amount != 1:
                            clvp_out = clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)
                        # if there are autoregressive conditions and cvvp_amount is not 0, then the cvvp will have
                        # effect and the following code will be run. if the condition of the following if is not met,
                        # the clvp_out will be append to clip_results.
                        if auto_conds is not None and cvvp_amount > 0:
                            cvvp_accumulator = 0
                            for cl in range(auto_conds.shape[1]):
                                cvvp_accumulator = cvvp_accumulator + self.cvvp(auto_conds[:, cl].repeat(batch.shape[0], 1, 1), batch, return_loss=False)
                            cvvp = cvvp_accumulator / auto_conds.shape[1]
                            # if cvvp_amount is 1, cvvp is the only factor in selecting the best output and the 1st
                            # section will be run, otherwise, the else section.
                            if cvvp_amount == 1:
                                clip_results.append(cvvp)
                            else:
                                clip_results.append(cvvp * cvvp_amount + clvp_out * (1-cvvp_amount))
                        else:
                            clip_results.append(clvp_out)
                    # concatenating the results of the loop into a single tensor.
                    clip_results = torch.cat(clip_results, dim=0)
                    samples = torch.cat(samples, dim=0)
                    # selecting the k best results from the samples using the clip_results.
                    best_results = samples[torch.topk(clip_results, k=k).indices]
            else:
                # half precision is not supported in M1/M2 chips, so we do not need to enable autocast.
                # rest of the codes are exactly same as the above section.
                with self.temporary_cuda(self.clvp) as clvp:
                    if cvvp_amount > 0:
                        if self.cvvp is None:
                            self.load_cvvp()
                        self.cvvp = self.cvvp.to(self.device)
                    if verbose:
                        if self.cvvp is None:
                            print("Computing best candidates using CLVP")
                        else:
                            print(f"Computing best candidates using CLVP {((1-cvvp_amount) * 100):2.0f}% and CVVP {(cvvp_amount * 100):2.0f}%")
                    for batch in tqdm(samples, disable=not verbose):
                        for i in range(batch.shape[0]):
                            batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                        if cvvp_amount != 1:
                            clvp_out = clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)
                        if auto_conds is not None and cvvp_amount > 0:
                            cvvp_accumulator = 0
                            for cl in range(auto_conds.shape[1]):
                                cvvp_accumulator = cvvp_accumulator + self.cvvp(auto_conds[:, cl].repeat(batch.shape[0], 1, 1), batch, return_loss=False)
                            cvvp = cvvp_accumulator / auto_conds.shape[1]
                            if cvvp_amount == 1:
                                clip_results.append(cvvp)
                            else:
                                clip_results.append(cvvp * cvvp_amount + clvp_out * (1-cvvp_amount))
                        else:
                            clip_results.append(clvp_out)
                    clip_results = torch.cat(clip_results, dim=0)
                    samples = torch.cat(samples, dim=0)
                    best_results = samples[torch.topk(clip_results, k=k).indices]
            # if the cvvp is used (cvvp_amount is not 0), then we move it back to cpu
            if self.cvvp is not None:
                self.cvvp = self.cvvp.cpu()
            del samples

            # Original Comment: The diffusion model actually wants the last hidden layer from the autoregressive model
            # as conditioning inputs. Re-produce those for the top results. This could be made more efficient by
            # storing all of these results, but will increase memory usage.

            # this part loads the autoregressive model and generates the conditioning latents for the best results for
            # further use. the first if is for the case that the code is NOT running on M1/M2 chips, otherwise,
            # the else section.
            if not torch.backends.mps.is_available():
                # models are moved to cuda and autocast is enabled if the half precision is enabled.
                with self.temporary_cuda(
                    self.autoregressive
                ) as autoregressive, torch.autocast(
                    device_type="cuda" if not torch.backends.mps.is_available() else 'mps', dtype=torch.float16, enabled=self.half
                ):
                    # best_latents is restored using best results and text tokens.
                    best_latents = autoregressive(
                        auto_conditioning.repeat(k, 1), text_tokens.repeat(k, 1),
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), best_results,
                        torch.tensor(
                            [best_results.shape[-1]*self.autoregressive.mel_length_compression],
                            device=text_tokens.device),
                        return_latent=True, clip_inputs=False)
                    del auto_conditioning
            else:
                # the following section is exactly same as the above section. except that autocast is not enabled.
                with self.temporary_cuda(
                    self.autoregressive
                ) as autoregressive:
                    best_latents = autoregressive(auto_conditioning.repeat(k, 1), text_tokens.repeat(k, 1),
                                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), best_results,
                                                    torch.tensor([best_results.shape[-1]*self.autoregressive.mel_length_compression], device=text_tokens.device),
                                                    return_latent=True, clip_inputs=False)
                    del auto_conditioning

            if verbose:
                print("Transforming autoregressive outputs into audio..")
            # the following sections transforms the output of autoregressive model into audio. There are again two
            # sections, one for running code on M1/M2 chips and the other for the rest of the cases.

            # wav candidates are stored in this list.
            wav_candidates = []
            if not torch.backends.mps.is_available():
                # moving diffusion and vocoder models to cuda
                with self.temporary_cuda(self.diffusion) as diffusion, self.temporary_cuda(
                    self.vocoder
                ) as vocoder:
                    # for each of the k best results:
                    for b in range(best_results.shape[0]):
                        # generated codes and their corresponding latents are stored in these variables.
                        codes = best_results[b].unsqueeze(0)
                        latents = best_latents[b].unsqueeze(0)

                        # Find the first occurrence of the "calm" token and trim the codes to that.
                        ctokens = 0
                        for k in range(codes.shape[-1]):
                            if codes[0, k] == calm_token:
                                ctokens += 1
                            else:
                                ctokens = 0
                            if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                                latents = latents[:, :k]
                                break
                        mel = do_spectrogram_diffusion(diffusion, diffuser, latents, diffusion_conditioning, temperature=diffusion_temperature, 
                                                    verbose=verbose)
                        wav = vocoder.inference(mel)
                        wav_candidates.append(wav.cpu())
            else:
                diffusion, vocoder = self.diffusion, self.vocoder
                diffusion_conditioning = diffusion_conditioning.cpu()
                for b in range(best_results.shape[0]):
                    codes = best_results[b].unsqueeze(0).cpu()
                    latents = best_latents[b].unsqueeze(0).cpu()

                    # Find the first occurrence of the "calm" token and trim the codes to that. calm token is the token
                    # for coding silence, which is fixed in place with "fix_autoregressive_output".
                    #
                    # This is how it works: for each code in the codes, if it is equal to calm_token, then the counter
                    # which is named ctokens start to count number of calm_tokens in a row after the first calm_token.
                    # if the counter reaches 8, then the latents are trimmed to that point, because 8 tokens gives the
                    # diffusion model some "breathing room" to terminate speech. Also, if it reaches a token which is not
                    # equal to calm_token, then the counter is reset to 0 again.
                    ctokens = 0
                    for k in range(codes.shape[-1]):
                        if codes[0, k] == calm_token:
                            ctokens += 1
                        else:
                            ctokens = 0
                        if ctokens > 8:
                            latents = latents[:, :k]
                            break
                    # the trimmed latenet is given to do_spectrogram_diffusion function to generate a mel spectrogram.
                    # then the mel spectrogram is given to vocoder to generate the audio clip. the details are in
                    # the related functions.
                    mel = do_spectrogram_diffusion(diffusion, diffuser, latents, diffusion_conditioning,
                                                   temperature=diffusion_temperature, verbose=verbose)
                    wav = vocoder.inference(mel)
                    # finally the generated audio is moved to cpu and append to wav_candidates list.
                    wav_candidates.append(wav.cpu())

            def potentially_redact(clip, text):
                """
                This function redacts the given clip if the enable_redaction is True. otherwise, it will return the clip
                """

                # if enable_redaction is true, the audio parts aligned with the text in [] will be removed from the
                # final audio. the details are in the aligner class.
                if self.enable_redaction:
                    return self.aligner.redact(clip.squeeze(1), text).unsqueeze(1)
                # if enable_redaction is False, the clip will be returned without change.
                return clip
            # applies the potentially_redact function to all the wav_candidates.
            wav_candidates = [potentially_redact(wav_candidate, text) for wav_candidate in wav_candidates]
            # if the number of wav_candidates is more than 1, then the wav_candidates will be returned.
            # otherwise, the only element of wav_candidate will be returned instead of a size-one-list.
            if len(wav_candidates) > 1:
                res = wav_candidates
            else:
                res = wav_candidates[0]

            # if return_deterministic_state is true, then the deterministic_seed, text, voice_samples, and
            # conditioning_latents will be returned as well.
            if return_deterministic_state:
                return res, (deterministic_seed, text, voice_samples, conditioning_latents)
            else:
                return res

    def deterministic_state(self, seed=None):
        """
        Sets the random seeds that tortoise uses to the current time() and returns that seed so results can be
        reproduced.
        """
        seed = int(time()) if seed is None else seed
        torch.manual_seed(seed)
        random.seed(seed)
        # Can't currently set this because of CUBLAS. TODO: potentially enable it if necessary.
        # torch.use_deterministic_algorithms(True)

        return seed

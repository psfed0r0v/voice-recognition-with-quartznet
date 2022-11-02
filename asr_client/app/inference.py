import numpy as np
from .speech_utils import AudioSegment, SpeechClient
from omegaconf import OmegaConf
import yaml


def read_config(conf_path):
    with open(conf_path, "r") as f:
        parsed_yaml = yaml.safe_load(f)
    return OmegaConf.create(parsed_yaml)


async def get_transcription(filename):
    config = read_config("config.yaml")
    feature_preprocessing_params = config.feature_preprocessing_params
    inference_params = config.inference_params

    protocol = inference_params.protocol.lower()
    model_name = inference_params.model_name

    speech_client = SpeechClient(
        inference_params.triton_url,
        protocol,
        model_name,
        1,
        inference_params.batch_size,
        verbose=inference_params.verbose,
        mode="synchronous",
    )

    filenames = [filename]
    # Read the audio files
    # Group requests in batches
    audio_idx = 0
    last_request = False
    predictions = []
    while not last_request:
        batch_audio_samples = []
        batch_filenames = []

        for idx in range(inference_params.batch_size):
            filename = filenames[audio_idx]
            audio = AudioSegment.from_file(
                filename, offset=0, duration=inference_params.fixed_size
            ).samples
            if inference_params.fixed_size:
                audio = np.resize(audio, inference_params.fixed_size)

            audio_idx = (audio_idx + 1) % len(filenames)
            if audio_idx == 0:
                last_request = True

            batch_audio_samples.append(audio)
            batch_filenames.append(filename)

        predictions += speech_client.recognize(
            batch_audio_samples, batch_filenames, feature_preprocessing_params
        )

    return "".join(predictions[0])

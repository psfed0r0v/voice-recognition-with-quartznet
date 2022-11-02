import soundfile as sf
import librosa
import torch
import numpy as np
import tritonclient.grpc
from tritonclient.utils import triton_to_np_dtype, np_to_triton_dtype
from .common.features import FilterbankFeatures

import sys
import os

if "./triton" not in sys.path:
    sys.path.append(os.path.join(sys.path[0], "../"))
from .common.text import _clean_text

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}


def normalize_string(s, labels, table, **unused_kwargs):
    """
    Normalizes string. For example:
    'call me at 8:00 pm!' -> 'call me at eight zero pm'

    Args:
        s: string to normalize
        labels: labels used during model training.

    Returns:
            Normalized string
    """

    def good_token(token, labels):
        s = set(labels)
        for t in token:
            if not t in s:
                return False
        return True

    try:
        text = _clean_text(s, ["english_cleaners"], table).strip()
        return "".join([t for t in text if good_token(t, labels=labels)])
    except:
        print("WARNING: Normalizing {} failed".format(s))
        return None


def ctc_decoder_predictions_tensor(prediction_cpu_tensor, batch_size, labels):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Returns prediction
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    blank_id = len(labels) - 1
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over batch
    prediction_cpu_tensor = prediction_cpu_tensor.reshape(
        (batch_size, int(prediction_cpu_tensor.size()[0] / batch_size))
    )
    for ind in range(batch_size):
        prediction = prediction_cpu_tensor[ind].tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels) - 1  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = "".join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses


class SpeechClient(object):
    def __init__(
            self,
            url,
            protocol,
            model_name,
            model_version,
            batch_size,
            verbose=False,
            mode="batch",
    ):

        self.model_name = model_name
        self.model_version = model_version
        self.verbose = verbose
        self.batch_size = batch_size
        self.transpose_audio_features = False
        self.grpc_stub = None
        self.ctx = None
        self.correlation_id = 0
        self.first_run = True
        if mode == "streaming" or mode == "asynchronous":
            self.correlation_id = 1

        self.buffer = []

        if protocol == "grpc":
            self.prtcl_client = tritonclient.grpc
        else:
            raise Exception("Support only grpc protocol")

        self.triton_client = self.prtcl_client.InferenceServerClient(
            url=url, verbose=self.verbose
        )
        self.labels = [
            " ",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "'",
            "<BLANK>",
        ]

        (
            self.audio_signals_name,
            self.transcripts_name,
            self.audio_signals_type,
            self.num_samples_type,
        ) = ("audio_signal", "logprobs", np.float32, np.int32)

    def postprocess(self, transcript_values, labels):
        res = []
        for transcript, filename in zip(transcript_values, labels):
            # print("---")
            # print("File: ", filename)
            t = ctc_decoder_predictions_tensor(transcript, self.batch_size, self.labels)
            # print("Final transcript: ", t)
            # print("---")
            res.append(t)
        return res

    def recognize(self, audio_signal, filenames, feature_preprocessing_params):
        input_batch = []
        input_filenames = []

        for idx in range(self.batch_size):
            input_batch.append(audio_signal[idx].astype(self.audio_signals_type))
            input_filenames.append(filenames[idx])

        print("Sending request to transcribe file(s):", ",".join(input_filenames))

        inputs = []
        FeaturesCalc = FilterbankFeatures(**feature_preprocessing_params)
        input_batch, _ = FeaturesCalc.calculate_features(
            torch.tensor(input_batch), torch.tensor([len(input_batch[0])])
        )
        input_batch = np.asarray(input_batch)

        inputs.append(
            self.prtcl_client.InferInput(
                self.audio_signals_name,
                input_batch.shape,
                np_to_triton_dtype(input_batch.dtype),
            )
        )

        inputs[0].set_data_from_numpy(input_batch)

        outputs = []
        outputs.append(self.prtcl_client.InferRequestedOutput(self.transcripts_name))

        triton_result = self.triton_client.infer(
            self.model_name, inputs=inputs, outputs=outputs
        )
        transcripts = triton_result.as_numpy(self.transcripts_name)

        transcripts = torch.tensor(transcripts).argmax(dim=-1, keepdim=False).int()

        result = self.postprocess(transcripts, input_filenames)

        return result


def normalize_signal(signal, gain=None):
    """
    Normalize float32 signal to [-1, 1] range
    """
    if gain is None:
        gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=16000, trim=False, trim_db=60):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype("float32")
        if samples.dtype in np.sctypes["int"]:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes["float"]:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(
            cls,
            filename,
            target_sr=16000,
            int_values=False,
            offset=0,
            duration=0,
            trim=False,
    ):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        if ".ogg" in filename:
            data, samplerate = librosa.load(filename, sr=target_sr)
            filename = f"{filename[:-4]}.wav"
            sf.write(filename, data, samplerate)
        with sf.SoundFile(filename, "r") as f:
            dtype = "int32" if int_values else "float32"
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

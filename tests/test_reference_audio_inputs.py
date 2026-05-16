import os

import numpy as np
import pytest
import soundfile as sf

from vieneu.base import coerce_ref_audio_path
from vieneu.turbo import TurboVieNeuTTS


def test_coerce_ref_audio_path_accepts_gradio_dict():
    assert coerce_ref_audio_path({"path": "sample.wav"}) == "sample.wav"
    assert coerce_ref_audio_path({"name": "upload.mp3"}) == "upload.mp3"
    assert coerce_ref_audio_path({"orig_name": "upload.mp3"}) is None


def test_coerce_ref_audio_path_saves_gradio_tuple():
    audio = np.zeros(1600, dtype=np.float32)

    path = coerce_ref_audio_path((16000, audio))

    try:
        assert os.path.exists(path)
        saved_audio, sample_rate = sf.read(path)
        assert sample_rate == 16000
        assert len(saved_audio) == len(audio)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_turbo_rejects_reference_audio_cloning():
    tts = TurboVieNeuTTS.__new__(TurboVieNeuTTS)

    with pytest.raises(ValueError, match="does not support reference-audio voice cloning"):
        tts.infer("Xin chao", ref_audio="sample.wav", ref_text="Xin chao")

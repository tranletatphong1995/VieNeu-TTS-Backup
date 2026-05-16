"""
OmniVoice Vietnamese Backend for VieNeu-TTS
============================================

Wrapper around `splendor1811/omnivoice-vietnamese` — a fine-tuned version of
OmniVoice (k2-fsa) trained on 1,000h of Vietnamese speech data.

Architecture: Diffusion Language Model with Qwen3-0.6B backbone.
License: Apache 2.0

Installation (thêm vào môi trường đã có GPU deps):
    pip install omnivoice

References:
    - Model: https://huggingface.co/splendor1811/omnivoice-vietnamese
    - Base:  https://huggingface.co/k2-fsa/OmniVoice
    - Paper: https://arxiv.org/abs/2604.00688
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base import coerce_ref_audio_path

logger = logging.getLogger("OmniVoiceBackend")

# Default model hosted on HuggingFace
_DEFAULT_MODEL_ID = "splendor1811/omnivoice-vietnamese"

# Default inference parameters (tradeoff tốc độ / chất lượng)
_DEFAULT_NUM_STEP = 8          # 8 steps = nhanh + chất lượng tốt
_DEFAULT_GUIDANCE = 2.0        # Độ bám sát giọng tham chiếu


class OmniVoiceTTS:
    """
    VieNeu-TTS compatible wrapper cho OmniVoice Vietnamese.

    Cung cấp interface tương thích với các engine hiện có:
    - `infer(text, ref_audio, ref_text)` → np.ndarray (float32, 24kHz)
    - `sample_rate` property
    - `list_preset_voices()` → [] (OmniVoice không dùng preset)
    - `close()` để giải phóng bộ nhớ GPU

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Mặc định: 'splendor1811/omnivoice-vietnamese'.
    device : str
        Device để inference. 'cuda' (GPU), 'cuda:0', 'cpu'.
        Nếu None, tự detect (ưu tiên CUDA nếu có).
    dtype : torch.dtype | None
        Kiểu dữ liệu. Mặc định float16 trên GPU, float32 trên CPU.
    num_step : int
        Số diffusion steps. Ít bước = nhanh hơn nhưng chất lượng có thể giảm.
        Khuyến nghị: 8 (nhanh), 16 (cân bằng), 32 (chất lượng cao nhất).
    guidance_scale : float
        Hệ số bám sát giọng tham chiếu. Mặc định 2.0.
    use_compile : bool
        Bật torch.compile để tăng tốc inference (~2-3x). Lần đầu chạy
        sẽ cần warmup 30-60s. Khuyến nghị bật cho production.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype=None,
        num_step: int = _DEFAULT_NUM_STEP,
        guidance_scale: float = _DEFAULT_GUIDANCE,
        use_compile: bool = False,
    ):
        self._model_id = model_id
        self._num_step = num_step
        self._guidance_scale = guidance_scale
        self._use_compile = use_compile
        self._model = None
        self._voice_prompt_cache: dict = {}   # cache voice prompt để tái dụng

        # Detect device
        self._device = self._resolve_device(device)
        logger.info(f"🌟 OmniVoice backend sẽ dùng device: {self._device}")

        # Detect dtype
        import torch
        if dtype is not None:
            self._dtype = dtype
        elif "cuda" in self._device:
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32
            logger.warning(
                "⚠️  OmniVoice đang chạy trên CPU — inference sẽ rất chậm. "
                "Khuyến nghị dùng GPU (NVIDIA CUDA) để có hiệu suất tốt nhất."
            )

        self._load_model()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_device(self, device: Optional[str]) -> str:
        """Tự detect CUDA nếu device không được chỉ định."""
        if device is not None:
            return device
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(f"🎮 OmniVoice: Phát hiện GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self) -> None:
        """Load OmniVoice model từ HuggingFace."""
        try:
            import torch
            from omnivoice import OmniVoice
        except ImportError as e:
            raise ImportError(
                "Thiếu package 'omnivoice'. Hãy cài đặt:\n"
                "  pip install omnivoice\n"
                "Nếu dùng GPU, cài PyTorch trước:\n"
                "  pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 "
                "--extra-index-url https://download.pytorch.org/whl/cu128"
            ) from e

        logger.info(f"🔄 Đang tải OmniVoice Vietnamese từ '{self._model_id}'...")
        logger.info(f"   Device: {self._device} | Dtype: {self._dtype}")

        self._model = OmniVoice.from_pretrained(
            self._model_id,
            device_map=self._device,
            dtype=self._dtype,
        )

        logger.info("✅ OmniVoice Vietnamese tải xong!")

        # Bật torch.compile nếu yêu cầu (chỉ trên GPU)
        if self._use_compile and "cuda" in self._device:
            self._apply_torch_compile()

    def _apply_torch_compile(self) -> None:
        """Áp dụng torch.compile và warmup model."""
        import torch
        from omnivoice import OmniVoiceGenerationConfig

        logger.info("⚡ Đang áp dụng torch.compile (mode='reduce-overhead')...")
        torch.set_float32_matmul_precision("high")

        try:
            self._model.llm = torch.compile(
                self._model.llm,
                mode="reduce-overhead",
                dynamic=True,
            )
            logger.info("🔥 torch.compile thành công! Đang warmup (3 lần)...")

            # Warmup với dummy audio để trigger compilation
            config = OmniVoiceGenerationConfig(
                num_step=self._num_step,
                guidance_scale=self._guidance_scale,
            )
            _dummy_voice = self._model.create_voice_clone_prompt(
                ref_audio=str(Path(__file__).parent / "assets" / "samples" / "vi_male_1.wav"),
                ref_text="Xin chào.",
            ) if (Path(__file__).parent / "assets" / "samples" / "vi_male_1.wav").exists() else None

            if _dummy_voice:
                for i in range(3):
                    logger.info(f"   Warmup {i+1}/3...")
                    self._model.generate(
                        text="Xin chào.",
                        language="vietnamese",
                        voice_clone_prompt=_dummy_voice,
                        generation_config=config,
                    )
                logger.info("✅ Warmup hoàn thành! OmniVoice sẵn sàng ở tốc độ tối đa.")
            else:
                logger.info("⚠️  Không tìm thấy audio mẫu để warmup. "
                            "Lần generate đầu tiên sẽ chậm hơn do JIT compilation.")
        except Exception as e:
            logger.warning(f"⚠️  torch.compile thất bại: {e}. Dùng chế độ bình thường.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface (tương thích với VieNeu-TTS engines)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        """Sample rate đầu ra: 24,000 Hz."""
        return 24_000

    def list_preset_voices(self) -> list:
        """
        OmniVoice Vietnamese không có preset voices.
        Trả về list rỗng để UI xử lý đúng cách.
        """
        return []

    def get_preset_voice(self, voice_name=None):
        """OmniVoice không support preset voice. Raise ValueError."""
        raise ValueError(
            "OmniVoice Vietnamese không hỗ trợ preset voices.\n"
            "Vui lòng dùng tab 'Voice Cloning' và upload audio mẫu (3-10 giây)."
        )

    def create_voice_prompt(self, ref_audio: str, ref_text: str):
        """
        Tạo và cache voice prompt cho ref_audio.
        Dùng cache để tránh encode lại nhiều lần khi generate nhiều chunks.

        Parameters
        ----------
        ref_audio : str
            Đường dẫn đến file audio tham chiếu (WAV/MP3, 3-10 giây).
        ref_text : str
            Transcript chính xác của audio tham chiếu.

        Returns
        -------
        voice_prompt : OmniVoice voice prompt object
        """
        ref_audio_path = coerce_ref_audio_path(ref_audio)
        cache_key = f"{ref_audio_path}::{ref_text}"
        if cache_key not in self._voice_prompt_cache:
            logger.info("🎙️  Đang encode giọng tham chiếu...")
            self._voice_prompt_cache[cache_key] = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )
            logger.info("✅ Encode giọng tham chiếu xong (đã cache).")
        return self._voice_prompt_cache[cache_key]

    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: Optional[str] = None,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Tổng hợp giọng nói từ văn bản.

        Parameters
        ----------
        text : str
            Văn bản cần tổng hợp (tiếng Việt).
        ref_audio : str | Path | None
            File audio tham chiếu cho voice cloning. BẮT BUỘC với OmniVoice.
        ref_text : str | None
            Transcript chính xác của audio tham chiếu. BẮT BUỘC với OmniVoice.
        num_step : int | None
            Override số diffusion steps cho lần generate này.
        guidance_scale : float | None
            Override guidance scale cho lần generate này.

        Returns
        -------
        np.ndarray
            Audio waveform, float32, sample_rate=24000.

        Raises
        ------
        ValueError
            Nếu không cung cấp ref_audio và ref_text.
        RuntimeError
            Nếu model chưa được load.
        """
        if self._model is None:
            raise RuntimeError("OmniVoice model chưa được load. Gọi _load_model() trước.")

        ref_audio_path = coerce_ref_audio_path(ref_audio)
        if not ref_audio_path or not ref_text or not ref_text.strip():
            raise ValueError(
                "OmniVoice cần audio tham chiếu và transcript để clone giọng.\n"
                "Vui lòng upload file WAV/MP3 (3-10 giây) và điền transcript chính xác."
            )

        # Resolve inference config
        try:
            from omnivoice import OmniVoiceGenerationConfig
            config = OmniVoiceGenerationConfig(
                num_step=num_step or self._num_step,
                guidance_scale=guidance_scale or self._guidance_scale,
            )
        except ImportError:
            config = None   # Dùng default config của model

        # Lấy voice prompt (có cache)
        voice_prompt = self.create_voice_prompt(ref_audio_path, ref_text.strip())

        # Generate
        logger.debug(f"🎙️  OmniVoice generate: {len(text)} ký tự, "
                     f"num_step={num_step or self._num_step}")

        generate_kwargs = {
            "text": text,
            "language": "vietnamese",
            "voice_clone_prompt": voice_prompt,
        }
        if config is not None:
            generate_kwargs["generation_config"] = config

        audio_output = self._model.generate(**generate_kwargs)

        # audio_output là tensor [1, T] hoặc [T], convert sang np.ndarray float32
        if hasattr(audio_output, "__iter__"):
            # Có thể là list hoặc tuple
            audio_tensor = audio_output[0] if isinstance(audio_output, (list, tuple)) else audio_output
        else:
            audio_tensor = audio_output

        # Convert torch tensor → numpy
        if hasattr(audio_tensor, "cpu"):
            wav = audio_tensor.cpu().numpy()
        elif hasattr(audio_tensor, "numpy"):
            wav = audio_tensor.numpy()
        else:
            wav = np.array(audio_tensor, dtype=np.float32)

        # Đảm bảo 1D
        wav = np.squeeze(wav).astype(np.float32)
        return wav

    def close(self) -> None:
        """Giải phóng model khỏi bộ nhớ GPU/RAM."""
        if self._model is not None:
            try:
                import torch
                del self._model
                self._model = None
                self._voice_prompt_cache.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("✅ OmniVoice model đã được giải phóng.")
            except Exception as e:
                logger.warning(f"Lỗi khi giải phóng OmniVoice model: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "not loaded"
        return (
            f"OmniVoiceTTS("
            f"model='{self._model_id}', "
            f"device='{self._device}', "
            f"num_step={self._num_step}, "
            f"compiled={self._use_compile}, "
            f"status={status})"
        )

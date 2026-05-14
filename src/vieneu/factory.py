

def Vieneu(mode="turbo", **kwargs):
    """
    Factory function for VieNeu-TTS.

    Args:
        mode: 'standard' (CPU/GPU-GGUF), 'fast' (GPU-LMDeploy), 'remote' (API),
              'xpu' (Intel GPU), 'turbo' (CPU llama.cpp),
              'omnivoice' (OmniVoice Vietnamese — zero-shot voice cloning)
        **kwargs: Arguments for chosen class

    Returns:
        An instance of the chosen TTS implementation.

    OmniVoice mode kwargs:
        model_id (str): HuggingFace model ID. Default: 'splendor1811/omnivoice-vietnamese'
        device (str): 'cuda', 'cuda:0', or 'cpu'. Auto-detects if None.
        num_step (int): Diffusion steps. Default 8. More = higher quality, slower.
        guidance_scale (float): Voice adherence strength. Default 2.0.
        use_compile (bool): Enable torch.compile for faster inference. Default False.
    """
    match mode:
        case "remote" | "api":
            from .remote import RemoteVieNeuTTS
            return RemoteVieNeuTTS(**kwargs)
        case "fast" | "gpu":
            from .fast import FastVieNeuTTS
            return FastVieNeuTTS(**kwargs)
        case "turbo":
            from .turbo import TurboVieNeuTTS
            return TurboVieNeuTTS(**kwargs)
        case "xpu":
            try:
                from .core_xpu import XPUVieNeuTTS
                return XPUVieNeuTTS(**kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to load XPU backend. Ensure Intel GPU drivers and torch.xpu are installed: {e}") from e
        case "standard":
            from .standard import VieNeuTTS
            return VieNeuTTS(**kwargs)
        case "omnivoice":
            try:
                from .omnivoice_backend import OmniVoiceTTS
                return OmniVoiceTTS(**kwargs)
            except ImportError as e:
                raise ImportError(
                    "Không thể load OmniVoice backend. Hãy cài đặt:\n"
                    "  pip install omnivoice\n"
                    f"Chi tiết lỗi: {e}"
                ) from e
        case _:
            raise ValueError(
                f"Mode '{mode}' không hợp lệ. Các mode được hỗ trợ: "
                "'turbo', 'standard', 'fast', 'remote', 'xpu', 'omnivoice'"
            )

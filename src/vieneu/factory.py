def Vieneu(mode="v3turbo", **kwargs):
    """
    Factory function for VieNeu-TTS.

    Args:
        mode: 'v3turbo' (DEFAULT) — VieNeu-TTS v3 Turbo, 48 kHz. CPU runs torch-free
              via ONNX Runtime; GPU uses PyTorch. Works with the minimal install.
              Other modes need extras (``pip install vieneu[gpu]``):
              'standard' (CPU/GPU-GGUF), 'fast' (GPU-LMDeploy), 'turbo'/'turbo_gpu',
              'remote' (API), 'xpu' (Intel GPU),
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
        case "v3turbo":
            from .v3turbo import V3TurboVieNeuTTS
            return V3TurboVieNeuTTS(**kwargs)
        case "remote" | "api":
            from .remote import RemoteVieNeuTTS
            return RemoteVieNeuTTS(**kwargs)
        case "fast" | "gpu":
            from .fast import FastVieNeuTTS
            return FastVieNeuTTS(**kwargs)
        case "turbo":
            from .turbo import TurboVieNeuTTS
            return TurboVieNeuTTS(**kwargs)
        case "turbo_gpu":
            from .turbo import TurboGPUVieNeuTTS
            return TurboGPUVieNeuTTS(**kwargs)
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

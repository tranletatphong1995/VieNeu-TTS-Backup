import gradio as gr
import numpy as np
import os
import re
import sys
import tempfile
import scipy.io.wavfile as wavfile
from pathlib import Path

import threading

try:
    import torch
except ImportError:
    torch = None

# Ensure this app imports the local source tree first (avoid stale editable installs).
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_SRC = _PROJECT_ROOT / "src"
if _LOCAL_SRC.exists():
    _local_src_str = str(_LOCAL_SRC)
    if _local_src_str not in sys.path:
        sys.path.insert(0, _local_src_str)

from vieneu_utils.srt_audio import format_timecode, plan_subtitle_segments, read_srt
from vieneu.base import coerce_ref_audio_path

# Avoid Windows console UnicodeEncodeError when printing emoji/status logs.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# ── Lazy-loaded engine instances (mỗi engine load riêng, chỉ load khi cần) ──
_tts_vieneu = None       # VieNeu-TTS engine (turbo/standard/fast)
_tts_omnivoice = None    # OmniVoice Vietnamese engine
_lock_vieneu = threading.Lock()
_lock_omnivoice = threading.Lock()

# ── Labels hiển thị cho từng engine ──────────────────────────────────────────
ENGINE_CHOICES = [
    ("🎯 VieNeu-TTS (Auto — turbo/standard/fast)", "vieneu"),
    ("🌟 OmniVoice Vietnamese (Zero-shot Voice Cloning)", "omnivoice"),
]

ENGINE_DESCRIPTIONS = {
    "vieneu": (
        "**VieNeu-TTS** — Engine mặc định.\n"
        "Tự động chọn mode tốt nhất: **Fast** (LMDeploy GPU) → **Standard** (PyTorch GPU) → **Turbo** (CPU).\n"
        "Hỗ trợ cả giọng preset và voice cloning."
    ),
    "omnivoice": (
        "**OmniVoice Vietnamese** — Fine-tuned trên 1,000h tiếng Việt.\n"
        "Kiến trúc: Diffusion Language Model + Qwen3-0.6B.\n"
        "**Chỉ hỗ trợ Voice Cloning** — cần upload audio mẫu (3-10 giây).\n"
        "📦 Model: `splendor1811/omnivoice-vietnamese`"
    ),
}


# ── Khởi tạo VieNeu-TTS (lazy, thread-safe, chỉ load 1 lần) ──────────────────────────
def _is_turbo_engine(tts) -> bool:
    return type(tts).__name__ == "TurboVieNeuTTS"


def get_vieneu_tts(require_voice_cloning: bool = False):
    global _tts_vieneu
    if _tts_vieneu is not None and not (require_voice_cloning and _is_turbo_engine(_tts_vieneu)):
        return _tts_vieneu
    with _lock_vieneu:
        if _tts_vieneu is not None and not (require_voice_cloning and _is_turbo_engine(_tts_vieneu)):
            return _tts_vieneu
        from vieneu import Vieneu

        has_cuda = torch is not None and torch.cuda.is_available()
        if require_voice_cloning:
            device = "cuda" if has_cuda else "cpu"
            print(f"Loading VieNeu-TTS Standard mode for voice cloning on {device}...")
            try:
                kwargs = {
                    "mode": "standard",
                    "backbone_device": device,
                    "codec_device": device,
                }
                if has_cuda:
                    kwargs["backbone_repo"] = "pnnbao-ump/VieNeu-TTS"
                _tts_vieneu = Vieneu(**kwargs)
                print("VieNeu-TTS Standard voice cloning engine loaded!")
                return _tts_vieneu
            except Exception as e:
                raise RuntimeError(
                    "VieNeu-TTS voice cloning requires the Standard/Fast backend. "
                    "Turbo CPU does not support reference-audio cloning. "
                    f"Could not load Standard backend: {e}"
                ) from e

        if has_cuda:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

            # Fast mode (LMDeploy) — chất lượng cao nhất, nhanh nhất trên GPU
            try:
                print("🚀 Loading VieNeu-TTS in FAST mode (LMDeploy + GPU)...")
                _tts_vieneu = Vieneu(
                    mode="fast",
                    backbone_repo="pnnbao-ump/VieNeu-TTS",
                    backbone_device="cuda",
                    codec_device="cuda",
                    memory_util=0.4,
                )
                print("✅ Fast mode loaded successfully!")
                return _tts_vieneu
            except Exception as e:
                print(f"⚠️ Fast mode failed: {e}")
                print("🔄 Falling back to Standard mode...")

            # Fallback: Standard mode (PyTorch trên GPU)
            try:
                _tts_vieneu = Vieneu(
                    mode="standard",
                    backbone_repo="pnnbao-ump/VieNeu-TTS",
                    backbone_device="cuda",
                    codec_device="cuda",
                )
                print("✅ Standard mode loaded on GPU!")
                return _tts_vieneu
            except Exception as e:
                print(f"⚠️ Standard GPU mode failed: {e}")
                print("🔄 Falling back to Turbo (CPU) mode...")

        # Fallback cuối: Turbo mode (CPU, luôn work)
        print("💡 Loading VieNeu-TTS in Turbo mode (CPU)...")
        _tts_vieneu = Vieneu(mode="turbo")
        print("✅ Turbo mode loaded!")

    return _tts_vieneu


# ── Khởi tạo OmniVoice (lazy, thread-safe, chỉ load 1 lần) ─────────────────
def get_omnivoice_tts():
    global _tts_omnivoice
    if _tts_omnivoice is not None:       # Fast path — không cần lock
        return _tts_omnivoice
    with _lock_omnivoice:
        if _tts_omnivoice is not None:   # Double-checked locking
            return _tts_omnivoice
        from vieneu import Vieneu
        print("🌟 Loading OmniVoice Vietnamese...")
        _tts_omnivoice = Vieneu(
            mode="omnivoice",
            # device=None → tự detect CUDA / CPU
            num_step=8,
            guidance_scale=2.0,
            use_compile=False,   # Tắt mặc định — user có thể bật qua UI
        )
        if _tts_omnivoice is None:
            raise RuntimeError(
                "Factory Vieneu(mode='omnivoice') trả về None. "
                "Hãy đảm bảo app đang import đúng mã nguồn local trong thư mục hiện tại."
            )
        print("✅ OmniVoice Vietnamese loaded!")
    return _tts_omnivoice


def get_tts(engine: str = "vieneu", require_voice_cloning: bool = False):
    """Trả về TTS engine tương ứng với lựa chọn của user."""
    if engine == "omnivoice":
        return get_omnivoice_tts()
    return get_vieneu_tts(require_voice_cloning=require_voice_cloning)


# ── Chia văn bản thành chunks ≤ max_chars ký tự ─────────────────────────────
def chunk_text(text: str, max_chars: int = 250) -> list[str]:
    """Chia văn bản dài thành các đoạn ngắn, ưu tiên cắt tại dấu câu."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]

        # Ưu tiên 1: dấu kết thúc câu (. ! ?) kèm dấu ngoặc kép nếu có
        cut = -1
        for pattern in [r'[.!?]["\']?\s', r'[.!?]["\']?$']:
            matches = list(re.finditer(pattern, window))
            if matches:
                m = matches[-1]
                pos = m.end()
                if pos > max_chars // 3:
                    cut = pos
                    break

        # Ưu tiên 2: dấu phẩy, chấm phẩy
        if cut == -1:
            m = max(window.rfind(', '), window.rfind('; '))
            if m > max_chars // 3:
                cut = m + 2

        # Ưu tiên 3: khoảng trắng
        if cut == -1:
            m = window.rfind(' ')
            cut = m + 1 if m > 0 else max_chars

        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)

    return [c for c in chunks if c]


# ── Lấy danh sách giọng preset (chỉ VieNeu-TTS) ────────────────────────────
def fetch_voices(engine: str):
    """Load danh sách preset voices. OmniVoice không có preset."""
    if engine == "omnivoice":
        return gr.update(
            choices=[("OmniVoice không có preset voice", "__none__")],
            value="__none__",
            interactive=False,
        )
    try:
        t = get_vieneu_tts()
        voices = t.list_preset_voices()          # [(description, id), ...]
        choices = [(desc, vid) for desc, vid in voices]
        if choices:
            return gr.update(choices=choices, value=choices[0][1], interactive=True)
        return gr.update(
            choices=[('Giọng mặc định', 'default')],
            value='default',
            interactive=True,
        )
    except Exception:
        return gr.update(
            choices=[('Giọng mặc định', 'default')],
            value='default',
            interactive=True,
        )


# ── Cập nhật UI khi user đổi engine ──────────────────────────────────────────────
def on_engine_change(engine: str):
    """
    Khi user chọn engine khác:
    - Pre-load engine trong nền (tránh delay lần generate đầu tiên)
    - Cập nhật UI phần mô tả và voice dropdown
    """
    desc = ENGINE_DESCRIPTIONS.get(engine, "")
    is_omnivoice = engine == "omnivoice"

    # Pre-load OmniVoice ngay khi user chọn (warm up trong cùng thread)
    if is_omnivoice:
        try:
            get_omnivoice_tts()   # sắc load, set _tts_omnivoice trước khi synthesize
        except Exception as e:
            desc += f"\n\n⚠️ **Load OmniVoice thất bại:** {e}"

    load_btn_label = (
        "🌟 OmniVoice không dùng preset — hãy dùng Voice Cloning ↓"
        if is_omnivoice
        else "🔄 Tải danh sách giọng"
    )

    voice_dd_update = gr.update(
        choices=[("OmniVoice không có preset voice", "__none__")],
        value="__none__",
        interactive=False,
    ) if is_omnivoice else gr.update(interactive=True)

    omnivoice_note_update = gr.update(visible=is_omnivoice)
    load_btn_update = gr.update(
        value=load_btn_label,
        interactive=not is_omnivoice,
    )

    return desc, voice_dd_update, omnivoice_note_update, load_btn_update


# ── Hàm tổng hợp chính ──────────────────────────────────────────────────────
def synthesize(
    text,
    engine,
    voice_id,
    ref_audio_path,
    ref_text,
    num_step,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Tổng hợp giọng nói từ văn bản với engine được chọn.
    - VieNeu-TTS: hỗ trợ preset voice HOẶC voice cloning
    - OmniVoice: CHỈ hỗ trợ voice cloning (bắt buộc ref_audio + ref_text)
    """
    if not text or not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản cần đọc."

    try:
        ref_audio_path = coerce_ref_audio_path(ref_audio_path)
    except Exception as e:
        return None, f"❌ Không thể đọc audio tham chiếu: {e}"

    use_cloning = bool(ref_audio_path and ref_text and ref_text.strip())

    if ref_audio_path and not (ref_text and ref_text.strip()):
        return None, (
            "❌ Cần transcript của audio tham chiếu để clone giọng.\n"
            "Vui lòng điền đúng nội dung nói trong file audio tham chiếu."
        )

    # Validate OmniVoice requirements
    if engine == "omnivoice":
        if not ref_audio_path:
            return None, (
                "❌ OmniVoice cần audio tham chiếu!\n"
                "Vui lòng upload file WAV/MP3 (3-10 giây) trong tab 'Voice Cloning'."
            )
        if not ref_text or not ref_text.strip():
            return None, (
                "❌ OmniVoice cần transcript của audio tham chiếu!\n"
                "Vui lòng điền nội dung nói trong file audio vào ô 'Nội dung nói trong file tham chiếu'."
            )

    # Load engine — bắt lỗi nếu package chưa cài hoặc model load thất bại
    try:
        t = get_tts(engine, require_voice_cloning=(engine == "vieneu" and use_cloning))
    except ImportError:
        return None, (
            "❌ Package 'omnivoice' chưa được cài đặt!\n"
            "Chạy lệnh sau để cài:\n"
            "  .venv\\Scripts\\pip install omnivoice"
        )
    except Exception as e:
        return None, f"❌ Không thể khởi động engine {engine}: {e}"

    if t is None:
        return None, f"❌ Engine '{engine}' không khởi động được."

    chunks = chunk_text(text.strip(), max_chars=250)
    total = len(chunks)

    if total == 0:
        return None, "⚠️ Văn bản trống sau khi xử lý."

    # Chuẩn bị voice preset (chỉ VieNeu-TTS)
    voice_data = None
    if engine == "vieneu" and not use_cloning and voice_id and voice_id != "__none__":
        try:
            voice_data = t.get_preset_voice(voice_id)
        except Exception:
            voice_data = None

    # OmniVoice: pre-warm voice prompt cache bằng public API
    if engine == "omnivoice" and use_cloning:
        try:
            progress(0.02, desc="🎙️ Đang encode giọng tham chiếu (OmniVoice)...")
            t.create_voice_prompt(ref_audio_path, ref_text.strip())  # warm cache
        except Exception as e:
            return None, f"❌ Không thể encode giọng tham chiếu: {e}"

    audio_arrays = []     # lưu np.ndarray trực tiếp từ infer()
    failed_chunks = []    # theo dõi các chunk bị lỗi

    for i, chunk in enumerate(chunks):
        progress((i / total) * 0.9 + 0.05, desc=f"🔊 Đang tổng hợp đoạn {i + 1}/{total}...")

        try:
            if engine == "omnivoice":
                # OmniVoice: dùng t.infer() — voice prompt cache được xử lý bên trong
                wav = t.infer(
                    text=chunk,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text.strip(),
                    num_step=int(num_step),
                )

            elif use_cloning:
                wav = t.infer(
                    text=chunk,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text.strip(),
                    max_chars=300,
                )
            elif voice_data is not None:
                wav = t.infer(text=chunk, voice=voice_data, max_chars=300)
            else:
                wav = t.infer(text=chunk, max_chars=300)

            # infer() trả về np.ndarray (float32, sample_rate=24000)
            if wav is not None and len(wav) > 0:
                audio_arrays.append(wav)
            else:
                failed_chunks.append((i + 1, "Kết quả rỗng"))

        except Exception as e:
            failed_chunks.append((i + 1, str(e)))

    # Nếu tất cả chunk đều lỗi
    if not audio_arrays:
        error_detail = "\n".join([f"  • Đoạn {idx}: {msg}" for idx, msg in failed_chunks])
        return None, f"❌ Không tổng hợp được đoạn nào!\n{error_detail}"

    # ── Ghép audio ────────────────────────────────────────────────────────────
    progress(0.96, desc="🔗 Đang ghép các đoạn âm thanh...")

    sample_rate = t.sample_rate   # 24000 Hz

    # Thêm khoảng lặng 200ms giữa các đoạn
    silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    merged = []
    for idx, arr in enumerate(audio_arrays):
        # Đảm bảo array là 1D
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        merged.append(arr.astype(np.float32))
        if idx < len(audio_arrays) - 1:
            merged.append(silence)

    combined = np.concatenate(merged)

    # Chuẩn hóa sang int16 để lưu WAV
    combined_int16 = np.clip(combined, -1.0, 1.0)
    combined_int16 = (combined_int16 * 32767).astype(np.int16)

    # Lưu file đầu ra
    out_tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = out_tmp.name
    out_tmp.close()
    wavfile.write(out_path, sample_rate, combined_int16)

    progress(1.0, desc="✅ Hoàn thành!")

    # ── Tạo thông báo kết quả ──────────────────────────────────────────────────
    if engine == "omnivoice":
        mode_label = "voice cloning (OmniVoice)"
        engine_display = f"🌟 OmniVoice Vietnamese (num_step={int(num_step)})"
    else:
        mode_label = "giọng tham chiếu (voice cloning)" if use_cloning else "giọng preset"
        engine_label = type(t).__name__
        engine_mapping = {
            "FastVieNeuTTS":    "⚡ Fast (LMDeploy GPU)",
            "VieNeuTTS":        "🎯 Standard (GPU)",
            "TurboVieNeuTTS":   "🚀 Turbo (CPU)",
        }
        engine_display = engine_mapping.get(engine_label, engine_label)

    status_msg = (
        f"✅ Hoàn thành! Đã tổng hợp {len(audio_arrays)}/{total} đoạn "
        f"({len(text):,} ký tự) bằng {mode_label}.\n"
        f"🔧 Engine: {engine_display}"
    )
    if failed_chunks:
        fail_detail = ", ".join([f"đoạn {idx}" for idx, _ in failed_chunks])
        status_msg += f"\n⚠️ Bỏ qua {len(failed_chunks)} đoạn lỗi: {fail_detail}."

    return out_path, status_msg


def _as_file_path(file_value):
    if file_value is None:
        return None
    if isinstance(file_value, str):
        return file_value
    if hasattr(file_value, "name"):
        return file_value.name
    if isinstance(file_value, dict):
        return file_value.get("name") or file_value.get("path")
    return str(file_value)


def _to_mono_float32(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False)


def _save_wav_temp(audio: np.ndarray, sample_rate: int) -> str:
    audio = _to_mono_float32(audio)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    combined_int16 = np.clip(audio, -1.0, 1.0)
    combined_int16 = (combined_int16 * 32767).astype(np.int16)

    out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = out_tmp.name
    out_tmp.close()
    wavfile.write(out_path, sample_rate, combined_int16)
    return out_path


def _engine_display_name(t, engine: str, num_step: int) -> str:
    if engine == "omnivoice":
        return f"OmniVoice Vietnamese (num_step={int(num_step)})"
    engine_mapping = {
        "FastVieNeuTTS": "Fast (LMDeploy GPU)",
        "VieNeuTTS": "Standard (GPU)",
        "TurboVieNeuTTS": "Turbo (CPU)",
    }
    return engine_mapping.get(type(t).__name__, type(t).__name__)


def _infer_tts_chunk(t, engine, text, voice_data, ref_audio_path, ref_text, num_step, use_cloning):
    if engine == "omnivoice":
        return t.infer(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text.strip(),
            num_step=int(num_step),
        )
    if use_cloning:
        return t.infer(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text.strip(),
            max_chars=300,
        )
    if voice_data is not None:
        return t.infer(text=text, voice=voice_data, max_chars=300)
    return t.infer(text=text, max_chars=300)


def _compose_srt_timeline(rendered_segments, sample_rate: int, timing_mode: str):
    if not rendered_segments:
        return np.array([], dtype=np.float32), []

    warnings = []
    timing_mode = (timing_mode or "balanced").lower()

    if timing_mode == "natural":
        pieces = []
        previous_end_ms = None
        for segment, wav in rendered_segments:
            if previous_end_ms is not None:
                gap_ms = max(0, segment.start_ms - previous_end_ms)
                pause_ms = min(1000, max(160, gap_ms))
                pieces.append(np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32))
            pieces.append(_to_mono_float32(wav))
            previous_end_ms = segment.end_ms
        return np.concatenate(pieces), warnings

    last_segment_end = max(segment.end_ms for segment, _ in rendered_segments)
    total_samples = int((last_segment_end + 1000) * sample_rate / 1000)
    timeline = np.zeros(max(total_samples, 1), dtype=np.float32)
    cursor = 0

    for segment, wav in rendered_segments:
        wav = _to_mono_float32(wav)
        desired_start = int(segment.start_ms * sample_rate / 1000)
        start = max(desired_start, cursor)
        end = start + len(wav)
        if end > len(timeline):
            timeline = np.pad(timeline, (0, end - len(timeline)))

        slot_samples = max(1, int((segment.end_ms - segment.start_ms) * sample_rate / 1000))
        overflow_ms = max(0, int((len(wav) - slot_samples) * 1000 / sample_rate))
        delay_ms = max(0, int((start - desired_start) * 1000 / sample_rate))
        if overflow_ms > 0:
            warnings.append(
                f"cue {segment.cue_indices[0]}-{segment.cue_indices[-1]} dai hon slot {overflow_ms} ms"
            )
        if delay_ms > 40:
            warnings.append(
                f"cue {segment.cue_indices[0]}-{segment.cue_indices[-1]} bi day tre {delay_ms} ms"
            )

        timeline[start:end] += wav
        cursor = end

    return timeline, warnings


def synthesize_srt(
    srt_file,
    engine,
    voice_id,
    ref_audio_path,
    ref_text,
    num_step,
    timing_mode,
    srt_max_chars,
    srt_max_gap_ms,
    remove_bracketed,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Convert an SRT file to speech while reusing the currently selected TTS engine.
    The SRT layer plans natural speech segments; the model layer stays unchanged.
    """
    srt_path = _as_file_path(srt_file)
    if not srt_path:
        return None, "Vui long upload file .srt."

    try:
        ref_audio_path = coerce_ref_audio_path(ref_audio_path)
    except Exception as e:
        return None, f"Khong the doc audio tham chieu: {e}"

    use_cloning = bool(ref_audio_path and ref_text and ref_text.strip())

    if ref_audio_path and not (ref_text and ref_text.strip()):
        return None, "Can transcript cua audio tham chieu de clone giong."

    if engine == "omnivoice":
        if not ref_audio_path:
            return None, "OmniVoice can audio tham chieu trong tab Voice Cloning."
        if not ref_text or not ref_text.strip():
            return None, "OmniVoice can transcript cua audio tham chieu."

    try:
        t = get_tts(engine, require_voice_cloning=(engine == "vieneu" and use_cloning))
    except ImportError:
        return None, (
            "Package 'omnivoice' chua duoc cai dat.\n"
            "Chay: .venv\\Scripts\\pip install omnivoice"
        )
    except Exception as e:
        return None, f"Khong the khoi dong engine {engine}: {e}"

    try:
        cues = read_srt(srt_path)
    except Exception as e:
        return None, f"Khong doc duoc file SRT: {e}"

    if not cues:
        return None, "File SRT khong co cue hop le."

    segments = plan_subtitle_segments(
        cues,
        mode=timing_mode,
        max_chars=int(srt_max_chars),
        max_gap_ms=int(srt_max_gap_ms),
        remove_bracketed=bool(remove_bracketed),
    )
    if not segments:
        return None, "Khong con noi dung de doc sau khi lam sach SRT."

    voice_data = None
    if engine == "vieneu" and not use_cloning and voice_id and voice_id != "__none__":
        try:
            voice_data = t.get_preset_voice(voice_id)
        except Exception:
            voice_data = None

    if engine == "omnivoice" and use_cloning:
        try:
            progress(0.02, desc="Dang encode giong tham chieu (OmniVoice)...")
            t.create_voice_prompt(ref_audio_path, ref_text.strip())
        except Exception as e:
            return None, f"Khong the encode giong tham chieu: {e}"

    rendered_segments = []
    failed_segments = []
    total = len(segments)

    for i, segment in enumerate(segments):
        progress((i / total) * 0.88 + 0.05, desc=f"Dang tong hop SRT segment {i + 1}/{total}...")
        try:
            wav = _infer_tts_chunk(
                t=t,
                engine=engine,
                text=segment.text,
                voice_data=voice_data,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                num_step=num_step,
                use_cloning=use_cloning,
            )
            if wav is not None and len(wav) > 0:
                rendered_segments.append((segment, wav))
            else:
                failed_segments.append((segment, "ket qua rong"))
        except Exception as e:
            failed_segments.append((segment, str(e)))

    if not rendered_segments:
        details = "\n".join(
            f"  - cue {segment.cue_indices[0]}-{segment.cue_indices[-1]}: {message}"
            for segment, message in failed_segments[:8]
        )
        return None, f"Khong tong hop duoc segment nao.\n{details}"

    progress(0.95, desc="Dang can audio vao timeline SRT...")
    sample_rate = t.sample_rate
    combined, timing_warnings = _compose_srt_timeline(rendered_segments, sample_rate, timing_mode)
    out_path = _save_wav_temp(combined, sample_rate)

    progress(1.0, desc="Hoan thanh!")

    duration_s = len(combined) / sample_rate if sample_rate else 0
    first_time = format_timecode(cues[0].start_ms)
    last_time = format_timecode(cues[-1].end_ms)
    status_msg = (
        f"Hoan thanh SRT -> audio: {len(rendered_segments)}/{total} segment, "
        f"{len(cues)} cue, thoi luong {duration_s:.1f}s.\n"
        f"Timeline SRT: {first_time} -> {last_time}\n"
        f"Che do: {timing_mode} | Engine: {_engine_display_name(t, engine, num_step)}"
    )

    warnings = timing_warnings[:]
    if failed_segments:
        warnings.extend(
            f"cue {segment.cue_indices[0]}-{segment.cue_indices[-1]} loi: {message}"
            for segment, message in failed_segments
        )
    if warnings:
        preview = "\n".join(f"  - {item}" for item in warnings[:8])
        extra = "" if len(warnings) <= 8 else f"\n  - ... va {len(warnings) - 8} canh bao khac"
        status_msg += f"\nCanh bao timing/chat luong:\n{preview}{extra}"

    return out_path, status_msg


# ── Giao diện Gradio ──────────────────────────────────────────────────────────
with gr.Blocks(
    title="VieNeu-TTS — Tiếng Việt",
    theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    css="""
        #header { text-align: center; padding: 12px 0 4px; }
        #header h1 { font-size: 1.7rem; margin-bottom: 2px; }
        #header p  { color: #64748b; font-size: 0.9rem; }
        .status-box textarea { font-size: 0.9rem !important; }
        #gen-btn { min-height: 48px; font-size: 1rem; font-weight: 600; }
        .engine-desc { background: #f0fdfa; border-left: 4px solid #0d9488;
                       padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
        .omnivoice-note { background: #fef3c7; border-left: 4px solid #f59e0b;
                          padding: 8px 12px; border-radius: 4px; }
    """
) as demo:

    gr.HTML("""
        <div id="header">
            <h1>🦜 VieNeu-TTS</h1>
            <p>Chuyển văn bản tiếng Việt thành giọng nói · Offline · Voice Cloning · Đa mô hình</p>
        </div>
    """)

    # ── Chọn Engine ─────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 Chọn mô hình TTS")
            engine_radio = gr.Radio(
                choices=ENGINE_CHOICES,
                value="vieneu",
                label="Engine",
                info="Mỗi engine có ưu điểm khác nhau. Xem mô tả bên dưới.",
            )
            engine_desc = gr.Markdown(
                value=ENGINE_DESCRIPTIONS["vieneu"],
                elem_classes=["engine-desc"],
            )

    with gr.Row():
        # ── Cột trái: Input ──────────────────────────────────────────────────
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="📝 Văn bản cần đọc",
                placeholder="Nhập văn bản tiếng Việt tại đây. Không giới hạn độ dài — hệ thống sẽ tự chia chunk và ghép âm thanh...",
                lines=10,
                max_lines=100,
                show_copy_button=True,
            )
            char_count = gr.Markdown("_0 ký tự_")

            text_input.change(
                fn=lambda t: f"_{len(t):,} ký tự · ~{max(1, len(t)//250)} đoạn_",
                inputs=text_input,
                outputs=char_count,
            )

            with gr.Accordion("SRT sang am thanh", open=False):
                srt_file = gr.File(
                    label="Upload file .srt",
                    file_types=[".srt"],
                    type="filepath",
                )
                srt_mode = gr.Radio(
                    choices=[
                        ("Can bang", "balanced"),
                        ("Doc tu nhien", "natural"),
                        ("Khop timeline", "sync"),
                    ],
                    value="balanced",
                    label="Che do timing",
                    info="Can bang giu moc SRT nhung uu tien giong doc tu nhien.",
                )
                with gr.Row():
                    srt_max_chars = gr.Slider(
                        minimum=120,
                        maximum=360,
                        value=260,
                        step=10,
                        label="Ky tu toi da / segment",
                    )
                    srt_max_gap_ms = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        value=700,
                        step=50,
                        label="Gop cue neu cach nhau duoi (ms)",
                    )
                remove_bracketed = gr.Checkbox(
                    value=True,
                    label="Bo dong phu de trong ngoac nhu [music], (tieng cuoi)",
                )

        # ── Cột phải: Cài đặt ────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Cài đặt giọng nói")

            with gr.Tab("🎙️ Giọng có sẵn"):
                # Lưu ý khi dùng OmniVoice
                omnivoice_note = gr.Markdown(
                    "⚠️ **OmniVoice không hỗ trợ preset voices.** "
                    "Vui lòng dùng tab **Voice Cloning** bên dưới.",
                    elem_classes=["omnivoice-note"],
                    visible=False,
                )
                voice_dd = gr.Dropdown(
                    label="Chọn giọng preset",
                    choices=[],
                    interactive=True,
                )
                load_btn = gr.Button(
                    "🔄 Tải danh sách giọng",
                    size="sm",
                    variant="secondary",
                )
                load_btn.click(
                    fn=fetch_voices,
                    inputs=engine_radio,
                    outputs=voice_dd,
                )

            with gr.Tab("🔬 Voice Cloning"):
                gr.Markdown(
                    "Upload file audio mẫu (WAV/MP3, 3–10 giây, giọng rõ, không tiếng ồn). "
                    "Khi dùng voice cloning, **tất cả các đoạn** sẽ bám sát giọng mẫu này.\n\n"
                    "💡 **OmniVoice bắt buộc dùng tab này.**"
                )
                ref_audio = gr.Audio(
                    label="🎵 File âm thanh tham chiếu",
                    type="filepath",
                    sources=["upload"],
                )
                ref_text = gr.Textbox(
                    label="📄 Nội dung nói trong file tham chiếu (phải khớp chính xác)",
                    placeholder="Nhập đúng nội dung lời nói có trong file audio tham chiếu...",
                    lines=3,
                )
                gr.Markdown(
                    "> ⚠️ **Quan trọng:** `ref_text` phải khớp chính xác với nội dung trong "
                    "file audio. Nếu không khớp, chất lượng sẽ giảm."
                )

            # OmniVoice advanced settings
            with gr.Accordion("⚡ Cài đặt OmniVoice nâng cao", open=False):
                num_step_slider = gr.Slider(
                    minimum=4,
                    maximum=64,
                    value=8,
                    step=1,
                    label="Số diffusion steps (num_step)",
                    info="Ít bước = nhanh hơn. Nhiều bước = chất lượng cao hơn. Khuyến nghị: 8.",
                )
                gr.Markdown(
                    "| Steps | Tốc độ | Chất lượng |\n"
                    "|-------|--------|------------|\n"
                    "| 4 | Rất nhanh | Trung bình |\n"
                    "| 8 | Nhanh | Tốt ✅ (khuyến nghị) |\n"
                    "| 16 | Vừa | Rất tốt |\n"
                    "| 32 | Chậm | Chất lượng cao |\n"
                    "| 64 | Rất chậm | Thử nghiệm / cải thiện nhẹ |"
                )

    with gr.Row():
        gen_btn = gr.Button("▶ Tổng hợp giọng nói", variant="primary", elem_id="gen-btn", scale=3)
        srt_btn = gr.Button("SRT -> audio", variant="secondary", scale=2)
        clear_btn = gr.Button("🗑️ Xóa", variant="stop", scale=1)

    with gr.Row():
        with gr.Column():
            status_box = gr.Textbox(
                label="📊 Trạng thái",
                interactive=False,
                elem_classes=["status-box"],
            )
            audio_out = gr.Audio(
                label="🔊 Âm thanh đầu ra",
                type="filepath",
                show_download_button=True,
            )

    gr.Markdown("""
    ---
    **💡 Hướng dẫn nhanh:**
    - **VieNeu-TTS (Auto):** Chọn giọng preset hoặc dùng Voice Cloning. Tự động chọn mode GPU/CPU.
    - **OmniVoice Vietnamese:** Chỉ dùng Voice Cloning — upload audio mẫu (3-10s) + điền transcript.
    - **Text dài (hàng nghìn ký tự):** Cả hai engine đều tự chia chunk và ghép âm thanh.
    - **Lần đầu chạy:** Model tải tự động từ HuggingFace (~vài GB), vui lòng chờ.
    """)

    # ── Kết nối sự kiện ────────────────────────────────────────────────────────
    engine_radio.change(
        fn=on_engine_change,
        inputs=engine_radio,
        outputs=[engine_desc, voice_dd, omnivoice_note, load_btn],
    )

    gen_btn.click(
        fn=synthesize,
        inputs=[text_input, engine_radio, voice_dd, ref_audio, ref_text, num_step_slider],
        outputs=[audio_out, status_box],
    )

    srt_btn.click(
        fn=synthesize_srt,
        inputs=[
            srt_file,
            engine_radio,
            voice_dd,
            ref_audio,
            ref_text,
            num_step_slider,
            srt_mode,
            srt_max_chars,
            srt_max_gap_ms,
            remove_bracketed,
        ],
        outputs=[audio_out, status_box],
    )

    clear_btn.click(
        fn=lambda: (None, None, "", None, "", None, None),
        outputs=[audio_out, ref_audio, ref_text, text_input, status_box, voice_dd, srt_file],
    )

if __name__ == "__main__":
    server_port = int(os.getenv("VIENEU_PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        show_error=True,
        max_threads=4,           # Giới hạn thread pool — bảo toàn global state
    )

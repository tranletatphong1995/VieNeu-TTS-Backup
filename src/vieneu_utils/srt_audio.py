from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
import re
from typing import Iterable, List


_TIMECODE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_ASS_TAG_RE = re.compile(r"\{\\[^}]+\}")
_BRACKETED_RE = re.compile(r"^\s*[\[(][^\])]+[\])]\s*$")
_SPACE_RE = re.compile(r"\s+")
_STRONG_END_RE = re.compile(r"[.!?…][\"')\]]*$")


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class SubtitleSegment:
    start_ms: int
    end_ms: int
    text: str
    cue_indices: tuple[int, ...]


def parse_timecode(value: str) -> int:
    """Convert an SRT timecode to milliseconds."""
    time_part, ms_part = re.split(r"[,\.]", value.strip(), maxsplit=1)
    hours, minutes, seconds = [int(part) for part in time_part.split(":")]
    milliseconds = int(ms_part.ljust(3, "0")[:3])
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def format_timecode(milliseconds: int) -> str:
    """Format milliseconds as HH:MM:SS,mmm for reports."""
    milliseconds = max(0, int(milliseconds))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, ms = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def read_srt(path: str | Path) -> List[SubtitleCue]:
    """Read and parse an SRT file."""
    raw = _read_text_with_fallbacks(Path(path))
    return parse_srt(raw)


def parse_srt(raw: str) -> List[SubtitleCue]:
    """Parse SRT content into subtitle cues."""
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n").strip())
    cues: List[SubtitleCue] = []

    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        time_line_index = next((i for i, line in enumerate(lines) if _TIMECODE_RE.search(line)), None)
        if time_line_index is None:
            continue

        match = _TIMECODE_RE.search(lines[time_line_index])
        if match is None:
            continue

        cue_number = len(cues) + 1
        if time_line_index > 0 and lines[0].isdigit():
            cue_number = int(lines[0])

        text_lines = lines[time_line_index + 1 :]
        text = "\n".join(text_lines).strip()
        if not text:
            continue

        cues.append(
            SubtitleCue(
                index=cue_number,
                start_ms=parse_timecode(match.group("start")),
                end_ms=parse_timecode(match.group("end")),
                text=text,
            )
        )

    return cues


def clean_subtitle_text(text: str, remove_bracketed: bool = True) -> str:
    """Clean subtitle markup while preserving natural Vietnamese punctuation."""
    cleaned_lines = []
    for line in text.replace("\\N", "\n").splitlines():
        line = unescape(line)
        line = _ASS_TAG_RE.sub("", line)
        line = _TAG_RE.sub("", line)
        line = line.replace("\u200b", "")
        line = line.strip()
        if not line:
            continue
        if remove_bracketed and _BRACKETED_RE.match(line):
            continue
        cleaned_lines.append(line)

    cleaned = " ".join(cleaned_lines)
    cleaned = cleaned.replace("...", "…")
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def plan_subtitle_segments(
    cues: Iterable[SubtitleCue],
    mode: str = "balanced",
    max_chars: int = 260,
    max_gap_ms: int | None = None,
    remove_bracketed: bool = True,
) -> List[SubtitleSegment]:
    """
    Group subtitle cues into speech-friendly segments.

    mode:
        natural  - groups more aggressively for the smoothest narration.
        balanced - default; keeps timing shape but avoids choppy cue-by-cue TTS.
        sync     - groups conservatively for tighter subtitle timing.
    """
    cleaned_cues = [
        SubtitleCue(cue.index, cue.start_ms, cue.end_ms, clean_subtitle_text(cue.text, remove_bracketed))
        for cue in sorted(cues, key=lambda item: (item.start_ms, item.end_ms))
    ]
    cleaned_cues = [cue for cue in cleaned_cues if cue.text]
    if not cleaned_cues:
        return []

    mode = (mode or "balanced").lower()
    if max_gap_ms is None:
        max_gap_ms = {"natural": 1400, "sync": 300}.get(mode, 700)

    segments: List[SubtitleSegment] = []
    current = cleaned_cues[0]
    current_text = current.text
    current_indices = [current.index]
    current_start = current.start_ms
    current_end = current.end_ms

    for cue in cleaned_cues[1:]:
        gap_ms = cue.start_ms - current_end
        candidate = f"{current_text} {cue.text}".strip()
        can_join = gap_ms <= max_gap_ms and len(candidate) <= max_chars

        if mode == "sync" and _STRONG_END_RE.search(current_text) and gap_ms > 120:
            can_join = False
        elif mode == "balanced" and _STRONG_END_RE.search(current_text) and gap_ms > 350:
            can_join = False

        if can_join:
            current_text = candidate
            current_indices.append(cue.index)
            current_end = max(current_end, cue.end_ms)
            continue

        segments.append(
            SubtitleSegment(
                start_ms=current_start,
                end_ms=current_end,
                text=current_text,
                cue_indices=tuple(current_indices),
            )
        )
        current_text = cue.text
        current_indices = [cue.index]
        current_start = cue.start_ms
        current_end = cue.end_ms

    segments.append(
        SubtitleSegment(
            start_ms=current_start,
            end_ms=current_end,
            text=current_text,
            cue_indices=tuple(current_indices),
        )
    )
    return segments


def _read_text_with_fallbacks(path: Path) -> str:
    encodings = ("utf-8-sig", "utf-8", "cp1258", "cp1252")
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return path.read_text()

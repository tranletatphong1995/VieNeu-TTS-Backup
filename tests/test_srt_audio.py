from vieneu_utils.srt_audio import (
    clean_subtitle_text,
    format_timecode,
    parse_srt,
    parse_timecode,
    plan_subtitle_segments,
)


def test_parse_timecode_round_trip():
    assert parse_timecode("01:02:03,045") == 3_723_045
    assert parse_timecode("00:00:01.5") == 1_500
    assert format_timecode(3_723_045) == "01:02:03,045"


def test_parse_srt_blocks():
    raw = """1
00:00:01,000 --> 00:00:02,500
Xin chao.

2
00:00:02,700 --> 00:00:04,000
Day la dong tiep theo.
"""
    cues = parse_srt(raw)
    assert len(cues) == 2
    assert cues[0].index == 1
    assert cues[0].start_ms == 1000
    assert cues[1].text == "Day la dong tiep theo."


def test_clean_subtitle_text_removes_markup_and_noise_lines():
    assert clean_subtitle_text("<i>Xin chao</i>\n[music]") == "Xin chao"
    assert clean_subtitle_text("(cuoi)", remove_bracketed=False) == "(cuoi)"


def test_plan_segments_groups_adjacent_cues():
    cues = parse_srt(
        """1
00:00:01,000 --> 00:00:01,800
Toi khong biet

2
00:00:01,950 --> 00:00:03,000
chuyen nay co dung khong.

3
00:00:05,000 --> 00:00:06,000
Tam biet.
"""
    )
    segments = plan_subtitle_segments(cues, mode="balanced", max_gap_ms=700)
    assert len(segments) == 2
    assert segments[0].cue_indices == (1, 2)
    assert segments[0].text == "Toi khong biet chuyen nay co dung khong."

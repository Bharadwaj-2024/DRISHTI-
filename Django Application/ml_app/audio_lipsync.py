"""
DRISHTI — Audio & Lip-Sync Deepfake Analysis Module
====================================================
Detects AI-synthesized politician propaganda videos by analysing:
  1. Audio deepfake markers  (via librosa)
       - Spectral flatness   : AI voices are unnaturally flat / tonal
       - MFCC variance       : Natural speech has high MFCC variation
       - Pitch (F0) variance : Natural speech varies expressively
       - Silence regularity  : AI speech lacks natural breath pauses
  2. Lip-sync correlation     (via mediapipe FaceMesh + librosa)
       - Mouth openness per frame vs. audio RMS energy per frame
       - Low Pearson correlation = lip movements don't match speech = FAKE

PERFORMANCE OPTIMISED:
  - Reduced lip sample rate to 3fps (was 8-10fps)
  - Replaced slow librosa.pyin() with librosa.yin() (~10x faster)
  - Accept pre-extracted WAV to avoid redundant ffmpeg calls
  - Downscale frames for face detection
  - Limit audio to 15s max
"""

import math
import os
import subprocess

# ── optional heavy deps (graceful fallback if not installed) ──────────────────
try:
    import numpy as np
except ImportError:
    np = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    mp = None
    _mp_face_mesh = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import dlib as _dlib
except ImportError:
    _dlib = None

# ── dlib 68-landmark shape predictor path ─────────────────────────────────────
_DLIB_PREDICTOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ml_models",
    "shape_predictor_68_face_landmarks.dat",
)
# dlib inner-lip landmark indices (68-point model): 60-67
_DLIB_INNER_LIP = list(range(60, 68))
# dlib outer-lip landmark indices: 48-59
_DLIB_OUTER_LIP = list(range(48, 60))

# ── MediaPipe mouth landmark indices (inner lip) ──────────────────────────────
# Point 13 = upper inner lip centre, Point 14 = lower inner lip centre
MOUTH_TOP_IDX    = 13
MOUTH_BOTTOM_IDX = 14

# ── Performance tuning constants ─────────────────────────────────────────────
_MAX_AUDIO_SECONDS = 15
_LIP_DETECT_MAX_WIDTH = 480  # Downscale for face detection


def _compute_mar(landmarks_pts):
    """
    Compute Mouth Aspect Ratio from dlib 68-landmark inner lip points (60-67).
    MAR = (|p62-p66| + |p63-p65|) / (2 * |p60-p64|)
    Higher MAR = mouth more open.
    """
    if np is None or len(landmarks_pts) < 8:
        return 0.0
    pts = np.array(landmarks_pts, dtype=np.float64)
    # Vertical distances
    v1 = np.linalg.norm(pts[2] - pts[6])  # p62-p66
    v2 = np.linalg.norm(pts[3] - pts[5])  # p63-p65
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[4])   # p60-p64
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_audio_wav(video_path: str, wav_path: str) -> bool:
    """Extract a mono 16 kHz PCM WAV from video using ffmpeg. Limit to 15s."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(_MAX_AUDIO_SECONDS),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.exists(wav_path)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LIP-SYNC: dlib 68-landmark MAR tracking (primary) — OPTIMISED
# ─────────────────────────────────────────────────────────────────────────────

def _mouth_openness_timeline_dlib(video_path: str, sample_fps: float = 3.0,
                                    max_frames: int = 50):
    """
    Use dlib 68-landmark predictor to compute Mouth Aspect Ratio (MAR)
    per sampled frame. Downscales for detection, limits max frames.
    Returns list of (time_sec, mar_value).
    """
    if cv2 is None or _dlib is None:
        return []
    if not os.path.exists(_DLIB_PREDICTOR_PATH):
        return []

    try:
        detector = _dlib.get_frontal_face_detector()
        predictor = _dlib.shape_predictor(_DLIB_PREDICTOR_PATH)
    except Exception:
        return []

    timeline = []
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(video_fps / sample_fps))

    idx = 0
    frames_sampled = 0
    while frames_sampled < max_frames:
        if total_frames and idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        frames_sampled += 1
        height, width = frame.shape[:2]

        # Downscale for face detection speed
        if width > _LIP_DETECT_MAX_WIDTH:
            scale = _LIP_DETECT_MAX_WIDTH / width
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)  # 0 = no upsampling (fast)
        mar = 0.0
        if faces:
            shape = predictor(gray, faces[0])
            inner_lip_pts = [(shape.part(i).x, shape.part(i).y) for i in _DLIB_INNER_LIP]
            mar = _compute_mar(inner_lip_pts)

        timeline.append((idx / video_fps, mar))
        idx += frame_step

    cap.release()
    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# LIP-SYNC: MediaPipe fallback
# ─────────────────────────────────────────────────────────────────────────────

def _mouth_openness_timeline(video_path: str, sample_fps: float = 3.0):
    """
    Primary: dlib 68-landmark MAR tracking.
    Fallback: MediaPipe FaceMesh 2-point distance.
    Returns list of (time_sec: float, openness: float).
    """
    # Try dlib first (more precise)
    dlib_result = _mouth_openness_timeline_dlib(video_path, sample_fps)
    if dlib_result and len(dlib_result) >= 5:
        return dlib_result

    # Fallback to MediaPipe
    if cv2 is None or mp is None or _mp_face_mesh is None:
        return []

    timeline = []
    cap = cv2.VideoCapture(video_path)
    video_fps  = cap.get(cv2.CAP_PROP_FPS)  or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(video_fps / sample_fps))

    with _mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as face_mesh:
        idx = 0
        frames_sampled = 0
        while frames_sampled < 50:
            if total_frames and idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            frames_sampled += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            openness = 0.0
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                top    = lm[MOUTH_TOP_IDX]
                bottom = lm[MOUTH_BOTTOM_IDX]
                openness = math.sqrt(
                    (top.x - bottom.x) ** 2 +
                    (top.y - bottom.y) ** 2 +
                    (top.z - bottom.z) ** 2
                )

            timeline.append((idx / video_fps, openness))
            idx += frame_step

    cap.release()
    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# LIP-SYNC: audio RMS energy timeline
# ─────────────────────────────────────────────────────────────────────────────

def _audio_energy_timeline(wav_path: str, video_duration: float,
                            sample_fps: float = 3.0):
    """
    Compute RMS energy per time-slice at the same rate as lip sampling.
    Returns list of (time_sec: float, rms_energy: float).
    """
    if librosa is None or np is None:
        return []
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
    except Exception:
        return []

    hop = max(1, int(sr / sample_fps))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]

    timeline = []
    for i, energy in enumerate(rms):
        t = i * hop / sr
        if t > video_duration + 2:
            break
        timeline.append((t, float(energy)))
    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(x_list, y_list) -> float:
    """Pearson correlation coefficient between two same-length lists."""
    n = len(x_list)
    if n < 5:
        return 0.0
    if np is not None:
        r = float(np.corrcoef(x_list, y_list)[0, 1])
        return 0.0 if math.isnan(r) else r

    mx = sum(x_list) / n
    my = sum(y_list) / n
    cov  = sum((a - mx) * (b - my) for a, b in zip(x_list, y_list))
    sx   = math.sqrt(sum((a - mx) ** 2 for a in x_list))
    sy   = math.sqrt(sum((b - my) ** 2 for b in y_list))
    return 0.0 if sx == 0 or sy == 0 else cov / (sx * sy)


def _correlate_timelines(lip_timeline, audio_timeline) -> float:
    """Align two (time, value) timelines by nearest second-bucket and correlate."""
    if not lip_timeline or not audio_timeline:
        return 0.0

    lip_dict   = {}
    audio_dict = {}
    for t, v in lip_timeline:
        lip_dict[round(t, 1)] = v
    for t, v in audio_timeline:
        audio_dict[round(t, 1)] = v

    common = sorted(set(lip_dict) & set(audio_dict))
    if len(common) < 5:
        return 0.0

    return _pearson(
        [lip_dict[t]   for t in common],
        [audio_dict[t] for t in common],
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO DEEPFAKE MARKERS (librosa) — OPTIMISED
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_audio_markers(wav_path: str) -> dict:
    """
    Returns individual audio-AI scores (0 = natural, 100 = AI-synthesised)
    and a composite fake_score.
    OPTIMISED: Uses librosa.yin() instead of librosa.pyin() (~10x faster).
    """
    _unavailable = {
        "available": False, "fake_score": 50.0,
        "spectral_flatness_score": 50.0, "mfcc_variance_score": 50.0,
        "pitch_variance_score": 50.0, "silence_pattern_score": 50.0,
        "note": "",
    }

    if librosa is None or np is None:
        _unavailable["note"] = "librosa not installed"
        return _unavailable

    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True,
                             duration=_MAX_AUDIO_SECONDS)
    except Exception as exc:
        _unavailable["note"] = str(exc)
        return _unavailable

    if len(y) < sr:          # less than 1 second of audio
        _unavailable["note"] = "Audio too short for analysis"
        return _unavailable

    # ── 1. Spectral flatness ─────────────────────────────────────────────────
    # AI voices: often very flat (noise-like) OR unnaturally tonal (< 0.005)
    sf       = librosa.feature.spectral_flatness(y=y)[0]
    mean_sf  = float(np.mean(sf))
    if mean_sf < 0.005:
        sf_score = 65.0     # Too tonal — AI over-synthesis
    elif mean_sf < 0.04:
        sf_score = 15.0     # Natural speech range
    elif mean_sf < 0.18:
        sf_score = 35.0
    else:
        sf_score = 80.0     # Very flat — AI noise artifact

    # ── 2. MFCC variance ────────────────────────────────────────────────────
    # Natural speech: high MFCC variance (expressive)
    # AI speech: unnaturally uniform / scripted MFCCs
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))
    if mfcc_var > 500:
        mfcc_score = 8.0
    elif mfcc_var > 200:
        mfcc_score = 20.0
    elif mfcc_var > 80:
        mfcc_score = 48.0
    elif mfcc_var > 30:
        mfcc_score = 70.0
    else:
        mfcc_score = 88.0   # Suspiciously uniform

    # ── 3. Pitch (F0) variance — FAST with librosa.yin() ─────────────────────
    # Natural speech: varies pitch expressively (f0_std > 20 Hz)
    # AI speech: monotone (< 10 Hz) or unnaturally perfect
    f0_std = 0.0
    pitch_score = 50.0
    try:
        # yin() is ~10x faster than pyin()
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        # Filter out likely unvoiced (very low or very high values)
        voiced_mask = (f0 > 50) & (f0 < 500)
        voiced_f0 = f0[voiced_mask]
        if len(voiced_f0) > 10:
            f0_std = float(np.std(voiced_f0))
            voiced_ratio = len(voiced_f0) / max(len(f0), 1)
            if f0_std > 45:
                pitch_score = 8.0
            elif f0_std > 22:
                pitch_score = 22.0
            elif f0_std > 10:
                pitch_score = 55.0
            else:
                pitch_score = 82.0    # Monotone
            if voiced_ratio < 0.25:   # Too many unvoiced = AI glitch
                pitch_score = min(100.0, pitch_score + 12.0)
    except Exception:
        pitch_score = 50.0

    # ── 4. Silence / pause regularity ────────────────────────────────────────
    # Natural speech: irregular pause intervals (breathing, thinking)
    # AI speech: very regular or zero pauses
    silence_score = 50.0
    try:
        intervals = librosa.effects.split(y, top_db=32)
        if len(intervals) > 2:
            gaps = [intervals[i][0] - intervals[i - 1][1]
                    for i in range(1, len(intervals))]
            gap_mean = float(np.mean(gaps)) if gaps else 1.0
            gap_std  = float(np.std(gaps))  if gaps else 0.0
            regularity = gap_std / max(gap_mean, 1.0)
            if regularity > 0.9:
                silence_score = 8.0
            elif regularity > 0.55:
                silence_score = 22.0
            elif regularity > 0.2:
                silence_score = 55.0
            else:
                silence_score = 78.0
    except Exception:
        silence_score = 50.0

    # ── Composite weighted score ─────────────────────────────────────────────
    fake_score = (
        sf_score       * 0.20 +
        mfcc_score     * 0.35 +
        pitch_score    * 0.30 +
        silence_score  * 0.15
    )

    return {
        "available":               True,
        "fake_score":              round(fake_score, 1),
        "spectral_flatness_score": round(sf_score, 1),
        "mfcc_variance_score":     round(mfcc_score, 1),
        "pitch_variance_score":    round(pitch_score, 1),
        "silence_pattern_score":   round(silence_score, 1),
        "mean_spectral_flatness":  round(mean_sf, 4),
        "mfcc_variance":           round(mfcc_var, 1),
        "pitch_std_hz":            round(f0_std, 1),
        "note":                    "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def full_audio_lipsync_analysis(video_path: str, sample_fps: float = 3.0,
                                  wav_path: str = None) -> dict:
    """
    Run audio deepfake marker analysis AND lip-sync correlation.
    If wav_path is provided, reuses the pre-extracted audio file.

    Returns a result dict:
      available         : bool — True if at least one analysis ran
      lipsync_score     : 0-100  (0 = perfect sync / natural, 100 = mismatch / fake)
      audio_fake_score  : 0-100  (0 = natural audio, 100 = AI-synthesised)
      combined_score    : 0-100  weighted combination
      correlation       : Pearson r  lip-openness vs audio energy
      audio_markers     : sub-scores dict from _analyze_audio_markers
      lip_frames_sampled: int
      note              : diagnostic string
    """
    result = {
        "available":          False,
        "lipsync_score":      50.0,
        "audio_fake_score":   50.0,
        "combined_score":     50.0,
        "correlation":        0.0,
        "audio_markers":      {},
        "lip_frames_sampled": 0,
        "note":               "",
    }

    # ── Check ffmpeg is available ────────────────────────────────────────────
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except Exception:
        result["note"] = "ffmpeg not found — audio analysis skipped"
        return result

    # Use pre-extracted WAV or extract new one
    own_wav = False
    if wav_path and os.path.exists(wav_path):
        tmp_wav = wav_path
    else:
        tmp_wav = video_path + "_drishti_audio.wav"
        own_wav = True

    try:
        # ── Extract audio (only if not pre-extracted) ────────────────────────
        if own_wav:
            audio_ok = _extract_audio_wav(video_path, tmp_wav)
            if not audio_ok:
                result["note"] = "No audio track in video"
                return result

        # ── Get video duration ───────────────────────────────────────────────
        video_duration = 30.0
        if cv2 is not None:
            try:
                cap = cv2.VideoCapture(video_path)
                _fps    = cap.get(cv2.CAP_PROP_FPS)  or 25.0
                _frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                video_duration = _frames / _fps if _fps else 30.0
                cap.release()
            except Exception:
                pass

        # ── Run both analyses ────────────────────────────────────────────────
        audio_markers  = _analyze_audio_markers(tmp_wav)
        lip_timeline   = _mouth_openness_timeline(video_path, sample_fps=sample_fps)
        audio_timeline = _audio_energy_timeline(tmp_wav, video_duration, sample_fps=sample_fps)

        # ── Lip-sync score ───────────────────────────────────────────────────
        correlation = _correlate_timelines(lip_timeline, audio_timeline)
        # correlation → lipsync_score  (higher correlation = more natural)
        if correlation >= 0.65:
            lipsync_score = 10.0 + (1.0 - correlation) * 25.0
        elif correlation >= 0.35:
            lipsync_score = 35.0 + (0.65 - correlation) * 70.0
        elif correlation >= 0.0:
            lipsync_score = 70.0 + (0.35 - correlation) * 60.0
        else:
            lipsync_score = 85.0 + abs(correlation) * 15.0
        lipsync_score = round(max(0.0, min(100.0, lipsync_score)), 1)

        audio_fake_score = audio_markers.get("fake_score", 50.0)
        has_audio = audio_markers.get("available", False)
        has_lip   = len(lip_timeline) >= 5

        if has_audio and has_lip:
            combined = audio_fake_score * 0.55 + lipsync_score * 0.45
        elif has_audio:
            combined = audio_fake_score
        elif has_lip:
            combined = lipsync_score
        else:
            combined = 50.0

        result.update({
            "available":          has_audio or has_lip,
            "lipsync_score":      lipsync_score,
            "audio_fake_score":   round(audio_fake_score, 1),
            "combined_score":     round(combined, 1),
            "correlation":        round(correlation, 3),
            "audio_markers":      audio_markers,
            "lip_frames_sampled": len(lip_timeline),
            "note":               "",
        })

    except Exception as exc:
        result["note"] = f"Analysis error: {exc}"
    finally:
        try:
            if own_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            pass

    return result

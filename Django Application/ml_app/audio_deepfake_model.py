"""
DRISHTI — Audio Deepfake Detection via Wav2Vec2
=================================================
Uses a HuggingFace Wav2Vec2 model fine-tuned for audio spoofing detection
to classify audio as real human speech vs. AI-generated/cloned speech.

Model: MelodyMachine/Deepfake-audio-detection-V2
  - Fine-tuned Wav2Vec2 for binary classification (real vs. fake audio)
  - Trained on speech anti-spoofing datasets

PERFORMANCE OPTIMISED:
  - Accepts pre-extracted WAV path to avoid redundant ffmpeg calls
  - Clips audio to 10s (was 30s) — enough for classification
  - Half-precision on CUDA
  - Caches feature extractor
"""

import os
import subprocess
import threading

# ── Optional heavy dependencies ──────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F_torch
except ImportError:
    torch = None
    F_torch = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForSequenceClassification,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 10  # 10s is enough for classification (was 30s)

_detector_instance = None
_detector_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_audio_wav(video_path: str, wav_path: str) -> bool:
    """Extract mono 16kHz PCM WAV from video using ffmpeg. Limit to first 10s."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(MAX_AUDIO_SECONDS),  # Only extract first 10s
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.exists(wav_path)
    except Exception:
        return False


def _load_audio(wav_path: str):
    """Load audio from WAV file, return (waveform_np, sample_rate)."""
    if librosa is not None:
        try:
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            # Clip to max duration
            max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
            if len(y) > max_samples:
                y = y[:max_samples]
            return y, sr
        except Exception:
            pass

    if sf is not None:
        try:
            y, sr = sf.read(wav_path)
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            # Resample if needed
            if sr != SAMPLE_RATE and librosa is not None:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE
            max_samples = MAX_AUDIO_SECONDS * sr
            if len(y) > max_samples:
                y = y[:max_samples]
            return y, sr
        except Exception:
            pass

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AudioDeepfakeDetector:
    """
    Wav2Vec2-based audio deepfake detector.
    OPTIMISED: half precision, shorter audio clips, reuse extracted WAV.

    Usage:
        detector = AudioDeepfakeDetector.get_instance()
        result = detector.detect_audio("/path/to/audio.wav")
        result = detector.detect_video("/path/to/video.mp4")
    """

    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.device = None
        self._model_loaded = False
        self._use_half = False
        self._load_model()

    def _load_model(self):
        """Load the Wav2Vec2 model from HuggingFace."""
        if not _HAS_TRANSFORMERS or torch is None:
            print("[DRISHTI-Audio] transformers or torch not installed")
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._use_half = self.device.type == "cuda"
            print(f"[DRISHTI-Audio] Loading {MODEL_NAME}...")

            # Try loading with AutoModel first
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    MODEL_NAME, trust_remote_code=True
                )
                self.model = AutoModelForAudioClassification.from_pretrained(
                    MODEL_NAME, trust_remote_code=True
                )
            except Exception as e1:
                print(f"[DRISHTI-Audio] AutoModel failed ({e1}), trying Wav2Vec2...")
                try:
                    self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        MODEL_NAME
                    )
                    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        MODEL_NAME
                    )
                except Exception as e2:
                    print(f"[DRISHTI-Audio] Wav2Vec2 also failed: {e2}")
                    return

            self.model = self.model.to(self.device)
            self.model.eval()

            if self._use_half:
                self.model = self.model.half()
                print("[DRISHTI-Audio] Using FP16 half-precision on CUDA")

            self._model_loaded = True
            print(f"[DRISHTI-Audio] Model loaded on {self.device}")

        except Exception as e:
            print(f"[DRISHTI-Audio] Failed to load model: {e}")
            self._model_loaded = False

    @classmethod
    def get_instance(cls):
        """Thread-safe singleton."""
        global _detector_instance
        if _detector_instance is None:
            with _detector_lock:
                if _detector_instance is None:
                    try:
                        _detector_instance = cls()
                    except Exception as e:
                        print(f"[DRISHTI-Audio] Singleton init failed: {e}")
                        return None
        return _detector_instance

    @property
    def is_available(self):
        return self._model_loaded and self.model is not None

    def detect_audio(self, wav_path: str) -> dict:
        """
        Classify audio file as real or fake.

        Returns:
            available: bool
            fake_probability: float 0-1
            real_probability: float 0-1
            label: str ("FAKE" or "REAL")
            confidence: float 0-100
            audio_fake_score: float 0-100 (for fusion: 0=natural, 100=AI)
            note: str
        """
        result = {
            "available": False,
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "label": "UNKNOWN",
            "confidence": 50.0,
            "audio_fake_score": 50.0,
            "note": "",
        }

        if not self.is_available:
            result["note"] = "Audio model not loaded"
            return result

        waveform, sr = _load_audio(wav_path)
        if waveform is None:
            result["note"] = "Could not load audio file"
            return result

        if len(waveform) < SAMPLE_RATE:  # Less than 1 second
            result["note"] = "Audio too short for analysis"
            return result

        try:
            # Process audio through feature extractor
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device)
            if self._use_half:
                input_values = input_values.half()

            # Inference
            with torch.no_grad():
                logits = self.model(input_values).logits
                probs = F_torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

            # Map labels — model may use different label ordering
            id2label = getattr(self.model.config, "id2label", {})
            label_map = {}
            for idx, label_name in id2label.items():
                label_lower = str(label_name).lower()
                if "fake" in label_lower or "spoof" in label_lower or "bonafide" not in label_lower:
                    label_map["fake"] = int(idx)
                if "real" in label_lower or "bonafide" in label_lower or "genuine" in label_lower:
                    label_map["real"] = int(idx)

            # If no clear mapping, assume: 0=real, 1=fake (most common)
            fake_idx = label_map.get("fake", 1 if len(probs) > 1 else 0)
            real_idx = label_map.get("real", 0)

            fake_prob = float(probs[fake_idx]) if fake_idx < len(probs) else 0.5
            real_prob = float(probs[real_idx]) if real_idx < len(probs) else 0.5

            # Normalize
            total = fake_prob + real_prob
            if total > 0:
                fake_prob /= total
                real_prob /= total

            label = "FAKE" if fake_prob > real_prob else "REAL"
            confidence = max(fake_prob, real_prob) * 100.0

            result.update({
                "available": True,
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "label": label,
                "confidence": round(confidence, 1),
                "audio_fake_score": round(fake_prob * 100.0, 1),
                "note": f"Wav2Vec2 classification: {label} ({confidence:.1f}%)",
            })

        except Exception as e:
            result["note"] = f"Inference error: {e}"

        return result

    def detect_video(self, video_path: str, wav_path: str = None) -> dict:
        """
        Extract audio from video and classify it.
        If wav_path is provided, reuse the already-extracted WAV file
        to avoid redundant ffmpeg calls.

        Returns same dict as detect_audio() with additional fields.
        """
        result = {
            "available": False,
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "label": "UNKNOWN",
            "confidence": 50.0,
            "audio_fake_score": 50.0,
            "note": "",
        }

        if not self.is_available:
            result["note"] = "Audio model not loaded"
            return result

        # If pre-extracted WAV provided, use it directly
        if wav_path and os.path.exists(wav_path):
            return self.detect_audio(wav_path)

        # Check ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        except Exception:
            result["note"] = "ffmpeg not found"
            return result

        tmp_wav = video_path + "_drishti_wav2vec2.wav"
        try:
            if not _extract_audio_wav(video_path, tmp_wav):
                result["note"] = "No audio track in video"
                return result

            return self.detect_audio(tmp_wav)

        except Exception as e:
            result["note"] = f"Video audio extraction error: {e}"
            return result
        finally:
            try:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_deepfake_detector():
    """Get the singleton AudioDeepfakeDetector, or None if unavailable."""
    if not _HAS_TRANSFORMERS or torch is None:
        return None
    try:
        return AudioDeepfakeDetector.get_instance()
    except Exception:
        return None

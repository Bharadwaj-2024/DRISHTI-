"""
DRISHTI — AI-Generated Image Detector
=======================================
Detects whether an image is AI-generated or authentic using:
  1. XceptionNet face analysis (if faces present)
  2. Frequency-domain analysis (DCT spectral artifacts)
  3. Edge consistency analysis (Laplacian variance patterns)
  4. Color distribution analysis (GAN color fingerprints)
  5. JPEG artifact analysis (compression inconsistencies)

Produces a fused confidence score with detailed evidence breakdown.
"""

import math
import os
import threading

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

try:
    import torch
    import torch.nn.functional as F_t
except ImportError:
    torch = None
    F_t = None


def _clamp(value, lo=0.0, hi=100.0):
    return round(max(lo, min(hi, float(value))), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FREQUENCY DOMAIN (DCT) ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_frequency_domain(img_bgr):
    """
    GAN-generated images have distinctive spectral artifacts in the
    frequency domain — periodic peaks, low high-frequency energy, and
    abnormal mid-band energy ratios.
    Returns: (score 0-100, evidence_str)
    """
    if cv2 is None or np is None:
        return 50.0, "Frequency analysis unavailable"

    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # Resize to square power-of-2 for clean DCT
        size = min(512, min(h, w))
        gray = cv2.resize(gray, (size, size))

        # 2D DCT via DFT
        dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude = np.log1p(magnitude)

        cy, cx = size // 2, size // 2

        # Create radial masks
        Y, X = np.ogrid[:size, :size]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        low = R < (size * 0.1)
        mid = (R >= size * 0.1) & (R < size * 0.35)
        high = R >= size * 0.35

        low_energy = float(np.mean(magnitude[low])) if np.any(low) else 0
        mid_energy = float(np.mean(magnitude[mid])) if np.any(mid) else 0
        high_energy = float(np.mean(magnitude[high])) if np.any(high) else 0

        total = low_energy + mid_energy + high_energy + 1e-6

        # GAN artifacts: low high-frequency ratio, unusual mid-band
        high_ratio = high_energy / total
        mid_ratio = mid_energy / total

        score = 50.0
        if high_ratio < 0.08:
            score += 25  # Very low high-freq = smoothed/AI
        elif high_ratio < 0.15:
            score += 10
        elif high_ratio > 0.35:
            score -= 15  # Natural images have rich high-freq

        if mid_ratio > 0.45:
            score += 15  # Unusual mid-band concentration
        elif mid_ratio < 0.2:
            score -= 10

        evidence = (
            f"Spectral energy: low {low_energy:.1f}, mid {mid_energy:.1f}, "
            f"high {high_energy:.1f}. High-freq ratio {high_ratio:.2%}."
        )
        return _clamp(score), evidence

    except Exception as e:
        return 50.0, f"Frequency analysis error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. EDGE CONSISTENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_edge_consistency(img_bgr):
    """
    AI-generated images often have unnaturally smooth or uniform edge
    distributions. Measure Laplacian variance across image patches.
    Returns: (score 0-100, evidence_str)
    """
    if cv2 is None or np is None:
        return 50.0, "Edge analysis unavailable"

    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Global Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        global_var = float(lap.var())

        # Patch-level variance (8x8 grid)
        patch_h = max(1, h // 8)
        patch_w = max(1, w // 8)
        patch_vars = []
        for r in range(0, h - patch_h + 1, patch_h):
            for c in range(0, w - patch_w + 1, patch_w):
                patch = gray[r:r + patch_h, c:c + patch_w]
                pv = cv2.Laplacian(patch, cv2.CV_64F).var()
                patch_vars.append(pv)

        if patch_vars:
            patch_std = float(np.std(patch_vars))
            patch_mean = float(np.mean(patch_vars))
            # Coefficient of variation
            cv_val = patch_std / max(patch_mean, 1e-6)
        else:
            cv_val = 0.5

        score = 50.0

        # Very low global variance = overly smooth = AI
        if global_var < 50:
            score += 30
        elif global_var < 150:
            score += 15
        elif global_var < 400:
            score -= 5
        else:
            score -= 15

        # Low CoV = unnaturally uniform edges = AI
        if cv_val < 0.3:
            score += 15
        elif cv_val < 0.6:
            score += 5
        elif cv_val > 1.2:
            score -= 10

        evidence = (
            f"Laplacian variance: global {global_var:.1f}, "
            f"patch CoV {cv_val:.2f}. "
            + ("Edges are unnaturally smooth." if score > 60 else "Normal edge distribution.")
        )
        return _clamp(score), evidence

    except Exception as e:
        return 50.0, f"Edge analysis error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. COLOR DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_color_distribution(img_bgr):
    """
    GAN images often have subtle color distribution anomalies —
    oversaturated channels, limited dynamic range, or periodic patterns.
    Returns: (score 0-100, evidence_str)
    """
    if cv2 is None or np is None:
        return 50.0, "Color analysis unavailable"

    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        sat_mean = float(np.mean(s_ch))
        sat_std = float(np.std(s_ch))
        val_mean = float(np.mean(v_ch))
        val_std = float(np.std(v_ch))

        # Channel correlation (natural images have cross-channel correlation)
        b, g, r = cv2.split(img_bgr)
        rg_corr = float(np.corrcoef(r.flatten()[:10000], g.flatten()[:10000])[0, 1])
        rb_corr = float(np.corrcoef(r.flatten()[:10000], b.flatten()[:10000])[0, 1])

        if math.isnan(rg_corr):
            rg_corr = 0.0
        if math.isnan(rb_corr):
            rb_corr = 0.0

        score = 50.0

        # Abnormally high saturation = AI over-vivid
        if sat_mean > 140:
            score += 15
        elif sat_mean > 100:
            score += 5
        elif sat_mean < 30:
            score += 8  # Unnaturally desaturated

        # Low saturation variance = uniform/synthetic
        if sat_std < 25:
            score += 12
        elif sat_std < 40:
            score += 5

        # Very high channel correlation = synthetic
        if rg_corr > 0.95 and rb_corr > 0.95:
            score += 10
        elif rg_corr < 0.3 or rb_corr < 0.3:
            score -= 5

        evidence = (
            f"Saturation: mean {sat_mean:.0f}, std {sat_std:.0f}. "
            f"Channel correlation R-G {rg_corr:.2f}, R-B {rb_corr:.2f}."
        )
        return _clamp(score), evidence

    except Exception as e:
        return 50.0, f"Color analysis error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. JPEG ARTIFACT & NOISE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_noise_patterns(img_bgr):
    """
    AI-generated images have different noise characteristics than
    camera-captured images. Check noise level and uniformity.
    Returns: (score 0-100, evidence_str)
    """
    if cv2 is None or np is None:
        return 50.0, "Noise analysis unavailable"

    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Estimate noise via high-pass filter
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blur
        noise_std = float(np.std(noise))
        noise_mean = float(np.mean(np.abs(noise)))

        # Check noise uniformity (AI = uniform noise, real = varying)
        h, w = gray.shape
        quad_noise = []
        for qr in range(2):
            for qc in range(2):
                quad = noise[
                    qr * h // 2:(qr + 1) * h // 2,
                    qc * w // 2:(qc + 1) * w // 2,
                ]
                quad_noise.append(float(np.std(quad)))

        noise_uniformity = float(np.std(quad_noise)) / max(float(np.mean(quad_noise)), 1e-6)

        score = 50.0

        # Very low noise = AI-generated (no sensor noise)
        if noise_std < 2.0:
            score += 25
        elif noise_std < 4.0:
            score += 12
        elif noise_std < 8.0:
            score -= 5
        else:
            score -= 10  # High noise = real camera

        # Very uniform noise = AI
        if noise_uniformity < 0.05:
            score += 15
        elif noise_uniformity < 0.1:
            score += 8
        elif noise_uniformity > 0.3:
            score -= 10

        evidence = (
            f"Noise std {noise_std:.2f}, uniformity {noise_uniformity:.3f}. "
            + ("Suspiciously clean/uniform." if score > 60 else "Natural noise patterns.")
        )
        return _clamp(score), evidence

    except Exception as e:
        return 50.0, f"Noise analysis error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. FACE-LEVEL XCEPTION ANALYSIS (if faces present)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_face_xception(img_bgr):
    """
    If a face is detected, run it through the XceptionNet model.
    Returns: (score 0-100, evidence_str, face_found: bool)
    """
    try:
        from .xception_detector import get_xception_detector
    except Exception:
        return 50.0, "XceptionNet not available", False

    detector = get_xception_detector() if get_xception_detector else None
    if detector is None or not detector.is_available:
        return 50.0, "XceptionNet model not loaded", False

    try:
        pred, conf, found = detector.detect_frame(img_bgr)
        if not found:
            return 50.0, "No face detected in image", False

        if pred == 1:  # fake
            score = _clamp(conf, 55, 99)
        else:
            score = _clamp(100 - conf, 1, 45)

        label = "FAKE" if pred == 1 else "REAL"
        evidence = f"XceptionNet: {label} ({conf:.1f}% confidence). Face artifacts {'detected' if pred == 1 else 'not detected'}."
        return score, evidence, True

    except Exception as e:
        return 50.0, f"XceptionNet error: {e}", False


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image(image_path: str) -> dict:
    """
    Full AI-generated image detection pipeline.

    Returns dict:
        available: bool
        is_ai_generated: bool
        confidence: float 0-100
        verdict: str ("AI-GENERATED" or "AUTHENTIC")
        signals: list of {name, score, summary, evidence, label, tone}
        face_detected: bool
        detection_mode: str
        note: str
    """
    result = {
        "available": False,
        "is_ai_generated": False,
        "confidence": 50.0,
        "verdict": "UNKNOWN",
        "signals": [],
        "face_detected": False,
        "detection_mode": "image_forensics",
        "note": "",
    }

    if cv2 is None or np is None:
        result["note"] = "OpenCV/numpy not available"
        return result

    # Load image
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            result["note"] = "Could not load image file"
            return result
    except Exception as e:
        result["note"] = f"Image load error: {e}"
        return result

    h, w = img_bgr.shape[:2]
    if h < 32 or w < 32:
        result["note"] = "Image too small for analysis"
        return result

    # Resize if very large (for speed)
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)

    # ── Run all analyses ─────────────────────────────────────────────────────
    freq_score, freq_evidence = _analyze_frequency_domain(img_bgr)
    edge_score, edge_evidence = _analyze_edge_consistency(img_bgr)
    color_score, color_evidence = _analyze_color_distribution(img_bgr)
    noise_score, noise_evidence = _analyze_noise_patterns(img_bgr)
    face_score, face_evidence, face_found = _analyze_face_xception(img_bgr)

    def _risk_band(score):
        if score >= 85:
            return "Critical", "critical"
        if score >= 70:
            return "High", "high"
        if score >= 50:
            return "Elevated", "elevated"
        return "Low", "low"

    def _signal(name, score, summary, evidence):
        label, tone = _risk_band(score)
        return {
            "name": name,
            "score": _clamp(score),
            "summary": summary,
            "evidence": evidence,
            "label": label,
            "tone": tone,
        }

    signals = [
        _signal(
            "Frequency Domain Analysis",
            freq_score,
            "Checks for GAN spectral artifacts in the frequency domain.",
            freq_evidence,
        ),
        _signal(
            "Edge Consistency",
            edge_score,
            "Measures edge sharpness uniformity — AI images have unnaturally smooth edges.",
            edge_evidence,
        ),
        _signal(
            "Color Distribution",
            color_score,
            "Detects unnatural color patterns common in AI-generated images.",
            color_evidence,
        ),
        _signal(
            "Noise Pattern Analysis",
            noise_score,
            "AI images lack camera sensor noise and show uniform noise profiles.",
            noise_evidence,
        ),
    ]

    if face_found:
        signals.insert(0, _signal(
            "Face Artifact Detection (XceptionNet)",
            face_score,
            "XceptionNet deep learning model detects face manipulation artifacts.",
            face_evidence,
        ))

    # ── Fusion ────────────────────────────────────────────────────────────────
    if face_found:
        # With face: weight face analysis heavily
        fused = (
            face_score * 0.35 +
            freq_score * 0.20 +
            edge_score * 0.15 +
            color_score * 0.15 +
            noise_score * 0.15
        )
        detection_mode = "xception+forensics"
    else:
        # No face: rely on forensic signals
        fused = (
            freq_score * 0.30 +
            edge_score * 0.25 +
            color_score * 0.20 +
            noise_score * 0.25
        )
        detection_mode = "image_forensics"

    is_ai = fused >= 55.0
    confidence = _clamp(fused, 50.0, 99.0)

    if is_ai:
        if confidence >= 75:
            verdict = "AI-GENERATED"
        else:
            verdict = "LIKELY AI-GENERATED"
    else:
        if confidence <= 35 or (100 - confidence) >= 75:
            verdict = "AUTHENTIC"
        else:
            verdict = "LIKELY AUTHENTIC"

    result.update({
        "available": True,
        "is_ai_generated": is_ai,
        "confidence": confidence,
        "verdict": verdict,
        "signals": signals,
        "face_detected": face_found,
        "detection_mode": detection_mode,
        "image_dimensions": f"{w}×{h}",
        "note": "",
    })

    return result

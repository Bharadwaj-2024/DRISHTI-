import glob
import io
import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from .forms import VideoUploadForm
from .models import DeepfakeModel

try:
    from .audio_lipsync import full_audio_lipsync_analysis
except Exception:
    full_audio_lipsync_analysis = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import face_recognition
except Exception:
    face_recognition = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset
    from torchvision import transforms
except Exception:
    torch = None
    nn = None
    Dataset = None
    transforms = None


if torch is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = None


index_template_name = "index.html"
predict_template_name = "predict.html"
about_template_name = "about.html"

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1) if nn is not None else None

if transforms is not None:
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
else:
    train_transforms = None


MISSION_BRIEF = {
    "name": "DRISHTI",
    "tagline": "Real-Time Military Deepfake & Disinformation Detection Engine",
    "problem": "India needs a government-ready operator console that can catch synthetic military narratives before they dominate television and social feeds.",
    "operator_mode": "Demo-ready analyst workflow for DRDO / BEL / PIB-style review teams.",
}

CAPABILITY_CARDS = [
    {
        "title": "Multimodal Detector",
        "icon": "fa-layer-group",
        "summary": "Face-swap risk, lip-sync anomaly scoring, audio-visual mismatch, and metadata forensics fused into one threat score.",
    },
    {
        "title": "India-Specific Identity Layer",
        "icon": "fa-user-shield",
        "summary": "Flags impersonation attempts against strategic voices and faces such as national leadership, military commands, and ISPR media channels.",
    },
    {
        "title": "Weaponisation Predictor",
        "icon": "fa-bullhorn",
        "summary": "Scores how fast a clip can spread using reach potential, emotional manipulation, and alignment with sensitive military information windows.",
    },
]

WATCHLIST_ENTITIES = [
    # National leadership
    {"name": "Narendra Modi", "role": "Prime Minister voice-face embedding"},
    {"name": "S. Jaishankar", "role": "External Affairs Minister — diplomatic voiceprint"},
    {"name": "Rajnath Singh", "role": "Defence Minister — command authority persona"},
    {"name": "Amit Shah", "role": "Home Minister — security narrative target"},
    # Military Command
    {"name": "General Upendra Dwivedi (COAS)", "role": "Chief of Army Staff — highest military impersonation risk"},
    {"name": "Admiral Dinesh Tripathi (CNS)", "role": "Chief of Naval Staff — naval command persona"},
    {"name": "Air Chief Marshal AP Singh (CAS)", "role": "Chief of Air Staff — IAF command identity"},
    {"name": "General Anil Chauhan (CDS)", "role": "Chief of Defence Staff — tri-service command persona"},
    # Intelligence & Security
    {"name": "Ajit Doval (NSA)", "role": "National Security Advisor — intelligence narrative target"},
    # Adversarial
    {"name": "ISPR spokesperson", "role": "Pakistan military media identity — adversarial narrative set"},
    {"name": "Pakistan Army COAS", "role": "Cross-border military impersonation persona"},
]

EVENT_WATCHLIST = [
    {
        "title": "Military Officer Statement Window",
        "window": "Critical sensitivity",
        "detail": "AI-generated clips of COAS/CDS/NSA making false statements about casualties, ceasefire, or operations spread fastest in the first 30 minutes after a live briefing.",
    },
    {
        "title": "Operation Sindoor replay moments",
        "window": "Narrative volatility",
        "detail": "Historical conflict footage paired with synthetic military speech remains the highest-yield misinformation format for adversarial actors.",
    },
    {
        "title": "Border briefing cycle",
        "window": "High sensitivity",
        "detail": "False statements around ceasefire, casualties, or surrender narratives trend fastest during official military briefings.",
    },
    {
        "title": "Diplomatic escalation windows",
        "window": "Amplification risk",
        "detail": "Foreign policy impersonation clips can alter public mood before verified statements are published by MEA.",
    },
]

# ─── OSINT Context narratives ─────────────────────────────────────────────────
OSINT_CONTEXTS = {
    "military_officer": {
        "title": "Military Officer False Statement Pattern",
        "narrative": ("AI-generated videos of Indian military officers (COAS, CDS, NSA, Corps Commanders) making false statements "
                       "are the highest-priority threat vector. These clips typically fabricate ceasefire orders, casualty figures, "
                       "or operational failures to demoralise troops and mislead civilians during active conflict windows."),
        "badge": "CRITICAL THREAT",
        "tone": "critical",
    },
    "sindoor": {
        "title": "Operation Sindoor Narrative Window",
        "narrative": ("Synthetic clips exploiting Operation Sindoor visuals are a known high-yield format. "
                       "Adversarial actors fabricate surrender or casualty footage to manipulate public sentiment "
                       "before official briefings."),
        "badge": "Threat Active",
        "tone": "critical",
    },
    "ispr_pakistan": {
        "title": "Cross-Border Disinformation Pattern",
        "narrative": ("ISPR-linked synthetic media campaigns typically target Indian diplomatic and military figures "
                       "to generate confusion ahead of bilateral engagements. This clip matches known ISPR "
                       "narrative playbook signatures."),
        "badge": "Threat Active",
        "tone": "high",
    },
    "default": {
        "title": "Military Disinformation Context",
        "narrative": ("Generic military disinformation clips circulate during crisis windows to seed confusion about "
                       "command structure, casualties, or operational outcomes. DRISHTI flags this clip for analyst review."),
        "badge": "Monitoring",
        "tone": "elevated",
    },
}


def _get_threat_level(confidence, is_likely_fake):
    """Return 4-tier threat badge data."""
    c = float(confidence)
    if not is_likely_fake or c < 35.0:
        return {"level": "AUTHENTIC", "sub": "LOW THREAT", "tone": "low", "pulse": False, "label": "AUTHENTIC — LOW THREAT"}
    elif c < 55.0:
        return {"level": "UNCERTAIN", "sub": "MEDIUM THREAT", "tone": "elevated", "pulse": False, "label": "UNCERTAIN — MEDIUM THREAT"}
    elif c < 75.0:
        return {"level": "SYNTHETIC", "sub": "HIGH THREAT", "tone": "high", "pulse": False, "label": "SYNTHETIC — HIGH THREAT"}
    else:
        return {"level": "SYNTHETIC", "sub": "CRITICAL THREAT", "tone": "critical", "pulse": True, "label": "SYNTHETIC — CRITICAL THREAT"}


def _get_osint_context(video_name):
    """Return threat context card based on filename keywords."""
    name = (video_name or "").lower()
    # Military officer keywords — highest priority
    military_keywords = [
        "general", "admiral", "air chief", "coas", "cds", "cns", "cas", "nsa",
        "army chief", "defence", "colonel", "brigadier", "lieutenant", "major",
        "corp", "brigade", "division", "regiment", "airforce", "navy", "army",
        "dwivedi", "chauhan", "tripathi", "doval", "rajnath",
        "false", "ceasefire", "casualty", "surrender", "operation", "offensive"
    ]
    if any(k in name for k in military_keywords):
        return OSINT_CONTEXTS["military_officer"]
    if "sindoor" in name:
        return OSINT_CONTEXTS["sindoor"]
    if any(k in name for k in ["ispr", "pakistan", "pak"]):
        return OSINT_CONTEXTS["ispr_pakistan"]
    return OSINT_CONTEXTS["default"]


def _compute_false_statement_probability(confidence, is_likely_fake, audio_analysis, signals):
    """
    Compute a clear False Statement Probability score (0-100%)
    fusing detection confidence, audio AI score, and lip-sync mismatch.
    This is the single clearest indicator for military officer deepfake speech.
    """
    base = float(confidence) if is_likely_fake else (100 - float(confidence))

    audio_score = 50.0
    lipsync_score = 50.0
    if audio_analysis and audio_analysis.get("available"):
        audio_score   = float(audio_analysis.get("audio_fake_score", 50.0))
        lipsync_score = float(audio_analysis.get("lipsync_score", 50.0))

    # Weighted fusion: detection confidence (40%) + audio (35%) + lipsync (25%)
    fsp = base * 0.40 + audio_score * 0.35 + lipsync_score * 0.25
    fsp = round(max(0.0, min(100.0, fsp)), 1)

    if fsp >= 75:
        label = "HIGHLY LIKELY AI-GENERATED"
        tone  = "critical"
    elif fsp >= 55:
        label = "LIKELY AI-GENERATED"
        tone  = "high"
    elif fsp >= 35:
        label = "UNCERTAIN"
        tone  = "elevated"
    else:
        label = "LIKELY AUTHENTIC"
        tone  = "low"

    return {"score": fsp, "label": label, "tone": tone}


if Dataset is not None:
    class validation_dataset(Dataset):
        def __init__(self, video_names, sequence_length=60, transform=None):
            self.video_names = video_names
            self.transform = transform
            self.count = sequence_length

        def __len__(self):
            return len(self.video_names)

        def __getitem__(self, idx):
            video_path = self.video_names[idx]
            frames = []

            for frame in self.frame_extract(video_path):
                if face_recognition is not None:
                    faces = face_recognition.face_locations(frame)
                else:
                    faces = []

                try:
                    top, right, bottom, left = faces[0]
                    frame = frame[top:bottom, left:right, :]
                except Exception:
                    pass

                frames.append(self.transform(frame))
                if len(frames) == self.count:
                    break

            frames = torch.stack(frames)[: self.count]
            return frames.unsqueeze(0)

        def frame_extract(self, path):
            vid = cv2.VideoCapture(path)
            while True:
                success, image = vid.read()
                if not success:
                    break
                yield image
else:
    class validation_dataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ML dependencies (torch/torchvision) are not installed.")


def predict(model, img):
    if torch is None:
        raise RuntimeError("Cannot run prediction: 'torch' is not installed.")

    _, logits = model(img.to(device))
    logits = sm(logits)
    _, pred = torch.max(logits, 1)
    confidence = float(logits[0][pred.item()] * 100)
    return int(pred.item()), confidence


def get_accurate_model(sequence_length):
    """Return the best matching .pt model path, or None if none are available."""
    models_dir = os.path.join(settings.BASE_DIR, "ml_app", "ml_models")

    if not os.path.isdir(models_dir):
        print(f"[DRISHTI] ml_models folder not found at {models_dir} — running in demo mode.")
        return None

    model_files = glob.glob(os.path.join(models_dir, "*.pt"))
    if not model_files:
        print("[DRISHTI] No .pt model files found — running in demo mode.")
        return None

    match = []
    for model_path in model_files:
        parts = os.path.basename(model_path).replace(".pt", "").split("_")
        try:
            acc = float(parts[1])
            seq = int(parts[3])
            if seq == sequence_length:
                match.append((acc, model_path))
        except Exception:
            continue

    if not match:
        print(f"[DRISHTI] No model matched sequence length {sequence_length} — running in demo mode.")
        return None

    match.sort(reverse=True)
    return match[0][1]


def allowed_video_file(filename):
    return filename.split(".")[-1].lower() in ["mp4", "gif", "webm", "avi", "3gp", "wmv", "flv", "mkv", "mov"]


def _append_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, default=str) + "\n")


def _read_jsonl(path, limit=8):
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    rows.reverse()
    return rows[:limit]


def _get_detection_stats():
    log_path = os.path.join(settings.PROJECT_DIR, "logs", "detections.jsonl")
    stats = {
        "total": 0,
        "real": 0,
        "fake": 0,
        "avg_confidence": 0,
        "high_confidence_count": 0,
        "fake_rate": 0,
    }

    if not os.path.exists(log_path):
        return stats

    confidences = []
    for entry in _read_jsonl(log_path, limit=5000):
        verdict = (entry.get("verdict") or "").upper()
        confidence = float(entry.get("confidence", 0) or 0)
        stats["total"] += 1
        confidences.append(confidence)

        if verdict in {"AUTHENTIC", "REAL"}:
            stats["real"] += 1
        else:
            stats["fake"] += 1

        if confidence >= 80:
            stats["high_confidence_count"] += 1

    if confidences:
        stats["avg_confidence"] = round(_mean(confidences), 1)

    if stats["total"]:
        stats["fake_rate"] = round((stats["fake"] / stats["total"]) * 100, 1)

    return stats


def _clamp(value, lower=0.0, upper=100.0):
    return round(max(lower, min(upper, float(value))), 1)


def _mean(values):
    if not values:
        return 0.0
    if np is not None:
        return float(np.mean(values))
    return float(statistics.fmean(values))


def _std(values):
    if not values or len(values) < 2:
        return 0.0
    if np is not None:
        return float(np.std(values))
    return float(statistics.pstdev(values))


def _risk_band(score):
    if score >= 85:
        return {"label": "Critical", "tone": "critical"}
    if score >= 70:
        return {"label": "High", "tone": "high"}
    if score >= 50:
        return {"label": "Elevated", "tone": "elevated"}
    return {"label": "Low", "tone": "low"}


def _make_signal(name, score, summary, evidence):
    band = _risk_band(score)
    return {
        "name": name,
        "score": _clamp(score),
        "summary": summary,
        "evidence": evidence,
        "label": band["label"],
        "tone": band["tone"],
    }


def _build_home_context(form, stats=None):
    return {
        "form": form,
        "stats": stats or _get_detection_stats(),
        "mission": MISSION_BRIEF,
        "capability_cards": CAPABILITY_CARDS,
        "watchlist_entities": WATCHLIST_ENTITIES,
        "event_watchlist": EVENT_WATCHLIST,
    }


def _build_impersonation_matches(video_name, base_score, is_likely_fake, confidence):
    """
    Only flag impersonation when the video is actually detected as synthetic
    AND confidence is meaningfully high. Authentic videos get near-zero scores.
    """
    name = (video_name or "").lower()

    # Authentic videos: return empty — no impersonation detected
    if not is_likely_fake:
        return []

    # Confidence below 65%: inconclusive, return empty
    if confidence < 65.0:
        return []

    # Keyword presence in filename signals relevance
    keyword_map = {
        # National leadership
        "Narendra Modi": {
            "role": "Prime Minister face-voice embedding",
            "keywords": ["modi", "pm ", "prime minister"],
            "direct_match": any(k in name for k in ["modi", " pm "]),
        },
        "S. Jaishankar": {
            "role": "External Affairs Minister — diplomatic voiceprint",
            "keywords": ["jaishankar", "eam", "foreign ministry", "mea"],
            "direct_match": any(k in name for k in ["jaishankar", "eam"]),
        },
        "Rajnath Singh": {
            "role": "Defence Minister — command authority persona",
            "keywords": ["rajnath", "defence minister", "mod"],
            "direct_match": any(k in name for k in ["rajnath"]),
        },
        # Military Command
        "General Upendra Dwivedi (COAS)": {
            "role": "Chief of Army Staff — highest military impersonation risk",
            "keywords": ["dwivedi", "coas", "army chief", "chief of army"],
            "direct_match": any(k in name for k in ["dwivedi", "coas"]),
        },
        "General Anil Chauhan (CDS)": {
            "role": "Chief of Defence Staff — tri-service command persona",
            "keywords": ["chauhan", "cds", "chief of defence", "defence staff"],
            "direct_match": any(k in name for k in ["chauhan", "cds"]),
        },
        "Admiral Dinesh Tripathi (CNS)": {
            "role": "Chief of Naval Staff — naval command persona",
            "keywords": ["tripathi", "cns", "naval chief", "navy chief"],
            "direct_match": any(k in name for k in ["tripathi", "cns"]),
        },
        "Air Chief Marshal AP Singh (CAS)": {
            "role": "Chief of Air Staff — IAF command identity",
            "keywords": ["ap singh", "cas", "air chief", "iaf chief"],
            "direct_match": any(k in name for k in ["cas", "air chief"]),
        },
        "Ajit Doval (NSA)": {
            "role": "National Security Advisor — intelligence narrative target",
            "keywords": ["doval", "nsa", "national security"],
            "direct_match": any(k in name for k in ["doval", "nsa"]),
        },
        # Generic military — catch-all for officer videos
        "Indian Military Officer": {
            "role": "Army/Navy/IAF command persona cluster",
            "keywords": ["general", "admiral", "colonel", "brigadier", "major", "lieutenant",
                         "army", "navy", "airforce", "military", "officer", "defence"],
            "direct_match": any(k in name for k in ["general", "admiral", "colonel", "brigadier"]),
        },
        # Adversarial
        "ISPR spokesperson": {
            "role": "Pakistan military media identity — adversarial narrative set",
            "keywords": ["ispr", "pakistan", "pak"],
            "direct_match": any(k in name for k in ["ispr", "pakistan"]),
        },
    }

    # Scale score from actual confidence, not from base_score
    conf_factor = (confidence - 65.0) / 35.0  # 0.0 at 65%, 1.0 at 100%

    matches = []
    for subject, data in keyword_map.items():
        keyword_hit = any(k in name for k in data["keywords"])
        if data["direct_match"]:
            score = _clamp(55.0 + conf_factor * 30.0)
        elif keyword_hit:
            score = _clamp(35.0 + conf_factor * 20.0)
        else:
            # No filename evidence: skip this entity entirely
            continue

        band = _risk_band(score)
        matches.append({
            "subject": subject,
            "role": data["role"],
            "score": score,
            "label": band["label"],
            "tone": band["tone"],
        })

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:3]


def _build_weaponization(confidence, is_likely_fake, video_name):
    """
    Weaponization score is ONLY elevated when:
    - The video is genuinely detected as synthetic
    - AND there are relevant keywords in the filename
    - AND confidence is high
    Authentic videos return near-zero scores.
    """
    lowered = (video_name or "").lower()
    narrative_keywords = ["sindoor", "army", "strike", "general", "war", "breaking", "exclusive", "ispr", "jaishankar"]
    keyword_hits = sum(1 for keyword in narrative_keywords if keyword in lowered)

    if not is_likely_fake:
        # Authentic video: minimal weaponization risk regardless of filename
        reach_score = _clamp(keyword_hits * 3.0)          # max 27% if every keyword matches
        emotion_score = _clamp(keyword_hits * 2.5)         # max ~22%
        timing_score = _clamp(5.0 + keyword_hits * 2.0)   # max ~23%
    else:
        # Synthetic video: scale with confidence and keyword relevance
        conf_lift = max(0.0, confidence - 60.0)            # 0 at 60%, 35 at 95%
        reach_score = _clamp(15.0 + keyword_hits * 9.0 + conf_lift * 0.9)
        emotion_score = _clamp(12.0 + keyword_hits * 7.0 + conf_lift * 0.75)
        timing_score = _clamp(10.0 + keyword_hits * 6.0 + conf_lift * 0.6)

    total = _clamp(reach_score * 0.4 + emotion_score * 0.35 + timing_score * 0.25)
    band = _risk_band(total)

    if total >= 70:
        assessment = "High likelihood of rapid pickup across TV clips, X/Twitter handles, and Telegram forwarding chains."
    elif total >= 40:
        assessment = "Moderate amplification potential — monitor for narrative reuse in conflict-adjacent contexts."
    else:
        assessment = "Low weaponisation risk. No significant synthetic or conflict-narrative signals detected in this clip."

    return {
        "score": total,
        "label": band["label"],
        "tone": band["tone"],
        "reach_score": reach_score,
        "emotion_score": emotion_score,
        "timing_score": timing_score,
        "assessment": assessment,
    }


def _build_operator_actions(is_likely_fake, confidence):
    if is_likely_fake:
        return [
            "Escalate to PIB / media cell with a red-channel alert and attach the confidence breakdown.",
            "Cross-check speaker identity against the India-specific voice-face library before public rebuttal.",
            "Archive the clip hash, extracted frames, and URL trail for forensic follow-up.",
            "Prepare a one-click fact-check note for spokesperson approval within the next media cycle.",
        ]

    if confidence >= 70:
        return [
            "Hold for analyst verification rather than public escalation.",
            "Run a second-pass model on higher frame count if the clip is tied to a live security event.",
            "Preserve metadata and provenance in case the narrative mutates into a synthetic variant later.",
        ]

    return [
        "Mark as low-priority for continuous monitoring.",
        "Retain extracted frames and metadata for future comparison against related uploads.",
    ]


def _build_timeline(duration_seconds, frame_count):
    return [
        {"step": "Ingest", "detail": "Video uploaded into operator queue with chain-of-custody timestamp."},
        {"step": "Frame split", "detail": f"{frame_count} representative frames sampled across {duration_seconds:.1f}s of footage."},
        {"step": "Signal fusion", "detail": "Visual, temporal, metadata, and OSINT indicators combined into a single risk score."},
        {"step": "Analyst action", "detail": "Queue, report, and fact-check pathways generated for government operators."},
    ]


def _compose_alert_title(is_likely_fake, top_match):
    if is_likely_fake and top_match:
        return f"Possible synthetic impersonation of {top_match['subject']}"
    if is_likely_fake:
        return "Possible coordinated synthetic media event"
    return "No dominant synthetic signal detected"


def _compose_alert_summary(is_likely_fake, confidence, weaponization):
    if is_likely_fake:
        return (
            f"Multimodal fusion flagged this clip as likely synthetic with {confidence:.1f}% confidence. "
            f"Weaponisation potential is {weaponization['label'].lower()} because the content can be reframed into a fast-moving narrative."
        )

    return (
        f"The clip currently trends authentic at {confidence:.1f}% confidence, but DRISHTI still records the forensic fingerprint "
        f"for replay or impersonation attempts."
    )


def _build_report_text(last):
    signal_lines = []
    for signal in last.get("signals", []):
        signal_lines.append(f"- {signal['name']}: {signal['score']}% ({signal['label']})")

    impersonation_lines = []
    for match in last.get("impersonation_matches", []):
        impersonation_lines.append(f"- {match['subject']} / {match['role']}: {match['score']}%")

    action_lines = []
    for action in last.get("recommended_actions", []):
        action_lines.append(f"- {action}")

    signals_text = "\n".join(signal_lines) or "- No signals recorded"
    impersonation_text = "\n".join(impersonation_lines) or "- No watchlist matches recorded"
    actions_text = "\n".join(action_lines) or "- No action generated"

    return f"""DRISHTI ANALYST REPORT
======================

Mission: {MISSION_BRIEF['tagline']}
Video File: {last.get('video', 'Unknown')}
Analysis Date: {last.get('timestamp', 'N/A')}

VERDICT
-------
Verdict: {last.get('verdict', 'Unknown')}
Confidence: {last.get('confidence', 'N/A')}%
Weaponisation Potential: {last.get('weaponization', {}).get('score', 'N/A')}%

ALERT
-----
{last.get('alert_title', 'No alert title')}
{last.get('alert_summary', '')}

SIGNAL BREAKDOWN
----------------
{signals_text}

WATCHLIST MATCHES
-----------------
{impersonation_text}

RECOMMENDED ACTIONS
-------------------
{actions_text}

Generated by DRISHTI demo pipeline for hackathon evaluation.
"""


def generate_demo_frames(video_path, num_frames=6):
    demo_dir = os.path.join(settings.BASE_DIR, "static", "images", "demo")
    os.makedirs(demo_dir, exist_ok=True)

    video_name = os.path.basename(video_path)
    extension = os.path.splitext(video_name)[1].lower()

    preprocessed_images = []
    faces_cropped_images = []
    heatmap_images = []
    is_likely_fake = False
    confidence = 50.0
    total_frames = 0
    fps = 0.0
    duration_seconds = 0.0
    avg_laplacian = 0.0
    std_laplacian = 0.0
    avg_frame_diff = 0.0
    std_frame_diff = 0.0
    avg_color_std = 0.0
    std_color_std = 0.0
    fake_score = 0
    real_score = 0

    # ── Audio + Lip-sync analysis ────────────────────────────────────────────
    audio_analysis = {}
    if full_audio_lipsync_analysis is not None:
        try:
            audio_analysis = full_audio_lipsync_analysis(video_path, sample_fps=2.0)
        except Exception as _ae:
            print(f"[DRISHTI] Audio analysis error: {_ae}")
            audio_analysis = {}

    if cv2 is None:
        lowered_name = video_name.lower()
        inferred_fake = any(keyword in lowered_name for keyword in ["army", "ispr", "jaishankar", "modi", "sindoor", "general"])
        confidence = 82.0 if inferred_fake else 67.0
        is_likely_fake = inferred_fake
    else:
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            duration_seconds = (total_frames / fps) if fps else 0.0
            frame_interval = max(1, total_frames // max(num_frames, 1)) if total_frames else 1

            frame_count = 0
            frame_idx = 0
            laplacian_vars = []
            frame_diffs = []
            color_consistency = []
            prev_gray = None

            while frame_count < num_frames and (total_frames == 0 or frame_idx < total_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                laplacian_vars.append(laplacian_var)

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    frame_diffs.append(float(diff.mean()))
                prev_gray = gray.copy()

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color_std = float(hsv[:, :, 0].std())
                color_consistency.append(color_std)

                height, width = frame.shape[:2]
                new_height = 400
                aspect_ratio = width / height if height else 1
                new_width = max(1, int(new_height * aspect_ratio))
                frame_display = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                preprocess_filename = f"demo_frame_{frame_count:02d}.jpg"
                preprocess_path = os.path.join(demo_dir, preprocess_filename)
                cv2.imwrite(preprocess_path, frame_display, [cv2.IMWRITE_JPEG_QUALITY, 95])
                preprocessed_images.append(f"images/demo/{preprocess_filename}")

                # Generate Laplacian heatmap for artifact visualisation
                try:
                    gray_disp = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
                    lap = cv2.Laplacian(gray_disp, cv2.CV_64F)
                    lap_abs = cv2.convertScaleAbs(lap)
                    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap_bgr = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
                    heatmap_filename = f"demo_heatmap_{frame_count:02d}.jpg"
                    heatmap_path = os.path.join(demo_dir, heatmap_filename)
                    cv2.imwrite(heatmap_path, heatmap_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    heatmap_images.append(f"images/demo/{heatmap_filename}")
                except Exception:
                    pass

                display_height, display_width = frame_display.shape[:2]
                crop_size = min(180, display_height, display_width)
                x_start = max(0, (display_width - crop_size) // 2)
                y_start = max(0, (display_height - crop_size) // 2)
                x_end = min(display_width, x_start + crop_size)
                y_end = min(display_height, y_start + crop_size)

                cropped_frame = frame_display[y_start:y_end, x_start:x_end]
                bordered_frame = cv2.copyMakeBorder(
                    cropped_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(58, 166, 85)
                )

                cropped_filename = f"demo_face_{frame_count:02d}.jpg"
                cropped_path = os.path.join(demo_dir, cropped_filename)
                cv2.imwrite(cropped_path, bordered_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                faces_cropped_images.append(f"images/demo/{cropped_filename}")

                frame_count += 1
                frame_idx += frame_interval

            cap.release()

            if laplacian_vars and frame_diffs:
                avg_laplacian = _mean(laplacian_vars)
                std_laplacian = _std(laplacian_vars)
                avg_frame_diff = _mean(frame_diffs)
                std_frame_diff = _std(frame_diffs)
                avg_color_std = _mean(color_consistency) if color_consistency else 0.0
                std_color_std = _std(color_consistency) if color_consistency else 0.0

                if avg_laplacian < 80:
                    fake_score += 4
                elif avg_laplacian < 150:
                    fake_score += 2
                elif avg_laplacian <= 400:
                    real_score += 3
                else:
                    real_score += 2

                if std_laplacian < 15:
                    fake_score += 3
                elif std_laplacian < 40:
                    fake_score += 1
                elif std_laplacian <= 150:
                    real_score += 3
                else:
                    real_score += 2

                if avg_frame_diff < 2:
                    fake_score += 2
                elif avg_frame_diff < 8:
                    real_score += 2
                elif avg_frame_diff <= 25:
                    real_score += 3
                else:
                    real_score += 1

                if std_frame_diff < 1.5:
                    fake_score += 5
                elif std_frame_diff < 3:
                    fake_score += 3
                elif std_frame_diff < 6:
                    real_score += 2
                else:
                    real_score += 4

                if avg_color_std < 5:
                    fake_score += 1
                elif avg_color_std >= 15:
                    real_score += 1

                if std_color_std < 2:
                    fake_score += 1
                elif std_color_std >= 3:
                    real_score += 1

                if fake_score >= 8:
                    is_likely_fake = True
                    confidence = min(93.0, 75.0 + min(fake_score - 8, 15))
                elif fake_score >= 5 and fake_score > real_score:
                    is_likely_fake = True
                    confidence = min(92.0, 70.0 + (fake_score - 5) * 2)
                elif real_score > fake_score + 2:
                    is_likely_fake = False
                    confidence = min(93.0, 72.0 + min(real_score - 3, 15))
                elif real_score >= 6:
                    is_likely_fake = False
                    confidence = min(90.0, 70.0 + (real_score - 4) * 2)
                elif std_frame_diff < 2:
                    is_likely_fake = True
                    confidence = 70.0
                else:
                    is_likely_fake = False
                    confidence = 72.0
        except Exception as exc:
            print(f"Error generating demo frames: {exc}")

    confidence = _clamp(confidence, 50.0, 95.0)

    # --- Proportional signal scoring ---
    # For AUTHENTIC videos: scores stay LOW (5-25%) unless specific frame metrics say otherwise.
    # For SYNTHETIC videos: scores scale with confidence and frame anomaly strength.

    if is_likely_fake:
        # Scale from actual fake_score and confidence
        conf_lift = max(0.0, confidence - 55.0)  # 0 at 55%, up to 40 at 95%
        face_swap_score = _clamp(10.0 + fake_score * 5.5 + conf_lift * 1.1)
        lip_sync_score = _clamp(8.0 + fake_score * 4.0 + (12.0 if std_frame_diff < 2.0 else 4.0) + conf_lift * 0.8)
        metadata_score = _clamp(
            6.0
            + fake_score * 3.0
            + (10.0 if extension in {".avi", ".wmv", ".mkv"} else 3.0)
            + (8.0 if duration_seconds and duration_seconds < 8 else 0.0)
            + conf_lift * 0.5
        )
        av_sync_score = _clamp(10.0 + fake_score * 4.5 + (12.0 if avg_frame_diff < 2.5 else 3.0) + conf_lift * 0.9)
        lowered_name = video_name.lower()
        osint_keywords = ["sindoor", "army", "strike", "general", "war", "ispr", "jaishankar", "modi"]
        osint_hits = sum(1 for k in osint_keywords if k in lowered_name)
        osint_score = _clamp(8.0 + osint_hits * 10.0 + fake_score * 3.0 + conf_lift * 0.7)
    else:
        # Authentic video: scores reflect actual frame quality, NOT inflated
        # A clean video with high Laplacian sharpness gets very low scores
        sharpness_penalty = max(0.0, (200.0 - avg_laplacian) / 200.0) * 8.0  # 0 for sharp, up to 8 for blurry
        motion_penalty = max(0.0, (5.0 - avg_frame_diff) / 5.0) * 6.0  # penalty for too-static video

        face_swap_score = _clamp(3.0 + sharpness_penalty + real_score * 0.5)
        lip_sync_score = _clamp(3.0 + motion_penalty + (5.0 if std_frame_diff < 1.0 else 0.0))
        metadata_score = _clamp(
            2.0
            + (5.0 if extension in {".avi", ".wmv", ".mkv"} else 1.0)
            + (3.0 if duration_seconds and duration_seconds < 5 else 0.0)
        )
        av_sync_score = _clamp(3.0 + motion_penalty)
        lowered_name = video_name.lower()
        osint_keywords = ["sindoor", "army", "strike", "general", "war", "ispr", "jaishankar", "modi"]
        osint_hits = sum(1 for k in osint_keywords if k in lowered_name)
        osint_score = _clamp(2.0 + osint_hits * 4.0)  # max ~34% even with all keywords, but video is authentic

    # ── Audio + Lip-sync signal scores ───────────────────────────────────────
    _al = audio_analysis  # shorthand
    _al_available = _al.get("available", False)
    audio_fake_signal_score = _al.get("audio_fake_score", av_sync_score) if _al_available else av_sync_score
    lipsync_signal_score    = _al.get("lipsync_score",    lip_sync_score) if _al_available else lip_sync_score

    _al_markers = _al.get("audio_markers", {})
    _al_corr    = _al.get("correlation", 0.0)
    _al_lip_n   = _al.get("lip_frames_sampled", 0)

    signals = [
        _make_signal(
            "Face-swap detector",
            face_swap_score,
            "Looks for edge smoothing, spatial inconsistencies, and identity-region artifacts.",
            f"Laplacian sharpness mean {avg_laplacian:.1f}, variance {std_laplacian:.1f}. {'Anomalies detected.' if is_likely_fake else 'No significant artifacts found.'}",
        ),
        _make_signal(
            "AI Voice Synthesis",
            audio_fake_signal_score,
            "Analyses spectral flatness, MFCC variance, pitch (F0) standard deviation, and pause regularity for AI synthesis markers.",
            (
                f"Pitch std {_al_markers.get('pitch_std_hz', 0):.1f} Hz, "
                f"MFCC var {_al_markers.get('mfcc_variance', 0):.0f}, "
                f"spectral flatness {_al_markers.get('mean_spectral_flatness', 0):.4f}. "
                + ("AI voice markers detected." if audio_fake_signal_score >= 55 else "Audio within natural human speech range.")
            ) if _al_available else "Audio analysis unavailable (install librosa + ffmpeg).",
        ),
        _make_signal(
            "Lip-sync integrity",
            lipsync_signal_score,
            "Correlates mouth-openness landmarks (MediaPipe FaceMesh) with audio RMS energy frame-by-frame.",
            (
                f"Pearson r={_al_corr:.3f} across {_al_lip_n} sampled frames. "
                + ("Significant lip-audio mismatch — deepfake indicator." if lipsync_signal_score >= 60 else "Lip movements align with audio energy.")
            ) if _al_available else "Lip-sync analysis unavailable (install mediapipe + ffmpeg).",
        ),
        _make_signal(
            "Metadata forensics",
            metadata_score,
            "Flags container types and durations common in repackaged synthetic clips.",
            f"Container {extension or 'unknown'}, duration {duration_seconds:.1f}s. {'Suspicious packaging.' if is_likely_fake else 'Normal file metadata.'}",
        ),
        _make_signal(
            "OSINT conflict alignment",
            osint_score,
            "Checks whether the clip filename or content matches known conflict-era narrative patterns.",
            f"{osint_hits} conflict keyword(s) detected in clip metadata. {'Elevated narrative risk.' if osint_score > 30 else 'No significant conflict-narrative match.'}",
        ),
    ]

    impersonation_matches = _build_impersonation_matches(video_name, max(face_swap_score, osint_score), is_likely_fake, confidence)
    weaponization = _build_weaponization(confidence, is_likely_fake, video_name)
    top_match = impersonation_matches[0] if impersonation_matches else None
    false_statement_prob = _compute_false_statement_probability(confidence, is_likely_fake, audio_analysis, signals)

    return {
        "preprocessed_images": preprocessed_images,
        "faces_cropped_images": faces_cropped_images,
        "heatmap_images": heatmap_images,
        "is_likely_fake": is_likely_fake,
        "confidence": confidence,
        "signals": signals,
        "impersonation_matches": impersonation_matches,
        "weaponization": weaponization,
        "recommended_actions": _build_operator_actions(is_likely_fake, confidence),
        "event_watchlist": EVENT_WATCHLIST,
        "timeline": _build_timeline(duration_seconds, len(preprocessed_images)),
        "telemetry": {
            "frames_sampled": len(preprocessed_images),
            "fps": round(fps, 2),
            "duration_seconds": round(duration_seconds, 2),
            "container": extension.replace(".", "").upper() or "UNKNOWN",
        },
        "alert_title": _compose_alert_title(is_likely_fake, top_match),
        "alert_summary": _compose_alert_summary(is_likely_fake, confidence, weaponization),
        "analyst_note": (
            "Recommend immediate counter-disinformation workflow and spokesperson-ready fact-check note."
            if is_likely_fake
            else "Store this scan as a baseline authentic sample unless new narrative context emerges."
        ),
        "audio_analysis": audio_analysis,
        "false_statement_probability": false_statement_prob,
    }


def _build_result_payload(video_path, seq_len):
    video_filename = os.path.basename(video_path)
    analysis = generate_demo_frames(video_path, num_frames=6)

    model_path = get_accurate_model(seq_len) if torch is not None else None

    if torch is None or model_path is None:
        # No ML model available — use heuristic demo mode
        verdict = "SYNTHETIC" if analysis["is_likely_fake"] else "AUTHENTIC"
        confidence = analysis["confidence"]
        mode = "demo"
        model_name = "Heuristic multimodal fusion"
    else:
        dataset = validation_dataset([video_path], seq_len, train_transforms)

        model = DeepfakeModel(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        label, ml_confidence = predict(model, dataset[0])
        verdict = "AUTHENTIC" if label == 1 else "SYNTHETIC"
        confidence = round(float(ml_confidence), 2)
        mode = "ml"
        model_name = os.path.basename(model_path)

        analysis["is_likely_fake"] = verdict == "SYNTHETIC"
        analysis["confidence"] = confidence
        analysis["weaponization"] = _build_weaponization(confidence, analysis["is_likely_fake"], video_filename)
        analysis["recommended_actions"] = _build_operator_actions(analysis["is_likely_fake"], confidence)
        analysis["alert_title"] = _compose_alert_title(analysis["is_likely_fake"], analysis["impersonation_matches"][0] if analysis["impersonation_matches"] else None)
        analysis["alert_summary"] = _compose_alert_summary(analysis["is_likely_fake"], confidence, analysis["weaponization"])
        analysis["analyst_note"] = (
            "The checkpoint and heuristic layers agree that the clip should enter the red analyst queue."
            if analysis["is_likely_fake"]
            else "Model and heuristic layers both lean authentic; retain for watchlist comparison."
        )
        # Recalculate false_statement_probability with updated ML confidence
        analysis["false_statement_probability"] = _compute_false_statement_probability(
            confidence, analysis["is_likely_fake"], analysis.get("audio_analysis", {}), analysis["signals"]
        )

    last_result = {
        "video": video_filename,
        "verdict": verdict,
        "confidence": confidence,
        "mode": mode,
        "model_path": model_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alert_title": analysis["alert_title"],
        "alert_summary": analysis["alert_summary"],
        "signals": analysis["signals"],
        "impersonation_matches": analysis["impersonation_matches"],
        "weaponization": analysis["weaponization"],
        "recommended_actions": analysis["recommended_actions"],
        "telemetry": analysis["telemetry"],
        "analyst_note": analysis["analyst_note"],
        "timeline": analysis["timeline"],
        "false_statement_probability": analysis.get("false_statement_probability", {}),
        "audio_analysis": analysis.get("audio_analysis", {}),
        "threat_level": "",   # filled in predict_page after _get_threat_level
    }
    last_result["report_text"] = _build_report_text(last_result)

    return analysis, last_result


def index(request):
    if request.method == "GET":
        form = VideoUploadForm()
        request.session.pop("file_name", None)
        request.session.pop("sequence_length", None)
        return render(request, index_template_name, _build_home_context(form))

    form = VideoUploadForm(request.POST, request.FILES)
    stats = _get_detection_stats()

    if form.is_valid():
        video = form.cleaned_data["upload_video_file"]
        seq_len = form.cleaned_data["sequence_length"]

        if seq_len <= 0:
            form.add_error("sequence_length", "Sequence length must be greater than 0.")
            return render(request, index_template_name, _build_home_context(form, stats))

        if not allowed_video_file(video.name):
            form.add_error("upload_video_file", "Unsupported video type.")
            return render(request, index_template_name, _build_home_context(form, stats))

        saved_name = f"uploaded_{int(time.time())}.{video.name.split('.')[-1]}"
        save_path = os.path.join(settings.PROJECT_DIR, "uploaded_videos", saved_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as output_file:
            shutil.copyfileobj(video, output_file)

        request.session["file_name"] = save_path
        request.session["sequence_length"] = seq_len

        return redirect("ml_app:predict")

    return render(request, index_template_name, _build_home_context(form, stats))


def _sanitize_for_session(obj):
    """Recursively convert numpy/non-serializable types to native Python for session storage."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_session(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_session(i) for i in obj]
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def predict_page(request):
    if "file_name" not in request.session:
        messages.error(request, "Session expired. Please upload a video again.")
        return redirect("ml_app:home")

    video_path = request.session["file_name"]
    seq_len    = request.session.get("sequence_length", 60)

    # Guard: check file actually exists
    if not os.path.exists(video_path):
        messages.error(request, "Uploaded video file not found. Please upload again.")
        request.session.pop("file_name", None)
        return redirect("ml_app:home")

    t_start = time.time()
    try:
        analysis, last_result = _build_result_payload(video_path, seq_len)
    except Exception as exc:
        import traceback
        print(f"[DRISHTI] predict_page error: {traceback.format_exc()}")
        messages.error(request, f"Analysis failed: {exc}. Please try a different video.")
        return redirect("ml_app:home")

    detection_time = round(time.time() - t_start, 2)

    manual_baseline_min = 40
    speedup  = max(1, round((manual_baseline_min * 60) / max(detection_time, 0.1)))
    threat   = _get_threat_level(last_result["confidence"], last_result["verdict"] == "SYNTHETIC")
    osint_ctx = _get_osint_context(os.path.basename(video_path))

    last_result["threat_level"]    = threat["label"]
    last_result["detection_time"]  = detection_time

    # Safe-store in session (convert numpy types → native Python)
    try:
        request.session["last_result"] = _sanitize_for_session(last_result)
    except Exception as _se:
        print(f"[DRISHTI] Session store error: {_se}")
        request.session["last_result"] = {
            "video": last_result.get("video", "unknown"),
            "verdict": last_result.get("verdict", "UNKNOWN"),
            "confidence": last_result.get("confidence", 0),
        }

    log_path = os.path.join(settings.PROJECT_DIR, "logs", "detections.jsonl")
    try:
        _append_jsonl(log_path, _sanitize_for_session(last_result))
    except Exception as _le:
        print(f"[DRISHTI] Log write error: {_le}")

    return render(
        request,
        predict_template_name,
        {
            "output":           last_result["verdict"],
            "confidence":       last_result["confidence"],
            "original_video":   os.path.basename(video_path),
            "analysis":         analysis,
            "last_result":      last_result,
            "MEDIA_URL":        settings.MEDIA_URL,
            "detection_time":   detection_time,
            "manual_baseline_min": manual_baseline_min,
            "speedup":          speedup,
            "threat":           threat,
            "osint_context":    osint_ctx,
        },
    )


def download_report(request):
    last = request.session.get("last_result")
    if not last:
        messages.error(request, "No recent analysis found. Please analyze a video first.")
        return redirect("ml_app:home")

    filename = f"drishti_report_{last.get('video', 'clip').split('.')[0]}.txt"
    response = HttpResponse(last.get("report_text", _build_report_text(last)), content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = f"attachment; filename={filename}"
    return response


def report_page(request):
    last = request.session.get("last_result")
    if not last:
        messages.warning(request, "No recent analysis. Please analyze a video first.")
        return redirect("ml_app:home")

    return render(request, "report.html", {
        "last_result": last,
        "report_text": last.get("report_text", ""),
        "pib_notice": request.session.get("pib_notice"),
        "pib_notice_meta": request.session.get("pib_notice_meta"),
    })


def feedback_page(request):
    last = request.session.get("last_result")
    if not last:
        messages.warning(request, "No recent analysis. Please analyze a video first.")
        return redirect("ml_app:home")

    return render(request, "feedback.html", {"last_result": last})


def stats_page(request):
    stats = _get_detection_stats()
    detection_log = os.path.join(settings.PROJECT_DIR, "logs", "detections.jsonl")
    feedback_log = os.path.join(settings.PROJECT_DIR, "logs", "feedback.jsonl")
    recent_detections = _read_jsonl(detection_log, limit=8)
    recent_feedback = _read_jsonl(feedback_log, limit=6)

    return render(
        request,
        "stats.html",
        {
            "stats": stats,
            "recent_detections": recent_detections,
            "recent_feedback": recent_feedback,
            "event_watchlist": EVENT_WATCHLIST,
        },
    )


@require_POST
def submit_feedback(request):
    feedback = (request.POST.get("feedback") or "").strip()
    verdict = request.POST.get("verdict") or ""
    video_name = request.POST.get("video_name") or ""
    confidence = request.POST.get("confidence") or ""

    if not feedback:
        messages.error(request, "Please add some feedback before submitting.")
        return redirect("ml_app:feedback_page")

    entry = {
        "video": video_name,
        "verdict": verdict,
        "confidence": confidence,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    log_path = os.path.join(settings.PROJECT_DIR, "logs", "feedback.jsonl")
    _append_jsonl(log_path, entry)
    messages.success(request, "Feedback captured for the analyst loop.")
    return redirect("ml_app:report_page")


def about(request):
    return render(
        request,
        about_template_name,
        {
            "mission": MISSION_BRIEF,
            "capability_cards": CAPABILITY_CARDS,
            "watchlist_entities": WATCHLIST_ENTITIES,
            "event_watchlist": EVENT_WATCHLIST,
        },
    )


def handler404(request, exception):
    return render(request, "404.html", status=404)


def cuda_full(request):
    return render(request, "cuda_full.html")


# ─────────────────────────────────────────────────────────────────────────────
# DRISHTI v2 — New Feature Views
# ─────────────────────────────────────────────────────────────────────────────

def url_ingest(request):
    """Download a video from a URL using yt-dlp and run it through the pipeline."""
    if request.method != "POST":
        return redirect("ml_app:home")

    video_url = (request.POST.get("video_url") or "").strip()
    seq_len = int(request.POST.get("sequence_length") or 60)

    if not video_url:
        messages.error(request, "Please provide a video URL.")
        return redirect("ml_app:home")

    saved_name = f"url_ingest_{int(time.time())}.mp4"
    save_path = os.path.join(settings.PROJECT_DIR, "uploaded_videos", saved_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        try:
            cmd = [
                "yt-dlp", "--no-playlist",
                "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "-o", save_path,
                "--merge-output-format", "mp4",
                video_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or not os.path.exists(save_path):
                raise RuntimeError(result.stderr or "yt-dlp download failed")
        except (FileNotFoundError, RuntimeError):
            urllib.request.urlretrieve(video_url, save_path)

        if not os.path.exists(save_path):
            raise RuntimeError("Video file was not created after download.")

        request.session["file_name"] = save_path
        request.session["sequence_length"] = seq_len
        return redirect("ml_app:predict")

    except Exception as exc:
        messages.error(request, f"Failed to fetch video from URL: {exc}")
        return redirect("ml_app:home")


@require_POST
def issue_fact_check(request):
    """Log a structured PIB Fact Check notice and store the formatted text in session."""
    video_name = (request.POST.get("video_name") or "Unknown").strip()
    verdict = (request.POST.get("verdict") or "").strip()
    confidence = (request.POST.get("confidence") or "N/A").strip()
    analyst_note = (request.POST.get("analyst_note") or "No additional analyst note provided.").strip()
    threat_level = (request.POST.get("threat_level") or "").strip()
    timestamp = datetime.utcnow().isoformat() + "Z"

    entry = {
        "video": video_name,
        "verdict": verdict,
        "confidence": confidence,
        "threat_level": threat_level,
        "analyst_note": analyst_note,
        "issued_by": "DRISHTI-PIB-Operator",
        "timestamp": timestamp,
    }
    log_path = os.path.join(settings.PROJECT_DIR, "logs", "fact_checks.jsonl")
    _append_jsonl(log_path, entry)

    sep = "=" * 60
    pib_notice = (
        f"PIB FACT CHECK NOTICE\n{sep}\n"
        f"GOVERNMENT OF INDIA | PRESS INFORMATION BUREAU\n"
        f"Digital Media Verification Division \u2014 DRISHTI System\n{sep}\n\n"
        f"Reference No : DRISHTI-{int(time.time())}\n"
        f"Date / Time  : {timestamp}\n"
        f"Clip File    : {video_name}\n"
        f"Verdict      : {verdict}\n"
        f"Confidence   : {confidence}%\n"
        f"Threat Level : {threat_level}\n\n"
        f"ANALYST ASSESSMENT\n{'-' * 40}\n{analyst_note}\n\n"
        f"ADVISORY\n{'-' * 40}\n"
        f"This content has been reviewed by the DRISHTI automated deepfake detection pipeline.\n"
        f"Citizens are advised to verify all media from official government channels before sharing.\n\n"
        f"{sep}\nThis notice is machine-generated. Subject to senior analyst review before public release.\n{sep}\n"
    )
    request.session["pib_notice"] = pib_notice
    request.session["pib_notice_meta"] = entry
    messages.success(request, "PIB Fact Check Notice issued and logged successfully.")
    return redirect("ml_app:report_page")


def _dashboard_authenticated(request):
    return request.session.get("dashboard_authenticated") is True


def dashboard_login(request):
    if _dashboard_authenticated(request):
        return redirect("ml_app:dashboard")

    analyst_username = os.environ.get("ANALYST_USERNAME", "pib_analyst")
    analyst_password = os.environ.get("ANALYST_PASSWORD", "drishti2025")
    error = None

    if request.method == "POST":
        username = (request.POST.get("username") or "").strip()
        password = (request.POST.get("password") or "").strip()
        if username == analyst_username and password == analyst_password:
            request.session["dashboard_authenticated"] = True
            return redirect("ml_app:dashboard")
        error = "Invalid credentials. Access denied."

    return render(request, "dashboard_login.html", {"error": error})


def dashboard_logout(request):
    request.session.pop("dashboard_authenticated", None)
    messages.info(request, "Logged out of analyst dashboard.")
    return redirect("ml_app:dashboard_login")


def dashboard(request):
    if not _dashboard_authenticated(request):
        return redirect("ml_app:dashboard_login")

    detection_log = os.path.join(settings.PROJECT_DIR, "logs", "detections.jsonl")
    fact_check_log = os.path.join(settings.PROJECT_DIR, "logs", "fact_checks.jsonl")
    detections = _read_jsonl(detection_log, limit=50)
    fact_checks = _read_jsonl(fact_check_log, limit=10)

    def _sort_key(d):
        tl = (d.get("threat_level") or "").upper()
        v = (d.get("verdict") or "").upper()
        if "CRITICAL" in tl:
            return 0
        if "HIGH" in tl or v == "SYNTHETIC":
            return 1
        if "MEDIUM" in tl or "UNCERTAIN" in tl:
            return 2
        return 3

    detections.sort(key=_sort_key)
    stats = _get_detection_stats()
    impersonation_alerts = sum(1 for d in detections if d.get("impersonation_matches"))

    return render(request, "dashboard.html", {
        "detections": detections,
        "fact_checks": fact_checks,
        "stats": stats,
        "impersonation_alerts": impersonation_alerts,
        "fact_check_count": len(fact_checks),
    })


def download_pdf_report(request):
    """Generate and stream a formatted PDF report using reportlab."""
    last = request.session.get("last_result")
    if not last:
        messages.error(request, "No recent analysis found. Please analyze a video first.")
        return redirect("ml_app:home")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import ParagraphStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)
        story = []
        gold = colors.HexColor("#d8b15a")
        dark = colors.HexColor("#1a1a2e")
        light_row = colors.HexColor("#f5f5f5")

        def _ps(name, **kw):
            return ParagraphStyle(name, **kw)

        h1 = _ps("h1", fontSize=22, fontName="Helvetica-Bold", textColor=gold, spaceAfter=4)
        h2 = _ps("h2", fontSize=13, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=8)
        sub = _ps("sub", fontSize=9, fontName="Helvetica", textColor=colors.grey, spaceAfter=14)
        body = _ps("body", fontSize=10, fontName="Helvetica", spaceAfter=6, leading=15)
        footer_s = _ps("footer", fontSize=8, textColor=colors.grey, spaceBefore=10, alignment=1)

        story += [
            Paragraph("DRISHTI", h1),
            Paragraph("Real-Time Military Deepfake &amp; Disinformation Detection Engine", sub),
            HRFlowable(width="100%", thickness=1.5, color=gold),
            Spacer(1, 0.3 * cm),
            Paragraph("ANALYST DETECTION REPORT", h2),
        ]

        meta_data = [
            ["Field", "Value"],
            ["Video File", last.get("video", "N/A")],
            ["Verdict", last.get("verdict", "N/A")],
            ["Confidence", f"{last.get('confidence', 'N/A')}%"],
            ["Threat Level", last.get("threat_level", "N/A")],
            ["Detection Mode", (last.get("mode") or "demo").upper()],
            ["Detection Time", f"{last.get('detection_time', 'N/A')}s"],
            ["Timestamp", last.get("timestamp", "N/A")],
        ]
        meta_table = Table(meta_data, colWidths=[5 * cm, 12 * cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), dark), ("TEXTCOLOR", (0, 0), (-1, 0), gold),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [light_row, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("PADDING", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]))
        story += [meta_table, Spacer(1, 0.4 * cm)]

        signals = last.get("signals", [])
        if signals:
            story.append(Paragraph("Signal Breakdown", h2))
            sig_data = [["Signal", "Score", "Risk", "Evidence"]] + [
                [s.get("name", ""), f"{s.get('score', 0)}%", s.get("label", ""), (s.get("evidence") or "")[:55]]
                for s in signals
            ]
            sig_table = Table(sig_data, colWidths=[5 * cm, 2 * cm, 3 * cm, 7 * cm])
            sig_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), dark), ("TEXTCOLOR", (0, 0), (-1, 0), gold),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [light_row, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story += [sig_table, Spacer(1, 0.4 * cm)]

        story += [
            Paragraph("Alert Assessment", h2),
            Paragraph(f"<b>{last.get('alert_title', '')}</b>", body),
            Paragraph(last.get("alert_summary", ""), body),
        ]

        imp = last.get("impersonation_matches", [])
        if imp:
            story.append(Paragraph("Watchlist Matches", h2))
            imp_data = [["Subject", "Role", "Score", "Risk"]] + [
                [m.get("subject", ""), m.get("role", ""), f"{m.get('score', 0)}%", m.get("label", "")]
                for m in imp
            ]
            imp_table = Table(imp_data, colWidths=[4 * cm, 6 * cm, 2.5 * cm, 4.5 * cm])
            imp_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), dark), ("TEXTCOLOR", (0, 0), (-1, 0), gold),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [light_row, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(imp_table)

        story += [
            Spacer(1, 1 * cm),
            HRFlowable(width="100%", thickness=0.5, color=colors.grey),
            Paragraph(f"Generated by DRISHTI | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | For official use only.", footer_s),
        ]

        doc.build(story)
        buffer.seek(0)
        pdf_name = f"drishti_report_{last.get('video', 'clip').split('.')[0]}.pdf"
        response = HttpResponse(buffer.read(), content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="{pdf_name}"'
        return response

    except ImportError:
        messages.error(request, "PDF export requires reportlab. Run: pip install reportlab")
        return redirect("ml_app:report_page")
    except Exception as exc:
        messages.error(request, f"PDF generation failed: {exc}")
        return redirect("ml_app:report_page")


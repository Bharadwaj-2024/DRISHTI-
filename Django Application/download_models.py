"""
DRISHTI — Model Download Utility
==================================
Downloads all required ML models for the production pipeline.
Run: python download_models.py
"""
import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_app", "ml_models")

DLIB_PREDICTOR_URL = (
    "https://github.com/itcomusic/face-recognition-svc/raw/master/"
    "shape_predictor_68_face_landmarks.dat"
)
DLIB_PREDICTOR_BZ2_URL = (
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
)
DLIB_PREDICTOR_FILE = "shape_predictor_68_face_landmarks.dat"


def download_file(url, dest, label="file"):
    if os.path.exists(dest):
        print(f"  [OK] {label} already exists: {dest}")
        return True
    print(f"  Downloading {label}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] Downloaded to {dest}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def download_dlib_predictor():
    """Download dlib 68-landmark shape predictor."""
    dest = os.path.join(MODELS_DIR, DLIB_PREDICTOR_FILE)
    if os.path.exists(dest):
        print(f"  [OK] dlib predictor already exists")
        return True

    # Try direct download first
    if download_file(DLIB_PREDICTOR_URL, dest, "dlib 68-landmark predictor"):
        return True

    # Try bz2 version
    bz2_dest = dest + ".bz2"
    if download_file(DLIB_PREDICTOR_BZ2_URL, bz2_dest, "dlib predictor (bz2)"):
        try:
            import bz2
            print("  Decompressing bz2...")
            with open(bz2_dest, "rb") as f_in:
                data = bz2.decompress(f_in.read())
            with open(dest, "wb") as f_out:
                f_out.write(data)
            os.remove(bz2_dest)
            print(f"  [OK] Decompressed to {dest}")
            return True
        except Exception as e:
            print(f"  [FAIL] Decompression error: {e}")
    return False


def download_huggingface_audio_model():
    """Pre-download the HuggingFace audio detection model."""
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        model_name = "MelodyMachine/Deepfake-audio-detection-V2"
        print(f"  Downloading {model_name}...")
        AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
        print(f"  [OK] HuggingFace model cached")
        return True
    except ImportError:
        print("  [SKIP] transformers not installed — run: pip install transformers")
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def download_xception_weights():
    """Pre-download XceptionNet ImageNet weights."""
    try:
        import torch.utils.model_zoo as model_zoo
        url = "http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"
        print("  Downloading XceptionNet ImageNet weights...")
        model_zoo.load_url(url)
        print("  [OK] XceptionNet weights cached")
        return True
    except ImportError:
        print("  [SKIP] torch not installed")
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    print("=" * 60)
    print("DRISHTI — Model Download Utility")
    print("=" * 60)
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    print("\n[1/3] dlib 68-landmark shape predictor")
    results["dlib"] = download_dlib_predictor()

    print("\n[2/3] XceptionNet ImageNet pretrained weights")
    results["xception"] = download_xception_weights()

    print("\n[3/3] HuggingFace Audio Deepfake Detection V2")
    results["audio"] = download_huggingface_audio_model()

    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, ok in results.items():
        status = "READY" if ok else "MISSING"
        print(f"  {name:20s} [{status}]")

    all_ok = all(results.values())
    print(f"\nAll models ready: {'YES' if all_ok else 'NO'}")
    if not all_ok:
        print("Note: DRISHTI will still run with available models.")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

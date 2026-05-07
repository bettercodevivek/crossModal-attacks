"""
Download sample images for cross-modal adversarial attacks.

Uses Lorem Picsum (https://picsum.photos/) — reliable, no API key.
(Unsplash Source URLs were removed / often return 503.)

Usage:
    python download_images.py
"""
import os

import requests

PICSUM_BASE = "https://picsum.photos/id/{pic_id}/{w}/{h}"


def _download_one(session: requests.Session, url: str, timeout: float = 25.0) -> bytes:
    r = session.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    if len(r.content) < 512:
        raise ValueError("response too small")
    return r.content


def _picsum_url(seed_index: int, slot: int, w: int = 800, h: int = 600) -> str:
    """Deterministic image id in [0, 999] — spreads picks across the catalog."""
    pic_id = (seed_index * 47 + slot * 19 + seed_index * slot * 3) % 1000
    return PICSUM_BASE.format(pic_id=pic_id, w=w, h=h)


def download_images(num_training=30, num_evaluation=10):
    """
    Download JPEGs from Lorem Picsum into data/images/ and data/holdout/.
    """
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/holdout", exist_ok=True)

    print("=" * 60)
    print("Downloading test images (Lorem Picsum)")
    print("=" * 60)

    session = requests.Session()
    session.headers.setdefault(
        "User-Agent",
        "crossModal-attacks/download_images (educational; contact: local)",
    )

    print(f"\nDownloading {num_training} training images to data/images/...")
    ok_train = 0
    for i in range(num_training):
        url = _picsum_url(i, slot=0)
        try:
            data = _download_one(session, url)
            path = f"data/images/img_{i + 1}.jpg"
            with open(path, "wb") as f:
                f.write(data)
            print(f"  [ok] [{i + 1}/{num_training}] {path}")
            ok_train += 1
        except Exception as e:
            # Retry with alternate offsets if one id is missing
            fallback = False
            for bump in range(1, 25):
                url2 = _picsum_url(i + bump * 31, slot=bump)
                try:
                    data = _download_one(session, url2)
                    path = f"data/images/img_{i + 1}.jpg"
                    with open(path, "wb") as f:
                        f.write(data)
                    print(f"  [ok] [{i + 1}/{num_training}] {path}  (fallback #{bump})")
                    ok_train += 1
                    fallback = True
                    break
                except Exception:
                    continue
            if not fallback:
                print(f"  [fail] [{i + 1}/{num_training}] {e}")

    print(f"\nTraining images: {ok_train}/{num_training} OK")

    print(f"\nDownloading {num_evaluation} holdout images to data/holdout/...")
    ok_eval = 0
    for i in range(num_evaluation):
        # Different slot than training so holdout != training picks
        url = _picsum_url(i, slot=50)
        try:
            data = _download_one(session, url)
            path = f"data/holdout/img_{i + 1}.jpg"
            with open(path, "wb") as f:
                f.write(data)
            print(f"  [ok] [{i + 1}/{num_evaluation}] {path}")
            ok_eval += 1
        except Exception as e:
            fallback = False
            for bump in range(1, 25):
                url2 = _picsum_url(i + bump * 29 + 100, slot=50 + bump)
                try:
                    data = _download_one(session, url2)
                    path = f"data/holdout/img_{i + 1}.jpg"
                    with open(path, "wb") as f:
                        f.write(data)
                    print(f"  [ok] [{i + 1}/{num_evaluation}] {path}  (fallback #{bump})")
                    ok_eval += 1
                    fallback = True
                    break
                except Exception:
                    continue
            if not fallback:
                print(f"  [fail] [{i + 1}/{num_evaluation}] {e}")

    print(f"\nHoldout images: {ok_eval}/{num_evaluation} OK")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print(f"\nTraining: data/images/   Holdout: data/holdout/")
    print("\nExample:")
    print("  cd src")
    print("  python demo_attack.py --attack fgsm")


if __name__ == "__main__":
    try:
        download_images()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nInstall requests if needed:  pip install requests")

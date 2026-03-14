"""
Simple script to download test images for cross-modal adversarial attacks.

This script downloads multiple images from Unsplash Source API and saves them
to the appropriate directories for training and evaluation.

Usage:
    python download_images.py
"""
import requests
import os
from pathlib import Path

def download_images(num_training=30, num_evaluation=10):
    """
    Download multiple test images from Unsplash Source API.
    
    Args:
        num_training: Number of training images to download (default: 30)
        num_evaluation: Number of evaluation images to download (default: 10)
    """
    
    # Create directories
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/holdout", exist_ok=True)
    
    # Diverse keywords for varied image content
    keywords = [
        "dog", "cat", "car", "building", "nature", "food", "person", "animal", 
        "landscape", "city", "beach", "mountain", "forest", "bird", "flower",
        "sunset", "ocean", "tree", "sky", "street", "house", "bridge", "river"
    ]
    
    print("=" * 60)
    print("Downloading Test Images for Cross-Modal Adversarial Attacks")
    print("=" * 60)
    print(f"\nDownloading {num_training} training images to data/images/...")
    
    # Download training images
    success_count = 0
    for i in range(num_training):
        keyword = keywords[i % len(keywords)]
        url = f"https://source.unsplash.com/800x600/?{keyword}"
        try:
            response = requests.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            filepath = f"data/images/img_{i+1}.jpg"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ [{i+1}/{num_training}] Downloaded {filepath} (keyword: {keyword})")
            success_count += 1
        except Exception as e:
            print(f"  ✗ [{i+1}/{num_training}] Failed to download: {e}")
    
    print(f"\n✓ Training images: {success_count}/{num_training} downloaded successfully")
    
    print(f"\nDownloading {num_evaluation} evaluation images to data/holdout/...")
    
    # Download evaluation images
    success_count = 0
    for i in range(num_evaluation):
        keyword = keywords[(i + 15) % len(keywords)]  # Use different keywords
        url = f"https://source.unsplash.com/800x600/?{keyword}"
        try:
            response = requests.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            filepath = f"data/holdout/img_{i+1}.jpg"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ [{i+1}/{num_evaluation}] Downloaded {filepath} (keyword: {keyword})")
            success_count += 1
        except Exception as e:
            print(f"  ✗ [{i+1}/{num_evaluation}] Failed to download: {e}")
    
    print(f"\n✓ Evaluation images: {success_count}/{num_evaluation} downloaded successfully")
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nTraining images saved to: data/images/ ({num_training} images)")
    print(f"Evaluation images saved to: data/holdout/ ({num_evaluation} images)")
    print("\nYou can now run the attacks:")
    print("  cd src")
    print("  python demo_attack.py --attack fgsm")
    print("  python demo_attack.py --attack pgd")
    print("  python demo_attack.py --attack patch")

if __name__ == "__main__":
    try:
        download_images()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nMake sure you have 'requests' installed:")
        print("  pip install requests")


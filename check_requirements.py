import sys

# Dictionary of package names to their import names
requirements = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'timm': 'timm',
    'opencv-python': 'cv2',
    'albumentations': 'albumentations',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'matplotlib': 'matplotlib',
    'grad-cam': 'pytorch_grad_cam',
    'PyYAML': 'yaml',
    'tqdm': 'tqdm'
}

def check_requirements():
    print("Checking required libraries...")
    all_passed = True
    failed = []

    for package, import_name in requirements.items():
        try:
            __import__(import_name)
            print(f"{package}: PASS")
        except ImportError:
            print(f"{package}: FAIL")
            all_passed = False
            failed.append(package)

    if all_passed:
        print("\nAll requirements satisfied!")
    else:
        print(f"\nMissing requirements: {', '.join(failed)}")
        print("Please install them using: pip install -r requirements.txt")

if __name__ == '__main__':
    check_requirements()

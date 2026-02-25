#!/bin/bash
# Setup script for Supernan AI Video Dubbing Pipeline
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo "Supernan AI Video Dubbing – Setup"
echo "=================================="

# 0. Check Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 1. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel --quiet

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet

# 3. Create output directory structure
echo "Creating project directories..."
mkdir -p input output/clips output/audio output/transcripts output/translations output/final models
touch input/.gitkeep output/clips/.gitkeep output/audio/.gitkeep \
      output/transcripts/.gitkeep output/translations/.gitkeep \
      output/final/.gitkeep models/.gitkeep

# 4. Clone Wav2Lip
echo "Cloning Wav2Lip..."
if [ ! -d "Wav2Lip" ]; then
    git clone https://github.com/Rudrabha/Wav2Lip.git --quiet
else
    echo "Wav2Lip already exists, skipping"
fi
cd Wav2Lip && pip install -r requirements.txt --quiet && cd ..

# 5. Download Wav2Lip checkpoint
echo "Downloading Wav2Lip checkpoint..."
if [ ! -f "Wav2Lip/checkpoints/wav2lip.pth" ]; then
    mkdir -p Wav2Lip/checkpoints
    wget -q --show-progress \
        "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z5DVux1K-153ZHjg" \
        -O Wav2Lip/checkpoints/wav2lip.pth || \
    echo "Auto-download failed. Download wav2lip.pth manually from:"
    echo "  https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
    echo "  Place it in: Wav2Lip/checkpoints/wav2lip.pth"
else
    echo "wav2lip.pth already exists, skipping"
fi

# 6. Clone and install GFPGAN
echo "Cloning GFPGAN..."
if [ ! -d "GFPGAN" ]; then
    git clone https://github.com/TencentARC/GFPGAN.git --quiet
    cd GFPGAN
    pip install -r requirements.txt --quiet
    python setup.py develop --quiet
    cd ..
else
    echo "GFPGAN already exists, skipping"
fi

# Download GFPGAN weights
if [ ! -f "GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth" ]; then
    mkdir -p GFPGAN/experiments/pretrained_models
    wget -q --show-progress \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth \
        -O GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth
fi

# 7. Install IndicTrans2 tokenizer (falls back to googletrans if this fails)
echo "Installing IndicTrans2 tokenizer..."
pip install git+https://github.com/VarunGumma/IndicTransTokenizer --quiet || \
    echo "IndicTrans2 install skipped — pipeline will use googletrans fallback"

# 8. Check FFmpeg
echo "Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "FFmpeg NOT found. Install it:"
    echo "  Mac:   brew install ffmpeg"
    echo "  Linux: sudo apt install ffmpeg"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Place your source video in: input/"
echo "  2. Edit config.yaml with your timestamps"
echo "  3. Colab: open notebooks/ai_dubbing_colab.ipynb (Runtime > T4 GPU)"
echo "  4. Local:  python dub_video.py --help"

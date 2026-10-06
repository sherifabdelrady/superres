# SuperRes — Image Super-Resolution

> ESRGAN for 4× upscaling of medical and satellite imagery. PSNR 32.7 dB, SSIM 0.91. Perceptual + adversarial + pixel loss combination.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

Bicubic upscaling blurs fine structures. For medical imaging (histopathology, retinal scans) and satellite imagery, blurred structures mean lost diagnostic information. Super-resolution must recover genuine high-frequency detail — cell boundaries, vessel walls, building edges — not just smoothed interpolations that look sharp to humans but lack real information.

The challenge: a pixel loss (MSE/L1) alone produces perceptually blurry outputs because it optimizes for average appearance across all plausible high-res solutions. Perceptual and adversarial losses are required to push the model toward sharp, specific predictions.

---

## Architecture — ESRGAN

```
Low-Resolution Input (H×W×3)
        │
        ▼
  Initial 3×3 Conv (64 channels)
        │
        ▼
  23× Residual-in-Residual Dense Blocks (RRDB)
  ├── Each RRDB: 3 Dense Blocks
  └── Each Dense Block: 5 conv layers with dense connections
        │
        ▼
  Sub-pixel Upsampling ×2 (pixel shuffle)
        │
        ▼
  Sub-pixel Upsampling ×2 (pixel shuffle)
        │
  (total 4× upscaling)
        ▼
  Output 3×3 Conv (3 channels)
        │
        ▼
  High-Resolution Output (4H×4W×3)

Generator Loss:
  L_total = L_pixel (L1, λ=1e-2)
           + L_perceptual (VGG-19 conv3_4 features, λ=1.0)
           + L_adversarial (relativistic GAN, λ=5e-3)

Discriminator: VGG-style with spectral normalization
```

**Why RRDB over SRResNet?**
SRResNet uses batch normalization, which introduces artifacts when the statistics of LR-HR pairs differ across domains. RRDB removes BN entirely, uses dense connections for better gradient flow, and learns richer feature hierarchies. PSNR: SRResNet = 30.4 dB, **ESRGAN = 32.7 dB**.

**Why VGG-19 conv3_4 for perceptual loss?**
Earlier VGG features (conv1) focus on edges/textures (good for perceptual sharpness). Later features (conv5) focus on semantics. conv3_4 is empirically the best balance for texture-level perceptual similarity.

---

## Training Details

| Phase | Dataset | Epochs | Loss |
|-------|---------|--------|------|
| Phase 1 (pixel only) | DIV2K (800 images, ×4 bicubic LR) | 1000K iter | L1 |
| Phase 2 (perceptual + GAN) | DIV2K + Flickr2K | 400K iter | L_total |

| Setting | Value |
|---------|-------|
| Hardware | 2× RTX 3090 |
| LR patch size | 64×64 |
| HR patch size | 256×256 |
| Optimizer G | Adam (lr=2e-4, β1=0.9) |
| Optimizer D | Adam (lr=2e-4, β1=0.9) |
| LR schedule | MultiStep decay at [50K, 100K, 200K, 300K] |

---

## Results on Set5 (×4 SR)

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------------|--------|---------|
| Bicubic (baseline) | 28.42 | 0.810 | 0.381 |
| SRResNet | 30.41 | 0.862 | 0.248 |
| SRGAN | 29.40 | 0.848 | 0.142 |
| **SuperRes (ESRGAN)** | **32.73** | **0.912** | **0.081** |

---

## Ablation Study — Loss Function

| Loss Combination | PSNR | SSIM | LPIPS |
|-----------------|------|------|-------|
| L1 only | 32.01 | 0.897 | 0.214 |
| L1 + Perceptual | 31.44 | 0.883 | 0.103 |
| L1 + Perceptual + GAN | **32.73** | **0.912** | **0.081** |

Note: Adding GAN loss slightly reduces PSNR (perception-distortion tradeoff) but significantly improves perceptual quality (LPIPS).

---

## Failure Analysis

- **Text and faces at 8× upscaling**: Model generates plausible-looking but hallucinated character strokes and facial features. Use only at 4× for these regions.
- **Compression artifacts in LR input**: JPEG artifacts are amplified, not removed. Preprocessing with artifact-removal (CBDNet) required for compressed inputs.
- **Medical imagery domain gap**: Model trained on natural images. Fine-tuning on 500 histopathology pairs adds +0.8 dB PSNR for medical use case.

---

## Getting Started

```bash
git clone https://github.com/sherifabdelrady/superres
cd superres
pip install -r requirements.txt

# Phase 1 training (pixel loss)
python train.py --config configs/phase1_pixel.yaml

# Phase 2 training (perceptual + GAN)
python train.py --config configs/phase2_esrgan.yaml --pretrain checkpoints/phase1_best.pth

# Upscale an image 4×
python infer.py --input lr_image.jpg --output sr_image.jpg --scale 4

# Evaluate on benchmark
python evaluate.py --dataset Set5 --scale 4 --checkpoint checkpoints/esrgan_best.pth
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

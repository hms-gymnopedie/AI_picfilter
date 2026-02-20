# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI_picfilter is a project for AI-based image style/mood learning and automated filter generation. The goal is to learn the visual characteristics (color, lighting, texture, atmosphere) of reference images and generate reusable filters that can transfer those styles to other images.

## Key Technical Domains

The project draws on these core technologies (documented in `pre_report.md`):

- **Neural Style Transfer** — Deep Photo Style Transfer with photorealism regularization to preserve photo structure while transferring color/mood
- **Image-Adaptive 3D LUTs** — Lightweight CNN that predicts blending weights for basis LUTs; real-time 4K capable (<2ms)
- **Neural Implicit LUTs (NILUT)** — MLP-based continuous color transform; supports multi-style blending in a single model (<0.25MB for 512 styles)
- **GAN-based style transfer** — StarGAN v2 for multi-domain image translation with style encoder
- **Diffusion-based LUT (D-LUT)** — Score-matching to learn color distributions; outputs standard `.cube` files compatible with Photoshop/Premiere

## Reference Open-Source Projects

- `luanfujun/deep-photo-styletransfer` — Torch/Lua + MATLAB, requires CUDA
- `HuiZeng/Image-Adaptive-3DLUT` — PyTorch, supports paired/unpaired training
- `mv-lab/nilut` — Neural implicit LUT with multi-style blending tutorials
- `clovaai/stargan-v2` — Multi-domain image translation (Naver Clova AI)

## Quality Metrics

- Color accuracy: CIE76 ΔE < 1.0 (perceptually indistinguishable)
- Inference speed: < 16ms at 4K (real-time 60fps target)
- Model size: < 10MB (edge device deployable)
- Structure preservation: PSNR and SSIM for texture/edge retention

## Current State

The project is in the research/planning phase. `pre_report.md` contains the technical analysis (in Korean). No implementation code exists yet.

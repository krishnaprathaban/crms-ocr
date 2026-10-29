---
title: CRMS Newspaper OCR
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# CRMS Self-Hosted Broadsheet OCR API

This Space runs **OpenCV (ad border detection) + PaddleOCR (text recognition)** on Hugging Face's free 2 vCPU / 16 GB RAM tier and exposes a `POST /scan` endpoint compatible with `crms.tnie.top`.

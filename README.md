# Through the Eyes of the Beholder - Presentation Website

[![Paper](https://img.shields.io/badge/Paper-CLEF%20EXIST%202026-7c3aed)](https://When-Paper-Appears-it-Will-Work.com)
[![Code Implementation](https://img.shields.io/badge/Code-Implementation-green)](https://github.com/DS4AI-UPB/VANGUARD-CLEF2026-EXIST)
[![arXiv](https://img.shields.io/badge/arXiv-WIP-b31b1b.svg)](https://arxiv.org/abs/WIP)
[![Leaderboard](https://img.shields.io/badge/EXIST%202026%20Task%202.2-29th%2F114%20%C2%B7%20soft-b8860b)](https://github.com/DS4AI-UPB/VANGUARD-CLEF2026-EXIST)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the source code and assets for the official project website of the paper **"Through the Eyes of the Beholder: Biometric and Demographic Conditioning for Multimodal Sexism Detection"**, to be presented at the **EXIST Lab @ CLEF 2026 (Task 2)** by team **VANGUARD**.

**Live Website:** [ds4ai-upb.github.io/VANGUARD-CLEF2026-EXIST](https://ds4ai-upb.github.io/VANGUARD-CLEF2026-EXIST/)

## Paper Summary

**Authors:** Ana-Maria Luisa Mocanu, Sebastian Mocanu, Ciprian-Octavian Truică, Elena-Simona Apostol

**Abstract:**

> Detecting sexism on the internet is a fundamentally subjective task; our team, VANGUARD, addresses this challenge in the EXIST 2026 Task 2 by proposing a human-centered multimodal framework that analyses and incorporates the psychological and demographic characteristics of human annotators into the detection pipeline. We fuse five input modalities through a cross-attention architecture with Feature-wise Linear Modulation conditioning. Meme text is extracted and visually described with Gemma 4, then augmented by automatic translation between English and Spanish with NLLB-200. Text and image representations are produced by LoRA-adapted XLM-RoBERTa and CLIP encoders and fused with sensor features encoded by a pretrained autoencoder. To model annotator subjectivity, we frame Subtask 2.1 as a label distribution learning problem, optimizing a Kullback-Leibler divergence loss over the full annotator label distribution. At inference time, predictions are produced by soft-voting between the deep multimodal network and a complementary SVM trained on stylometric and physiological features. Our best submission ranks 29th out of 114 on Subtask 2.2 (source intention) under soft evaluation, and the normalized ICM scores remain above the baseline on Subtasks 2.1 and 2.2, indicating that annotator-centered conditioning contributes a usable signal. We release our full pipeline and analysis to support reproducible human-centered modeling. 

## Resources
- [Paper (CLEF EXIST 2026 Working Notes)](https://When-Paper-Appears-it-Will-Work.com) - placeholder until the official proceedings entry is available
- [arXiv](https://arxiv.org/abs/WIP) - WIP
- [Code](https://github.com/DS4AI-UPB/VANGUARD-CLEF2026-EXIST) - full five-stage pipeline, training, and ablation scripts

## Local Development

Simply open `index.html` in a browser, or serve with any static file server:

```bash
python -m http.server 8000
```

## Deployment

The site auto-deploys to GitHub Pages via the included workflow
[.github/workflows/github-pages.yml](.github/workflows/github-pages.yml),
which handles image compression and CSS/JS minification.

## Citation
```bibtex
@InProceedings{Mocanu_2026_EXIST_CLEF,
    author    = {Mocanu, Ana-Maria Luisa and Mocanu, Sebastian and Truică, Ciprian-Octavian and Apostol, Elena-Simona},
    title     = {Through the Eyes of the Beholder: Biometric and Demographic Conditioning for Multimodal Sexism Detection},
    booktitle = {Conference and Labs of the Evaluation Forum (CLEF), EXIST 2026 Lab, Task 2},
    month     = {September},
    year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
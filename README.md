---
title: Chess Opening Recommender
emoji: 💻
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Recommend fun chess openings
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

# Chess Opening Recommender – Model Space

This Hugging Face Space hosts the **inference layer** for a chess opening recommender system.

The underlying project trains separate machine-learning models for **White** and **Black** using a player’s Lichess game history and predicts expected performance (score) across chess openings. This Space exposes those models for lightweight inference and integration with a web frontend.

## Repository contents

* Trained model weights (`.pt`) for White and Black
* Supporting model artifacts (ID mappings, side information, lookup tables)
* Application code used to load the models and serve predictions

## Large files / Git LFS

Model weights and auxiliary artifacts are tracked using **Git Large File Storage (LFS)**.
The repository contains lightweight pointer files; the actual binaries are stored and retrieved transparently by Hugging Face during build and runtime.

This keeps the git history clean while allowing the models to be versioned alongside the inference code.

## Related project

This Space is part of a larger personal project combining:

* Lichess game data processing
* Machine learning for opening recommendation
* A Next.js / TypeScript frontend consuming this model as a service (TBA)

---

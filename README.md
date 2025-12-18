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

The underlying project trains separate machine-learning models for **White** and **Black** using a player's Lichess game history and predicts expected performance (score) across chess openings. This Space exposes those models for lightweight inference and integration with a web frontend.

## Local Development

### Start the server
```bash
# Create and activate virtual environment (first time only)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic

# Start the server
uvicorn app:app --port 7860
```

### Kill the server
Press `Ctrl+C` in the terminal, or if running in background:
```bash
kill %1
```

### Test the API

**Health check:**
```bash
curl http://127.0.0.1:7860/health
```

**Predict for White:**
```bash
curl -X POST http://127.0.0.1:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "side": "white",
    "player": {
      "player_id": "test123",
      "username": "TestPlayer",
      "rating_normalized": 0.65,
      "site": "lichess"
    },
    "openings": []
  }'
```

**Predict for Black:**
```bash
curl -X POST http://127.0.0.1:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "side": "black",
    "player": {
      "player_id": "test456",
      "username": "AnotherPlayer",
      "rating_normalized": 0.72,
      "site": "lichess"
    },
    "openings": []
  }'
```

**Run test script:**
```bash
source venv/bin/activate
pip install requests
python test_api.py
```

**API Documentation:**
Visit http://127.0.0.1:7860/docs for interactive Swagger UI

---

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

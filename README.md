# NeerWatch

NeerWatch is a water evidence and prototype risk dashboard for Northeast India. The interface combines published historical water observations, a field verification workflow, and a separate ML Model Studio.

## Model data notice

The repository contains saved prediction and SHAP CSV outputs dated August 2024, but no executable XGBoost weights or live sensor feeds. It also includes a separately trained logistic-regression demo model that scores CSV rows from generated synthetic training labels. This demonstrates inference only; it is not validated for real-world water-safety decisions. All other risk values are synthetic demonstration outputs, not current findings. Verify conditions with field and laboratory testing.

## Run locally

Requirements: Node.js 18 or newer. There are no third-party npm dependencies.

```sh
npm start
```

Then open [http://localhost:8000](http://localhost:8000). The saved model inspection page is at [http://localhost:8000/model-studio](http://localhost:8000/model-studio), and the separate CSV inference experience is at [http://localhost:8000/synthetic-demo](http://localhost:8000/synthetic-demo). Set `PORT` to use another port. Local mode allows field verification records to be written to `backend/data/verification.json`.

## Pages and API

- Home: published evidence workspace, map, historical readings, and field test flow.
- Synthetic Demo: upload and validate a CSV, then run the separate synthetic logistic-regression demo and inspect row-level results.
- About: the project story, evidence workflow, model-output explanation, limitations, and walkthrough for judges.
- ML Model Studio: saved zone estimates, SHAP explanations, model flow, and snapshot integrity check.
- `GET /health`: deployment and data status.
- `GET /api/v1/locations` and `GET /api/v1/risk`: location and saved risk records.
- `GET /api/v1/risk/{location_id}` and `GET /api/v1/locations/{location_id}/history`: zone details and historical observations.
- `GET /api/v1/model/check`: checks saved prediction and SHAP record integrity; it does not run inference.
- `GET /synthetic-demo.csv`: downloads the three-row synthetic demo fixture.
- `POST /api/v1/model/demo-score`: runs the separately trained synthetic logistic-regression demo on 1–100 uploaded CSV rows. Output is demonstration-only.
- `POST /api/v1/locations/{location_id}/verification`: saves a field verification record in local mode.

## Deployment

This repository includes a Render Blueprint (`render.yaml`) and a Dockerfile. The Render configuration deploys a **read-only public demo**: `NEERWATCH_READ_ONLY=true` blocks public field record writes and dataset uploads. To enable writes, deploy behind an authentication layer and use persistent storage for `backend/data/verification.json`; do not expose the unauthenticated local write API directly to the public internet.

For Render, push this project to a Git provider connected to your Render account, then create a new Blueprint instance from `render.yaml`. The Docker image can be built with `docker build -t neerwatch .` and run with `docker run --rm -p 8000:10000 neerwatch`.

Retrain the deterministic synthetic demonstration weights with `npm run train:demo-model`. A three-row synthetic CSV fixture is included for inference demonstrations.

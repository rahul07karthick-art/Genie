# NeerWatch prototype API

Small Node.js HTTP API for the Northeast India water-monitoring dashboard. It uses Node's built-in modules, so there is no package install step.

## Run locally

From the workspace root:

```powershell
node backend/server.mjs
```

The dashboard and API listen on `http://localhost:8000` by default. Open that address to use the dashboard; it fetches its records from this backend. Set `$env:PORT` before starting to use another port. Check `/health` or `/api/v1` for status and route discovery.

The standalone ML Model Studio is available at `http://localhost:8000/model-studio`. It lets you inspect assessment zones and saved SHAP contributors. The separate Synthetic Demo page is available at `http://localhost:8000/synthetic-demo`; it validates CSV inputs and runs the synthetic logistic-regression demo. Its training labels are generated, so these outputs demonstrate inference mechanics and are not validated water-safety predictions. The bundled zone predictions remain synthetic August 2024 outputs.

For a public demo, set `NEERWATCH_READ_ONLY=true`. Field-verification writes and the legacy dataset upload route are rejected. The bounded, stateless synthetic model-score route is available because it does not persist uploaded data. The Render Blueprint and Dockerfile use read-only mode.

`POST /api/v1/model/demo-score` accepts CSV text (maximum 100 KB, 1–100 records) with the required fields `location_id,state,date,ph,e_coli,rainfall_mm,historical_cases,source_consistency,sensor_reliability,turbidity_ntu`. Run `npm run train:demo-model` to regenerate `ml-artifacts/synthetic_demo_model.json` from deterministically generated synthetic examples. The reported holdout accuracy is synthetic-only and must not be interpreted as real-world model performance.

## Routes

- `GET /api/v1/locations` — location catalog. Optional `?state=Assam` filter.
- `GET /api/v1/risk` — map/dashboard shape for all locations. Optional `?state=Assam` filter.
- `GET /api/v1/risk/{location_id}` — risk, reliability, environment, water quality, health, explanation, verification, and published-sample detail.
- `GET /api/v1/locations/{location_id}/history` — historical observations; unknown dates remain `null` and are accompanied by their source period.
- `POST /api/v1/locations/{location_id}/verification` — append a field-verification record to `backend/data/verification.json`.

### Verification request example

```json
{
  "verification_status": "COMPLETED",
  "recommended_verification_priority": "HIGH",
  "reason_for_priority": "Fresh field sample collected after historical exceedance.",
  "verification_date": "2026-09-29",
  "actual_field_result": {
    "result": "INCONCLUSIVE",
    "tested_parameters": [],
    "notes": "Replace this example with the actual field team's result."
  }
}
```

## Data and safety semantics

- The bundled catalog contains the 44 sample locations already shown in the prototype, scoped to Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, and Tripura.
- The bundled environmental readings are **published historical observations**, not live measurements. Each has a period, source label/link, and screening summary.
- Population and location precision are unavailable/approximate in the source dataset, so population is `null` and the coordinate precision is labeled `APPROXIMATE_LOCALITY`.
- Risk probability, risk level, priority, reliability, rainfall/flood evidence, current water-quality readings, and disease cases are `null` until real feeds/model outputs are connected. `risk_status: UNAVAILABLE` distinguishes that from a measured zero or low risk.
- An empty `top_risk_drivers` list means explanations are not connected; it does not mean there are no risk drivers.
- Historical source observations are returned separately as `published_observation` and in the history endpoint. They are not substituted for current values or ML predictions.
- The API has permissive CORS and no authentication. Public deployments must use `NEERWATCH_READ_ONLY=true`; do not enable public write routes or put identifiable health data into it without adding authentication and privacy controls.

## Frontend connection

For a local frontend, request `http://localhost:8000/api/v1/locations` and `http://localhost:8000/api/v1/risk`. For a deployed site, set the API base URL in the frontend to the backend host and replace the prototype's permissive CORS policy with the exact frontend origin.

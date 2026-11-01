import { createServer } from 'node:http';
import { randomUUID } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { predictSyntheticDemo } from './synthetic_demo_model.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const seed = JSON.parse(await readFile(resolve(here, 'data/locations.json'), 'utf8'));
const dashboardFile = resolve(here, '../outputs/waterguard-dashboard/index.html');
const modelStudioFile = resolve(here, '../outputs/waterguard-dashboard/model-studio.html');
const aboutFile = resolve(here, '../outputs/waterguard-dashboard/about.html');
const syntheticDemoFile = resolve(here, '../outputs/waterguard-dashboard/synthetic-demo.html');
const syntheticDemoCsvFile = resolve(here, '../ml-artifacts/synthetic_smoke_inputs.csv');
const verificationFile = resolve(here, 'data/verification.json');
const demoModel = JSON.parse(await readFile(resolve(here, '../ml-artifacts/synthetic_demo_model.json'), 'utf8'));
let diseaseTrends = {};
try {
  diseaseTrends = JSON.parse(await readFile(resolve(here, 'data/history_v2.json'), 'utf8'));
} catch (e) {
  console.warn('No history_v2.json found');
}
const PORT = Number(process.env.PORT) || 8000;
const READ_ONLY = ['true', '1'].includes(String(process.env.NEERWATCH_READ_ONLY || '').toLowerCase());
const stateSet = new Set(seed.states);
const byId = new Map(seed.locations.map(location => [location.location_id, location]));

// -- V2 ML INTEGRATION --
let mlData = new Map(); // loc_id -> latest ML record
let analysisState = { status: 'MODEL_SNAPSHOT_READY', errors: [] };
let uploadedCsvText = null;

function parseCSV(csvText) {
  const lines = String(csvText || '').replace(/^\uFEFF/, '').split(/\r?\n/).filter(line => line.trim());
  if (lines.length < 2) return [];
  const parseLine = line => {
    const fields = [];
    let value = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"' && quoted && line[i + 1] === '"') { value += '"'; i++; }
      else if (char === '"') quoted = !quoted;
      else if (char === ',' && !quoted) { fields.push(value.trim()); value = ''; }
      else value += char;
    }
    fields.push(value.trim());
    return fields;
  };
  const headers = parseLine(lines[0]);
  return lines.slice(1).map(line => {
    const values = parseLine(line);
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = values[i]?.trim();
    });
    return obj;
  });
}

try {
  const { existsSync } = await import('node:fs');
  const xgbPriorityFile = resolve(here, '../ml-artifacts/xgb_v2_verification_priority.csv');
  const shapExplanationsFile = resolve(here, '../ml-artifacts/shap_v2_test_explanations.csv');

  if (existsSync(xgbPriorityFile)) {
    const priorityCsv = await readFile(xgbPriorityFile, 'utf8');
    const priorityRows = parseCSV(priorityCsv);
    for (const row of priorityRows) {
      if (row.date === '2024-08-01') {
        mlData.set(row.location_id, row);
      }
    }
  }
  if (existsSync(shapExplanationsFile)) {
    const shapCsv = await readFile(shapExplanationsFile, 'utf8');
    const shapRows = parseCSV(shapCsv);
    for (const row of shapRows) {
      if (mlData.has(row.location_id)) {
        mlData.get(row.location_id).top_positive_contributors = row.top_positive_contributors;
        mlData.get(row.location_id).top_negative_contributors = row.top_negative_contributors;
      }
    }
  }
} catch (e) {
  console.error('Warning: ML data not loaded', e.message);
}

function getMlRecordFor(location_id) {
  return mlData.get(location_id) || null;
}

function checkModelSnapshot() {
  const rows = [...mlData.values()].filter(row => row.date === '2024-08-01');
  const errors = [];
  const probabilityFields = ['raw_risk_probability', 'calibrated_risk_probability'];
  const scoreFields = ['priority_score', 'reliability_component', 'data_completeness', 'sensor_reliability', 'source_consistency'];
  for (const row of rows) {
    for (const field of [...probabilityFields, ...scoreFields]) {
      const value = Number(row[field]);
      if (!Number.isFinite(value) || value < 0 || value > 1) errors.push(`${row.location_id}: invalid ${field}`);
    }
    if (!row.top_positive_contributors || !row.top_negative_contributors) errors.push(`${row.location_id}: SHAP contributions missing`);
  }
  return {
    snapshot_status: rows.length === 200 && errors.length === 0 ? 'PASS' : 'CHECK_REQUIRED',
    snapshot_date: '2024-08-01',
    zones_checked: rows.length,
    valid_predictions: rows.length - new Set(errors.map(error => error.split(':')[0])).size,
    shap_explanations: rows.filter(row => row.top_positive_contributors && row.top_negative_contributors).length,
    errors,
    model_weights_available: false,
    inference_available: false,
    demo_model_available: true,
    message: 'This check validates the saved August 2024 XGBoost prediction and SHAP files. New rows can be scored separately by the clearly labeled synthetic logistic demo model.'
  };
}

const demoRequiredColumns = ['location_id', 'state', 'date', 'ph', 'e_coli', 'rainfall_mm', 'historical_cases', 'source_consistency', 'sensor_reliability', 'turbidity_ntu'];

function validateDemoRows(rows) {
  if (!rows.length) return ['CSV is empty or has no data rows'];
  if (rows.length > 100) return ['Upload up to 100 rows per model run'];
  const missing = demoRequiredColumns.filter(column => !(column in rows[0]));
  if (missing.length) return missing.map(column => `Missing required column: ${column}`);
  const errors = [];
  rows.forEach((row, index) => {
    const line = index + 2;
    if (!row.location_id || !row.state || !row.date) errors.push(`Row ${line}: location_id, state, and date are required`);
    for (const column of ['ph', 'e_coli', 'rainfall_mm', 'historical_cases', 'source_consistency', 'sensor_reliability', 'turbidity_ntu']) {
      const value = Number(row[column]);
      if (!Number.isFinite(value)) errors.push(`Row ${line}: ${column} must be numeric`);
    }
    if (Number(row.ph) < 0 || Number(row.ph) > 14) errors.push(`Row ${line}: ph must be between 0 and 14`);
    for (const column of ['e_coli', 'rainfall_mm', 'historical_cases', 'turbidity_ntu']) {
      if (Number(row[column]) < 0) errors.push(`Row ${line}: ${column} cannot be negative`);
    }
    for (const column of ['source_consistency', 'sensor_reliability']) {
      if (Number(row[column]) < 0 || Number(row[column]) > 1) errors.push(`Row ${line}: ${column} must be between 0 and 1`);
    }
  });
  return errors;
}

if (byId.size !== seed.locations.length) throw new Error('Duplicate location_id in locations.json');
for (const location of seed.locations) {
  if (!stateSet.has(location.state)) throw new Error(`Out-of-scope state in seed data: ${location.state}`);
  if (!Number.isFinite(location.latitude) || !Number.isFinite(location.longitude)) {
    throw new Error(`Invalid coordinates for ${location.location_id}`);
  }
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': READ_ONLY ? 'GET, OPTIONS' : 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400'
};

function sendJson(response, status, payload) {
  response.writeHead(status, {
    ...corsHeaders,
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store'
  });
  response.end(JSON.stringify(payload));
}

function sendHtml(response, status, html) {
  response.writeHead(status, {
    ...corsHeaders,
    'Content-Type': 'text/html; charset=utf-8',
    'Cache-Control': 'no-store'
  });
  response.end(html);
}

function sendError(response, status, code, message) {
  sendJson(response, status, { error: { code, message } });
}

function locationType(location_id) {
  return location_id.startsWith('LOC_') ? 'ML_ASSESSMENT_ZONE' : 'HISTORICAL_EVIDENCE';
}

function locationFields(location) {
  return {
    location_id: location.location_id,
    location_type: locationType(location.location_id),
    state: location.state,
    district: location.district,
    location_name: location.location_name,
    latitude: location.latitude,
    longitude: location.longitude,
    population: location.population,
    coordinate_precision: location.coordinate_precision
  };
}

function selectedLocations(state) {
  if (!state) return seed.locations;
  if (!stateSet.has(state)) return null;
  return seed.locations.filter(location => location.state === state);
}

function mapRisk(location) {
  const sample = location.published_observation;
  const ml = getMlRecordFor(location.location_id);
  return {
    ...locationFields(location),
    risk_probability: ml ? parseFloat(ml.raw_risk_probability) : null,
    risk_level: ml ? ml.risk_level : null,
    priority_score: ml ? parseFloat(ml.priority_score) : null,
    priority_level: ml ? (ml.priority_level ? ml.priority_level.replace(' Priority', '').replace(' PRIORITY', '').toUpperCase() : null) : null,
    confidence: ml ? parseFloat(ml.reliability_component) : null,
    reliability: ml ? parseFloat(ml.reliability_component) : null,
    data_completeness: ml ? parseFloat(ml.data_completeness) : null,
    data_freshness_days: ml ? parseFloat(ml.data_freshness_days) : null,
    source_consistency: ml ? parseFloat(ml.source_consistency) : null,
    last_updated: ml ? ml.date : null,
    risk_status: ml ? 'AVAILABLE' : 'UNAVAILABLE',
    published_observation_summary: sample?.summary ?? null,
    published_observation_period: sample?.period ?? null,
    published_observation_source: sample?.source_id ?? null,
    published_observation_source_label: sample?.source_label ?? null,
    published_observation_source_url: sample ? seed.sources[sample.source_id] ?? null : null,
    published_observation: sample ? {
      ...sample,
      source_url: seed.sources[sample.source_id] ?? null
    } : null
  };
}

async function loadVerifications() {
  try {
    const value = JSON.parse(await readFile(verificationFile, 'utf8'));
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch (error) {
    if (error.code === 'ENOENT') return {};
    throw error;
  }
}

function latestVerification(records) {
  return records.length ? records[records.length - 1] : {
    verification_status: 'NOT_STARTED',
    reason_for_priority: null,
    verification_date: null,
    actual_field_result: null
  };
}

function detailFor(location, verificationRecords) {
  const rawSample = location.published_observation ?? null;
  const ml = getMlRecordFor(location.location_id);
  const sample = rawSample ? {
    ...rawSample,
    source_url: seed.sources[rawSample.source_id] ?? null
  } : null;
  const verification = latestVerification(verificationRecords);
  return {
    location_id: location.location_id,
    location_type: locationType(location.location_id),
    location: {
      state: location.state,
      district: location.district,
      village: location.location_name,
      location_name: location.location_name,
      latitude: location.latitude,
      longitude: location.longitude,
      population: location.population,
      coordinate_precision: location.coordinate_precision
    },
    risk: { 
      probability: ml ? parseFloat(ml.raw_risk_probability) : null, 
      level: ml ? ml.risk_level : null, 
      last_updated: ml ? ml.date : null, 
      status: ml ? 'AVAILABLE' : 'UNAVAILABLE' 
    },
    priority: { 
      score: ml ? parseFloat(ml.priority_score) : null, 
      level: ml ? ml.priority_level : null, 
      recommended_verification_priority: ml ? (ml.priority_level ? ml.priority_level.replace(' Priority', '').replace(' PRIORITY', '').toUpperCase() : null) : null, 
      reason: null 
    },
    reliability: {
      score: ml ? parseFloat(ml.reliability_component) : null,
      data_completeness: ml ? parseFloat(ml.data_completeness) : null,
      data_freshness_days: ml ? parseFloat(ml.data_freshness_days) : null,
      source_consistency: ml ? parseFloat(ml.source_consistency) : null,
      evidence_strength: null
    },
    environment: {
      rainfall_mm: null,
      rainfall_3d: null,
      rainfall_7d: null,
      rainfall_14d: null,
      flood_occurred: null,
      flood_duration_days: null,
      flood_severity: null
    },
    water_quality: {
      ph: null,
      turbidity: null,
      tds: null,
      fecal_coliform: null,
      e_coli: null,
      bod: null,
      iron: null,
      arsenic: null,
      sulphate: null,
      current_measurements_status: 'UNAVAILABLE'
    },
    health: {
      disease_indicator: null,
      recent_cases: null,
      historical_cases: null,
      case_change: null,
      trend: null
    },
    data_quality: {
      data_completeness: null,
      data_freshness_days: null,
      source_consistency: null,
      evidence_strength: null,
      status: 'UNAVAILABLE'
    },
    top_risk_drivers: (ml && ml.top_positive_contributors) ? ml.top_positive_contributors.split('|').map(s => s.trim()) : [],
    top_negative_drivers: (ml && ml.top_negative_contributors) ? ml.top_negative_contributors.split('|').map(s => s.trim()) : [],
    explanation_status: (ml && ml.top_positive_contributors) ? 'AVAILABLE' : 'UNAVAILABLE',
    recommended_action: null,
    verification: {
      verification_status: verification.verification_status,
      recommended_verification_priority: verification.recommended_verification_priority ?? null,
      reason_for_priority: verification.reason_for_priority ?? null,
      verification_date: verification.verification_date ?? null,
      actual_field_result: verification.actual_field_result ?? null,
      records: verificationRecords
    },
    published_observation: sample
  };
}

function historyFor(location) {
  const sample = location.published_observation;
  const locationId = location.location_id;
  const trends = diseaseTrends[locationId] || [];

  const baseHistory = sample ? [{
    date: null,
    period: sample.period,
    record_type: 'PUBLISHED_WATER_OBSERVATION',
    risk_probability: null,
    risk_level: null,
    rainfall: null,
    rainfall_7d: null,
    flood_occurred: null,
    water_quality: sample.readings,
    water_quality_summary: sample.summary,
    disease_observations: null,
    source_label: sample.source_label,
    source_url: seed.sources[sample.source_id] ?? null
  }] : [];
  
  const mappedTrends = trends.map(t => ({
    date: t.date,
    record_type: 'SYNTHETIC_ML_HISTORY',
    disease_cases: t.disease_cases,
    rainfall_mm: t.rainfall_mm,
    flood_occurred: t.flood_occurred,
    e_coli: t.e_coli,
    bod: t.bod
  }));
  
  return [...mappedTrends, ...baseHistory];
}

async function readRequestJson(request) {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of request) {
    bytes += chunk.length;
    if (bytes > 1_000_000) throw Object.assign(new Error('Request body exceeds 1 MB'), { status: 413 });
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw Object.assign(new Error('Body must be valid JSON'), { status: 400 });
  }
}

async function readRequestText(request, maximumBytes = 10_000_000) {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of request) {
    bytes += chunk.length;
    if (bytes > maximumBytes) throw Object.assign(new Error(`Request body exceeds ${Math.round(maximumBytes / 1000)} KB`), { status: 413 });
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function validateVerification(payload) {
  const statuses = new Set(['NOT_STARTED', 'PLANNED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED']);
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return 'Body must be a JSON object';
  if (!statuses.has(payload.verification_status)) return 'verification_status must be NOT_STARTED, PLANNED, IN_PROGRESS, COMPLETED, or CANCELLED';
  if (payload.verification_date !== undefined && payload.verification_date !== null) {
    if (typeof payload.verification_date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(payload.verification_date) || Number.isNaN(Date.parse(payload.verification_date))) {
      return 'verification_date must be an ISO date (YYYY-MM-DD) or null';
    }
  }
  if (payload.reason_for_priority !== undefined && payload.reason_for_priority !== null && typeof payload.reason_for_priority !== 'string') {
    return 'reason_for_priority must be a string or null';
  }
  if (payload.recommended_verification_priority !== undefined && payload.recommended_verification_priority !== null && !['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'].includes(payload.recommended_verification_priority)) {
    return 'recommended_verification_priority must be LOW, MEDIUM, HIGH, VERY_HIGH, or null';
  }
  if (payload.actual_field_result !== undefined && payload.actual_field_result !== null && (typeof payload.actual_field_result !== 'object' || Array.isArray(payload.actual_field_result))) {
    return 'actual_field_result must be a JSON object or null';
  }
  return null;
}

async function handle(request, response) {
  for (const [key, value] of Object.entries(corsHeaders)) response.setHeader(key, value);
  if (request.method === 'OPTIONS') {
    response.writeHead(204);
    response.end();
    return;
  }

  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
  const path = url.pathname.replace(/\/+$/, '') || '/';

  if (request.method === 'GET' && path === '/health') {
    const historicalCount = seed.locations.filter(l => !l.location_id.startsWith('LOC_')).length;
    const mlCount = seed.locations.filter(l => l.location_id.startsWith('LOC_')).length;
    sendJson(response, 200, {
      status: 'ok',
      region: seed.region,
      locations_loaded: seed.locations.length,
      historical_evidence_locations: historicalCount,
      ml_assessment_zones: mlCount,
      model_status: 'SNAPSHOT_ONLY',
      read_only: READ_ONLY
    });
    return;
  }

  if (request.method === 'GET' && (path === '/' || path === '/dashboard')) {
    try {
      sendHtml(response, 200, await readFile(dashboardFile, 'utf8'));
    } catch (error) {
      console.error(error);
      sendError(response, 500, 'DASHBOARD_NOT_FOUND', 'Could not load the dashboard HTML file');
    }
    return;
  }

  if (request.method === 'GET' && (path === '/model-studio' || path === '/model-lab')) {
    try {
      sendHtml(response, 200, await readFile(modelStudioFile, 'utf8'));
    } catch (error) {
      console.error(error);
      sendError(response, 500, 'MODEL_STUDIO_NOT_FOUND', 'Could not load the model studio page');
    }
    return;
  }

  if (request.method === 'GET' && path === '/about') {
    try {
      sendHtml(response, 200, await readFile(aboutFile, 'utf8'));
    } catch (error) {
      console.error(error);
      sendError(response, 500, 'ABOUT_PAGE_NOT_FOUND', 'Could not load the NeerWatch project overview');
    }
    return;
  }

  if (request.method === 'GET' && path === '/synthetic-demo') {
    try {
      sendHtml(response, 200, await readFile(syntheticDemoFile, 'utf8'));
    } catch (error) {
      console.error(error);
      sendError(response, 500, 'SYNTHETIC_DEMO_NOT_FOUND', 'Could not load the synthetic model demo page');
    }
    return;
  }

  if (request.method === 'GET' && path === '/synthetic-demo.csv') {
    try {
      const csv = await readFile(syntheticDemoCsvFile, 'utf8');
      response.writeHead(200, {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': 'attachment; filename="neerwatch-synthetic-demo.csv"',
        'Content-Length': Buffer.byteLength(csv)
      });
      response.end(csv);
    } catch (error) {
      console.error(error);
      sendError(response, 500, 'SYNTHETIC_CSV_NOT_FOUND', 'Could not load the synthetic demo CSV');
    }
    return;
  }

  if (request.method === 'GET' && path === '/api/v1/locations') {
    const state = url.searchParams.get('state');
    const locations = selectedLocations(state);
    if (!locations) return sendError(response, 400, 'INVALID_STATE', 'state must be one of the eight supported Northeast states');
    sendJson(response, 200, { region: seed.region, states: seed.states, total: locations.length, locations: locations.map(locationFields) });
    return;
  }

  if (request.method === 'GET' && path === '/api/v1/risk') {
    const state = url.searchParams.get('state');
    const locations = selectedLocations(state);
    if (!locations) return sendError(response, 400, 'INVALID_STATE', 'state must be one of the eight supported Northeast states');
    sendJson(response, 200, {
      generated_at: null,
      region: seed.region,
      model_status: 'SNAPSHOT_ONLY',
      message: 'Prototype ML risk outputs are generated using synthetic demonstration data. Historical water observations are returned as evidence.',
      total: locations.length,
      locations: locations.map(mapRisk)
    });
    return;
  }

  if (request.method === 'GET' && path === '/api/v1/model/check') {
    sendJson(response, 200, checkModelSnapshot());
    return;
  }

  if (request.method === 'POST' && path === '/api/v1/model/demo-score') {
    let csvText;
    try {
      csvText = await readRequestText(request, 100_000);
    } catch (error) {
      return sendError(response, error.status || 400, 'CSV_UPLOAD_FAILED', error.message);
    }
    const rows = parseCSV(csvText);
    const errors = validateDemoRows(rows);
    if (errors.length) {
      sendJson(response, 400, { status: 'VALIDATION_FAILED', errors });
      return;
    }
    const predictions = rows.map(row => predictSyntheticDemo(row, demoModel));
    sendJson(response, 200, {
      status: 'DEMO_INFERENCE_COMPLETE',
      model: {
        name: demoModel.model_name,
        version: demoModel.model_version,
        type: demoModel.model_type,
        training_examples: demoModel.training_examples,
        training_data: demoModel.training_data,
        synthetic_holdout_accuracy: demoModel.synthetic_holdout_accuracy
      },
      rows_scored: predictions.length,
      predictions,
      warning: 'These are new inferences from a model trained only on generated synthetic scenarios. They are not validated water-safety predictions and must not guide real-world decisions.'
    });
    return;
  }

  const riskDetailMatch = path.match(/^\/api\/v1\/risk\/([^/]+)$/);
  if (request.method === 'GET' && riskDetailMatch) {
    const locationId = decodeURIComponent(riskDetailMatch[1]);
    const location = byId.get(locationId);
    if (!location) return sendError(response, 404, 'LOCATION_NOT_FOUND', `No monitored location found for ${locationId}`);
    const verifications = await loadVerifications();
    sendJson(response, 200, detailFor(location, verifications[locationId] ?? []));
    return;
  }

  const historyMatch = path.match(/^\/api\/v1\/locations\/([^/]+)\/history$/);
  if (request.method === 'GET' && historyMatch) {
    const locationId = decodeURIComponent(historyMatch[1]);
    const location = byId.get(locationId);
    if (!location) return sendError(response, 404, 'LOCATION_NOT_FOUND', `No monitored location found for ${locationId}`);
    sendJson(response, 200, { location_id: locationId, history: historyFor(location) });
    return;
  }

  const verificationMatch = path.match(/^\/api\/v1\/locations\/([^/]+)\/verification$/);
  if (request.method === 'POST' && verificationMatch) {
    if (READ_ONLY) return sendError(response, 403, 'READ_ONLY_DEPLOYMENT', 'Field verification submissions are disabled in this read-only deployment.');
    const locationId = decodeURIComponent(verificationMatch[1]);
    if (!byId.has(locationId)) return sendError(response, 404, 'LOCATION_NOT_FOUND', `No monitored location found for ${locationId}`);
    let payload;
    try {
      payload = await readRequestJson(request);
    } catch (error) {
      return sendError(response, error.status || 400, 'INVALID_REQUEST_BODY', error.message);
    }
    const validationError = validateVerification(payload);
    if (validationError) return sendError(response, 400, 'INVALID_VERIFICATION', validationError);
    const verifications = await loadVerifications();
    const record = {
      verification_id: randomUUID(),
      location_id: locationId,
      verification_status: payload.verification_status,
      recommended_verification_priority: payload.recommended_verification_priority ?? null,
      reason_for_priority: payload.reason_for_priority ?? null,
      verification_date: payload.verification_date ?? null,
      actual_field_result: payload.actual_field_result ?? null,
      created_at: new Date().toISOString()
    };
    verifications[locationId] ??= [];
    verifications[locationId].push(record);
    await writeFile(verificationFile, `${JSON.stringify(verifications, null, 2)}\n`, 'utf8');
    sendJson(response, 201, record);
    return;
  }

  if (request.method === 'GET' && path === '/api/v1') {
    sendJson(response, 200, {
      name: 'Northeast India Water-Borne Disease Early Warning API',
      version: '1.0.0',
      model_status: 'SNAPSHOT_ONLY',
      read_only: READ_ONLY,
      demo_model: {
        name: demoModel.model_name,
        version: demoModel.model_version,
        type: demoModel.model_type,
        inference_available: true,
        training_examples: demoModel.training_examples
      },
      endpoints: [
        'GET /api/v1/locations',
        'GET /api/v1/risk',
        'GET /api/v1/model/check',
        'POST /api/v1/model/demo-score',
        'GET /api/v1/risk/{location_id}',
        'GET /api/v1/locations/{location_id}/history',
        'POST /api/v1/locations/{location_id}/verification'
      ]
    });
    return;
  }

  if (request.method === 'POST' && path === '/api/v1/dataset/upload') {
    if (READ_ONLY) return sendError(response, 403, 'READ_ONLY_DEPLOYMENT', 'Dataset uploads are disabled in this read-only deployment.');
    try {
      uploadedCsvText = await readRequestText(request);
      analysisState = { status: 'UPLOAD_RECEIVED', errors: [] };
      sendJson(response, 202, { message: 'Dataset received', status: analysisState.status });
    } catch (e) {
      sendError(response, 400, 'UPLOAD_FAILED', e.message);
    }
    return;
  }

  if (request.method === 'POST' && path === '/api/v1/dataset/validate') {
    let csvText;
    try {
      csvText = await readRequestText(request, 100_000);
    } catch (error) {
      return sendError(response, error.status || 400, 'VALIDATION_FAILED', error.message);
    }
    if (!csvText.trim()) csvText = uploadedCsvText || '';
    analysisState.status = 'VALIDATING';
    analysisState.errors = [];
    const rows = parseCSV(csvText);
    analysisState.errors = validateDemoRows(rows);
    if (analysisState.errors.length > 0) {
      analysisState.status = 'VALIDATION_FAILED';
      sendJson(response, 400, { message: 'Validation failed', status: analysisState.status, errors: analysisState.errors });
    } else {
      analysisState.status = 'READY_FOR_ANALYSIS';
      sendJson(response, 200, { message: 'Validation successful', status: analysisState.status, rows_detected: rows.length });
    }
    return;
  }

  if (request.method === 'POST' && path === '/api/v1/dataset/analyze') {
    if (analysisState.status !== 'READY_FOR_ANALYSIS') {
      return sendError(response, 400, 'INVALID_STATE', 'Dataset must be validated before analysis');
    }
    analysisState.status = 'INFERENCE_UNAVAILABLE';
    analysisState.errors = ['CSV schema validation passed, but no executable XGBoost model weights are bundled; no predictions were generated.'];
    sendJson(response, 409, {
      message: 'New inference is unavailable in this build.',
      status: analysisState.status,
      errors: analysisState.errors,
      model_check: checkModelSnapshot()
    });
    return;
  }

  if (request.method === 'GET' && path === '/api/v1/dataset/status') {
    sendJson(response, 200, analysisState);
    return;
  }

  sendError(response, 404, 'ROUTE_NOT_FOUND', 'No API route matches this request');
}

const server = createServer((request, response) => {
  handle(request, response).catch(error => {
    console.error(error);
    if (!response.headersSent) sendError(response, 500, 'INTERNAL_ERROR', 'Unexpected server error');
    else response.destroy();
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`NeerWatch API listening on http://localhost:${PORT}`);
  console.log(`Loaded ${seed.locations.length} published sample locations; ML risk outputs use August 2024 prototype snapshot — synthetic demonstration data.`);
});

const FEATURES = [
  { key: 'ph_stress', label: 'pH distance from neutral' },
  { key: 'e_coli', label: 'E. coli' },
  { key: 'rainfall_mm', label: 'Rainfall' },
  { key: 'historical_cases', label: 'Historical cases' },
  { key: 'turbidity_ntu', label: 'Turbidity' }
];

export function transformDemoInputs(row) {
  return {
    ph_stress: Math.min(1, Math.abs(Number(row.ph) - 7) / 3.5),
    e_coli: Math.log1p(Number(row.e_coli)) / Math.log1p(1000),
    rainfall_mm: Math.log1p(Number(row.rainfall_mm)) / Math.log1p(500),
    historical_cases: Math.log1p(Number(row.historical_cases)) / Math.log1p(200),
    turbidity_ntu: Math.log1p(Number(row.turbidity_ntu)) / Math.log1p(50)
  };
}

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

export function predictSyntheticDemo(row, model) {
  const transformed = transformDemoInputs(row);
  const standardized = {};
  for (const feature of FEATURES) {
    const key = feature.key;
    standardized[key] = (transformed[key] - model.means[key]) / model.scales[key];
  }

  const contributions = FEATURES.map(feature => ({
    feature: feature.label,
    value: standardized[feature.key] * model.weights[feature.key]
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const logOdds = model.intercept + contributions.reduce((sum, item) => sum + item.value, 0);
  const probability = sigmoid(logOdds);
  const reliability = (Number(row.source_consistency) + Number(row.sensor_reliability)) / 2;
  const priority = Math.min(1, Math.max(0, probability * 0.7 + (1 - reliability) * 0.3));
  const riskLevel = probability < 0.2 ? 'LOW' : probability < 0.6 ? 'MODERATE' : 'HIGH';

  return {
    location_id: row.location_id,
    state: row.state,
    scenario: row.scenario || null,
    probability: Number(probability.toFixed(4)),
    risk_level: riskLevel,
    priority_score: Number(priority.toFixed(4)),
    data_reliability: Number(reliability.toFixed(4)),
    top_contributors: contributions.slice(0, 3).map(item => ({
      feature: item.feature,
      log_odds_contribution: Number(item.value.toFixed(3))
    }))
  };
}

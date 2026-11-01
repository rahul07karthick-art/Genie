import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { transformDemoInputs } from '../backend/synthetic_demo_model.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const outputPath = resolve(here, '../ml-artifacts/synthetic_demo_model.json');
const trainingCount = 8000;
const holdoutCount = 2000;
let seed = 20261031;

function random() {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296;
}

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

function scenario() {
  const row = {
    ph: 4.5 + random() * 4.5,
    e_coli: Math.expm1(random() * Math.log1p(1000)),
    rainfall_mm: Math.expm1(random() * Math.log1p(500)),
    historical_cases: Math.expm1(random() * Math.log1p(200)),
    turbidity_ntu: Math.expm1(random() * Math.log1p(50))
  };
  const x = transformDemoInputs(row);
  const latentHazard = 0.12 * x.ph_stress
    + 0.24 * x.e_coli
    + 0.18 * x.rainfall_mm
    + 0.26 * x.historical_cases
    + 0.20 * x.turbidity_ntu;
  const labelProbability = sigmoid(-5.2 + 9 * latentHazard);
  return { x, y: random() < labelProbability ? 1 : 0 };
}

const training = Array.from({ length: trainingCount }, scenario);
const holdout = Array.from({ length: holdoutCount }, scenario);
const featureKeys = ['ph_stress', 'e_coli', 'rainfall_mm', 'historical_cases', 'turbidity_ntu'];
const means = Object.fromEntries(featureKeys.map(key => [key, training.reduce((sum, row) => sum + row.x[key], 0) / trainingCount]));
const scales = Object.fromEntries(featureKeys.map(key => {
  const variance = training.reduce((sum, row) => sum + (row.x[key] - means[key]) ** 2, 0) / trainingCount;
  return [key, Math.sqrt(variance) || 1];
}));
const weights = Object.fromEntries(featureKeys.map(key => [key, 0]));
let intercept = 0;

const epochs = 900;
const learningRate = 0.3;
const regularization = 0.002;
for (let epoch = 0; epoch < epochs; epoch++) {
  const gradients = Object.fromEntries(featureKeys.map(key => [key, 0]));
  let interceptGradient = 0;
  for (const row of training) {
    let logOdds = intercept;
    for (const key of featureKeys) logOdds += weights[key] * ((row.x[key] - means[key]) / scales[key]);
    const error = sigmoid(logOdds) - row.y;
    interceptGradient += error;
    for (const key of featureKeys) gradients[key] += error * ((row.x[key] - means[key]) / scales[key]);
  }
  intercept -= learningRate * interceptGradient / trainingCount;
  for (const key of featureKeys) {
    weights[key] -= learningRate * (gradients[key] / trainingCount + regularization * weights[key]);
  }
}

let correct = 0;
for (const row of holdout) {
  let logOdds = intercept;
  for (const key of featureKeys) logOdds += weights[key] * ((row.x[key] - means[key]) / scales[key]);
  if ((sigmoid(logOdds) >= 0.5 ? 1 : 0) === row.y) correct++;
}

const artifact = {
  model_name: 'NeerWatch synthetic logistic demo',
  model_version: '0.1.0',
  model_type: 'logistic_regression',
  training_data: 'Deterministically generated synthetic scenarios; labels come from a simulated risk formula, not real water-health outcomes.',
  feature_keys: featureKeys,
  means,
  scales,
  weights,
  intercept,
  training_examples: trainingCount,
  synthetic_holdout_examples: holdoutCount,
  synthetic_holdout_accuracy: Number((correct / holdoutCount).toFixed(4)),
  generator_seed: 20261031,
  warning: 'Demonstration model only. Not trained or validated for real-world water-safety decisions.'
};

await writeFile(outputPath, `${JSON.stringify(artifact, null, 2)}\n`, 'utf8');
console.log(`Trained ${artifact.model_name} on ${trainingCount} generated synthetic scenarios.`);
console.log(`Synthetic-only holdout accuracy: ${(artifact.synthetic_holdout_accuracy * 100).toFixed(1)}% (not real-world validation).`);
console.log(`Saved weights: ${outputPath}`);

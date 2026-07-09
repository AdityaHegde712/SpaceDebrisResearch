# Space Debris Research: Parallel Task Plan

## Direction

The project will move away from the current leakage-prone binary classification setup and toward a regression-first, hybrid physics-ML trajectory correction framework.

The main modeling question is no longer:

> Can we classify whether an object is high risk using features that already encode the risk threshold?

The new modeling question is:

> Can we predict future debris position error, uncertainty growth, and risk-relevant quantities using physically valid inputs available before the prediction time?

This keeps the research aligned with the final goal: producing corrected debris probability clouds that can be used inside a Dynamic Probabilistic Occupancy Map (DPOM) and launch-window risk simulator.

## Shift in Objective

### Previous Objective

Build a classifier for a synthetic `collision_risk` label derived from `log_risk_probability` or related uncertainty-volume variables.

### Problem With Previous Objective

The Random Forest classifier collapsed onto the target-defining variable because the risk label was created directly from the same information available in the input features. This produced a near-perfect score, but it did not demonstrate learned orbital dynamics or useful predictive capability.

The TCN sequence classifier also did not solve the core problem because the target was mostly a static thresholded quantity rather than a genuinely temporal prediction target. Global average pooling diluted the final-timestep signal across the sequence.

### New Objective

Use regression as the primary direction:

1. Predict future `log_risk_probability`, `log_uncertainty_volume`, or RTN uncertainty growth as continuous values.
2. Predict physics-model residuals such as:

   ```text
   delta_r = r_true - r_physics
   ```

3. Add the learned residual correction back to the RK4/J2 propagated state.
4. Feed the corrected state and predicted covariance values into the DPOM / launch-risk simulator.

Classification can still be used later as a downstream decision layer, but it should be trained only after removing leakage-prone variables and should preferably classify predicted future risk rather than current known risk.

## Reasoning

Regression preserves more information than binary labels. A continuous risk or uncertainty target allows the model to learn the magnitude of uncertainty growth instead of only learning whether a threshold was crossed.

The hybrid physics-ML setup is also more physically meaningful than direct coordinate prediction. The RK4/J2 propagator provides a nominal orbit, while the ML model learns the missing correction caused by drag, atmospheric density variation, solar activity, and other non-conservative effects.

This makes the model output more useful for the final launch-routing objective because the DPOM needs corrected probability clouds, not only a binary high-risk / low-risk label.

## Parallel Work Allocation

### Person A: Mir — Target Definition and Leakage Audit

**Main responsibility:** Build a clean, leakage-safe dataset definition.

Tasks:

- Identify all variables directly or indirectly derived from `log_risk_probability`, `log_uncertainty_volume`, covariance volume, or the classification threshold.
- Separate features into:
  - allowed prediction-time inputs
  - future-only labels
  - leakage-risk variables
  - diagnostic-only variables
- Define clean regression targets:
  - future `log_risk_probability`
  - future `log_uncertainty_volume`
  - future RTN sigmas: `sigma_R`, `sigma_T`, `sigma_W`
  - optional covariance growth deltas
- Define a leakage-safe classification variant with the target-defining variables removed from the input.
- Verify that train/validation/test splits are temporal and do not leak future information.

Expected output:

- A leakage-audited dataset schema.
- A cleaned feature list for regression.
- A cleaned feature list for classification.
- A documented target-generation script or notebook.

### Person B: Kshitij — Regression Baselines

**Main responsibility:** Build non-deep-learning and simple ML regression baselines.

Tasks:

- Train baseline regressors for future risk and uncertainty targets.
- Start with simple models before sequence models:
  - Linear Regression / Ridge Regression
  - Random Forest Regressor
  - XGBoost / LightGBM Regressor, if available
  - MLP Regressor, if useful
- Evaluate against naive baselines such as:
  - predict last observed value
  - predict altitude-band median
  - predict object-size-band median
- Report regression metrics:
  - MAE
  - RMSE
  - R²
  - calibration-style plots for predicted vs actual risk
  - error by size band and altitude band
- Compare whether snapshot features are sufficient or whether sequence history is needed.

Expected output:

- A baseline risk regressor for future `log_risk_probability` or `log_uncertainty_volume`.
- A baseline uncertainty-growth regressor for future RTN sigmas or covariance growth.
- A metrics report comparing simple baselines against ML regressors.

### Person C: Sanjeev — Hybrid Physics Residual Regressor

**Main responsibility:** Build the physics-ML correction model.

Tasks:

- Use RK4/J2 propagation outputs as the nominal physics baseline.
- Construct residual targets:

  ```text
  delta_x = x_true - x_physics
  delta_y = y_true - y_physics
  delta_z = z_true - z_physics
  ```

  or preferably in RTN coordinates:

  ```text
  delta_R, delta_T, delta_N
  ```

- Train a model to predict the residual correction using historical orbital state, object properties, and environmental features.
- Start with simple residual regressors before advanced sequence models.
- Evaluate whether corrected position improves over raw RK4/J2 propagation at 1-day, 3-day, and 7-day horizons.
- Add covariance-growth prediction if time permits.

Expected output:

- A hybrid residual regressor that predicts RK4/J2 position error.
- A corrected trajectory output:

  ```text
  r_corrected = r_physics + delta_r_ML
  ```

- An evaluation table comparing RK4/J2 error vs corrected error at multiple horizons.

### Aditya — Integration, DPOM Interface, and Evaluation Design

**Main responsibility:** Keep the research direction coherent and connect model outputs to the final launch-risk objective.

Tasks:

- Define the final evaluation interface between trajectory models and the DPOM / launch-risk simulator.
- Specify the model output format required by the downstream simulator:
  - corrected position
  - corrected velocity, if available
  - predicted RTN sigmas
  - timestamp
  - object ID
  - object size band
- Define the end-to-end evaluation question:

  ```text
  Does the corrected debris probability model change the estimated launch-path risk over candidate launch offsets?
  ```

- Coordinate merging outputs from Mir, Kshitij, and Sanjeev.
- Maintain the final research narrative:
  - why classification failed
  - why regression is better
  - why hybrid physics-ML is the main direction
  - how corrected uncertainty clouds feed the DPOM

Expected output:

- A DPOM-ready corrected trajectory and uncertainty interface.
- An integration notebook or script that accepts model outputs and computes launch-path risk inputs.
- A short evaluation report linking model performance to launch-window risk scoring.

## Expected Built Artifacts

By the end of this phase, the team should have built:

1. A leakage-audited feature and target schema.
2. A future-risk regression dataset.
3. A baseline `log_risk_probability` or `log_uncertainty_volume` regressor.
4. A baseline RTN uncertainty-growth regressor.
5. A hybrid RK4/J2 residual correction regressor.
6. A corrected trajectory output format.
7. A DPOM-ready interface for launch-path risk scoring.
8. A comparison report showing:
   - raw RK4/J2 propagation error
   - ML residual-corrected propagation error
   - baseline risk-regression performance
   - leakage-safe classification performance, if pursued

## Recommended Priority Order

1. Mir completes the leakage audit and clean target definitions.
2. Kshitij trains baseline regressors on the clean dataset.
3. Sanjeev trains residual correction models using the RK4/J2 baseline outputs.
4. Aditya integrates the best available outputs into the DPOM-facing evaluation pipeline.

Classification should be treated as secondary. It is useful only if it is leakage-safe and framed as a downstream decision layer over predicted future risk.

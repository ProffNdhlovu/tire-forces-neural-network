# Tire Forces Neural Network Model

**Author:** Sinanzwayinkosi  
**Field:** Aerospace Engineering | Autonomous Vehicles | Machine Learning

## Abstract

This project implements a neural network model for predicting tire forces in vehicle dynamics applications. The model uses physics-based synthetic data generation and deep learning to predict lateral and longitudinal tire forces based on vehicle operating conditions.

## Project Overview

Tire force prediction is critical for autonomous vehicle control systems and vehicle dynamics simulation. This project develops a machine learning approach to estimate tire forces using measurable vehicle parameters, providing a foundation for advanced control algorithms.

## Technical Approach

### Input Parameters
- Slip angle (degrees)
- Slip ratio (dimensionless)
- Normal force (N)
- Vehicle velocity (m/s)
- Tire pressure (kPa)
- Road friction coefficient

### Output Predictions
- Lateral tire force (N)
- Longitudinal tire force (N)

## Model Architecture

The neural network consists of:
- Input layer: 6 features
- Hidden layers: 128, 64, 32, 16 neurons with ReLU activation
- Dropout layers: 0.2 rate for regularization
- Output layer: 2 neurons with linear activation

## Data Generation

Synthetic data is generated using simplified Pacejka tire model equations:
- Lateral force based on cornering stiffness and slip angle
- Longitudinal force based on longitudinal stiffness and slip ratio
- Environmental effects including road friction and tire pressure
- Realistic noise addition for model robustness

## Installation and Usage

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
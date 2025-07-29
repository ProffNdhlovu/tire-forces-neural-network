#!/usr/bin/env python3
"""
Tire Forces Neural Network Model
Author: Sinanzwayinkosi
Description: Neural network model for predicting tire forces in vehicle dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

class TireForceModel:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic tire force data based on vehicle dynamics principles
        """
        np.random.seed(42)
        
        # Input parameters
        slip_angle = np.random.uniform(-15, 15, n_samples)  # degrees
        slip_ratio = np.random.uniform(-0.3, 0.3, n_samples)  # dimensionless
        normal_force = np.random.uniform(2000, 8000, n_samples)  # N
        velocity = np.random.uniform(5, 50, n_samples)  # m/s
        tire_pressure = np.random.uniform(180, 250, n_samples)  # kPa
        road_friction = np.random.uniform(0.3, 1.2, n_samples)  # coefficient
        
        # Physics-based tire force calculations (simplified Pacejka model)
        # Lateral force
        C_alpha = 80000  # cornering stiffness
        lateral_force = -C_alpha * np.tan(np.radians(slip_angle)) * normal_force / 4000
        lateral_force *= road_friction * (tire_pressure / 220)  # pressure effect
        
        # Longitudinal force
        C_sigma = 120000  # longitudinal stiffness
        longitudinal_force = C_sigma * slip_ratio * normal_force / 4000
        longitudinal_force *= road_friction * (tire_pressure / 220)
        
        # Add realistic noise
        lateral_force += np.random.normal(0, 100, n_samples)
        longitudinal_force += np.random.normal(0, 80, n_samples)
        
        # Create dataset
        X = np.column_stack([
            slip_angle, slip_ratio, normal_force, 
            velocity, tire_pressure, road_friction
        ])
        
        y = np.column_stack([lateral_force, longitudinal_force])
        
        # Create DataFrame for better handling
        feature_names = ['slip_angle', 'slip_ratio', 'normal_force', 
                        'velocity', 'tire_pressure', 'road_friction']
        target_names = ['lateral_force', 'longitudinal_force']
        
        df_X = pd.DataFrame(X, columns=feature_names)
        df_y = pd.DataFrame(y, columns=target_names)
        
        return df_X, df_y
    
    def build_model(self, input_dim, output_dim):
        """
        Build neural network architecture for tire force prediction
        """
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X, y, test_size=0.2, epochs=100, batch_size=32):
        """
        Train the neural network model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Build model
        self.model = self.build_model(X_train.shape[1], y_train.shape[1])
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return history, X_test, y_test, y_pred
    
    def predict(self, X):
        """
        Predict tire forces for given input parameters
        """
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled)
    
    def plot_results(self, history, y_test, y_pred):
        """
        Visualize training results and predictions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training history
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Lateral force prediction
        axes[0, 1].scatter(y_test.iloc[:, 0], y_pred[:, 0], alpha=0.5)
        axes[0, 1].plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 
                       [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 'r--')
        axes[0, 1].set_title('Lateral Force Prediction')
        axes[0, 1].set_xlabel('Actual Force (N)')
        axes[0, 1].set_ylabel('Predicted Force (N)')
        
        # Longitudinal force prediction
        axes[1, 0].scatter(y_test.iloc[:, 1], y_pred[:, 1], alpha=0.5)
        axes[1, 0].plot([y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], 
                       [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], 'r--')
        axes[1, 0].set_title('Longitudinal Force Prediction')
        axes[1, 0].set_xlabel('Actual Force (N)')
        axes[1, 0].set_ylabel('Predicted Force (N)')
        
        # Residuals
        residuals_lat = y_test.iloc[:, 0] - y_pred[:, 0]
        residuals_lon = y_test.iloc[:, 1] - y_pred[:, 1]
        axes[1, 1].scatter(y_pred[:, 0], residuals_lat, alpha=0.5, label='Lateral')
        axes[1, 1].scatter(y_pred[:, 1], residuals_lon, alpha=0.5, label='Longitudinal')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residuals Plot')
        axes[1, 1].set_xlabel('Predicted Force (N)')
        axes[1, 1].set_ylabel('Residuals (N)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('tire_force_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("Tire Forces Neural Network Model")
    print("================================")
    
    # Initialize model
    tire_model = TireForceModel()
    
    # Generate synthetic data
    print("\nGenerating synthetic tire force data...")
    X, y = tire_model.generate_synthetic_data(n_samples=10000)
    
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    print(f"Targets: {list(y.columns)}")
    
    # Train model
    print("\nTraining neural network...")
    history, X_test, y_test, y_pred = tire_model.train(X, y, epochs=50)
    
    # Plot results
    print("\nGenerating visualizations...")
    tire_model.plot_results(history, y_test, y_pred)
    
    # Save model
    tire_model.model.save('tire_force_model.h5')
    print("\nModel saved as 'tire_force_model.h5'")
    
    # Example prediction
    print("\nExample prediction:")
    example_input = pd.DataFrame({
        'slip_angle': [5.0],
        'slip_ratio': [0.1],
        'normal_force': [4000],
        'velocity': [25.0],
        'tire_pressure': [220],
        'road_friction': [0.8]
    })
    
    prediction = tire_model.predict(example_input)
    print(f"Input: Slip angle=5°, Slip ratio=0.1, Normal force=4000N")
    print(f"Predicted forces: Lateral={prediction[0][0]:.1f}N, Longitudinal={prediction[0][1]:.1f}N")
    
    print("\nProject complete. Ready for analysis.")

if __name__ == "__main__":
    main()
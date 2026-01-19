import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class AGBModelDashboard:
    def __init__(self, data_path='FEI data/opt_means_cleaned.csv'):
        """Initialize the dashboard with data loading and preprocessing"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.training_times = {}
        
    def load_and_preprocess_data(self):
        """Load data and perform preprocessing"""
        print("🔄 Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        
        # Detect target column (prefer AGB_2024, fallback to AGB_2017)
        agb_col = 'AGB_2024' if 'AGB_2024' in self.data.columns else 'AGB_2017'
        print(f"✅ Using target column: {agb_col}")
        
        # Features and target
        X = self.data.drop(columns=[agb_col])
        y = self.data[agb_col]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data split: Train {self.X_train.shape[0]}, Test {self.X_test.shape[0]}")
        print(f"✅ Features scaled and ready for training")
        
    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        print("\n🌲 Training Random Forest...")
        start_time = time.time()
        
        # Hyperparameter space
        rf_params = {
            'n_estimators': [100, 200, 500, 1000],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        # Random search
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            n_iter=50,  # Reduced for faster training
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        rf_search.fit(self.X_train_scaled, self.y_train)
        best_rf = rf_search.best_estimator_
        
        # Predictions and metrics
        y_pred = best_rf.predict(self.X_test_scaled)
        rmse = mean_squared_error(self.y_test, y_pred) ** 0.5
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        # Store results
        self.models['Random Forest'] = best_rf
        self.results['Random Forest'] = {
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae,
            'Best Params': rf_search.best_params_
        }
        self.training_times['Random Forest'] = time.time() - start_time
        
        print(f"✅ Random Forest trained in {self.training_times['Random Forest']:.2f}s")
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
    def train_lightgbm(self):
        """Train LightGBM with hyperparameter tuning"""
        print("\n💡 Training LightGBM...")
        start_time = time.time()
        
        # Hyperparameter space
        lgbm_params = {
            'num_leaves': [31, 50, 70, 100],
            'max_depth': [5, 7, 9, 12, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 500],
            'min_data_in_leaf': [20, 30, 50],
            'lambda_l1': [0, 0.1, 0.5],
            'lambda_l2': [0, 0.1, 0.5]
        }
        
        # Random search
        lgbm_search = RandomizedSearchCV(
            LGBMRegressor(random_state=42, verbose=-1),
            lgbm_params,
            n_iter=30,  # Reduced for faster training
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        lgbm_search.fit(self.X_train_scaled, self.y_train)
        best_lgbm = lgbm_search.best_estimator_
        
        # Predictions and metrics
        y_pred = best_lgbm.predict(self.X_test_scaled)
        rmse = mean_squared_error(self.y_test, y_pred) ** 0.5
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        # Store results
        self.models['LightGBM'] = best_lgbm
        self.results['LightGBM'] = {
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae,
            'Best Params': lgbm_search.best_params_
        }
        self.training_times['LightGBM'] = time.time() - start_time
        
        print(f"✅ LightGBM trained in {self.training_times['LightGBM']:.2f}s")
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning"""
        print("\n⚡ Training XGBoost...")
        start_time = time.time()
        
        # Hyperparameter space
        xgb_params = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 500],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Random search
        xgb_search = RandomizedSearchCV(
            XGBRegressor(random_state=42, verbosity=0),
            xgb_params,
            n_iter=30,  # Reduced for faster training
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        xgb_search.fit(self.X_train_scaled, self.y_train)
        best_xgb = xgb_search.best_estimator_
        
        # Predictions and metrics
        y_pred = best_xgb.predict(self.X_test_scaled)
        rmse = mean_squared_error(self.y_test, y_pred) ** 0.5
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        # Store results
        self.models['XGBoost'] = best_xgb
        self.results['XGBoost'] = {
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae,
            'Best Params': xgb_search.best_params_
        }
        self.training_times['XGBoost'] = time.time() - start_time
        
        print(f"✅ XGBoost trained in {self.training_times['XGBoost']:.2f}s")
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
    def train_svr(self):
        """Train Support Vector Regression"""
        print("\n🔧 Training SVR...")
        start_time = time.time()
        
        # SVR with default parameters (faster training)
        svr_model = SVR()
        svr_model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions and metrics
        y_pred = svr_model.predict(self.X_test_scaled)
        rmse = mean_squared_error(self.y_test, y_pred) ** 0.5
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        # Store results
        self.models['SVR'] = svr_model
        self.results['SVR'] = {
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae,
            'Best Params': 'Default'
        }
        self.training_times['SVR'] = time.time() - start_time
        
        print(f"✅ SVR trained in {self.training_times['SVR']:.2f}s")
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
    def train_all_models(self):
        """Train all models sequentially"""
        print("🚀 Starting model training pipeline...")
        print("=" * 60)
        
        self.train_random_forest()
        self.train_lightgbm()
        self.train_xgboost()
        self.train_svr()
        
        print("\n" + "=" * 60)
        print("🎉 All models trained successfully!")
        
    def generate_performance_summary(self):
        """Generate comprehensive performance summary"""
        print("\n📊 Generating Performance Summary...")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'R²': metrics['R²'],
                'MAE': metrics['MAE'],
                'Training Time (s)': self.training_times[model_name]
            })
        
        self.summary_df = pd.DataFrame(summary_data)
        self.summary_df = self.summary_df.sort_values('RMSE')
        
        print("\n🏆 Model Performance Ranking (by RMSE):")
        print(self.summary_df.to_string(index=False))
        
        return self.summary_df
        
    def create_performance_plots(self):
        """Create comprehensive performance visualization plots"""
        print("\n📈 Creating performance visualizations...")
        
        # 1. Performance Comparison Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AGB Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # RMSE Comparison
        axes[0, 0].bar(self.summary_df['Model'], self.summary_df['RMSE'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 0].set_title('RMSE Comparison (Lower is Better)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² Comparison
        axes[0, 1].bar(self.summary_df['Model'], self.summary_df['R²'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 1].set_title('R² Comparison (Higher is Better)')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE Comparison
        axes[1, 0].bar(self.summary_df['Model'], self.summary_df['MAE'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 0].set_title('MAE Comparison (Lower is Better)')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training Time Comparison
        axes[1, 1].bar(self.summary_df['Model'], self.summary_df['Training Time (s)'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Radar Chart for Top Model
        best_model = self.summary_df.iloc[0]['Model']
        print(f"\n🎯 Best performing model: {best_model}")
        
        # Create radar chart
        metrics = ['RMSE', 'R²', 'MAE', 'Training Time (s)']
        values = []
        
        for metric in metrics:
            if metric == 'RMSE' or metric == 'MAE':
                # Normalize to 0-1 scale (lower is better)
                max_val = self.summary_df[metric].max()
                min_val = self.summary_df[metric].min()
                normalized_val = 1 - (self.summary_df.loc[self.summary_df['Model'] == best_model, metric].iloc[0] - min_val) / (max_val - min_val)
                values.append(normalized_val)
            else:
                # For R² and Training Time, use actual values
                val = self.summary_df.loc[self.summary_df['Model'] == best_model, metric].iloc[0]
                if metric == 'R²':
                    values.append(val)
                else:
                    # Normalize training time
                    max_time = self.summary_df['Training Time (s)'].max()
                    values.append(1 - val / max_time)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Close the loop
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=best_model)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Radar Chart - {best_model}', pad=20)
        ax.grid(True)
        plt.show()
        
    def save_models_and_results(self):
        """Save all trained models and results"""
        print("\n💾 Saving models and results...")
        
        # Save models
        for model_name, model in self.models.items():
            filename = f'AGB_{model_name.replace(" ", "_")}_Model.pkl'
            joblib.dump(model, filename)
            print(f"   ✅ {model_name} saved as {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, 'AGB_Scaler.pkl')
        print("   ✅ Scaler saved as AGB_Scaler.pkl")
        
        # Save results summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_df.to_csv(f'model_performance_summary_{timestamp}.csv', index=False)
        print(f"   ✅ Performance summary saved as model_performance_summary_{timestamp}.csv")
        
    def run_full_pipeline(self):
        """Run the complete training and analysis pipeline"""
        print("🚀 AGB Model Dashboard - Full Pipeline")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train all models
        self.train_all_models()
        
        # Generate summary
        self.generate_performance_summary()
        
        # Create visualizations
        self.create_performance_plots()
        
        # Save everything
        self.save_models_and_results()
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        
        return self.summary_df

if __name__ == "__main__":
    # Initialize and run dashboard
    dashboard = AGBModelDashboard()
    results = dashboard.run_full_pipeline()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# Statistical analysis
from scipy import stats
from scipy.stats import normaltest, shapiro
import scipy.stats as stats

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class AGBModelDiagnosticsDashboard:
    def __init__(self, data_path='FEI data/opt_means_cleaned.csv'):
        """Initialize the model diagnostics dashboard"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.predictions = {}
        self.diagnostics = {}
        
    def load_and_preprocess_data(self):
        """Load data and perform preprocessing"""
        print("🔄 Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        
        # Features and target
        # Detect target column (prefer AGB_2024, fallback to AGB_2017)
        agb_col = 'AGB_2024' if 'AGB_2024' in self.data.columns else 'AGB_2017'
        X = self.data.drop(columns=[agb_col])
        y = self.data[agb_col]
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data preprocessed: {len(self.feature_names)} features ready for analysis")
        
    def train_models_for_diagnostics(self):
        """Train models with optimal parameters for diagnostics"""
        print("\n🚀 Training models for diagnostics...")
        
        # Use best parameters from previous analysis
        models_config = {
            'Random Forest': RandomForestRegressor(
                n_estimators=1000, max_depth=None, max_features='sqrt',
                min_samples_split=2, min_samples_leaf=4, bootstrap=True,
                random_state=42
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=500, num_leaves=50, max_depth=7,
                learning_rate=0.1, random_state=42, verbose=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Train models and make predictions
        for model_name, model in models_config.items():
            print(f"   Training {model_name}...")
            model.fit(self.X_train_scaled, self.y_train)
            self.models[model_name] = model
            
            # Store predictions
            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            self.predictions[model_name] = {
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_actual': self.y_train.values,
                'test_actual': self.y_test.values
            }
        
        print("✅ Models trained and predictions generated")
        
    def analyze_learning_curves(self):
        """Analyze learning curves to detect overfitting/underfitting"""
        print("\n📈 Analyzing learning curves...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for model_name, model in self.models.items():
            print(f"   Computing learning curve for {model_name}...")
            
            # Compute learning curve
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, self.X_train_scaled, self.y_train,
                train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, random_state=42
            )
            
            # Convert to RMSE
            train_rmse = np.sqrt(-train_scores)
            val_rmse = np.sqrt(-val_scores)
            
            self.diagnostics[f'{model_name}_learning'] = {
                'train_sizes': train_sizes_abs,
                'train_rmse_mean': np.mean(train_rmse, axis=1),
                'train_rmse_std': np.std(train_rmse, axis=1),
                'val_rmse_mean': np.mean(val_rmse, axis=1),
                'val_rmse_std': np.std(val_rmse, axis=1)
            }
        
        print("✅ Learning curves analysis completed")
        
    def analyze_residuals(self):
        """Analyze residuals for model validation"""
        print("\n🔍 Analyzing residuals...")
        
        for model_name in self.models.keys():
            pred_data = self.predictions[model_name]
            
            # Calculate residuals
            train_residuals = pred_data['train_actual'] - pred_data['train_pred']
            test_residuals = pred_data['test_actual'] - pred_data['test_pred']
            
            # Statistical tests
            # Normality test
            train_normality = normaltest(train_residuals)
            test_normality = normaltest(test_residuals)
            
            # Homoscedasticity test (Breusch-Pagan test approximation)
            # Correlation between squared residuals and predictions
            train_homo_corr = np.corrcoef(train_residuals**2, pred_data['train_pred'])[0, 1]
            test_homo_corr = np.corrcoef(test_residuals**2, pred_data['test_pred'])[0, 1]
            
            self.diagnostics[f'{model_name}_residuals'] = {
                'train_residuals': train_residuals,
                'test_residuals': test_residuals,
                'train_normality_pvalue': train_normality.pvalue,
                'test_normality_pvalue': test_normality.pvalue,
                'train_homo_correlation': train_homo_corr,
                'test_homo_correlation': test_homo_corr,
                'train_residuals_mean': np.mean(train_residuals),
                'test_residuals_mean': np.mean(test_residuals),
                'train_residuals_std': np.std(train_residuals),
                'test_residuals_std': np.std(test_residuals)
            }
        
        print("✅ Residuals analysis completed")
        
    def analyze_bias_variance(self):
        """Analyze bias-variance tradeoff"""
        print("\n⚖️ Analyzing bias-variance tradeoff...")
        
        for model_name, model in self.models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train,
                cv=10, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Training score
            train_pred = model.predict(self.X_train_scaled)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            
            # Test score
            test_pred = model.predict(self.X_test_scaled)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            
            # Bias-Variance metrics
            bias_squared = (np.mean(cv_rmse) - train_rmse) ** 2
            variance = np.var(cv_rmse)
            
            self.diagnostics[f'{model_name}_bias_variance'] = {
                'cv_rmse_mean': np.mean(cv_rmse),
                'cv_rmse_std': np.std(cv_rmse),
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'bias_squared': bias_squared,
                'variance': variance,
                'overfitting_gap': test_rmse - train_rmse
            }
        
        print("✅ Bias-variance analysis completed")
        
    def create_learning_curves_plot(self):
        """Create learning curves visualization"""
        print("\n📊 Creating learning curves visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learning Curves Analysis - Training vs Validation Performance', fontsize=16, fontweight='bold')
        
        model_names = list(self.models.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, model_name in enumerate(model_names):
            ax = axes[i//2, i%2]
            
            learning_data = self.diagnostics[f'{model_name}_learning']
            
            # Plot training curve
            ax.plot(learning_data['train_sizes'], learning_data['train_rmse_mean'], 
                   'o-', color=colors[i], alpha=0.8, label='Training RMSE')
            ax.fill_between(learning_data['train_sizes'], 
                           learning_data['train_rmse_mean'] - learning_data['train_rmse_std'],
                           learning_data['train_rmse_mean'] + learning_data['train_rmse_std'],
                           alpha=0.2, color=colors[i])
            
            # Plot validation curve
            ax.plot(learning_data['train_sizes'], learning_data['val_rmse_mean'], 
                   's-', color=colors[i], alpha=0.6, linestyle='--', label='Validation RMSE')
            ax.fill_between(learning_data['train_sizes'], 
                           learning_data['val_rmse_mean'] - learning_data['val_rmse_std'],
                           learning_data['val_rmse_mean'] + learning_data['val_rmse_std'],
                           alpha=0.1, color=colors[i])
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('RMSE')
            ax.set_title(f'{model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def create_residuals_analysis_plot(self):
        """Create comprehensive residuals analysis"""
        print("\n🔍 Creating residuals analysis visualization...")
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('Residuals Analysis - Model Validation', fontsize=16, fontweight='bold')
        
        model_names = list(self.models.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, model_name in enumerate(model_names):
            residual_data = self.diagnostics[f'{model_name}_residuals']
            pred_data = self.predictions[model_name]
            
            # Residuals vs Fitted (Test set)
            ax1 = axes[i, 0]
            ax1.scatter(pred_data['test_pred'], residual_data['test_residuals'], 
                       alpha=0.6, color=colors[i])
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Fitted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title(f'{model_name} - Residuals vs Fitted')
            ax1.grid(True, alpha=0.3)
            
            # Q-Q Plot (Test set)
            ax2 = axes[i, 1]
            stats.probplot(residual_data['test_residuals'], dist="norm", plot=ax2)
            ax2.set_title(f'{model_name} - Q-Q Plot')
            ax2.grid(True, alpha=0.3)
            
            # Histogram of residuals
            ax3 = axes[i, 2]
            ax3.hist(residual_data['test_residuals'], bins=15, alpha=0.7, color=colors[i], edgecolor='black')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'{model_name} - Residuals Distribution')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.grid(True, alpha=0.3)
            
            # Actual vs Predicted
            ax4 = axes[i, 3]
            ax4.scatter(pred_data['test_actual'], pred_data['test_pred'], alpha=0.6, color=colors[i])
            
            # Perfect prediction line
            min_val = min(min(pred_data['test_actual']), min(pred_data['test_pred']))
            max_val = max(max(pred_data['test_actual']), max(pred_data['test_pred']))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            
            ax4.set_xlabel('Actual AGB')
            ax4.set_ylabel('Predicted AGB')
            ax4.set_title(f'{model_name} - Actual vs Predicted')
            ax4.grid(True, alpha=0.3)
            
            # Add R² annotation
            r2 = r2_score(pred_data['test_actual'], pred_data['test_pred'])
            ax4.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    def create_bias_variance_analysis(self):
        """Create bias-variance analysis visualization"""
        print("\n⚖️ Creating bias-variance analysis visualization...")
        
        # Extract bias-variance data
        models_bv_data = []
        for model_name in self.models.keys():
            bv_data = self.diagnostics[f'{model_name}_bias_variance']
            models_bv_data.append({
                'Model': model_name,
                'CV_RMSE_Mean': bv_data['cv_rmse_mean'],
                'CV_RMSE_Std': bv_data['cv_rmse_std'],
                'Train_RMSE': bv_data['train_rmse'],
                'Test_RMSE': bv_data['test_rmse'],
                'Overfitting_Gap': bv_data['overfitting_gap'],
                'Variance': bv_data['variance']
            })
        
        bv_df = pd.DataFrame(models_bv_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bias-Variance Analysis', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Cross-validation performance with error bars
        ax1 = axes[0, 0]
        bars = ax1.bar(bv_df['Model'], bv_df['CV_RMSE_Mean'], 
                      yerr=bv_df['CV_RMSE_Std'], capsize=5, color=colors)
        ax1.set_title('Cross-Validation Performance')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Training vs Test performance
        ax2 = axes[0, 1]
        x = np.arange(len(bv_df))
        width = 0.35
        ax2.bar(x - width/2, bv_df['Train_RMSE'], width, label='Training RMSE', color=colors[0], alpha=0.7)
        ax2.bar(x + width/2, bv_df['Test_RMSE'], width, label='Test RMSE', color=colors[1], alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Training vs Test Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bv_df['Model'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Overfitting analysis
        ax3 = axes[1, 0]
        bars = ax3.bar(bv_df['Model'], bv_df['Overfitting_Gap'], color=colors)
        ax3.set_title('Overfitting Gap (Test - Train RMSE)')
        ax3.set_ylabel('RMSE Difference')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # Variance analysis
        ax4 = axes[1, 1]
        bars = ax4.bar(bv_df['Model'], bv_df['Variance'], color=colors)
        ax4.set_title('Model Variance (CV Stability)')
        ax4.set_ylabel('Variance')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def generate_diagnostic_insights(self):
        """Generate diagnostic insights and recommendations"""
        print("\n🧠 Generating diagnostic insights...")
        
        insights = {
            'overfitting_analysis': {},
            'residual_analysis': {},
            'stability_analysis': {},
            'recommendations': []
        }
        
        # Overfitting analysis
        for model_name in self.models.keys():
            bv_data = self.diagnostics[f'{model_name}_bias_variance']
            overfitting_gap = bv_data['overfitting_gap']
            
            if overfitting_gap > 0.5:
                insights['overfitting_analysis'][model_name] = 'High overfitting risk'
            elif overfitting_gap > 0.2:
                insights['overfitting_analysis'][model_name] = 'Moderate overfitting'
            else:
                insights['overfitting_analysis'][model_name] = 'Well-balanced'
        
        # Residual analysis
        for model_name in self.models.keys():
            residual_data = self.diagnostics[f'{model_name}_residuals']
            
            # Check normality
            normality_ok = residual_data['test_normality_pvalue'] > 0.05
            
            # Check homoscedasticity
            homo_ok = abs(residual_data['test_homo_correlation']) < 0.3
            
            insights['residual_analysis'][model_name] = {
                'normality': 'Good' if normality_ok else 'Violated',
                'homoscedasticity': 'Good' if homo_ok else 'Violated'
            }
        
        # Stability analysis
        best_stability = None
        best_stability_score = float('inf')
        
        for model_name in self.models.keys():
            bv_data = self.diagnostics[f'{model_name}_bias_variance']
            cv_std = bv_data['cv_rmse_std']
            
            if cv_std < best_stability_score:
                best_stability_score = cv_std
                best_stability = model_name
            
            insights['stability_analysis'][model_name] = cv_std
        
        # Generate recommendations
        insights['recommendations'].append(f"🏆 Most stable model: {best_stability} (CV std: {best_stability_score:.3f})")
        
        # Overfitting recommendations
        high_overfitting = [model for model, status in insights['overfitting_analysis'].items() 
                           if status == 'High overfitting risk']
        if high_overfitting:
            insights['recommendations'].append(f"⚠️ High overfitting detected in: {', '.join(high_overfitting)}")
            insights['recommendations'].append("💡 Consider regularization, cross-validation, or reducing model complexity")
        
        # Residual recommendations
        residual_issues = []
        for model, analysis in insights['residual_analysis'].items():
            if analysis['normality'] == 'Violated' or analysis['homoscedasticity'] == 'Violated':
                residual_issues.append(model)
        
        if residual_issues:
            insights['recommendations'].append(f"🔍 Residual assumption violations in: {', '.join(residual_issues)}")
            insights['recommendations'].append("💡 Consider data transformation or different model assumptions")
        
        # Print insights
        print("\n" + "="*70)
        print("🎯 MODEL DIAGNOSTICS INSIGHTS")
        print("="*70)
        
        print("\n📈 OVERFITTING ANALYSIS:")
        for model, status in insights['overfitting_analysis'].items():
            print(f"   {model}: {status}")
        
        print(f"\n🔍 RESIDUAL ANALYSIS:")
        for model, analysis in insights['residual_analysis'].items():
            print(f"   {model}: Normality={analysis['normality']}, Homoscedasticity={analysis['homoscedasticity']}")
        
        print(f"\n⚖️ STABILITY ANALYSIS (CV Standard Deviation):")
        for model, std in insights['stability_analysis'].items():
            print(f"   {model}: {std:.4f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("="*70)
        
        return insights
        
    def save_diagnostic_results(self):
        """Save all diagnostic results"""
        print("\n💾 Saving diagnostic results...")
        
        # Save learning curves data
        learning_data = {}
        for model_name in self.models.keys():
            learning_data[model_name] = self.diagnostics[f'{model_name}_learning']
        
        # Save to CSV
        for model_name, data in learning_data.items():
            df = pd.DataFrame(data)
            df.to_csv(f'learning_curve_{model_name.replace(" ", "_")}.csv', index=False)
        
        # Save residuals data
        residuals_summary = []
        for model_name in self.models.keys():
            residual_data = self.diagnostics[f'{model_name}_residuals']
            residuals_summary.append({
                'Model': model_name,
                'Train_Residuals_Mean': residual_data['train_residuals_mean'],
                'Test_Residuals_Mean': residual_data['test_residuals_mean'],
                'Train_Residuals_Std': residual_data['train_residuals_std'],
                'Test_Residuals_Std': residual_data['test_residuals_std'],
                'Normality_PValue': residual_data['test_normality_pvalue'],
                'Homoscedasticity_Correlation': residual_data['test_homo_correlation']
            })
        
        pd.DataFrame(residuals_summary).to_csv('residuals_analysis.csv', index=False)
        
        # Save bias-variance data
        bv_summary = []
        for model_name in self.models.keys():
            bv_data = self.diagnostics[f'{model_name}_bias_variance']
            bv_summary.append({
                'Model': model_name,
                'CV_RMSE_Mean': bv_data['cv_rmse_mean'],
                'CV_RMSE_Std': bv_data['cv_rmse_std'],
                'Train_RMSE': bv_data['train_rmse'],
                'Test_RMSE': bv_data['test_rmse'],
                'Overfitting_Gap': bv_data['overfitting_gap'],
                'Variance': bv_data['variance']
            })
        
        pd.DataFrame(bv_summary).to_csv('bias_variance_analysis.csv', index=False)
        
        print("   ✅ All diagnostic results saved")
        
    def run_full_diagnostics(self):
        """Run the complete diagnostics pipeline"""
        print("🚀 AGB Model Diagnostics Dashboard - Full Analysis")
        print("="*70)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train models
        self.train_models_for_diagnostics()
        
        # Run diagnostics
        self.analyze_learning_curves()
        self.analyze_residuals()
        self.analyze_bias_variance()
        
        # Create visualizations
        self.create_learning_curves_plot()
        self.create_residuals_analysis_plot()
        self.create_bias_variance_analysis()
        
        # Generate insights
        insights = self.generate_diagnostic_insights()
        
        # Save results
        self.save_diagnostic_results()
        
        print("\n🎉 Model diagnostics completed successfully!")
        print("="*70)
        
        return insights

if __name__ == "__main__":
    # Initialize and run diagnostics
    diagnostics_dashboard = AGBModelDiagnosticsDashboard()
    insights = diagnostics_dashboard.run_full_diagnostics()

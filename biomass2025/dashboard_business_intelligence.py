import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Statistical analysis
import scipy.stats as stats
from scipy import stats as scipy_stats

# Business analysis
from datetime import datetime, timedelta

class AGBBusinessIntelligenceDashboard:
    def __init__(self, data_path='FEI data/opt_means_cleaned.csv'):
        """Initialize the business intelligence dashboard"""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.business_metrics = {}
        self.cost_analysis = {}
        self.roi_analysis = {}
        self.deployment_readiness = {}
        
    def load_and_preprocess_data(self):
        """Load data and perform preprocessing"""
        print("🔄 Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        
        # Features and target
        X = self.data.drop(columns=['AGB_2017'])
        y = self.data['AGB_2017']
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
        
    def train_optimized_models(self):
        """Train models with business-optimized configurations"""
        print("\n🚀 Training business-optimized models...")
        
        # Model configurations optimized for business deployment
        models_config = {
            'Production RF': {
                'model': RandomForestRegressor(
                    n_estimators=200,  # Reduced for faster inference
                    max_depth=10,      # Controlled complexity
                    max_features='sqrt',
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_case': 'High accuracy, moderate speed',
                'deployment_complexity': 'Medium'
            },
            'Fast LightGBM': {
                'model': LGBMRegressor(
                    n_estimators=100,  # Fast inference
                    num_leaves=31,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                ),
                'use_case': 'Real-time predictions',
                'deployment_complexity': 'Low'
            },
            'Balanced XGBoost': {
                'model': XGBRegressor(
                    n_estimators=150,  # Balanced performance
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                ),
                'use_case': 'Production balance',
                'deployment_complexity': 'Medium'
            },
            'Simple SVR': {
                'model': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'use_case': 'Lightweight deployment',
                'deployment_complexity': 'Low'
            }
        }
        
        # Train models and collect business metrics
        for model_name, config in models_config.items():
            print(f"   Training {model_name}...")
            
            model = config['model']
            
            # Time training
            import time
            start_time = time.time()
            model.fit(self.X_train_scaled, self.y_train)
            training_time = time.time() - start_time
            
            # Time inference
            start_time = time.time()
            y_pred = model.predict(self.X_test_scaled)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            
            # Cross-validation for stability
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
            # Store comprehensive results
            self.models[model_name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'training_time': training_time,
                'inference_time': inference_time,
                'inference_time_per_sample': inference_time / len(self.y_test),
                'cv_rmse_mean': np.mean(cv_rmse),
                'cv_rmse_std': np.std(cv_rmse),
                'use_case': config['use_case'],
                'deployment_complexity': config['deployment_complexity'],
                'predictions': y_pred
            }
        
        print("✅ Business-optimized models trained")
        
    def analyze_cost_benefit(self):
        """Analyze cost-benefit scenarios for different models"""
        print("\n💰 Analyzing cost-benefit scenarios...")
        
        # Define cost parameters (example values - adjust based on real scenarios)
        cost_parameters = {
            'data_collection_cost_per_sample': 50,  # USD per sample
            'model_development_cost': 10000,        # USD one-time
            'model_maintenance_cost_annual': 5000,  # USD per year
            'infrastructure_cost_monthly': 500,     # USD per month
            'prediction_cost_per_1000': 1,          # USD per 1000 predictions
            'field_survey_cost_per_hectare': 200,   # USD per hectare
            'satellite_data_cost_per_hectare': 5,   # USD per hectare
        }
        
        # Define business scenarios
        scenarios = {
            'Small Scale': {
                'area_hectares': 1000,
                'predictions_per_month': 10000,
                'accuracy_requirement': 0.7,  # R² threshold
                'budget_annual': 50000
            },
            'Medium Scale': {
                'area_hectares': 10000,
                'predictions_per_month': 100000,
                'accuracy_requirement': 0.75,
                'budget_annual': 200000
            },
            'Large Scale': {
                'area_hectares': 100000,
                'predictions_per_month': 1000000,
                'accuracy_requirement': 0.8,
                'budget_annual': 1000000
            }
        }
        
        # Calculate costs and benefits for each model and scenario
        for scenario_name, scenario in scenarios.items():
            print(f"   Analyzing {scenario_name} scenario...")
            
            scenario_results = {}
            
            for model_name, model_data in self.models.items():
                # Calculate annual costs
                infrastructure_cost = cost_parameters['infrastructure_cost_monthly'] * 12
                prediction_cost = (scenario['predictions_per_month'] * 12 * 
                                 cost_parameters['prediction_cost_per_1000'] / 1000)
                satellite_cost = scenario['area_hectares'] * cost_parameters['satellite_data_cost_per_hectare']
                
                total_annual_cost = (
                    cost_parameters['model_maintenance_cost_annual'] +
                    infrastructure_cost +
                    prediction_cost +
                    satellite_cost
                )
                
                # Calculate benefits (cost savings vs traditional field surveys)
                traditional_survey_cost = (scenario['area_hectares'] * 
                                         cost_parameters['field_survey_cost_per_hectare'])
                cost_savings = traditional_survey_cost - total_annual_cost
                
                # ROI calculation
                roi = (cost_savings / total_annual_cost) * 100 if total_annual_cost > 0 else 0
                
                # Accuracy penalty (if model doesn't meet requirements)
                accuracy_penalty = 0
                if model_data['r2'] < scenario['accuracy_requirement']:
                    accuracy_penalty = (scenario['accuracy_requirement'] - model_data['r2']) * 50000  # USD penalty
                
                # Net benefit
                net_benefit = cost_savings - accuracy_penalty
                
                # Deployment feasibility
                meets_accuracy = model_data['r2'] >= scenario['accuracy_requirement']
                within_budget = total_annual_cost <= scenario['budget_annual']
                fast_enough = model_data['inference_time_per_sample'] < 0.001  # < 1ms per sample
                
                scenario_results[model_name] = {
                    'total_annual_cost': total_annual_cost,
                    'cost_savings': cost_savings,
                    'roi_percentage': roi,
                    'accuracy_penalty': accuracy_penalty,
                    'net_benefit': net_benefit,
                    'meets_accuracy': meets_accuracy,
                    'within_budget': within_budget,
                    'fast_enough': fast_enough,
                    'deployment_score': sum([meets_accuracy, within_budget, fast_enough]) / 3
                }
            
            self.cost_analysis[scenario_name] = scenario_results
        
        print("✅ Cost-benefit analysis completed")
        
    def analyze_prediction_confidence(self):
        """Analyze prediction confidence and uncertainty"""
        print("\n🎯 Analyzing prediction confidence...")
        
        for model_name, model_data in self.models.items():
            predictions = model_data['predictions']
            actual = self.y_test
            
            # Calculate prediction intervals using residuals
            residuals = actual - predictions
            residual_std = np.std(residuals)
            
            # 95% prediction intervals
            prediction_intervals = {
                'lower_95': predictions - 1.96 * residual_std,
                'upper_95': predictions + 1.96 * residual_std,
                'interval_width': 2 * 1.96 * residual_std
            }
            
            # Coverage probability (how many actual values fall within intervals)
            within_interval = ((actual >= prediction_intervals['lower_95']) & 
                             (actual <= prediction_intervals['upper_95']))
            coverage_probability = np.mean(within_interval)
            
            # Prediction reliability metrics
            reliability_metrics = {
                'mean_absolute_error': model_data['mae'],
                'rmse': model_data['rmse'],
                'coverage_probability': coverage_probability,
                'mean_interval_width': np.mean(prediction_intervals['interval_width']),
                'prediction_uncertainty': residual_std,
                'confidence_score': min(coverage_probability, model_data['r2']) * 100
            }
            
            self.models[model_name]['prediction_confidence'] = {
                'intervals': prediction_intervals,
                'reliability': reliability_metrics
            }
        
        print("✅ Prediction confidence analysis completed")
        
    def assess_deployment_readiness(self):
        """Assess models for deployment readiness"""
        print("\n🚀 Assessing deployment readiness...")
        
        deployment_criteria = {
            'accuracy_threshold': 0.75,      # Minimum R² required
            'stability_threshold': 0.3,      # Maximum CV std allowed
            'speed_threshold': 0.01,         # Maximum inference time per sample (seconds)
            'complexity_preference': 'Low',  # Preferred deployment complexity
            'reliability_threshold': 0.8     # Minimum confidence score
        }
        
        for model_name, model_data in self.models.items():
            # Accuracy check
            accuracy_pass = model_data['r2'] >= deployment_criteria['accuracy_threshold']
            
            # Stability check
            stability_pass = model_data['cv_rmse_std'] <= deployment_criteria['stability_threshold']
            
            # Speed check
            speed_pass = model_data['inference_time_per_sample'] <= deployment_criteria['speed_threshold']
            
            # Complexity check
            complexity_pass = model_data['deployment_complexity'] == deployment_criteria['complexity_preference']
            
            # Reliability check
            confidence_score = model_data['prediction_confidence']['reliability']['confidence_score']
            reliability_pass = confidence_score >= deployment_criteria['reliability_threshold'] * 100
            
            # Overall deployment score
            checks = [accuracy_pass, stability_pass, speed_pass, reliability_pass]
            deployment_score = sum(checks) / len(checks)
            
            # Risk assessment
            risks = []
            if not accuracy_pass:
                risks.append("Low accuracy - may lead to poor business decisions")
            if not stability_pass:
                risks.append("High variance - inconsistent performance")
            if not speed_pass:
                risks.append("Slow inference - may not meet real-time requirements")
            if not reliability_pass:
                risks.append("Low confidence - predictions may be unreliable")
            
            # Deployment recommendation
            if deployment_score >= 0.75:
                recommendation = "✅ Ready for production deployment"
            elif deployment_score >= 0.5:
                recommendation = "⚠️ Ready with monitoring and improvements"
            else:
                recommendation = "❌ Not ready - requires significant improvements"
            
            self.deployment_readiness[model_name] = {
                'accuracy_pass': accuracy_pass,
                'stability_pass': stability_pass,
                'speed_pass': speed_pass,
                'complexity_pass': complexity_pass,
                'reliability_pass': reliability_pass,
                'deployment_score': deployment_score,
                'risks': risks,
                'recommendation': recommendation
            }
        
        print("✅ Deployment readiness assessment completed")
        
    def create_business_visualizations(self):
        """Create business-focused visualizations"""
        print("\n📈 Creating business intelligence visualizations...")
        
        # 1. ROI Analysis Across Scenarios
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Business Intelligence Dashboard - ROI & Cost Analysis', fontsize=16, fontweight='bold')
        
        # ROI by scenario
        ax1 = axes[0, 0]
        scenarios = list(self.cost_analysis.keys())
        models = list(self.models.keys())
        
        x = np.arange(len(scenarios))
        width = 0.2
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, model in enumerate(models):
            roi_values = [self.cost_analysis[scenario][model]['roi_percentage'] for scenario in scenarios]
            ax1.bar(x + i*width, roi_values, width, label=model, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Business Scenarios')
        ax1.set_ylabel('ROI (%)')
        ax1.set_title('Return on Investment by Scenario')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(scenarios)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Cost breakdown
        ax2 = axes[0, 1]
        medium_scenario = self.cost_analysis['Medium Scale']
        cost_data = []
        
        for model, data in medium_scenario.items():
            cost_data.append({
                'Model': model,
                'Annual Cost': data['total_annual_cost'],
                'Cost Savings': data['cost_savings']
            })
        
        cost_df = pd.DataFrame(cost_data)
        
        x_pos = np.arange(len(cost_df))
        ax2.bar(x_pos - 0.2, cost_df['Annual Cost'], 0.4, label='Annual Cost', color='#E74C3C', alpha=0.7)
        ax2.bar(x_pos + 0.2, cost_df['Cost Savings'], 0.4, label='Cost Savings', color='#2ECC71', alpha=0.7)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Cost (USD)')
        ax2.set_title('Cost Analysis - Medium Scale Scenario')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cost_df['Model'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Deployment readiness radar
        ax3 = axes[1, 0]
        
        # Select best model for radar chart
        best_model = max(self.deployment_readiness.keys(), 
                        key=lambda x: self.deployment_readiness[x]['deployment_score'])
        
        readiness_data = self.deployment_readiness[best_model]
        categories = ['Accuracy', 'Stability', 'Speed', 'Reliability']
        values = [
            readiness_data['accuracy_pass'],
            readiness_data['stability_pass'], 
            readiness_data['speed_pass'],
            readiness_data['reliability_pass']
        ]
        
        # Convert boolean to numeric
        values = [1 if v else 0 for v in values]
        
        # Create radar chart data
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax3.remove()
        ax3 = fig.add_subplot(2, 2, 3, projection='polar')
        ax3.plot(angles, values, 'o-', linewidth=2, label=best_model, color=colors[0])
        ax3.fill(angles, values, alpha=0.25, color=colors[0])
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title(f'Deployment Readiness - {best_model}', pad=20)
        ax3.grid(True)
        
        # Prediction confidence
        ax4 = axes[1, 1]
        
        confidence_scores = []
        model_names = []
        
        for model_name, model_data in self.models.items():
            confidence_score = model_data['prediction_confidence']['reliability']['confidence_score']
            confidence_scores.append(confidence_score)
            model_names.append(model_name)
        
        bars = ax4.bar(model_names, confidence_scores, color=colors, alpha=0.8)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Confidence Score')
        ax4.set_title('Prediction Confidence Scores')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Minimum Threshold')
        ax4.legend()
        
        # Add value labels on bars
        for bar, score in zip(bars, confidence_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Business Decision Matrix
        print("   Creating business decision matrix...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create decision matrix data
        matrix_data = []
        for model_name, model_data in self.models.items():
            matrix_data.append({
                'Model': model_name,
                'Accuracy (R²)': model_data['r2'],
                'Speed (1/inference_time)': 1 / (model_data['inference_time_per_sample'] + 1e-6),
                'ROI (Medium Scale)': self.cost_analysis['Medium Scale'][model_name]['roi_percentage'],
                'Deployment Score': self.deployment_readiness[model_name]['deployment_score'] * 100
            })
        
        matrix_df = pd.DataFrame(matrix_data)
        
        # Create scatter plot with size as deployment score
        scatter = ax.scatter(
            matrix_df['Accuracy (R²)'], 
            matrix_df['ROI (Medium Scale)'],
            s=matrix_df['Deployment Score'] * 5,  # Size based on deployment score
            c=matrix_df['Speed (1/inference_time)'],  # Color based on speed
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add model labels
        for i, row in matrix_df.iterrows():
            ax.annotate(row['Model'], 
                       (row['Accuracy (R²)'], row['ROI (Medium Scale)']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Model Accuracy (R²)')
        ax.set_ylabel('ROI (%) - Medium Scale')
        ax.set_title('Business Decision Matrix\n(Bubble size = Deployment Score, Color = Speed)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Speed (1/inference_time)')
        
        # Add quadrant lines
        ax.axhline(y=matrix_df['ROI (Medium Scale)'].median(), color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=matrix_df['Accuracy (R²)'].median(), color='red', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.95, 0.95, 'High Accuracy\nHigh ROI', transform=ax.transAxes, 
               ha='right', va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
    def generate_business_insights(self):
        """Generate comprehensive business insights and recommendations"""
        print("\n🧠 Generating business insights...")
        
        insights = {
            'best_roi_model': {},
            'deployment_recommendations': {},
            'cost_optimization': {},
            'risk_assessment': {},
            'strategic_recommendations': []
        }
        
        # Find best ROI model for each scenario
        for scenario_name, scenario_data in self.cost_analysis.items():
            best_roi = -float('inf')
            best_model = None
            
            for model_name, model_data in scenario_data.items():
                if model_data['roi_percentage'] > best_roi:
                    best_roi = model_data['roi_percentage']
                    best_model = model_name
            
            insights['best_roi_model'][scenario_name] = {
                'model': best_model,
                'roi': best_roi,
                'net_benefit': scenario_data[best_model]['net_benefit']
            }
        
        # Deployment recommendations
        for model_name, readiness in self.deployment_readiness.items():
            insights['deployment_recommendations'][model_name] = {
                'score': readiness['deployment_score'],
                'recommendation': readiness['recommendation'],
                'risks': readiness['risks']
            }
        
        # Cost optimization insights
        medium_costs = self.cost_analysis['Medium Scale']
        lowest_cost_model = min(medium_costs.keys(), key=lambda x: medium_costs[x]['total_annual_cost'])
        highest_roi_model = max(medium_costs.keys(), key=lambda x: medium_costs[x]['roi_percentage'])
        
        insights['cost_optimization'] = {
            'lowest_cost_model': lowest_cost_model,
            'highest_roi_model': highest_roi_model,
            'cost_difference': (medium_costs[highest_roi_model]['total_annual_cost'] - 
                              medium_costs[lowest_cost_model]['total_annual_cost'])
        }
        
        # Risk assessment
        high_risk_models = []
        for model_name, readiness in self.deployment_readiness.items():
            if readiness['deployment_score'] < 0.7:
                high_risk_models.append(model_name)
        
        insights['risk_assessment'] = {
            'high_risk_models': high_risk_models,
            'risk_factors': []
        }
        
        # Generate strategic recommendations
        best_overall = max(self.deployment_readiness.keys(), 
                          key=lambda x: self.deployment_readiness[x]['deployment_score'])
        
        insights['strategic_recommendations'].extend([
            f"🏆 Recommended model for production: {best_overall}",
            f"💰 Best ROI for medium scale: {insights['best_roi_model']['Medium Scale']['model']} ({insights['best_roi_model']['Medium Scale']['roi']:.1f}% ROI)",
            f"⚡ Fastest model: {min(self.models.keys(), key=lambda x: self.models[x]['inference_time_per_sample'])}",
            f"🎯 Most accurate model: {max(self.models.keys(), key=lambda x: self.models[x]['r2'])}"
        ])
        
        if insights['cost_optimization']['cost_difference'] > 10000:
            insights['strategic_recommendations'].append(
                f"💡 Consider cost vs performance tradeoff: {insights['cost_optimization']['highest_roi_model']} costs ${insights['cost_optimization']['cost_difference']:,.0f} more but provides better ROI"
            )
        
        if high_risk_models:
            insights['strategic_recommendations'].append(
                f"⚠️ High-risk models requiring attention: {', '.join(high_risk_models)}"
            )
        
        # Print comprehensive insights
        print("\n" + "="*80)
        print("🎯 BUSINESS INTELLIGENCE INSIGHTS")
        print("="*80)
        
        print("\n💰 ROI ANALYSIS BY SCENARIO:")
        for scenario, data in insights['best_roi_model'].items():
            print(f"   {scenario}: {data['model']} (ROI: {data['roi']:.1f}%, Net Benefit: ${data['net_benefit']:,.0f})")
        
        print(f"\n🚀 DEPLOYMENT READINESS:")
        for model, rec in insights['deployment_recommendations'].items():
            print(f"   {model}: Score {rec['score']:.2f} - {rec['recommendation']}")
        
        print(f"\n💡 COST OPTIMIZATION:")
        print(f"   Lowest cost model: {insights['cost_optimization']['lowest_cost_model']}")
        print(f"   Highest ROI model: {insights['cost_optimization']['highest_roi_model']}")
        print(f"   Cost difference: ${insights['cost_optimization']['cost_difference']:,.0f}")
        
        print(f"\n⚠️ RISK ASSESSMENT:")
        if high_risk_models:
            print(f"   High-risk models: {', '.join(high_risk_models)}")
        else:
            print("   No high-risk models identified")
        
        print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(insights['strategic_recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("="*80)
        
        return insights
        
    def save_business_results(self):
        """Save all business intelligence results"""
        print("\n💾 Saving business intelligence results...")
        
        # Save cost analysis
        cost_summary = []
        for scenario, models in self.cost_analysis.items():
            for model, data in models.items():
                cost_summary.append({
                    'Scenario': scenario,
                    'Model': model,
                    'Annual_Cost': data['total_annual_cost'],
                    'Cost_Savings': data['cost_savings'],
                    'ROI_Percentage': data['roi_percentage'],
                    'Net_Benefit': data['net_benefit'],
                    'Deployment_Score': data['deployment_score']
                })
        
        pd.DataFrame(cost_summary).to_csv('business_cost_analysis.csv', index=False)
        
        # Save deployment readiness
        deployment_summary = []
        for model, readiness in self.deployment_readiness.items():
            deployment_summary.append({
                'Model': model,
                'Deployment_Score': readiness['deployment_score'],
                'Accuracy_Pass': readiness['accuracy_pass'],
                'Stability_Pass': readiness['stability_pass'],
                'Speed_Pass': readiness['speed_pass'],
                'Reliability_Pass': readiness['reliability_pass'],
                'Recommendation': readiness['recommendation']
            })
        
        pd.DataFrame(deployment_summary).to_csv('deployment_readiness.csv', index=False)
        
        # Save model performance summary
        performance_summary = []
        for model, data in self.models.items():
            performance_summary.append({
                'Model': model,
                'RMSE': data['rmse'],
                'R2': data['r2'],
                'MAE': data['mae'],
                'Training_Time': data['training_time'],
                'Inference_Time_Per_Sample': data['inference_time_per_sample'],
                'CV_RMSE_Mean': data['cv_rmse_mean'],
                'CV_RMSE_Std': data['cv_rmse_std'],
                'Use_Case': data['use_case'],
                'Deployment_Complexity': data['deployment_complexity']
            })
        
        pd.DataFrame(performance_summary).to_csv('business_model_performance.csv', index=False)
        
        print("   ✅ All business intelligence results saved")
        
    def run_full_business_analysis(self):
        """Run the complete business intelligence pipeline"""
        print("🚀 AGB Business Intelligence Dashboard - Full Analysis")
        print("="*80)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train business-optimized models
        self.train_optimized_models()
        
        # Analyze cost-benefit
        self.analyze_cost_benefit()
        
        # Analyze prediction confidence
        self.analyze_prediction_confidence()
        
        # Assess deployment readiness
        self.assess_deployment_readiness()
        
        # Create visualizations
        self.create_business_visualizations()
        
        # Generate insights
        insights = self.generate_business_insights()
        
        # Save results
        self.save_business_results()
        
        print("\n🎉 Business intelligence analysis completed successfully!")
        print("="*80)
        
        return insights

if __name__ == "__main__":
    # Initialize and run business analysis
    business_dashboard = AGBBusinessIntelligenceDashboard()
    insights = business_dashboard.run_full_business_analysis()

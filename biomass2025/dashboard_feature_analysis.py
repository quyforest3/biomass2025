import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

# Advanced analysis
import shap
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class AGBFeatureAnalysisDashboard:
    def __init__(self, data_path='FEI data/opt_means_cleaned.csv'):
        """Initialize the feature analysis dashboard"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.feature_importance = {}
        self.correlation_data = {}
        
    def load_and_preprocess_data(self):
        """Load data and perform preprocessing"""
        print("🔄 Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        
        # Features and target
        # Detect target column (prefer AGB_2024, fallback to AGB_2017)
        self.agb_col = 'AGB_2024' if 'AGB_2024' in self.data.columns else 'AGB_2017'
        X = self.data.drop(columns=[self.agb_col])
        y = self.data[self.agb_col]
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
        
    def train_models_for_analysis(self):
        """Train models specifically for feature analysis"""
        print("\n🚀 Training models for feature analysis...")
        
        # Train Random Forest (best performer)
        rf_model = RandomForestRegressor(n_estimators=500, max_depth=None, 
                                       max_features='sqrt', random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        self.models['Random Forest'] = rf_model
        
        # Train LightGBM
        lgbm_model = LGBMRegressor(n_estimators=500, random_state=42, verbose=-1)
        lgbm_model.fit(self.X_train_scaled, self.y_train)
        self.models['LightGBM'] = lgbm_model
        
        # Train XGBoost
        xgb_model = XGBRegressor(n_estimators=500, random_state=42, verbosity=0)
        xgb_model.fit(self.X_train_scaled, self.y_train)
        self.models['XGBoost'] = xgb_model
        
        print("✅ Models trained for feature analysis")
        
    def analyze_feature_importance(self):
        """Analyze feature importance across different models"""
        print("\n📊 Analyzing feature importance...")
        
        # Get feature importance from tree-based models
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.feature_importance[model_name] = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
        
        # Permutation importance for all models (model-agnostic)
        print("   Computing permutation importance...")
        for model_name, model in self.models.items():
            perm_importance = permutation_importance(
                model, self.X_test_scaled, self.y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            self.feature_importance[f'{model_name}_Permutation'] = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            }).sort_values('Importance', ascending=False)
        
        print("✅ Feature importance analysis completed")
        
    def analyze_correlations(self):
        """Analyze feature correlations and relationships with target"""
        print("\n🔗 Analyzing feature correlations...")
        
        # Correlation with target variable
        target_correlations = []
        for feature in self.feature_names:
            pearson_corr, p_val = pearsonr(self.data[feature], self.data[self.agb_col])
            target_correlations.append({
                'Feature': feature,
                'Pearson_Correlation': pearson_corr,
                'P_Value': p_val,
                'Abs_Correlation': abs(pearson_corr)
            })
        
        self.correlation_data['target_correlations'] = pd.DataFrame(target_correlations).sort_values(
            'Abs_Correlation', ascending=False
        )
        
        # Feature-to-feature correlations
        feature_corr_matrix = self.data[self.feature_names].corr()
        self.correlation_data['feature_matrix'] = feature_corr_matrix
        
        # Mutual information
        mi_scores = mutual_info_regression(self.X_train_scaled, self.y_train, random_state=42)
        self.correlation_data['mutual_info'] = pd.DataFrame({
            'Feature': self.feature_names,
            'Mutual_Info': mi_scores
        }).sort_values('Mutual_Info', ascending=False)
        
        print("✅ Correlation analysis completed")
        
    def create_feature_importance_plots(self):
        """Create comprehensive feature importance visualizations"""
        print("\n📈 Creating feature importance visualizations...")
        
        # 1. Feature Importance Comparison (Top 15 features)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Feature Importance Analysis - Top 15 Features', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
            if model_name in self.feature_importance:
                top_features = self.feature_importance[model_name].head(15)
                
                ax = axes[plot_idx // 2, plot_idx % 2]
                bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                              color=colors[plot_idx % len(colors)])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['Feature'])
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{model_name} - Feature Importance')
                ax.invert_yaxis()
                
                # Add value labels
                for i, (idx, row) in enumerate(top_features.iterrows()):
                    ax.text(row['Importance'] + 0.001, i, f'{row["Importance"]:.3f}', 
                           va='center', fontsize=8)
                
                plot_idx += 1
        
        # 4th subplot: Permutation importance comparison
        ax = axes[1, 1]
        perm_data = self.feature_importance['Random Forest_Permutation'].head(15)
        bars = ax.barh(range(len(perm_data)), perm_data['Importance'], 
                      xerr=perm_data['Std'], color='#FFA07A')
        ax.set_yticks(range(len(perm_data)))
        ax.set_yticklabels(perm_data['Feature'])
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Random Forest - Permutation Importance (with std)')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        # 2. Feature Importance Heatmap Comparison
        print("   Creating feature importance heatmap...")
        
        # Combine top features from all models
        all_important_features = set()
        for model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
            if model_name in self.feature_importance:
                top_10 = self.feature_importance[model_name].head(10)['Feature'].tolist()
                all_important_features.update(top_10)
        
        # Create comparison matrix
        importance_matrix = []
        for feature in all_important_features:
            row = {'Feature': feature}
            for model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
                if model_name in self.feature_importance:
                    importance_df = self.feature_importance[model_name]
                    feature_importance = importance_df[importance_df['Feature'] == feature]['Importance'].iloc[0] if feature in importance_df['Feature'].values else 0
                    row[model_name] = feature_importance
            importance_matrix.append(row)
        
        importance_comparison_df = pd.DataFrame(importance_matrix)
        importance_comparison_df = importance_comparison_df.set_index('Feature')
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(importance_comparison_df, annot=True, cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature Importance'}, fmt='.3f')
        plt.title('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
    def create_correlation_plots(self):
        """Create correlation analysis visualizations"""
        print("\n🔗 Creating correlation visualizations...")
        
        # 1. Target correlation plot
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Feature Correlation Analysis with AGB Target', fontsize=16, fontweight='bold')
        
        # Top positive correlations
        top_pos_corr = self.correlation_data['target_correlations'][
            self.correlation_data['target_correlations']['Pearson_Correlation'] > 0
        ].head(10)
        
        ax = axes[0, 0]
        bars = ax.barh(range(len(top_pos_corr)), top_pos_corr['Pearson_Correlation'], color='#2ECC71')
        ax.set_yticks(range(len(top_pos_corr)))
        ax.set_yticklabels(top_pos_corr['Feature'])
        ax.set_xlabel('Pearson Correlation')
        ax.set_title('Top Positive Correlations with AGB')
        ax.invert_yaxis()
        
        # Top negative correlations
        top_neg_corr = self.correlation_data['target_correlations'][
            self.correlation_data['target_correlations']['Pearson_Correlation'] < 0
        ].head(10)
        
        ax = axes[0, 1]
        bars = ax.barh(range(len(top_neg_corr)), top_neg_corr['Pearson_Correlation'], color='#E74C3C')
        ax.set_yticks(range(len(top_neg_corr)))
        ax.set_yticklabels(top_neg_corr['Feature'])
        ax.set_xlabel('Pearson Correlation')
        ax.set_title('Top Negative Correlations with AGB')
        ax.invert_yaxis()
        
        # Mutual Information
        top_mi = self.correlation_data['mutual_info'].head(15)
        ax = axes[1, 0]
        bars = ax.barh(range(len(top_mi)), top_mi['Mutual_Info'], color='#9B59B6')
        ax.set_yticks(range(len(top_mi)))
        ax.set_yticklabels(top_mi['Feature'])
        ax.set_xlabel('Mutual Information Score')
        ax.set_title('Top Features by Mutual Information')
        ax.invert_yaxis()
        
        # Correlation vs Importance scatter
        ax = axes[1, 1]
        
        # Combine correlation and importance data
        rf_importance = self.feature_importance['Random Forest']
        target_corr = self.correlation_data['target_correlations']
        
        merged_data = rf_importance.merge(target_corr, on='Feature')
        
        scatter = ax.scatter(merged_data['Abs_Correlation'], merged_data['Importance'], 
                           alpha=0.7, s=60, c='#3498DB')
        ax.set_xlabel('Absolute Correlation with AGB')
        ax.set_ylabel('Random Forest Importance')
        ax.set_title('Correlation vs Feature Importance')
        
        # Add labels for top points
        for idx, row in merged_data.nlargest(5, 'Importance').iterrows():
            ax.annotate(row['Feature'], (row['Abs_Correlation'], row['Importance']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Feature correlation heatmap
        plt.figure(figsize=(16, 14))
        
        # Select top correlated features for cleaner visualization
        top_features = self.correlation_data['target_correlations'].head(15)['Feature'].tolist()
        corr_subset = self.correlation_data['feature_matrix'].loc[top_features, top_features]
        
        mask = np.triu(np.ones_like(corr_subset, dtype=bool))
        sns.heatmap(corr_subset, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Feature-to-Feature Correlation Matrix (Top 15 Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def generate_feature_insights(self):
        """Generate actionable insights from feature analysis"""
        print("\n🧠 Generating Feature Insights...")
        
        insights = {
            'top_features': {},
            'correlation_insights': {},
            'model_agreement': {},
            'recommendations': []
        }
        
        # Top features across models
        for model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
            if model_name in self.feature_importance:
                top_5 = self.feature_importance[model_name].head(5)
                insights['top_features'][model_name] = top_5['Feature'].tolist()
        
        # Correlation insights
        high_corr_features = self.correlation_data['target_correlations'][
            self.correlation_data['target_correlations']['Abs_Correlation'] > 0.3
        ]
        insights['correlation_insights']['high_correlation_count'] = len(high_corr_features)
        insights['correlation_insights']['top_correlated'] = high_corr_features.head(5)['Feature'].tolist()
        
        # Model agreement analysis
        all_top_features = []
        for model_features in insights['top_features'].values():
            all_top_features.extend(model_features[:3])  # Top 3 from each model
        
        from collections import Counter
        feature_counts = Counter(all_top_features)
        agreed_features = [feature for feature, count in feature_counts.items() if count >= 2]
        insights['model_agreement']['agreed_features'] = agreed_features
        
        # Generate recommendations
        if len(agreed_features) > 0:
            insights['recommendations'].append(
                f"🎯 Focus on these {len(agreed_features)} features agreed upon by multiple models: {', '.join(agreed_features[:5])}"
            )
        
        if insights['correlation_insights']['high_correlation_count'] > 10:
            insights['recommendations'].append(
                "⚠️ Consider feature selection - many features show high correlation with target"
            )
        
        # Check for multicollinearity
        high_corr_pairs = []
        corr_matrix = self.correlation_data['feature_matrix']
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if len(high_corr_pairs) > 0:
            insights['recommendations'].append(
                f"🔗 Found {len(high_corr_pairs)} highly correlated feature pairs - consider dimensionality reduction"
            )
        
        # Print insights
        print("\n" + "="*60)
        print("🎯 FEATURE ANALYSIS INSIGHTS")
        print("="*60)
        
        print("\n📊 TOP FEATURES BY MODEL:")
        for model, features in insights['top_features'].items():
            print(f"   {model}: {', '.join(features[:3])}")
        
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print(f"   • {insights['correlation_insights']['high_correlation_count']} features with |correlation| > 0.3")
        print(f"   • Top correlated: {', '.join(insights['correlation_insights']['top_correlated'][:3])}")
        
        print(f"\n🤝 MODEL AGREEMENT:")
        if agreed_features:
            print(f"   • {len(agreed_features)} features agreed upon by multiple models")
            print(f"   • Consensus features: {', '.join(agreed_features)}")
        else:
            print("   • No strong consensus between models on top features")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("="*60)
        
        return insights
        
    def save_analysis_results(self):
        """Save all analysis results"""
        print("\n💾 Saving feature analysis results...")
        
        # Save feature importance results
        for model_name, importance_df in self.feature_importance.items():
            filename = f'feature_importance_{model_name.replace(" ", "_")}.csv'
            importance_df.to_csv(filename, index=False)
            print(f"   ✅ {model_name} importance saved as {filename}")
        
        # Save correlation results
        self.correlation_data['target_correlations'].to_csv('target_correlations.csv', index=False)
        self.correlation_data['feature_matrix'].to_csv('feature_correlation_matrix.csv')
        self.correlation_data['mutual_info'].to_csv('mutual_information_scores.csv', index=False)
        
        print("   ✅ Correlation analysis saved")
        print("   ✅ Feature analysis complete!")
        
    def run_full_analysis(self):
        """Run the complete feature analysis pipeline"""
        print("🚀 AGB Feature Analysis Dashboard - Full Analysis")
        print("="*60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train models for analysis
        self.train_models_for_analysis()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Analyze correlations
        self.analyze_correlations()
        
        # Create visualizations
        self.create_feature_importance_plots()
        self.create_correlation_plots()
        
        # Generate insights
        insights = self.generate_feature_insights()
        
        # Save results
        self.save_analysis_results()
        
        print("\n🎉 Feature analysis completed successfully!")
        print("="*60)
        
        return insights

if __name__ == "__main__":
    # Initialize and run feature analysis
    feature_dashboard = AGBFeatureAnalysisDashboard()
    insights = feature_dashboard.run_full_analysis()

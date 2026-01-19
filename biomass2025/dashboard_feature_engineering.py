import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Feature selection and engineering
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, RFE, RFECV,
    SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Advanced feature engineering
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import scipy.stats as stats

class AGBFeatureEngineeringDashboard:
    def __init__(self, data_path='FEI data/opt_means_cleaned.csv'):
        """Initialize the feature engineering dashboard"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.engineered_features = {}
        self.selection_results = {}
        self.transformation_results = {}
        self.optimized_models = {}
        
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
        
        print(f"✅ Data preprocessed: {len(self.feature_names)} features ready for engineering")
        
    def create_engineered_features(self):
        """Create new engineered features"""
        print("\n🔧 Creating engineered features...")
        
        # Start with original features
        X_train_eng = self.X_train.copy()
        X_test_eng = self.X_test.copy()
        
        # 1. Vegetation Indices Combinations (based on our feature analysis insights)
        print("   Creating vegetation indices combinations...")
        
        # Key features from previous analysis: ChlRe, MCARI, NDMI, NDCI
        # Create interaction terms
        X_train_eng['ChlRe_MCARI'] = X_train_eng['ChlRe'] * X_train_eng['MCARI']
        X_test_eng['ChlRe_MCARI'] = X_test_eng['ChlRe'] * X_test_eng['MCARI']
        
        X_train_eng['NDMI_NDCI'] = X_train_eng['NDMI'] * X_train_eng['NDCI']
        X_test_eng['NDMI_NDCI'] = X_test_eng['NDMI'] * X_test_eng['NDCI']
        
        X_train_eng['ChlRe_NDMI'] = X_train_eng['ChlRe'] * X_train_eng['NDMI']
        X_test_eng['ChlRe_NDMI'] = X_test_eng['ChlRe'] * X_test_eng['NDMI']
        
        # 2. Spectral band ratios
        print("   Creating spectral band ratios...")
        
        # NIR/Red ratio (classic vegetation index)
        X_train_eng['NIR_Red_Ratio'] = X_train_eng['B08'] / (X_train_eng['B04'] + 1e-6)
        X_test_eng['NIR_Red_Ratio'] = X_test_eng['B08'] / (X_test_eng['B04'] + 1e-6)
        
        # SWIR ratios
        X_train_eng['SWIR1_SWIR2_Ratio'] = X_train_eng['B11'] / (X_train_eng['B12'] + 1e-6)
        X_test_eng['SWIR1_SWIR2_Ratio'] = X_test_eng['B11'] / (X_test_eng['B12'] + 1e-6)
        
        # 3. Statistical aggregations
        print("   Creating statistical features...")
        
        # Mean of all spectral bands
        spectral_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        X_train_eng['Spectral_Mean'] = X_train_eng[spectral_bands].mean(axis=1)
        X_test_eng['Spectral_Mean'] = X_test_eng[spectral_bands].mean(axis=1)
        
        # Standard deviation of spectral bands
        X_train_eng['Spectral_Std'] = X_train_eng[spectral_bands].std(axis=1)
        X_test_eng['Spectral_Std'] = X_test_eng[spectral_bands].std(axis=1)
        
        # 4. Polynomial features for top vegetation indices
        print("   Creating polynomial features...")
        
        top_indices = ['ChlRe', 'MCARI', 'NDMI', 'NDCI']
        for idx in top_indices:
            X_train_eng[f'{idx}_squared'] = X_train_eng[idx] ** 2
            X_test_eng[f'{idx}_squared'] = X_test_eng[idx] ** 2
            
            X_train_eng[f'{idx}_log'] = np.log1p(X_train_eng[idx] - X_train_eng[idx].min() + 1)
            X_test_eng[f'{idx}_log'] = np.log1p(X_test_eng[idx] - X_test_eng[idx].min() + 1)
        
        self.engineered_features['X_train'] = X_train_eng
        self.engineered_features['X_test'] = X_test_eng
        
        print(f"✅ Feature engineering completed: {X_train_eng.shape[1]} features (original: {len(self.feature_names)})")
        
    def apply_feature_selection(self):
        """Apply various feature selection techniques"""
        print("\n🎯 Applying feature selection techniques...")
        
        X_train_eng = self.engineered_features['X_train']
        X_test_eng = self.engineered_features['X_test']
        
        # Scale features for selection
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_test_scaled = scaler.transform(X_test_eng)
        
        # 1. Variance Threshold
        print("   Applying variance threshold...")
        var_selector = VarianceThreshold(threshold=0.01)
        X_train_var = var_selector.fit_transform(X_train_scaled)
        selected_features_var = X_train_eng.columns[var_selector.get_support()].tolist()
        
        self.selection_results['variance_threshold'] = {
            'n_features': len(selected_features_var),
            'features': selected_features_var,
            'selector': var_selector
        }
        
        # 2. Univariate Selection (F-test)
        print("   Applying univariate F-test selection...")
        f_selector = SelectKBest(f_regression, k=15)
        X_train_f = f_selector.fit_transform(X_train_scaled, self.y_train)
        selected_features_f = X_train_eng.columns[f_selector.get_support()].tolist()
        
        self.selection_results['f_test'] = {
            'n_features': len(selected_features_f),
            'features': selected_features_f,
            'scores': f_selector.scores_,
            'selector': f_selector
        }
        
        # 3. Mutual Information
        print("   Applying mutual information selection...")
        mi_selector = SelectKBest(mutual_info_regression, k=15)
        X_train_mi = mi_selector.fit_transform(X_train_scaled, self.y_train)
        selected_features_mi = X_train_eng.columns[mi_selector.get_support()].tolist()
        
        self.selection_results['mutual_info'] = {
            'n_features': len(selected_features_mi),
            'features': selected_features_mi,
            'scores': mi_selector.scores_,
            'selector': mi_selector
        }
        
        # 4. Recursive Feature Elimination with Random Forest
        print("   Applying RFE with Random Forest...")
        rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe_selector = RFE(rf_estimator, n_features_to_select=15, step=1)
        X_train_rfe = rfe_selector.fit_transform(X_train_scaled, self.y_train)
        selected_features_rfe = X_train_eng.columns[rfe_selector.get_support()].tolist()
        
        self.selection_results['rfe'] = {
            'n_features': len(selected_features_rfe),
            'features': selected_features_rfe,
            'ranking': rfe_selector.ranking_,
            'selector': rfe_selector
        }
        
        # 5. L1-based feature selection (Lasso)
        print("   Applying L1-based selection...")
        from sklearn.linear_model import LassoCV
        lasso = LassoCV(cv=5, random_state=42)
        lasso_selector = SelectFromModel(lasso)
        X_train_lasso = lasso_selector.fit_transform(X_train_scaled, self.y_train)
        selected_features_lasso = X_train_eng.columns[lasso_selector.get_support()].tolist()
        
        self.selection_results['lasso'] = {
            'n_features': len(selected_features_lasso),
            'features': selected_features_lasso,
            'selector': lasso_selector
        }
        
        # 6. Create consensus features (appear in multiple selection methods)
        all_selected = []
        for method in ['f_test', 'mutual_info', 'rfe', 'lasso']:
            all_selected.extend(self.selection_results[method]['features'])
        
        from collections import Counter
        feature_counts = Counter(all_selected)
        consensus_features = [feature for feature, count in feature_counts.items() if count >= 2]
        
        self.selection_results['consensus'] = {
            'n_features': len(consensus_features),
            'features': consensus_features,
            'feature_counts': feature_counts
        }
        
        print("✅ Feature selection completed")
        
    def apply_dimensionality_reduction(self):
        """Apply PCA and other dimensionality reduction techniques"""
        print("\n🔄 Applying dimensionality reduction...")
        
        X_train_eng = self.engineered_features['X_train']
        X_test_eng = self.engineered_features['X_test']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_test_scaled = scaler.transform(X_test_eng)
        
        # 1. PCA Analysis
        print("   Applying PCA analysis...")
        pca_full = PCA()
        X_train_pca_full = pca_full.fit_transform(X_train_scaled)
        
        # Find number of components for 95% variance
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        n_components_90 = np.argmax(cumsum_var >= 0.90) + 1
        
        # Apply PCA with optimal components
        pca_optimal = PCA(n_components=n_components_95)
        X_train_pca = pca_optimal.fit_transform(X_train_scaled)
        X_test_pca = pca_optimal.transform(X_test_scaled)
        
        self.transformation_results['pca'] = {
            'n_components_90': n_components_90,
            'n_components_95': n_components_95,
            'explained_variance_ratio': pca_optimal.explained_variance_ratio_,
            'X_train_transformed': X_train_pca,
            'X_test_transformed': X_test_pca,
            'scaler': scaler,
            'pca': pca_optimal
        }
        
        print(f"   PCA: {n_components_95} components explain 95% variance (original: {X_train_eng.shape[1]})")
        print("✅ Dimensionality reduction completed")
        
    def evaluate_feature_sets(self):
        """Evaluate different feature sets with models"""
        print("\n📊 Evaluating feature sets...")
        
        X_train_eng = self.engineered_features['X_train']
        X_test_eng = self.engineered_features['X_test']
        
        # Prepare feature sets to evaluate
        feature_sets = {
            'original': self.feature_names,
            'consensus': self.selection_results['consensus']['features'],
            'f_test': self.selection_results['f_test']['features'],
            'mutual_info': self.selection_results['mutual_info']['features'],
            'rfe': self.selection_results['rfe']['features'],
            'lasso': self.selection_results['lasso']['features']
        }
        
        # Add PCA as a special case
        pca_results = self.transformation_results['pca']
        
        # Models to test
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=300, num_leaves=31, max_depth=6, 
                                    learning_rate=0.1, random_state=42, verbose=-1),
            'XGBoost': XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1, 
                                  random_state=42, verbosity=0)
        }
        
        evaluation_results = {}
        
        for feature_set_name, features in feature_sets.items():
            print(f"   Evaluating {feature_set_name} features ({len(features)} features)...")
            
            # Prepare data
            if len(features) > 0:
                X_train_subset = X_train_eng[features]
                X_test_subset = X_test_eng[features]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_subset)
                X_test_scaled = scaler.transform(X_test_subset)
                
                feature_set_results = {}
                
                for model_name, model in models.items():
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, self.y_train, 
                                              cv=5, scoring='neg_mean_squared_error')
                    cv_rmse = np.sqrt(-cv_scores)
                    
                    # Train and test
                    model.fit(X_train_scaled, self.y_train)
                    y_pred = model.predict(X_test_scaled)
                    test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                    test_r2 = r2_score(self.y_test, y_pred)
                    
                    feature_set_results[model_name] = {
                        'cv_rmse_mean': np.mean(cv_rmse),
                        'cv_rmse_std': np.std(cv_rmse),
                        'test_rmse': test_rmse,
                        'test_r2': test_r2,
                        'overfitting_gap': test_rmse - np.mean(cv_rmse)
                    }
                
                evaluation_results[feature_set_name] = feature_set_results
        
        # Evaluate PCA separately
        print(f"   Evaluating PCA features ({pca_results['n_components_95']} components)...")
        X_train_pca = pca_results['X_train_transformed']
        X_test_pca = pca_results['X_test_transformed']
        
        pca_results_eval = {}
        for model_name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_pca, self.y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
            # Train and test
            model.fit(X_train_pca, self.y_train)
            y_pred = model.predict(X_test_pca)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            test_r2 = r2_score(self.y_test, y_pred)
            
            pca_results_eval[model_name] = {
                'cv_rmse_mean': np.mean(cv_rmse),
                'cv_rmse_std': np.std(cv_rmse),
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'overfitting_gap': test_rmse - np.mean(cv_rmse)
            }
        
        evaluation_results['pca'] = pca_results_eval
        
        self.optimized_models = evaluation_results
        print("✅ Feature set evaluation completed")
        
    def create_feature_engineering_visualizations(self):
        """Create comprehensive feature engineering visualizations"""
        print("\n📈 Creating feature engineering visualizations...")
        
        # 1. Feature Selection Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Selection Methods Comparison', fontsize=16, fontweight='bold')
        
        # Number of features selected
        methods = ['f_test', 'mutual_info', 'rfe', 'lasso', 'consensus']
        n_features = [self.selection_results[method]['n_features'] for method in methods]
        
        ax1 = axes[0, 0]
        bars = ax1.bar(methods, n_features, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A'])
        ax1.set_title('Number of Features Selected')
        ax1.set_ylabel('Number of Features')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, n_features):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom')
        
        # Feature overlap analysis
        ax2 = axes[0, 1]
        consensus_counts = self.selection_results['consensus']['feature_counts']
        features_sorted = sorted(consensus_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        features, counts = zip(*features_sorted)
        bars = ax2.barh(range(len(features)), counts, color='#9B59B6')
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features)
        ax2.set_xlabel('Selection Count')
        ax2.set_title('Top Features by Selection Frequency')
        ax2.invert_yaxis()
        
        # PCA Explained Variance
        ax3 = axes[1, 0]
        pca_var = self.transformation_results['pca']['explained_variance_ratio']
        cumsum_var = np.cumsum(pca_var)
        
        ax3.bar(range(1, len(pca_var) + 1), pca_var, alpha=0.7, color='#3498DB', label='Individual')
        ax3.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-', color='#E74C3C', label='Cumulative')
        ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title('PCA Explained Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Feature Engineering Impact
        ax4 = axes[1, 1]
        original_features = len(self.feature_names)
        engineered_features = self.engineered_features['X_train'].shape[1]
        
        categories = ['Original', 'After Engineering']
        values = [original_features, engineered_features]
        
        bars = ax4.bar(categories, values, color=['#34495E', '#2ECC71'])
        ax4.set_title('Feature Engineering Impact')
        ax4.set_ylabel('Number of Features')
        
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Model Performance Comparison with Different Feature Sets
        print("   Creating model performance comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Model Performance with Different Feature Sets', fontsize=16, fontweight='bold')
        
        # Prepare data for visualization
        feature_sets = list(self.optimized_models.keys())
        models = ['Random Forest', 'LightGBM', 'XGBoost']
        
        # Test RMSE comparison
        ax1 = axes[0, 0]
        x = np.arange(len(feature_sets))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            rmse_values = [self.optimized_models[fs][model]['test_rmse'] for fs in feature_sets]
            ax1.bar(x + i*width, rmse_values, width, label=model, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Feature Sets')
        ax1.set_ylabel('Test RMSE')
        ax1.set_title('Test RMSE by Feature Set')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(feature_sets, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² comparison
        ax2 = axes[0, 1]
        for i, model in enumerate(models):
            r2_values = [self.optimized_models[fs][model]['test_r2'] for fs in feature_sets]
            ax2.bar(x + i*width, r2_values, width, label=model, color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Feature Sets')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Test R² by Feature Set')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(feature_sets, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Overfitting gap comparison
        ax3 = axes[1, 0]
        for i, model in enumerate(models):
            gap_values = [self.optimized_models[fs][model]['overfitting_gap'] for fs in feature_sets]
            ax3.bar(x + i*width, gap_values, width, label=model, color=colors[i], alpha=0.8)
        
        ax3.set_xlabel('Feature Sets')
        ax3.set_ylabel('Overfitting Gap (Test - CV RMSE)')
        ax3.set_title('Overfitting Analysis by Feature Set')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(feature_sets, rotation=45)
        ax3.legend()
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # CV Stability (Standard Deviation)
        ax4 = axes[1, 1]
        for i, model in enumerate(models):
            std_values = [self.optimized_models[fs][model]['cv_rmse_std'] for fs in feature_sets]
            ax4.bar(x + i*width, std_values, width, label=model, color=colors[i], alpha=0.8)
        
        ax4.set_xlabel('Feature Sets')
        ax4.set_ylabel('CV RMSE Standard Deviation')
        ax4.set_title('Model Stability by Feature Set')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(feature_sets, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def generate_optimization_insights(self):
        """Generate insights from feature engineering and optimization"""
        print("\n🧠 Generating optimization insights...")
        
        insights = {
            'best_feature_sets': {},
            'engineering_impact': {},
            'selection_insights': {},
            'recommendations': []
        }
        
        # Find best feature sets for each model
        for model in ['Random Forest', 'LightGBM', 'XGBoost']:
            best_rmse = float('inf')
            best_feature_set = None
            
            for feature_set, results in self.optimized_models.items():
                if results[model]['test_rmse'] < best_rmse:
                    best_rmse = results[model]['test_rmse']
                    best_feature_set = feature_set
            
            insights['best_feature_sets'][model] = {
                'feature_set': best_feature_set,
                'test_rmse': best_rmse,
                'test_r2': self.optimized_models[best_feature_set][model]['test_r2'],
                'overfitting_gap': self.optimized_models[best_feature_set][model]['overfitting_gap']
            }
        
        # Engineering impact
        original_features = len(self.feature_names)
        engineered_features = self.engineered_features['X_train'].shape[1]
        insights['engineering_impact']['feature_increase'] = engineered_features - original_features
        insights['engineering_impact']['percentage_increase'] = ((engineered_features - original_features) / original_features) * 100
        
        # Selection insights
        consensus_features = self.selection_results['consensus']['features']
        insights['selection_insights']['consensus_count'] = len(consensus_features)
        insights['selection_insights']['top_consensus'] = consensus_features[:10]
        
        # PCA insights
        pca_components = self.transformation_results['pca']['n_components_95']
        insights['selection_insights']['pca_reduction'] = ((original_features - pca_components) / original_features) * 100
        
        # Generate recommendations
        # Best overall performance
        all_best_rmse = [info['test_rmse'] for info in insights['best_feature_sets'].values()]
        overall_best_rmse = min(all_best_rmse)
        best_model = [model for model, info in insights['best_feature_sets'].items() 
                     if info['test_rmse'] == overall_best_rmse][0]
        
        insights['recommendations'].append(
            f"🏆 Best overall performance: {best_model} with {insights['best_feature_sets'][best_model]['feature_set']} features"
        )
        insights['recommendations'].append(
            f"   RMSE: {overall_best_rmse:.4f}, R²: {insights['best_feature_sets'][best_model]['test_r2']:.4f}"
        )
        
        # Overfitting reduction
        reduced_overfitting = []
        for model, info in insights['best_feature_sets'].items():
            if info['overfitting_gap'] < 0.3:  # Reasonable threshold
                reduced_overfitting.append(model)
        
        if reduced_overfitting:
            insights['recommendations'].append(
                f"✅ Reduced overfitting achieved in: {', '.join(reduced_overfitting)}"
            )
        
        # Feature engineering value
        if insights['engineering_impact']['percentage_increase'] > 50:
            insights['recommendations'].append(
                f"🔧 Feature engineering added {insights['engineering_impact']['feature_increase']} features ({insights['engineering_impact']['percentage_increase']:.1f}% increase)"
            )
        
        # Dimensionality reduction value
        if insights['selection_insights']['pca_reduction'] > 50:
            insights['recommendations'].append(
                f"🔄 PCA can reduce dimensionality by {insights['selection_insights']['pca_reduction']:.1f}% while preserving 95% variance"
            )
        
        # Consensus features value
        if len(consensus_features) > 0:
            insights['recommendations'].append(
                f"🎯 {len(consensus_features)} consensus features identified across multiple selection methods"
            )
        
        # Print insights
        print("\n" + "="*70)
        print("🎯 FEATURE ENGINEERING OPTIMIZATION INSIGHTS")
        print("="*70)
        
        print("\n🏆 BEST FEATURE SETS BY MODEL:")
        for model, info in insights['best_feature_sets'].items():
            print(f"   {model}: {info['feature_set']} (RMSE: {info['test_rmse']:.4f}, R²: {info['test_r2']:.4f})")
        
        print(f"\n🔧 FEATURE ENGINEERING IMPACT:")
        print(f"   Original features: {original_features}")
        print(f"   Engineered features: {engineered_features}")
        print(f"   Increase: +{insights['engineering_impact']['feature_increase']} ({insights['engineering_impact']['percentage_increase']:.1f}%)")
        
        print(f"\n🎯 FEATURE SELECTION INSIGHTS:")
        print(f"   Consensus features: {insights['selection_insights']['consensus_count']}")
        print(f"   Top consensus: {', '.join(insights['selection_insights']['top_consensus'][:5])}")
        print(f"   PCA reduction potential: {insights['selection_insights']['pca_reduction']:.1f}%")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("="*70)
        
        return insights
        
    def save_optimization_results(self):
        """Save all optimization results"""
        print("\n💾 Saving optimization results...")
        
        # Save feature selection results
        for method, results in self.selection_results.items():
            if 'features' in results:
                df = pd.DataFrame({'Feature': results['features']})
                if 'scores' in results:
                    df['Score'] = results['scores'][results['selector'].get_support()]
                df.to_csv(f'feature_selection_{method}.csv', index=False)
        
        # Save model evaluation results
        eval_summary = []
        for feature_set, models in self.optimized_models.items():
            for model_name, metrics in models.items():
                eval_summary.append({
                    'Feature_Set': feature_set,
                    'Model': model_name,
                    'CV_RMSE_Mean': metrics['cv_rmse_mean'],
                    'CV_RMSE_Std': metrics['cv_rmse_std'],
                    'Test_RMSE': metrics['test_rmse'],
                    'Test_R2': metrics['test_r2'],
                    'Overfitting_Gap': metrics['overfitting_gap']
                })
        
        pd.DataFrame(eval_summary).to_csv('feature_optimization_results.csv', index=False)
        
        # Save engineered features
        self.engineered_features['X_train'].to_csv('engineered_features_train.csv', index=False)
        self.engineered_features['X_test'].to_csv('engineered_features_test.csv', index=False)
        
        print("   ✅ All optimization results saved")
        
    def run_full_optimization(self):
        """Run the complete feature engineering optimization pipeline"""
        print("🚀 AGB Feature Engineering Dashboard - Full Optimization")
        print("="*70)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create engineered features
        self.create_engineered_features()
        
        # Apply feature selection
        self.apply_feature_selection()
        
        # Apply dimensionality reduction
        self.apply_dimensionality_reduction()
        
        # Evaluate feature sets
        self.evaluate_feature_sets()
        
        # Create visualizations
        self.create_feature_engineering_visualizations()
        
        # Generate insights
        insights = self.generate_optimization_insights()
        
        # Save results
        self.save_optimization_results()
        
        print("\n🎉 Feature engineering optimization completed successfully!")
        print("="*70)
        
        return insights

if __name__ == "__main__":
    # Initialize and run optimization
    engineering_dashboard = AGBFeatureEngineeringDashboard()
    insights = engineering_dashboard.run_full_optimization()

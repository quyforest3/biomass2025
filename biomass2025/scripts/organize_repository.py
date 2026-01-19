"""
Repository Organization Script for Biomass Estimation

This script systematically organizes the repository files into a professional structure.
Run this script ONCE to organize your files before pushing to GitHub.

Usage:
    python scripts/organize_repository.py --dry-run  # Preview changes
    python scripts/organize_repository.py            # Execute moves
"""

import os
import shutil
from pathlib import Path
import argparse


class RepositoryOrganizer:
    """Organize repository files into professional structure."""
    
    def __init__(self, root_dir='.', dry_run=False):
        self.root = Path(root_dir)
        self.dry_run = dry_run
        self.moves = []
        
    def organize(self):
        """Main organization logic."""
        print("🗂️  Starting repository organization...")
        print(f"📁  Root directory: {self.root.absolute()}")
        print(f"🔍  Dry run: {self.dry_run}\n")
        
        # Data preprocessing scripts
        self.move_data_preprocessing()
        
        # Model training scripts
        self.move_model_scripts()
        
        # Visualization scripts
        self.move_visualization_scripts()
        
        # Dashboard components
        self.move_dashboard_components()
        
        # Notebooks
        self.move_notebooks()
        
        # Scripts and launchers
        self.move_scripts()
        
        # Data files
        self.move_data_files()
        
        # Models
        self.move_models()
        
        # Outputs
        self.move_outputs()
        
        # Summary
        self.print_summary()
        
    def move_file(self, source, destination):
        """Move a file with logging."""
        src = self.root / source
        dst = self.root / destination
        
        if not src.exists():
            print(f"⚠️  SKIP: {source} (not found)")
            return
            
        if dst.exists():
            print(f"⚠️  SKIP: {source} → {destination} (destination exists)")
            return
            
        self.moves.append((source, destination))
        
        if not self.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"✅  {source} → {destination}")
        else:
            print(f"📋  [DRY RUN] {source} → {destination}")
    
    def move_data_preprocessing(self):
        """Move data preprocessing scripts."""
        print("\n📊 Moving data preprocessing scripts...")
        
        files = [
            ('p1.GEDI-preprocess.py', 'src/data_preprocessing/gedi_preprocessing.py'),
            ('p1.GEDI-preprocess-ROI.py', 'src/data_preprocessing/gedi_preprocessing_roi.py'),
            ('p2.Sen1-preprocessed.py', 'src/data_preprocessing/sentinel1_preprocessing.py'),
            ('p2.Sen2-preprocessed.py', 'src/data_preprocessing/sentinel2_preprocessing.py'),
            ('p2. sentinel1_data_extraction_and_cleaning.py', 'src/data_preprocessing/sentinel1_extraction_cleaning.py'),
            ('p2. sentinel2_data_extraction_and_cleaning.py', 'src/data_preprocessing/sentinel2_extraction_cleaning.py'),
            ('p3. dem_data_extraction_with_terrain_analysis.py', 'src/data_preprocessing/dem_extraction_terrain.py'),
            ('p4. merge_gedi_sentinel2_nearest_neighbor.py', 'src/data_preprocessing/merge_gedi_sentinel2.py'),
            ('p5. merge_gedi_sentinel_datasets.py', 'src/data_preprocessing/merge_gedi_sentinel.py'),
            ('p6. merge_gedi_sentinel_dem_datasets.py', 'src/data_preprocessing/merge_gedi_sentinel_dem.py'),
            ('p7. WorldCover_data_extraction_with_terrain_analysis.py', 'src/data_preprocessing/worldcover_extraction.py'),
            ('p8. final merge.py', 'src/data_preprocessing/final_merge.py'),
            ('S1. GEDI_AGBD_ROI_Visualization.py', 'src/data_preprocessing/gedi_visualization.py'),
            ('S2. merge_GEDI_Sentinel2_data.py', 'src/data_preprocessing/merge_gedi_s2.py'),
            ('S3. merge_GEDI_Sentinel1_Sentinel2_data.py', 'src/data_preprocessing/merge_gedi_s1_s2.py'),
            ('S4. merge_GEDI_Sentinel1_Sentinel2_DEM_data.py', 'src/data_preprocessing/merge_gedi_s1_s2_dem.py'),
            ('S5. merge_GEDI_Sentinel1_Sentinel2_DEM_LandCover_data.py', 'src/data_preprocessing/merge_all_sources.py'),
            ('S6. sentinel1_data_extraction.py', 'src/data_preprocessing/sentinel1_extraction.py'),
            ('accessGEDI.py', 'src/data_preprocessing/gedi_access.py'),
            ('collectData.py', 'src/data_preprocessing/data_collection.py'),
        ]
        
        for src, dst in files:
            self.move_file(src, dst)
    
    def move_model_scripts(self):
        """Move model training scripts."""
        print("\n🤖 Moving model training scripts...")
        
        # LightGBM models
        lgbm_files = [
            ('L1. initial LGBM.py', 'src/models/lightgbm/initial_lgbm.py'),
            ('L2. Gridsearch.py', 'src/models/lightgbm/lgbm_gridsearch.py'),
            ('L3. RandomizedSearchCV.py', 'src/models/lightgbm/lgbm_randomized_search.py'),
            ('L9. Finalized Model.py', 'src/models/lightgbm/finalized_lgbm.py'),
            ('M2. AGB_LGBM.py', 'src/models/lightgbm/train_lgbm.py'),
            ('M4. AGB_LGBM_GridSearch.py', 'src/models/lightgbm/lgbm_grid.py'),
            ('M4. AGB_LGBM_Genetic.py', 'src/models/lightgbm/lgbm_genetic.py'),
            ('M21. LGBM GA.py', 'src/models/lightgbm/lgbm_ga.py'),
            ('M21. LGBM TPE.py', 'src/models/lightgbm/lgbm_tpe.py'),
        ]
        
        # Random Forest models
        rf_files = [
            ('L12. Tuned Random Forest Model.py', 'src/models/random_forest/tuned_rf.py'),
            ('L13. RF Feature Importances.py', 'src/models/random_forest/rf_feature_importance.py'),
            ('L14. RF SHAP LIME.py', 'src/models/random_forest/rf_shap_lime.py'),
            ('M1. AGB_RandomForest.py', 'src/models/random_forest/train_rf.py'),
            ('M3. AGB_RandomForest_GridSearch.py', 'src/models/random_forest/rf_gridsearch.py'),
            ('M3. AGB_Genetic.py', 'src/models/random_forest/rf_genetic.py'),
            ('M20. RF TPE.py', 'src/models/random_forest/rf_tpe.py'),
        ]
        
        # XGBoost models
        xgb_files = [
            ('M6. XGBoost Baseline.py', 'src/models/xgboost/train_xgboost.py'),
            ('M6. XGBoost with Gridsearch.py', 'src/models/xgboost/xgboost_gridsearch.py'),
            ('M6. XGBoost with Genetic Algorithm.py', 'src/models/xgboost/xgboost_genetic.py'),
            ('M6. XGBoost with RandomizedSearchCV.py', 'src/models/xgboost/xgboost_randomized.py'),
            ('M18. XGBoostGenetic Algorithm.py', 'src/models/xgboost/xgboost_ga.py'),
            ('M22. XGboost GA.py', 'src/models/xgboost/xgb_ga.py'),
            ('M22. XGboost TPE.py', 'src/models/xgboost/xgb_tpe.py'),
        ]
        
        # SVR models
        svr_files = [
            ('M10.  SVR Baseline.py', 'src/models/svr/train_svr.py'),
            ('M10.  SVR with GridSearchCV.py', 'src/models/svr/svr_gridsearch.py'),
            ('M10.  SVR with RandomizedSearchCV.py', 'src/models/svr/svr_randomized.py'),
            ('M19. SVR GA.py', 'src/models/svr/svr_ga.py'),
            ('M23. SVR TPE.py', 'src/models/svr/svr_tpe.py'),
        ]
        
        # Ensemble and other models
        other_files = [
            ('M5. AGB RF LGBM baseline.py', 'src/models/ensemble/rf_lgbm_baseline.py'),
            ('M7. Stacking Ensemble Method.py', 'src/models/ensemble/stacking_ensemble.py'),
            ('M8. Blending Ensemble Method.py', 'src/models/ensemble/blending_ensemble.py'),
            ('M9. Bagging and Boosting.py', 'src/models/ensemble/bagging_boosting.py'),
            ('M11. DNN.py', 'src/models/deep_learning/dnn.py'),
            ('M12. DNN Hyperparameter tuning.py', 'src/models/deep_learning/dnn_tuning.py'),
            ('M14. Blend RF DNN weight opt.py', 'src/models/ensemble/blend_rf_dnn.py'),
        ]
        
        # Feature analysis
        feature_files = [
            ('L7.  Feature Importance.py', 'src/models/analysis/feature_importance.py'),
            ('L8. biomass_feature_selection.py', 'src/models/analysis/feature_selection.py'),
            ('L10.  Cross-Validation.py', 'src/models/analysis/cross_validation.py'),
            ('L11. SHAP and LIME.py', 'src/models/analysis/shap_lime.py'),
            ('M16. FI & SHAP.py', 'src/models/analysis/fi_shap.py'),
            ('M17. LIME.py', 'src/models/analysis/lime_analysis.py'),
        ]
        
        all_files = lgbm_files + rf_files + xgb_files + svr_files + other_files + feature_files
        
        for src, dst in all_files:
            self.move_file(src, dst)
    
    def move_visualization_scripts(self):
        """Move visualization scripts."""
        print("\n📊 Moving visualization scripts...")
        
        files = [
            ('visualization simple.py', 'src/visualization/simple_viz.py'),
            ('visualization+.py', 'src/visualization/advanced_viz.py'),
            ('Spider chart.py', 'src/visualization/spider_chart.py'),
            ('Histogram of AGB Values.py', 'src/visualization/agb_histogram.py'),
            ('M15. almost all VIS.py', 'src/visualization/comprehensive_viz.py'),
            ('M24. TREE_based VIS.py', 'src/visualization/tree_based_viz.py'),
            ('L6. plot learning curves.py', 'src/visualization/learning_curves.py'),
        ]
        
        for src, dst in files:
            self.move_file(src, dst)
    
    def move_dashboard_components(self):
        """Move dashboard components."""
        print("\n🎨 Moving dashboard components...")
        
        files = [
            ('dashboard_streamlit_app.py', 'src/dashboard/app.py'),
            ('dashboard_core.py', 'src/dashboard/core.py'),
            ('dashboard_feature_analysis.py', 'src/dashboard/feature_analysis.py'),
            ('dashboard_feature_engineering.py', 'src/dashboard/feature_engineering.py'),
            ('dashboard_model_diagnostics.py', 'src/dashboard/model_diagnostics.py'),
            ('AGB_Dashboard.py', 'src/dashboard/legacy_dashboard.py'),
        ]
        
        for src, dst in files:
            self.move_file(src, dst)
    
    def move_notebooks(self):
        """Move Jupyter notebooks."""
        print("\n📓 Moving notebooks...")
        
        files = [
            ('accessGEDI.ipynb', 'notebooks/01_gedi_access.ipynb'),
            ('collectData.ipynb', 'notebooks/02_data_collection.ipynb'),
        ]
        
        for src, dst in files:
            self.move_file(src, dst)
    
    def move_scripts(self):
        """Move utility scripts."""
        print("\n🛠️  Moving scripts...")
        
        files = [
            ('🚀_ONE_CLICK_LAUNCH.bat', 'scripts/one_click_launch.bat'),
            ('launch_dashboard.py', 'scripts/launch_dashboard.py'),
            ('launch_dashboard.bat', 'scripts/launch_dashboard.bat'),
            ('auto_launch.py', 'scripts/auto_launch.py'),
            ('run_dashboard.bat', 'scripts/run_dashboard.bat'),
            ('run_dashboard_auto.ps1', 'scripts/run_dashboard_auto.ps1'),
        ]
        
        for src, dst in files:
            self.move_file(src, dst)
    
    def move_data_files(self):
        """Move data files."""
        print("\n💾 Moving data files...")
        print("⚠️  Large files will be handled by .gitignore")
        
        # Note: Most data files should stay in place or be gitignored
        # Only move if you want to restructure
        
    def move_models(self):
        """Move trained models."""
        print("\n🤖 Moving trained models...")
        print("⚠️  Model files will be handled by .gitignore")
        
    def move_outputs(self):
        """Move output files."""
        print("\n📤 Moving output files...")
        print("⚠️  Output files will be handled by .gitignore")
        
    def print_summary(self):
        """Print summary of operations."""
        print("\n" + "="*60)
        print(f"📊 SUMMARY")
        print("="*60)
        print(f"Total files to move: {len(self.moves)}")
        
        if self.dry_run:
            print("\n⚠️  This was a DRY RUN - no files were actually moved")
            print("💡 Run without --dry-run to execute the moves")
        else:
            print("\n✅ Files have been organized successfully!")
            print("\n📝 Next steps:")
            print("  1. Review the changes")
            print("  2. Update imports in your Python files if needed")
            print("  3. Test the dashboard: streamlit run src/dashboard/app.py")
            print("  4. Commit changes: git add . && git commit -m 'Reorganize repository structure'")
            print("  5. Push to GitHub: git push origin main")


def main():
    parser = argparse.ArgumentParser(description='Organize Biomass Estimation repository')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without executing')
    parser.add_argument('--root', default='.', help='Root directory (default: current directory)')
    
    args = parser.parse_args()
    
    organizer = RepositoryOrganizer(root_dir=args.root, dry_run=args.dry_run)
    organizer.organize()


if __name__ == '__main__':
    main()


#!/usr/bin/env python3

import argparse
import yaml
import os
import joblib
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, randint
from tqdm import tqdm
import numpy as np
from tlpath.utils import (
    create_telomere_dict,
    build_dataset,
    process_all_tissues,
    save_results,
    save_result_summary
)

class TLPathConfig:
    """Configuration class for TLPath model."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Default configuration
        self.SEED = 42
        self.N_SPLITS_OUTER = 5
        self.N_SPLITS_INNER = 3
        self.TEST_SIZE = 0.2
        self.SEEDS = [1, 2, 3, 4, 5]
        self.MIN_SAMPLES_PER_TISSUE = 70
        self.TISSUES_TO_SKIP = []
        self.N_ITER = 20  # Number of parameter settings sampled in RandomizedSearchCV
        
        # Model hyperparameters - now using parameter distributions for RandomizedSearchCV
        self.RF_PARAM_DISTRIBUTIONS = {
            'n_estimators': randint(50, 200),
            'max_depth': [None, 10, 20],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 4),
            'max_features': ['sqrt', 'log2']
        }
        
        # Load custom configuration if provided
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update configuration parameters
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)


class TLPath:
    """Main class for TLPath model training and evaluation."""
    
    def __init__(self, config: TLPathConfig, 
                 telomere_file: Path,
                 UNI_features_dir: Path,
                 output_dir: Optional[Path] = None):
        self.config = config
        self.telomere_file = Path(telomere_file)
        self.UNI_features_dir = Path(UNI_features_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("results/TLPath")
        self.results_dir = self.output_dir / "results"
        self.models_dir = self.output_dir / "models"
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for results and models."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _train_fold(self, X_train: np.ndarray, y_train: np.ndarray, seed: int) -> Dict[str, Any]:
        """Train a single fold with randomized hyperparameter search."""
        inner_cv = KFold(n_splits=self.config.N_SPLITS_INNER, shuffle=True, random_state=seed)
        random_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            param_distributions=self.config.RF_PARAM_DISTRIBUTIONS,
            n_iter=self.config.N_ITER,
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=1,  # Use 1 for RandomizedSearchCV but -1 for the model
            verbose=0
        )
        random_search.fit(X_train, y_train)
        
        return {
            "best_model": random_search.best_estimator_,
            "best_params": random_search.best_params_,
            "best_score": float(random_search.best_score_)
        }
    
    def _evaluate_fold(self, model: RandomForestRegressor, X: np.ndarray, y: np.ndarray,
                       fold: int, seed: int, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate model performance on a single fold."""
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        pearson_r_value, _ = pearsonr(y, y_pred)
        
        result = {
            "fold_index": fold,
            "seed": seed,
            "val_mse": float(mse),
            "val_r2": float(r2),
            "pearson_r": float(pearson_r_value),
            "predictions": y_pred.tolist()
        }
        
        # Include feature importances if feature names are provided
        if feature_names is not None:
            result["feature_importances"] = dict(zip(feature_names, 
                                                    model.feature_importances_.astype(float)))
        
        return result
    
    def find_best_model(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best model configuration based on validation scores."""
        if not cv_results:
            return None
        
        # Sort by pearson_r (higher is better)
        sorted_results = sorted(cv_results, key=lambda x: x["pearson_r"], reverse=True)
        return sorted_results[0]
    
    def train_model(self, X: np.ndarray, y: np.ndarray, tissue_name: str, 
                   sample_ids: List[str] = None, feature_names: List[str] = None) -> Dict[str, Any]:
        """Train Random Forest model with nested cross-validation and sample tracking."""
        results = {
            "tissue": tissue_name,
            "seed_results": [],
            "overall_summary": {},
            "test_set": {},
            "sample_predictions": {}
        }
        
        # Split into train and test sets, tracking sample IDs
        if sample_ids is not None:
            X_train_full, X_test, y_train_full, y_test, sample_ids_train, sample_ids_test = train_test_split(
                X, y, sample_ids, test_size=self.config.TEST_SIZE, random_state=self.config.SEED
            )
            
            # Initialize predictions dictionary for each test sample
            for sample_id, true_val in zip(sample_ids_test, y_test):
                results["sample_predictions"][sample_id] = {
                    "true_value": float(true_val),
                    "predictions": []
                }
        else:
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=self.config.TEST_SIZE, random_state=self.config.SEED
            )
            sample_ids_test = None
        
        # Dictionary to store trained models
        trained_models = {}
        all_outer_folds = []
        
        # Nested cross-validation with tqdm progress bars
        seed_pbar = tqdm(self.config.SEEDS, desc=f"Training {tissue_name}", position=0)
        
        for seed in seed_pbar:
            seed_pbar.set_description(f"Seed {seed}")
            seed_results = {"seed": seed, "outer_folds": []}
            outer_cv = KFold(n_splits=self.config.N_SPLITS_OUTER, shuffle=True, random_state=seed)
            
            fold_pbar = tqdm(enumerate(outer_cv.split(X_train_full)), 
                           total=self.config.N_SPLITS_OUTER, 
                           desc="Outer Folds", 
                           position=1, 
                           leave=False)
            
            for fold, (train_idx, val_idx) in fold_pbar:
                fold_pbar.set_description(f"Fold {fold+1}/{self.config.N_SPLITS_OUTER}")
                
                X_train_fold, X_val = X_train_full[train_idx], X_train_full[val_idx]
                y_train_fold, y_val = y_train_full[train_idx], y_train_full[val_idx]
                
                # Train the model on this fold
                train_result = self._train_fold(X_train_fold, y_train_fold, seed)
                best_model = train_result["best_model"]
                
                # Evaluate on validation set
                fold_results = self._evaluate_fold(best_model, X_val, y_val, fold, seed, feature_names)
                fold_results.update({
                    "best_params": train_result["best_params"],
                    "best_score": train_result["best_score"]
                })
                
                # Make predictions on test set for this fold's model
                y_pred_test_fold = best_model.predict(X_test)
                
                # Store predictions for each test sample for this fold
                if sample_ids_test is not None:
                    for sample_id, pred_val in zip(sample_ids_test, y_pred_test_fold):
                        results["sample_predictions"][sample_id]["predictions"].append(float(pred_val))
                
                # Store model and results
                trained_models[(seed, fold)] = best_model
                seed_results["outer_folds"].append(fold_results)
                all_outer_folds.append(fold_results)
                
                fold_pbar.set_postfix({
                    'MSE': f'{fold_results["val_mse"]:.4f}',
                    'R²': f'{fold_results["val_r2"]:.4f}'
                })
            
            results["seed_results"].append(seed_results)
        
        # Calculate average predictions for each sample
        if sample_ids_test is not None:
            for sample_id in results["sample_predictions"]:
                predictions = results["sample_predictions"][sample_id]["predictions"]
                results["sample_predictions"][sample_id]["mean_prediction"] = float(np.mean(predictions))
                results["sample_predictions"][sample_id]["std_prediction"] = float(np.std(predictions))
        
        # Find the best model from cross-validation
        best_result = self.find_best_model(all_outer_folds)
        
        # Train final model using best parameters from CV
        if best_result is not None:
            best_params = best_result["best_params"]
            final_model = RandomForestRegressor(**best_params, random_state=self.config.SEED, n_jobs=-1)
            final_model.fit(X_train_full, y_train_full)
            
            # Evaluate final model on test set
            y_pred_test = final_model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            test_pearson_r, _ = pearsonr(y_test, y_pred_test)
            
            # Save test set results
            results["test_set"] = {
                "sample_ids": sample_ids_test,
                "y_test": y_test.tolist(),
                "y_pred_test": y_pred_test.tolist(),
                "test_mse": float(test_mse),
                "test_r2": float(test_r2),
                "test_pearson_r": float(test_pearson_r)
            }
            
            results["overall_summary"] = {
                "best_parameters": best_params,
                "feature_importances": dict(zip(range(X.shape[1]) if feature_names is None else feature_names, 
                                              final_model.feature_importances_.astype(float)))
            }
            
            # Save the final model
            model_path = self.models_dir / f"{tissue_name}_model.joblib"
            joblib.dump(final_model, model_path)
            results["overall_summary"]["model_path"] = str(model_path)
            
        else:
            print(f"Warning: No best model found for {tissue_name}")
        
        # Save results to file
        results_path = self.results_dir / f"{tissue_name}_results.pkl"
        save_results(results, results_path)
        
        return results
    
    def run_pipeline(self):
        """Run the complete TLPath pipeline."""
        # Load telomere data
        print("Loading telomere data...")
        telomere_df = pd.read_csv(self.telomere_file)
        
        # Process features
        print("Loading features...")
        telomere_dict = create_telomere_dict(telomere_df)
        GTEx_dataset = process_all_tissues(self.UNI_features_dir, telomere_dict)

        # Get tissue list
        tissue_list = telomere_df['TissueSiteDetail'].unique()
        if self.config.TISSUES_TO_SKIP:
            tissue_list = [t for t in tissue_list if t not in self.config.TISSUES_TO_SKIP]
        
        # Build dataset
        print("Building datasets...")
        datasets = build_dataset(
            GTEx_dataset,
            telomere_dict
        )
        
        # Train and evaluate models for each tissue
        all_results = {}
        tissue_pbar = tqdm(datasets.items(), desc="Processing tissues", position=0)
        
        for tissue, data in tissue_pbar:
            tissue_pbar.set_description(f"Processing tissue: {tissue}")
            
            try:
                # Extract sample IDs and feature names if available
                sample_ids = data.get("sample_ids", None)
                feature_names = data.get("feature_names", None)
                
                results = self.train_model(
                    data["X"],
                    data["y"],
                    tissue,
                    sample_ids,
                    feature_names
                )
                
                all_results[tissue] = results
                
                # Show test metrics in progress bar
                if "test_set" in results and results["test_set"]:
                    tissue_pbar.set_postfix({
                        'Test MSE': f'{results["test_set"]["test_mse"]:.4f}',
                        'Test R²': f'{results["test_set"]["test_r2"]:.4f}'
                    })
                
            except Exception as e:
                print(f"Error processing tissue {tissue}: {str(e)}")
        
        return all_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TLPath: Telomere Length Prediction from Histopathological Images")
    parser.add_argument("--telomere-file", required=True, type=str,
                      help="Path to telomere length data CSV file")
    parser.add_argument("--features_dir", required=True, type=str,
                      help="Directory containing patch features")
    parser.add_argument("--output-dir", type=str, default="results/TLPath",
                      help="Directory to save results and models")
    parser.add_argument("--config", type=str,
                      help="Path to configuration YAML file")
    parser.add_argument("--tissues-to-skip", type=str, nargs="*",
                      help="List of tissue types to exclude from analysis")
    return parser.parse_args()

def main():
    """Main entry point for TLPath."""
    args = parse_args()
    
    # Initialize configuration
    config = TLPathConfig(args.config)
    if args.tissues_to_skip:
        config.TISSUES_TO_SKIP = args.tissues_to_skip
    
    # Create TLPath instance
    tlpath = TLPath(
        config=config,
        telomere_file=args.telomere_file,
        UNI_features_dir=args.features_dir,
        output_dir=args.output_dir
    )
    
    # Run the pipeline
    results = tlpath.run_pipeline()
    
    # Save final results
    save_result_summary(results, args.output_dir)    
    print("TLPath pipeline completed successfully!")

if __name__ == "__main__":
    main()
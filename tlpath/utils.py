import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Any
import pickle
import json
import os
import base64
import io
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor
import joblib
import tqdm

def read_tissue_features(file_path, telomere_dict):
    """
    Read a tissue CSV file and convert base64 encoded features back to numpy arrays,
    filtering for samples that have telomere data.
    
    Args:
        file_path: Path to the CSV file
        telomere_dict: Dictionary with sample_ids as keys and telomere values
        
    Returns:
        Dictionary with sample_id as keys and features as numpy array values,
        only for samples that have telomere data
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert base64 strings back to numpy arrays
        features_dict = {}
        for _, row in df.iterrows():
            sample_id = row['sample_id']
            
            # Only process samples that have telomere data
            if sample_id in telomere_dict:
                # Decode base64 string
                binary_data = base64.b64decode(row['features'])
                # Load numpy array from binary data
                buffer = io.BytesIO(binary_data)
                features_array = np.load(buffer)
                features_dict[sample_id] = features_array
        
        return features_dict
    
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")


def process_all_tissues(directory_path, telomere_dict):
    """
    Process all tissue CSV files in the given directory and organize results
    in a dictionary where keys are tissue names (same as file names without extension)
    and values are the corresponding feature dictionaries.
    
    Args:
        directory_path: Path to the directory containing tissue CSV files
        telomere_dict: Dictionary with sample_ids as keys and telomere values
        
    Returns:
        Dictionary where keys are tissue names and values are dictionaries of features
    """
    tissues_data = {}
    
    try:
        # Get all CSV files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                # Extract tissue name (filename without extension)
                tissue_name = os.path.splitext(filename)[0]
                
                # Full path to the file
                file_path = os.path.join(directory_path, filename)
                
                # Process the file and store results
                tissue_features = read_tissue_features(file_path, telomere_dict)
                tissues_data[tissue_name] = tissue_features
                
                print(f"Processed {tissue_name}: {len(tissue_features)} samples with telomere data")
        
        return tissues_data
    
    except Exception as e:
        raise Exception(f"Error processing tissues directory: {str(e)}")    

def create_telomere_dict(telomere_df: pd.DataFrame) -> Dict[str, float]:
    """
    Creates a dictionary mapping sample IDs to telomere lengths.

    Args:
        telomere_df(pd.DataFrame): DataFrame containing telomere data.

    Returns:
        Dict[str, float]: Dictionary mapping sample IDs to TQImean (telomere length).
    """
    return {
        row["Sample.ID.for.Public.Release"].split("-SM")[0]: row["TQImean"]
        for _, row in telomere_df.iterrows()
    }

def build_dataset(feature_dict, telomere_dict):
    """
    Build a preprocessed dataset for the random_forest_regression_cv_optimised function
    
    Parameters:
    - feature_dict: Dictionary of tissues with sample_id to feature array mapping
    - telomere_dict: Dictionary of sample_id to telomere_length
    
    Returns:
    - preprocessed_data: Dictionary in the format expected by the model function
    """
    preprocessed_data = {}
    
    # Process each tissue
    for tissue, samples in feature_dict.items():
        # Initialize data structure for this tissue
        tissue_data = {
            "X": [],
            "y": [],
            "sample_ids": [],
            "feature_names": None  # Will be set if feature names are available
        }
        
        # For each sample in this tissue
        for sample_id, feature_array in samples.items():
            # Check if this sample has a corresponding telomere length
            if sample_id in telomere_dict:
                # Add the feature array to X
                tissue_data["X"].append(feature_array)
                # Add the telomere length to y
                tissue_data["y"].append(telomere_dict[sample_id])
                # Keep track of the sample ID
                tissue_data["sample_ids"].append(sample_id)
        
        # Check if we have any valid samples for this tissue
        if len(tissue_data["X"]) > 0:
            # Convert lists to numpy arrays
            tissue_data["X"] = np.array(tissue_data["X"])
            tissue_data["y"] = np.array(tissue_data["y"])
            tissue_data["sample_ids"] = np.array(tissue_data["sample_ids"])
            
            # Generate sequential feature names if not available
            tissue_data["feature_names"] = [f"feature_{i}" for i in range(tissue_data["X"].shape[1])]
            
            # Add to the preprocessed data
            preprocessed_data[tissue] = tissue_data
            print(f"Processed {tissue}: {len(tissue_data['X'])} valid samples")
        else:
            print(f"Warning: No valid samples found for {tissue}")
            
    return preprocessed_data

def save_results(results: Dict[str, Any], results_path: Path) -> None:
    """
    Save individual tissue model results to a pickle file.
    
    Args:
        results: Dictionary containing model results for a tissue
        results_path: Path where the results will be saved
    """
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")

def save_result_summary(results_dict, output_path):
    """
    Save a summary of results for all tissues to a CSV file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each tissue from TLPath pipeline
    output_path : str or Path
        Path where the CSV summary will be saved
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the summary results
    """
    import pandas as pd
    from pathlib import Path
    
    # Create lists to store data
    tissues = []
    test_mse = []
    test_r2 = []
    test_pearson_r = []
    cv_pearson_r = []
    sample_count = []
    best_params = []
    
    # Extract data for each tissue
    for tissue, result in results_dict.items():
        tissues.append(tissue)
        
        # Extract test set metrics
        if "test_set" in result and result["test_set"]:
            test_mse.append(result["test_set"].get("test_mse", float('nan')))
            test_r2.append(result["test_set"].get("test_r2", float('nan')))
            test_pearson_r.append(result["test_set"].get("test_pearson_r", float('nan')))
            
            # Count samples
            if "y_test" in result["test_set"]:
                test_count = len(result["test_set"]["y_test"])
                train_count = len(result["test_set"].get("y_pred_test", [])) * 4  # Assuming 80/20 split
                sample_count.append(train_count + test_count)
            else:
                sample_count.append(float('nan'))
        else:
            test_mse.append(float('nan'))
            test_r2.append(float('nan'))
            test_pearson_r.append(float('nan'))
            sample_count.append(float('nan'))
        
        # Extract cross-validation pearson_r (average across all folds)
        fold_pearson_values = []
        for seed_result in result.get("seed_results", []):
            for fold_result in seed_result.get("outer_folds", []):
                if "pearson_r" in fold_result:
                    fold_pearson_values.append(fold_result["pearson_r"])
        
        if fold_pearson_values:
            cv_pearson_r.append(sum(fold_pearson_values) / len(fold_pearson_values))
        else:
            cv_pearson_r.append(float('nan'))
        
        # Extract best parameters
        if "overall_summary" in result and "best_parameters" in result["overall_summary"]:
            best_params.append(str(result["overall_summary"]["best_parameters"]))
        else:
            best_params.append("")
    
    # Create DataFrame
    summary_df = pd.DataFrame({
        'Tissue': tissues,
        'Sample_Count': sample_count,
        'Test_MSE': test_mse,
        'Test_R2': test_r2,
        'Test_Pearson_R': test_pearson_r,
        'CV_Pearson_R': cv_pearson_r,
        'Best_Parameters': best_params
    })
    
    # Sort by Test_Pearson_R (descending)
    summary_df = summary_df.sort_values(by='Test_Pearson_R', ascending=False)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    
    print(f"Results summary saved to {output_path}")
    return summary_df

def save_result_summary(results_dict, output_dir):
    """
    Save a summary of results for all tissues to a CSV file in the specified output directory.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each tissue from TLPath pipeline
    output_dir : str or Path
        Directory where the CSV summary will be saved
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the summary results
    """
    import pandas as pd
    from pathlib import Path
    
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    output_path = output_dir / "model_summary.csv"
    
    # Create lists to store data
    tissues = []
    test_mse = []
    test_r2 = []
    test_pearson_r = []
    cv_pearson_r = []
    sample_count = []
    best_params = []
    
    # Extract data for each tissue
    for tissue, result in results_dict.items():
        tissues.append(tissue)
        
        # Extract test set metrics
        if "test_set" in result and result["test_set"]:
            test_mse.append(result["test_set"].get("test_mse", float('nan')))
            test_r2.append(result["test_set"].get("test_r2", float('nan')))
            test_pearson_r.append(result["test_set"].get("test_pearson_r", float('nan')))
            
            # Count samples
            if "y_test" in result["test_set"]:
                test_count = len(result["test_set"]["y_test"])
                train_count = len(result["test_set"].get("y_pred_test", [])) * 4  # Assuming 80/20 split
                sample_count.append(train_count + test_count)
            else:
                sample_count.append(float('nan'))
        else:
            test_mse.append(float('nan'))
            test_r2.append(float('nan'))
            test_pearson_r.append(float('nan'))
            sample_count.append(float('nan'))
        
        # Extract cross-validation pearson_r (average across all folds)
        fold_pearson_values = []
        for seed_result in result.get("seed_results", []):
            for fold_result in seed_result.get("outer_folds", []):
                if "pearson_r" in fold_result:
                    fold_pearson_values.append(fold_result["pearson_r"])
        
        if fold_pearson_values:
            cv_pearson_r.append(sum(fold_pearson_values) / len(fold_pearson_values))
        else:
            cv_pearson_r.append(float('nan'))
        
        # Extract best parameters
        if "overall_summary" in result and "best_parameters" in result["overall_summary"]:
            best_params.append(str(result["overall_summary"]["best_parameters"]))
        else:
            best_params.append("")
    
    # Create DataFrame
    summary_df = pd.DataFrame({
        'Tissue': tissues,
        'Sample_Count': sample_count,
        'Test_MSE': test_mse,
        'Test_R2': test_r2,
        'Test_Pearson_R': test_pearson_r,
        'CV_Pearson_R': cv_pearson_r,
        'Best_Parameters': best_params
    })
    
    # Sort by Test_Pearson_R (descending)
    summary_df = summary_df.sort_values(by='Test_Pearson_R', ascending=False)
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    
    print(f"Results summary saved to {output_path}")
    return summary_df

def find_best_model(cv_results: List[Dict[str, Any]]) -> tuple:
    """
    Find the best model configuration based on mean squared error.
    
    Args:
        cv_results: List of dictionaries containing cross-validation results
    
    Returns:
        tuple: (best_seed, best_fold) indicating the configuration of the best model
    """
    # Sort by MSE (lower is better)
    sorted_results = sorted(cv_results, key=lambda x: x["mse"])
    
    # Return the seed and fold of the best model
    if sorted_results:
        return sorted_results[0]["seed"], sorted_results[0]["fold"]
    return None, None



def predict_telomere_length(models_dir, features: np.ndarray, tissue_name: str) -> np.ndarray:
    """
    Run inference on a previously saved model for the specified tissue.
    
    Parameters:
    -----------
    features : np.ndarray
        Array of feature values with shape (n_samples, n_features)
    tissue_name : str
        Name of the tissue type for which to load the model
        
    Returns:
    --------
    np.ndarray
        Predicted telomere lengths for the input features
    
    Raises:
    -------
    FileNotFoundError
        If the model file for the specified tissue doesn't exist
    ValueError
        If the input features don't match the expected format
    """
    import os
    
    # Construct the model path
    model_path = os.path.join(models_dir, f"{tissue_name}_model.joblib")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for tissue '{tissue_name}' at {model_path}")
    
    # Load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model for tissue '{tissue_name}': {str(e)}")
    
    # Validate input features
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    
    if len(features.shape) == 1:
        # Single sample - reshape to 2D
        features = features.reshape(1, -1)
    
    # Check if features have the right dimensionality
    expected_features = model.n_features_in_
    if features.shape[1] != expected_features:
        raise ValueError(f"Model expects {expected_features} features, but got {features.shape[1]}")
    
    # Run prediction
    predictions = model.predict(features)
    
    return predictions

def predict_all_tissues(models_dir, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Run inference on multiple tissues using their respective models.
    
    Parameters:
    -----------
    features_dict : Dict[str, np.ndarray]
        Dictionary mapping tissue names to feature arrays
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping tissue names to their predictions
    """
    results = {}
    
    for tissue_name, features in features_dict.items():
        try:
            predictions = predict_telomere_length(models_dir, features, tissue_name)
            results[tissue_name] = predictions
        except Exception as e:
            print(f"Error predicting for tissue '{tissue_name}': {str(e)}")
            # Store the error in results
            results[tissue_name] = str(e)
    
    return results

def predict_batch(models_dir, batch_features: np.ndarray, tissue_name: str, batch_size: int = 32) -> np.ndarray:
    """
    Run batch inference on a large dataset for memory efficiency.
    
    Parameters:
    -----------
    batch_features : np.ndarray
        Array of feature values with shape (n_samples, n_features)
    tissue_name : str
        Name of the tissue type for which to load the model
    batch_size : int, default=32
        Number of samples to process in each batch
        
    Returns:
    --------
    np.ndarray
        Predicted telomere lengths for all input features
    """
    # Construct the model path
    model_path = models_dir / f"{tissue_name}_model.joblib"
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for tissue '{tissue_name}' at {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Convert to numpy array if not already
    if not isinstance(batch_features, np.ndarray):
        batch_features = np.array(batch_features)
    
    # Ensure 2D
    if len(batch_features.shape) == 1:
        batch_features = batch_features.reshape(1, -1)
    
    # Check dimensions
    if batch_features.shape[1] != model.n_features_in_:
        raise ValueError(f"Model expects {model.n_features_in_} features, but got {batch_features.shape[1]}")
    
    # Get total number of samples
    n_samples = batch_features.shape[0]
    
    # Initialize array for predictions
    all_predictions = np.zeros(n_samples)
    
    # Process in batches with progress bar
    for i in tqdm(range(0, n_samples, batch_size), desc=f"Predicting {tissue_name}"):
        end_idx = min(i + batch_size, n_samples)
        batch = batch_features[i:end_idx]
        predictions = model.predict(batch)
        all_predictions[i:end_idx] = predictions
    
    return all_predictions
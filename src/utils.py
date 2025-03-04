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

MIN_SAMPLES_PER_TISSUE = 70

def load_patch_features(
    uni_data_dir: str, telomere_dict: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """Loads patch features from .npy files.

    Args:
        uni_data_dir: Directory containing the GTEx UNI data.
        telomere_dict: Dictionary mapping sample IDs to telomere lengths.

    Returns:
        A dictionary mapping sample IDs to patch feature arrays.
    """
    all_tissue_patch_features = {}

    for folder_name in sorted(telomere_dict): #Ensuring deterministic output
        if folder_name.startswith("GTEX"):
            features_folder = os.path.join(uni_data_dir, folder_name, "_features")
            if os.path.exists(features_folder):
                expected_file = f"{folder_name}-uni.npy"
                npy_file = os.path.join(features_folder, expected_file)
                if os.path.isfile(npy_file):
                    try:
                        data = np.load(npy_file)
                        all_tissue_patch_features[folder_name] = data
                        print(f"Read data from {npy_file}, shape: {data.shape}")
                    except Exception as e:
                        print(f"Error loading {npy_file}: {e}")
                else:
                    print(f"Expected file {expected_file} not found in {features_folder}.")
            else:
                print(f"Features folder not found in {folder_name}.")

    return all_tissue_patch_features

def load_metadata(
    metadata_file: str, telomere_data_file: str, phenotype_data_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads GTEx metadata, telomere data, and phenotype data.

    Args:
        metadata_file: Path to the GTEx metadata file.
        telomere_data_file: Path to the GTEx telomere data file.
        phenotype_data_file: Path to the GTEx phenotype data file.

    Returns:
        A tuple containing the metadata, telomere, and phenotype DataFrames.
    """
    try:
        metadata = pd.read_csv(metadata_file, sep="\t", on_bad_lines="skip")
        telomere_df = pd.read_csv(telomere_data_file, sep=",")
        gtex_phenotype_df = pd.read_csv(phenotype_data_file, sep="\t")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Data file not found. {e}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error: Could not parse data file. {e}")

    return metadata, telomere_df, gtex_phenotype_df


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


def get_age_sex_covariates(gtex_phenotype_df, features_dict):
    """
    Adds age and sex covariates to the feature data of each sample in a new dictionary.

    Args:
        gtex_phenotype_df (pd.DataFrame): DataFrame containing GTEx phenotype information with columns 'SUBJID', 'AGE', and 'SEX'.
        features_dict (Dict[str, np.ndarray]): Dictionary with sample IDs as keys and feature arrays as values.

    Returns:
        Dict[str, np.ndarray]: New dictionary with updated feature arrays including age and sex covariates.
    """
    missing_samples = []  # To store sample_ids with missing SUBJID matches
    missing_sex_age = []  # To store sample_ids with missing or NaN SEX/AGE
    updated_features_dict = {}  # New dictionary for updated features

    for sample_id, features in features_dict.items():
        # Extract SUBJID part from the sample_id
        subjid = "-".join(sample_id.split("-")[:2])
        
        # Check for matches in gtex_phenotype_df
        matching_rows = gtex_phenotype_df[gtex_phenotype_df['SUBJID'] == subjid]
        
        if not matching_rows.empty:
            # Retrieve the index of the matched row
            matched_index = matching_rows.index[0]
            
            # Extract AGE and SEX
            age = matching_rows.loc[matched_index, 'AGE']
            sex = matching_rows.loc[matched_index, 'SEX']
            
            # Check if AGE or SEX is missing (NaN or invalid)
            if pd.isna(age) or pd.isna(sex):
                missing_sex_age.append(sample_id)
                updated_features_dict[sample_id] = features  # Keep original features
            else:
                # Add AGE and SEX to a new array
                updated_features_dict[sample_id] = np.append(features, [age, sex])
        else:
            # Add sample_id to the missing list if no match is found
            missing_samples.append(sample_id)
            updated_features_dict[sample_id] = features  # Keep original features
    
    # Display missing information
    print("Sample IDs with no matching SUBJID:", missing_samples)
    print("Sample IDs with missing or NaN AGE or SEX:", missing_sex_age)
    
    return updated_features_dict

def get_age_covariates(gtex_phenotype_df, features_dict):
    """
    Adds age and sex covariates to the feature data of each sample in a new dictionary.

    Args:
        gtex_phenotype_df (pd.DataFrame): DataFrame containing GTEx phenotype information with columns 'SUBJID', 'AGE', and 'SEX'.
        features_dict (Dict[str, np.ndarray]): Dictionary with sample IDs as keys and feature arrays as values.

    Returns:
        Dict[str, np.ndarray]: New dictionary with updated feature arrays including age and sex covariates.
    """
    missing_samples = []  # To store sample_ids with missing SUBJID matches
    updated_features_dict = {}  # New dictionary for updated features

    for sample_id, features in features_dict.items():
        # Extract SUBJID part from the sample_id
        subjid = "-".join(sample_id.split("-")[:2])
        
        # Check for matches in gtex_phenotype_df
        matching_rows = gtex_phenotype_df[gtex_phenotype_df['SUBJID'] == subjid]
        
        if not matching_rows.empty:
            # Retrieve the index of the matched row
            matched_index = matching_rows.index[0]
            
            # Extract AGE and SEX
            age = matching_rows.loc[matched_index, 'AGE']
            
            # Check if AGE or SEX is missing (NaN or invalid)
            if pd.isna(age) :
                updated_features_dict[sample_id] = features  # Keep original features
            else:
                # Add AGE and SEX to a new array
                updated_features_dict[sample_id] = np.append(features, [age])
        else:
            # Add sample_id to the missing list if no match is found
            missing_samples.append(sample_id)
            updated_features_dict[sample_id] = features  # Keep original features
    
    # Display missing information
    print("Sample IDs with no matching SUBJID:", missing_samples)
    
    return updated_features_dict 


def get_all_covariates(gtex_phenotype_df, features_dict):
    """
    Adds covariates (age, sex, ethnicity, smoking status, and BMI) to the feature data of each sample.

    Args:
        gtex_phenotype_df (pd.DataFrame): DataFrame containing GTEx phenotype information with columns 
            'SUBJID', 'AGE', 'SEX', 'ETHNCTY', 'MHSMKSTS', and 'BMI'.
        features_dict (Dict[str, np.ndarray]): Dictionary with sample IDs as keys and feature arrays as values.

    Returns:
        Dict[str, np.ndarray]: New dictionary with updated feature arrays including all covariates.
    """
    # Initialize label encoder for MHSMKSTS
    label_encoder = LabelEncoder()
    gtex_phenotype_df['MHSMKSTS_encoded'] = label_encoder.fit_transform(gtex_phenotype_df['MHSMKSTS'])
    
    # Store the mapping for reference
    smoking_status_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Smoking status encoding mapping:", smoking_status_mapping)

    missing_samples = []  # To store sample_ids with missing SUBJID matches
    missing_covariates = []  # To store sample_ids with missing covariate values
    updated_features_dict = {}  # New dictionary for updated features

    covariates = ['AGE', 'SEX', 'ETHNCTY', 'RACE' 'MHSMKSTS_encoded', 'BMI']

    for sample_id, features in features_dict.items():
        # Extract SUBJID part from the sample_id
        subjid = "-".join(sample_id.split("-")[:2])
        
        # Check for matches in gtex_phenotype_df
        matching_rows = gtex_phenotype_df[gtex_phenotype_df['SUBJID'] == subjid]
        
        if not matching_rows.empty:
            # Retrieve the index of the matched row
            matched_index = matching_rows.index[0]
            
            # Extract all covariates
            covariate_values = matching_rows.loc[matched_index, covariates]
            
            # Check if any covariate is missing (NaN or invalid)
            if covariate_values.isna().any():
                missing_covariates.append(sample_id)
                updated_features_dict[sample_id] = features  # Keep original features
            else:
                # Add all covariates to the features array
                updated_features_dict[sample_id] = np.append(features, covariate_values.values)
        else:
            # Add sample_id to the missing list if no match is found
            missing_samples.append(sample_id)
            updated_features_dict[sample_id] = features  # Keep original features
    
    # Display missing information
    print("Sample IDs with no matching SUBJID:", missing_samples)
    print("Sample IDs with missing covariates:", missing_covariates)
    
    # Print the order of added covariates for reference
    print("\nOrder of added covariates:", covariates)
    
    return updated_features_dict


def preprocess_data(
    tissue: str,
    telomere_df: pd.DataFrame,
    all_tissue_aggregated_features: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocesses data for a specific tissue, excluding covariates.

    Args:
        tissue (str): The tissue to process.
        telomere_df (pd.DataFrame): DataFrame containing telomere data.
        all_tissue_aggregated_features (Dict[str, np.ndarray]): Aggregated features.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: X (features), y (telomere lengths), and sample IDs.
    """
    tissue_telomere_df = telomere_df[telomere_df["TissueSiteDetail"] == tissue]
    tissue_telomere_dict = create_telomere_dict(tissue_telomere_df)

    tissue_common_samples = set(tissue_telomere_dict.keys()).intersection(
        set(all_tissue_aggregated_features.keys())
    )
    X_tissue = []
    y_tissue = []
    sample_ids_filtered = []

    for sample_id in sorted(tissue_common_samples):
        if sample_id in all_tissue_aggregated_features:
            X_tissue.append(all_tissue_aggregated_features[sample_id])
            y_tissue.append(tissue_telomere_dict[sample_id])
            sample_ids_filtered.append(sample_id)

    X_tissue = np.array(X_tissue)
    y_tissue = np.array(y_tissue)

    return X_tissue, y_tissue, sample_ids_filtered


def build_dataset(
    tissue_list: List[str],
    telomere_df: pd.DataFrame,
    all_tissue_aggregated_features: Dict[str, np.ndarray],
    tissues_to_skip: List[str],
    min_samples: int = MIN_SAMPLES_PER_TISSUE,
) -> Dict[str, Dict[str, Any]]:
    """
    Preprocesses data for each tissue, excluding covariates, and prepares it for modeling.

    Args:
        tissue_list (List[str]): List of tissue names.
        telomere_df (pd.DataFrame): DataFrame containing telomere data.
        all_tissue_aggregated_features (Dict[str, np.ndarray]): Dictionary mapping sample IDs to aggregated feature arrays.
        tissues_to_skip (List[str]): List of tissues to skip.
        min_samples (int): Minimum number of samples required for a tissue to be processed.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with preprocessed datasets for each tissue.
                                   Contains "X", "y", and "sample_ids".
    """
    preprocessed_data = {}

    for tissue in tissue_list:
        if tissue in tissues_to_skip:
            print(f"Skipping tissue: {tissue}")
            continue

        print(f"Processing tissue: {tissue}")

        X_tissue, y_tissue, sample_ids_filtered = preprocess_data(
            tissue, telomere_df, all_tissue_aggregated_features
        )

        if len(X_tissue) < min_samples:
            print(
                f"Skipping {tissue} due to insufficient data ({len(X_tissue)} samples)."
            )
            continue

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_tissue)

        preprocessed_data[tissue] = {
            "X": X_scaled,
            "y": y_tissue,
            "sample_ids": sample_ids_filtered,
        }

    return preprocessed_data


def save_dataset(dataset: Dict[str, Dict[str, Any]], file_name: str) -> None:
    """
    Saves the dataset to a file in the specified format.

    Args:
        dataset (Dict[str, Dict[str, Any]]): Preprocessed dataset for all tissues.
        file_name (str): Name of the file to save the data. The extension determines the format (e.g., .pkl, .csv, .json).

    Raises:
        ValueError: If the file format is not supported.
    """
    file_extension = file_name.split('.')[-1]
    
    if file_extension == "pkl":
        with open(file_name, "wb") as file:
            pickle.dump(dataset, file)
        print(f"Dataset saved as pickle file: {file_name}")
    
    elif file_extension == "json":
        # Convert numpy arrays to lists for JSON serialization
        dataset_serializable = {
            tissue: {
                "X": data["X"].tolist(),
                "y": data["y"].tolist(),
                "sample_ids": data["sample_ids"],
            }
            for tissue, data in dataset.items()
        }
        with open(file_name, "w") as file:
            json.dump(dataset_serializable, file)
        print(f"Dataset saved as JSON file: {file_name}")
    
    elif file_extension == "csv":
        # Flatten the dataset and save each tissue as a separate CSV file
        for tissue, data in dataset.items():
            tissue_file_name = f"{file_name.split('.')[0]}_{tissue}.csv"
            tissue_df = pd.DataFrame(data["X"])
            tissue_df["y"] = data["y"]
            tissue_df["sample_id"] = data["sample_ids"]
            tissue_df.to_csv(tissue_file_name, index=False)
        print(f"Dataset saved as CSV files with prefix: {file_name.split('.')[0]}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are .pkl, .json, and .csv.")

def build_age_telomere_dataset(
    telomere_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    tissue: str = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Builds a dataset with age as X and telomere length as Y.
    
    Args:
        telomere_df (pd.DataFrame): DataFrame with telomere data
        phenotype_df (pd.DataFrame): DataFrame with phenotype data including age
        tissue (str, optional): Specific tissue to filter for. If None, uses all tissues
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: X (ages), y (telomere lengths), and sample IDs
    """
    # Filter for specific tissue if provided
    if tissue:
        telomere_df = telomere_df[telomere_df["TissueSiteDetail"] == tissue]
    
    # Extract subject IDs from sample IDs
    telomere_df['SUBJID'] = telomere_df['CollaboratorParticipantID']
    
    # Merge telomere data with phenotype data
    merged_df = telomere_df.merge(
        phenotype_df[['SUBJID', 'AGE']], 
        on='SUBJID', 
        how='inner'
    )
    
    # Remove any rows with missing values
    merged_df = merged_df.dropna(subset=['AGE', 'TQImean'])
    
    # Extract features and target
    X = merged_df['AGE'].values.reshape(-1, 1)
    y = merged_df['TQImean'].values
    sample_ids = merged_df['CollaboratorParticipantID'].tolist()
    return X, y, sample_ids

def build_dataset_age(
    tissue_list: List[str],
    telomere_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    tissues_to_skip: List[str],
    min_samples: int = MIN_SAMPLES_PER_TISSUE,
) -> Dict[str, Dict[str, Any]]:
    """
    Preprocesses data for each tissue using age as target variable.

    Args:
        tissue_list (List[str]): List of tissue names.
        telomere_df (pd.DataFrame): DataFrame containing telomere data.
        phenotype_df (pd.DataFrame): DataFrame containing phenotype data.
        all_tissue_aggregated_features (Dict[str, np.ndarray]): Dictionary mapping sample IDs to aggregated feature arrays.
        tissues_to_skip (List[str]): List of tissues to skip.
        min_samples (int): Minimum number of samples required for a tissue to be processed.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with preprocessed datasets for each tissue.
                                   Contains "X", "y", and "sample_ids".
    """
    preprocessed_data = {}

    for tissue in tissue_list:
        if tissue in tissues_to_skip:
            print(f"Skipping tissue: {tissue}")
            continue

        print(f"Processing tissue: {tissue}")

        X_tissue, y_tissue, sample_ids_filtered = build_age_telomere_dataset(
             telomere_df, phenotype_df, tissue
        )

        if len(X_tissue) < min_samples:
            print(
                f"Skipping {tissue} due to insufficient data ({len(X_tissue)} samples)."
            )
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_tissue)

        preprocessed_data[tissue] = {
            "X": X_scaled,
            "y": y_tissue,
            "sample_ids": sample_ids_filtered,
        }

    return preprocessed_data



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
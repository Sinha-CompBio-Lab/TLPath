from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch
from PIL import Image
from typing import List, Tuple, Union, Optional
import os
from pathlib import Path
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import adjustText
import numpy as np    


class ConchAnalyzer:
    def __init__(self,auth_token):
        """
        Initialize Conch analyzer with the specified model
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_cfg: Model configuration name
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create model and preprocessing
        if auth_token == None:
            raise NotImplementedError(f'Access denied. Please request access from https://huggingface.co/MahmoodLab/CONCH')
        try:
            self.model, self.preprocess = create_model_from_pretrained(
                'conch_ViT-B-16',
                "hf_hub:MahmoodLab/conch",
                hf_auth_token=auth_token
            )
        except ImportError:
            raise ImportError('Access denied. Please request access from https://huggingface.co/MahmoodLab/CONCH')

        self.model.to(self.device)
        self.model.eval()
        
        # Set up tokenizer
        self.tokenizer = get_tokenizer()
        
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load and preprocess image"""
        image = Image.open(image_path)
        image = image.resize((224, 224))
        return image
     
    def analyze_image(self, 
        image_path: Union[str, Path], 
        classes: List[str], 
        templates: List[str] = None,
        tissues: List[str] = None) -> List[Tuple[str, float]]:
        """
        Analyze a single image against provided classes
        
        Args:
            image_path: Path to image file
            classes: List of possible classes
            template: Template string to prefix class names
            
        Returns:
            List of (class_name, confidence_score) tuples, sorted by confidence
        """
        # Prepare image
        image = self.load_image(image_path)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Prepare text prompts

        all_text = [template.format(c=label, t=tissue) for template in templates for label in classes for tissue in tissues]
        # all_text = [template.format(c=label) for template in templates for label in labels]

        tokenized_prompts = tokenize(texts=all_text, tokenizer=self.tokenizer).to(self.device)
        
        # Get predictions
        with torch.inference_mode():
            image_embeddings = self.model.encode_image(image_tensor)
            text_embeddings = self.model.encode_text(tokenized_prompts)
            
            # Calculate similarity scores
            sim_scores = (image_embeddings @ text_embeddings.T * 
                         self.model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()
        
        # Format results
        results = [(cls, float(score)) for cls, score in zip(all_text, sim_scores[0])]
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def analyze_images(self, 
                      image_paths: List[Union[str, Path]], 
                      
                    classes: List[str] = None,
                      template: List[str] = None) -> List[List[Tuple[str, float]]]:
        """
        Analyze multiple images against provided classes
        
        Args:
            image_paths: List of paths to image files
            classes: List of possible classes
            template: Template string to prefix class names
            
        Returns:
            List of results for each image, where each result is a list of 
            (class_name, confidence_score) tuples sorted by confidence
        """

        if template is None:
            template = [
                "{c}"
            ]
            # Past Templete 
            # "An H&E image of {t} Tissue with {c} present"
    
        if classes is None:
            # All 86 Words
            classes = [    
                    "mucosa","muscularis","congestion", "fibrosis","fibroadipose","fibrovascular","fibrofatty",
                    "fibromuscular","autolysis","squamous","epithelium","adipose",
                    "interstitial","lymphoid","dermal","epidermis","gland","ducts",
                    "stroma","atherosis","cortex","spermatogenesis","sloughed","foci","islets","vessel",
                    "hyperplasia","glomeruli","plaque","atrophy","adenohypophysis","nerve","atherosclerosis",
                    "ischemic","medulla","tubules", "inflammation","propria","sclerosis","lesions",
                    "hemorrhage","adventitia","subcutaneous","myometrium","macrovesicular","parenchyma",
                    "calcification","gynecomastoid","intimal","capsule","macrophages","alveolar",
                    "saponification","fascia","corpora","endometrium","edema","nodule","gastric",
                    "emphysema","cyst","pneumonia","scarring","monckeberg","atelectasis",
                    "hashimoto","infarction","hyalinization","hypertrophy","cirrhosis","necrosis","goiter",
                    "esophagitis", "hypereosinophilia","metaplasia", "desquamation",
                    "diabetic","nephritis","adenoma","amylacea","prostatitis","hepatitis","hypoxic",
                    "pancreatitis","mastopathy","dysplasia"
                ]

            # Pancreas specific words
            # classes = [ "autolysis", "interstitial", "saponification", "adipose", "islets", "fibrosis", "atrophy", "nodule", "parenchyma", "cyst", 
            #            "foci", "ducts", "sloughed", "pancreatitis", "squamous", "vessel", "metaplasia", "nerve", "sclerosis", "diabetic", "epithelium", 
            #            "fibrofatty", "fibrovascular", "congestion", "hemorrhage", "stroma", "macrophages", "calcification", "fibroadipose", "necrosis", 
            #            "gland", "inflammation", "desquamation", "hyalinization", "lesions", "lymphoid", "scarring"]

            
            tissues = [
                'Pancreas'
            ]
        
        return [self.analyze_image(img_path, classes, template, tissues) 
                for img_path in image_paths]



    
def generate_colors(labels):
    colors = [
            '#c91d1d', '#c9351d', '#c94d1d', '#c9651d', 
            '#c97d1d', '#c9951d', '#c9ad1d', '#c9c51d', 
            '#b5c91d', '#9dc91d', '#85c91d', '#6dc91d', 
            '#55c91d', '#3dc91d', '#25c91d', '#1dc92d', 
            '#1dc945', '#1dc95d', '#1dc975', '#1dc98d', 
            '#1dc9a5', '#1dc9bd', '#1dbdc9', '#1da5c9', 
            '#1d8dc9', '#1d75c9', '#1d5dc9', '#1d45c9', 
            '#1d2dc9', '#251dc9', '#3d1dc9', '#551dc9', 
            '#6d1dc9', '#851dc9', '#9d1dc9', '#b51dc9', 
            '#c91dc5', '#c91dad', '#c91d95', '#c91d7d', 
            '#c91d65', '#c91d4d', '#c91d35', '#e96363', 
            '#e97563', '#e98863', '#e99b63', '#e9ae63', 
            '#e9c063', '#e9d363', '#e9e663', '#d9e963', 
            '#c6e963', '#b4e963', '#a1e963', '#8ee963', 
            '#7ce963', '#69e963', '#63e96f', '#63e982', 
            '#63e995', '#63e9a7', '#63e9ba', '#63e9cd', 
            '#63e9df', '#63dfe9', '#63cde9', '#63bae9', 
            '#63a7e9', '#6395e9', '#6382e9', '#636fe9', 
            '#6963e9', '#7c63e9', '#8e63e9', '#a163e9', 
            '#b463e9', '#c663e9', '#d963e9', '#e963e6', 
            '#e963d3', '#e963c0', '#e963ae', '#e9639b', 
            '#e96388', '#e96375'
         ]
    return dict(zip(labels,colors))


def create_volcano_plot(low_results, high_results, feature_label):
    """
    Create a volcano plot with unique colors for significant labels
    """
    # Initialize dictionaries for each label
    low_data = {label: [] for label, _ in low_results[0]}
    high_data = {label: [] for label, _ in high_results[0]}
    
    # Collect probabilities for each label
    for low_res, high_res in zip(low_results, high_results):
        for (low_label, low_prob), (high_label, high_prob) in zip(low_res, high_res):
            low_data[low_label].append(low_prob)
            high_data[high_label].append(high_prob)
    
    # Calculate statistics for each label
    fold_changes = []
    p_values = []
    labels = []
    
    epsilon = 1e-10
    for label in low_data.keys():
        mean_high = np.mean(high_data[label])
        mean_low = np.mean(low_data[label])
        log2_fold_change = np.log2((mean_high + epsilon) / (mean_low + epsilon))
        fold_changes.append(log2_fold_change)
        
        t_stat, p_val = stats.ttest_ind(high_data[label], low_data[label])
        p_values.append(p_val)
        labels.append(label)
    
    plt.figure(figsize=(6,5))
    # plt.figure(figsize=(12,9))
    log_p_values = -np.log10(p_values)
    threshold_p = 0.05

    # Find significant labels and assign colors
    significant_labels = [label for label, pval in zip(labels, p_values) if pval < threshold_p]
    label_to_color = generate_colors(sorted(labels))
    

    # Assign colors to points
    colors = []
    for label, pval in zip(labels, p_values):
        if pval < threshold_p:
            colors.append(label_to_color[label])
        else:
            colors.append('grey')
    
    # Plot points
    scatter = plt.scatter(fold_changes, log_p_values, c=colors, alpha=0.6, s=180)
    
    # Add labels for significant points
    texts = []
    for i, (fc, pval, label) in enumerate(zip(fold_changes, p_values, labels)):
        if pval < threshold_p:
            texts.append(plt.text(fold_changes[i], -np.log10(pval), label, 
                                fontsize=9, fontweight='bold'))
    
    # Adjust labels position

    adjustText.adjust_text(texts, 
                         arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    # Add threshold lines
    plt.axhline(y=-np.log10(threshold_p), color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Log2 Fold Change', fontsize=12)
    plt.ylabel('-log10(p-value)', fontsize=12)
    plt.title(f'Volcano Plot: Feature {feature_label.split("_")[-1]}', 
              fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    return plt


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--patch_feature_dir', type=str, default=None)
parser.add_argument('--feature', type=str, default=None)
parser.add_argument('--auth',type=str, default=None )
args = parser.parse_args()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ConchAnalyzer(args.auth)

    if args.feature:
        feature_list = [args.feature]
    else:
        feature_list = ['mag_192','mag_628','mag_726','mag_852','mag_906',]
    for feature in feature_list:
        args.feature = feature

        high_image_paths = glob.glob(f'{args.patch_feature_dir}/patches_long/**/{args.feature}/high/*.png', recursive=True)
        low_image_paths = glob.glob(f'{args.patch_feature_dir}/patches_short/**/{args.feature}/high/*.png', recursive=True)

        # Process images
        print("Processing Conch analysis...")
        low_batch_results = analyzer.analyze_images(low_image_paths)
        high_batch_results = analyzer.analyze_images(high_image_paths)

        plot = create_volcano_plot(low_batch_results, high_batch_results,args.feature)
        plot.savefig(f'Conch_RF_{args.feature}_Long_Short_volcano_plot.png')
        plot.close()
        print("Done")
 
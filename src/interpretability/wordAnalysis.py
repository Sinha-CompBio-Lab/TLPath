import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from collections import defaultdict

def setup_nltk():
    """
    Download required NLTK data if not already present
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def analyze_word_frequencies(file_path):
    """
    Analyze word frequencies in specified columns of a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing word frequencies for each analyzed column
    """
    # Set up NLTK resources
    setup_nltk()
    
    # Get English stop words
    stop_words = set(stopwords.words('english'))
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Columns to analyze
    columns_to_analyze = ['Pathology Notes', 'Pathology Categories']
    results = Counter()
    
    for column in columns_to_analyze:
        # Combine all text in the column
        text = ' '.join(df[column].astype(str).fillna(''))
        text = text.lower()
        
        # 2. Remove non-English characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 3. Split into words
        words = text.split()
        
        # 4. Filter out stop words and short words
        words = [word for word in words 
                if word not in stop_words  # Remove stop words
                and len(word) > 2  # Remove very short words
                and word.isalpha()]  # Ensure only alphabetic characters
        
        # Count frequencies
        word_counts = Counter(words).most_common(200)
        results = results + Counter(dict(word_counts))
    
    return results

def print_results(results):
    print(f"{'Word':<20} | {'Occurrences':<10}")
    print("-" * 50)
    for word, count in results.items():
        print(f"{word:<20} | {count:<10}")

def frequency_by_tissue(file_path):
    df = pd.read_csv(file_path)
    columns_to_analyze = ['Pathology Notes', 'Pathology Categories']
    # 86 words from analyze_word_frequencies cleaned. (doc/words_cleaned.txt)
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
    # Initialize results dictionary with defaultdict to handle new tissues
    results = defaultdict(lambda: defaultdict(int))
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        tissue = row['Tissue']
        
        # Combine the text from both pathology columns
        combined_text = ' '.join(str(row[col]).lower() for col in columns_to_analyze if pd.notna(row[col]))
        
        # Check for each keyword in the combined text
        for keyword in classes:
            if keyword.lower() in combined_text:
                results[tissue][keyword] += 1
    
    # Convert defaultdict to regular dict with tuples
    final_results = {}
    for tissue, word_counts in results.items():
        # Convert word counts to list of tuples, only including words that appeared
        word_tuples = [(word, count) for word, count in word_counts.items() if count > 0]
        if word_tuples:  # Only include tissues that had matches
            final_results[tissue] = word_tuples
    
    return final_results,classes

def save_results_to_excel(results, classes, output_file):
    # Create a DataFrame with tissues as index and classes as columns
    df = pd.DataFrame(
        0,  # Fill with zeros initially
        index=classes,  # class as row indices
        columns=results.keys()  # All tissues as columns
    )
    
    for tissue, word_counts in results.items():
        for word, count in word_counts:
            df.loc[word, tissue] = count
    
    # Save to Excel
    df.to_excel(output_file)
  


if __name__ == "__main__":
    # Replace with your actual file path
    file_path = '..../GTEx_Portal.csv'
    
    try:
        # Most Common Words, uncleaned. 
        results = analyze_word_frequencies(file_path)
        print_results(results)

        # Pancrease specifict terms from our 86 Words cleaned. 
        tissue_results, classes = frequency_by_tissue(file_path)
        for tissue, word_counts in tissue_results.items():
            output = ""
            if tissue == 'Pancreas':
                for word, count in word_counts:
                    if count > 0:
                        output += f"{word}, "
       
  
        # Save results to a new Excel file
        output_df = save_results_to_excel(tissue_results, classes, 'tissue_analysis_results.xlsx')

    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
    except KeyError:
        print("Error: One or more required columns not found in the CSV file.")

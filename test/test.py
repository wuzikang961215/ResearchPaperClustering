import os
import sys

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data_processing import DataProcessing

def test_extract_abstracts():
    """
    Test the extraction of abstracts from PDF files in the data/raw directory.
    """
    dp = DataProcessing()
    
    # Directory containing raw PDF files
    raw_data_dir = os.path.join(parent_dir, 'data/raw')
    
    # Extract abstracts from all PDF files
    results = dp.extract_abstracts(raw_data_dir)
    
    # Print the results for verification
    for file_path, abstract in results:
        file_name = os.path.basename(file_path)
        # print(f"File: {file_name}\nAbstract:\n{dp.preprocess_text(abstract) if abstract != 'Abstract not found' else abstract}\n")
        print(f"File: {file_name}\nAbstract:\n{abstract}\n")

if __name__ == "__main__":
    test_extract_abstracts()

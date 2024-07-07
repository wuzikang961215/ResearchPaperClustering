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
    
    # Save the results to a CSV file
    output_csv = os.path.join(parent_dir, 'data/processed/abstracts.csv')
    dp.save_abstracts_to_csv(results, output_csv)
    print(f"Abstracts saved to {output_csv}")

if __name__ == "__main__":
    test_extract_abstracts()

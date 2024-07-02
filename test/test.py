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
    
    # List all PDF files in the data/raw directory
    raw_data_dir = os.path.join(parent_dir, 'data/raw')
    file_paths = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.pdf')]

    # Extract abstracts from all PDF files
    abstracts = dp.extract_abstracts(file_paths)
    
    # Print the results for verification
    for i, abstract in enumerate(abstracts):
        print(f"Abstract {i + 1}:\n{abstract}\n")

if __name__ == "__main__":
    test_extract_abstracts()


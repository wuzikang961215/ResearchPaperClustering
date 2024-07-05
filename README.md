# Research Paper Clustering

## Overview

(Introduction and how to run instructions will be added here once the project is complete.)

## Steps to Run the Project
1. **Setup the Environment**:
    - Create a virtual environment:
      ```bash
      python -m venv env
      ```
    - Activate the virtual environment:
      ```bash
      source env/bin/activate  # On Windows use `env\Scripts\activate`
      ```
    - Install the required packages:
      ```bash
      pip install -r requirements.txt
      ```

2. **Extract Abstracts**:
    - Run the data extraction script to extract abstracts from the PDF files:
      ```bash
      python src/data_processing.py
      ```

3. **Testing**:
    - Run the test script to verify the extraction process:
      ```bash
      python tests/test.py
      ```

## Future Work
- Further refinement of text preprocessing steps, including stop words removal and case normalization.
- Implementation of clustering algorithms to group similar research papers.
- Visualization of clustering results using dimensionality reduction techniques.

## Contributions
Feel free to contribute to this project by creating issues or submitting pull requests. Any contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Exploration Process

### Current Progress

#### Data Extraction and Preprocessing

- Implemented a `DataProcessing` class in `data_processing.py` to handle text extraction and preprocessing.
- Successfully extracted text from PDF files using `PdfReader` from PyPDF2.
- Implemented text preprocessing including lowercasing, removal of non-word characters, stop word removal, and lemmatization.

#### Testing

- Created a `tests` directory with a `test.py` script to test the text extraction functionality.
- The script lists all PDF files in the `data/raw` directory, extracts abstracts, and prints them for verification.
- Added sorting of file names to ensure consistent processing order.
- Modified the test script to include the PDF file names in the output for easier verification.

### Challenges

#### PDF Parsing

- Encountered deprecation issues with `PdfFileReader` in PyPDF2 version 3.0.0.
  - Solution: Switched to `PdfReader` for extracting text from PDF files.

#### Text Matching

- Difficulty in verifying the extracted abstracts against the original PDF files due to order mismatch.
  - Solution: Included the PDF file names in the test output to match abstracts with their respective files.

### Key Findings from Initial Data Processing
1. During the preprocessing phase, it was discovered that the abstracts typically end with the keywords "introduction", "index terms" and "keywords". This observation is crucial for accurately extracting the abstract section from the research papers.
2. In some PDFs, the introduction starts with "1. Introduction", causing "1." to be included in the abstract. (Consider using stop words from NLP to handle this.)
3. In `paper_69` and `paper_70`, the abstract title "ABSTRACT" in red was not recognized by the system.
4. In `paper_50`, the "K E Y W O R D S" were not recognized, resulting in the extraction of the entire article. This is likely due to the spaces in the keyword.
5. In `paper_51`, `paper_53`, `paper_58`, `paper_59`, `paper_63`, `paper_66`, `paper_72`, and `paper_73`, there is no keyword "abstract"; the abstracts are just sections at the beginning of the articles.
6. In the abstract for `paper_66`, the words are stuck together after extraction, possibly because the abstract section has a yellow background in the original PDF.
7. In `paper_79`, redundant words like university affiliations and email addresses were included in the extracted abstract. These are references at the bottom of page 1 but were mistakenly included in the abstract.

### Next Steps

- Continue refining the text extraction and preprocessing steps to handle edge cases and improve accuracy.
- Implement text vectorization using techniques like TF-IDF and word embeddings.
- Develop clustering algorithms to group similar research papers based on their abstracts.
- Perform thorough evaluation of the clustering results using appropriate metrics.
- Visualize the clustering results using dimensionality reduction techniques like t-SNE or PCA.


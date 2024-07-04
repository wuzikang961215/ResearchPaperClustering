# Research Paper Clustering

## Project Overview
This project focuses on clustering a given set of research papers based on their abstract similarity. It demonstrates skills in natural language processing, text preprocessing, and unsupervised machine learning.

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

## File Structure
├── data
│ ├── raw
│ │ └── *.pdf
├── src
│ ├── data_processing.py
├── tests
│ ├── test.py
├── requirements.txt
└── README.md

## Future Work
- Further refinement of text preprocessing steps, including stop words removal and case normalization.
- Implementation of clustering algorithms to group similar research papers.
- Visualization of clustering results using dimensionality reduction techniques.

## Contributions
Feel free to contribute to this project by creating issues or submitting pull requests. Any contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Progress and Exploration

### Key Findings
During the preprocessing phase, it was discovered that the abstracts typically end with the keywords "introduction" and "keywords". This observation is crucial for accurately extracting the abstract section from the research papers.

### Data Preprocessing
The preprocessing steps involve extracting the abstract content from each PDF file and handling any special characters or formatting issues.

### Current Progress
- Implemented a function to rename PDF files for consistency.
- Developed a method to extract abstracts from the PDF files.
- Identified the keywords "introduction" and "keywords" as critical markers for the end of abstracts.
- Implemented a cleaning function to remove redundant phrases from the abstracts.

### Challenges and Solutions
- Ensuring accurate extraction of abstracts despite diverse PDF formats.
- Handling redundant phrases that appear at the end of abstracts, such as "2014 Elsevier Ltd. All rights reserved."

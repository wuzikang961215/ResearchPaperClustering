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

## Contributions
Feel free to contribute to this project by creating issues or submitting pull requests. Any contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Exploration Process

### Initial Data Extraction and Preprocessing

The journey began with implementing a `DataProcessing` class in `data_processing.py` to handle text extraction and preprocessing. Successfully extracted text from PDF files using `PdfReader` from PyPDF2.

### Testing and Verification

A `tests` directory was created with a `test.py` script to test the text extraction functionality. The script lists all PDF files in the `data/raw` directory, extracts abstracts, and prints them for verification. Sorting of file names was added to ensure consistent processing order. The test script was also modified to include the PDF file names in the output for easier verification.

### Challenges Encountered

#### PDF Parsing Issues

Deprecation issues were encountered with `PdfFileReader` in PyPDF2 version 3.0.0. To address this, switched to `PdfReader` for extracting text from PDF files.

#### Text Matching Difficulties

Verifying the extracted abstracts against the original PDF files was challenging due to several reasons:
- **Order Mismatch**: The extracted text often did not match the order of the text in the original PDF, making manual verification difficult.
- **Complex File Names**: The original file names were long and complex, such as "1-s2.0-S0360128523000023-main.pdf", making it hard to read and compare the extraction results with the actual words in the PDF. To simplify this process, files were renamed to a consistent format (e.g., `paper_1` to `paper_79`). This renaming made it much easier to manually check the extracted abstracts.

### Key Findings from Initial Data Processing

1. During the preprocessing phase, it was discovered that the abstracts typically end with the keywords "introduction", "index terms" and "keywords". This observation is crucial for accurately extracting the abstract section from the research papers.
2. In some PDFs, the introduction starts with "1. Introduction", causing "1." to be included in the abstract. (Consider using stop words from NLP to handle this.)
3. In `paper_68` and `paper_69`, the abstract title "ABSTRACT" in red was not recognized by the system.
4. In `paper_49`, the "K E Y W O R D S" were not recognized, resulting in the extraction of the entire article. This is likely due to the spaces in the keyword.
5. In `paper_50`, `paper_52`, `paper_57`, `paper_58`, `paper_62`, `paper_65`, `paper_71`, and `paper_72`, there is no keyword "abstract"; the abstracts are just sections at the beginning of the articles.
6. In the abstract for `paper_65`, the words are stuck together after extraction, possibly because the abstract section has a yellow background in the original PDF.
7. In `paper_78`, redundant words like university affiliations and email addresses were included in the extracted abstract. These are references at the bottom of page 1 but were mistakenly included in the abstract.

### Addressing Key Issues with OCR Integration

To handle edge cases where the PDF's text is not recognized due to formatting issues (e.g., red "ABSTRACT"), OCR (Optical Character Recognition) was integrated using Tesseract and Poppler. This approach was particularly useful for PDFs like `paper_68` and `paper_69`, where the abstract title "ABSTRACT" in red was not recognized by the system.

- **Steps to integrate OCR**:
  - Installed Homebrew to manage dependencies.
  - Added Homebrew to PATH and installed Tesseract and Poppler.
  - Verified installations to ensure proper setup.
  - Updated the requirements file to include `pdf2image`.

The integration of OCR successfully addressed the issue in `paper_50`, where keywords were not recognized due to the library used. The OCR solution proved to be effective for extracting text in cases where the initial approach failed due to formatting issues.

### Exploration of Grobid and Other Libraries

Explored Grobid and other libraries like pdfminer, PyMuPDF, and transformers for abstract extraction. Grobid, despite its promising capabilities, was found to be complex to set up and did not yield significantly better results than the current approach. The same applied to other libraries and models, which either required extensive training or did not handle the specific edge cases encountered.

### Decided Abstract Extraction Approach and Future Steps

After several days of attempting to extract abstracts perfectly, it was found that the variations in PDF structures made it extremely difficult to achieve perfect results with the current approach. Given the constraints and time limitations, the current approach will use the regex patterns and keywords identification to extract abstracts. 

- Successfully extracted abstracts from 62 out of 79 PDFs.
- 17 PDFs remain unprocessed due to varied and complex formatting issues.
- For now, the extraction is based on the current regex-based approach, and more sophisticated machine learning models or classification methods may be considered in the future for handling the edge cases more effectively.
- The integration of OCR has improved the accuracy for some edge cases, but further refinement is needed.

This approach balances the need for accuracy with the practical constraints of the project timeline. Further enhancements will focus on improving the regex patterns and exploring more advanced text extraction techniques.

### Further Findings on OCR Usage

It was discovered that while OCR helps when "Abstract" is in a weird format and cannot be recognized, overusing OCR can lead to poor word extraction. For instance, in some cases where there are stuck words due to newline removal, using OCR introduced more issues. Therefore, it is more effective to use `PdfReader`'s `extract_text` function even if it results in stuck words. The stuck words can then be handled by adding a space when newline characters are removed.

It was discovered that while OCR helps when "Abstract" is in a weird format and cannot be recognized, overusing OCR can lead to poor word extraction. For instance, in some cases where there are stuck words due to newline removal, using OCR introduced more issues. Therefore, it is more effective to use `PdfReader`'s `extract_text` function even if it results in stuck words. The stuck words can then be handled by adding a space when newline characters are removed.

### Moving Forward with Text Vectorization and Clustering

Despite the challenges faced with perfect abstract extraction, progress was made to move forward with text vectorization and clustering. The following steps were implemented:

1. **Text Vectorization**:
    - TF-IDF vectorization was used to convert the preprocessed text data into numerical representation. This technique effectively captures the importance of terms within the abstracts.

2. **Clustering**:
    - K-Means clustering was applied to group similar research papers based on their abstracts.
    - The optimal number of clusters was determined to be 5.
    - Silhouette score was calculated to evaluate the quality of the clustering.

3. **Visualization**:
    - t-SNE was used to visualize the clustering results. This dimensionality reduction technique helped in displaying the high-dimensional abstract vectors in a 2D space.
    - The visualization provided insights into the clustering performance, showing that while some clusters were well-defined, others had overlapping points, indicating the need for further refinement.

The current approach provided a good starting point for clustering the research papers based on their abstracts. The process involved:
- Extracting and preprocessing abstracts.
- Vectorizing the text using TF-IDF.
- Clustering with K-Means.
- Visualizing the clusters using t-SNE.

The overall process highlighted the complexities involved in text extraction from PDFs and the importance of balancing accuracy with practical constraints. Future work will focus on improving the extraction techniques and exploring more advanced models for better clustering performance.

### Progress with Jupyter Notebook, Vectorization, and Clustering Visualization

#### Transition to Jupyter Notebook

To enhance the exploration and visualization of the clustering process, we transitioned the entire workflow to a Jupyter Notebook. This provided an interactive platform to document and visualize each step effectively.

#### Abstract Extraction and Saving

- Initially, we faced significant challenges in extracting abstracts accurately due to the varied and complex structures of the PDFs.
- We applied a combination of regex patterns and OCR to handle edge cases where the text format was non-standard.
- Extracted abstracts were saved to a CSV file for consistency and ease of further processing.

#### Text Vectorization

- We utilized TF-IDF vectorization to convert the preprocessed text data into numerical representation, capturing the importance of terms within the abstracts.
- This process involved validating the vectorization output by inspecting the shape of the vectorized data and the feature names.
- We printed sample vector values and corresponding feature words to ensure the vectorization was working as expected.

#### Determining Optimal Number of Clusters

- To find the optimal number of clusters, we conducted two key analyses:
    - **Elbow Method**: Plotted the within-cluster sum of squares (WCSS) for a range of cluster numbers to identify the "elbow point" where adding more clusters did not significantly improve the clustering.
    - **Silhouette Analysis**: Calculated silhouette scores for different cluster numbers to measure the similarity of points within their own cluster compared to other clusters. This helped determine the quality of the clustering.
- These analyses led us to set the number of clusters to 20 for the current implementation.

#### Clustering and Evaluation

- K-Means clustering was applied to group similar research papers based on their abstracts.
- We evaluated the clustering results using silhouette scores to ensure the quality of the clusters.
- The clustering results, including the file names and their respective cluster labels, were saved to a CSV file for further analysis.

#### Visualization

- Dimensionality reduction was performed using t-SNE to visualize the clusters in a 2D space.
- The clusters were plotted, providing a visual representation of how the abstracts were grouped.
- The visualization revealed that while some clusters were well-defined, others had overlapping points, indicating areas for potential improvement.

### Summary of Findings and Future Steps

- The initial approach using regex patterns and OCR for abstract extraction was challenging but provided a good starting point.
- Moving to a Jupyter Notebook allowed for a more interactive and iterative exploration process.
- TF-IDF vectorization and K-Means clustering were effective in grouping similar abstracts, although some clusters showed overlap.
- Further refinement is needed, especially in handling the extraction phase and improving clustering performance.
- Future work will explore more advanced text extraction techniques and clustering models to enhance the accuracy and quality of the results.



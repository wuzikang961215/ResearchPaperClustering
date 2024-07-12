## Overview

The Research Paper Clustering project aims to cluster research papers based on their abstracts. This involves extracting text from PDF files, preprocessing the text, vectorizing the text data, and applying clustering algorithms to group similar research papers together. The project is designed to help researchers and academics identify related papers quickly and efficiently.

### Steps to Run the Project

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

2. **Run the Jupyter Notebook**:
    - Launch the Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the notebook `ResearchPaperClustering.ipynb` and run it cell by cell.

### Details and Purposes of Each Step in the Notebook

1. **Data Extraction and Saving**: Extracts and preprocesses the text data from the PDF files. In this section:
    - Running the first cell extracts all text from the first two pages of all PDFs and saves the extractions in the `data/processed/texts` directory.
    - Running the next cell extracts abstracts from the text files and saves them in the `data/processed/abstracts` directory as text files. Additionally, it generates an `abstracts.csv` file in the `data/processed` directory for further data manipulation.

2. **Vectorization**: Converts the preprocessed text into numerical representation using various vectorization methods (TF-IDF, Word2Vec, BERT, etc.). After comparing the results among all vectorization methods, Word2Vec is used for further steps.

3. **Determine Optimal Number of Clusters**: This step utilizes the elbow method, silhouette score, and Davies-Bouldin index to determine the best number of clusters.

4. **Clustering and Visualization**: Applies different clustering algorithms (K-Means, Hierarchical Clustering, DBSCAN, Spectral Clustering, Gaussian Mixture Models) to group similar research papers. Hierarchical clustering is chosen for its effectiveness in this context. This step also prints out the silhouette score and Davies-Bouldin index scores, and visualizes the clusters in a 2-dimensional graph using t-SNE. It also outputs the `clustering_results.csv` and `summary_report.csv` to the `data/processed` directory, indicating which paper belongs to which cluster and also which cluster contains how many papers including their corresponding key topic terms, respectively.


### Contributions
Feel free to contribute to this project by creating issues or submitting pull requests. Any contributions are welcome!

### License
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

### Vectorization Methods and Silhouette Scores

To enhance the clustering results, we explored various vectorization methods and evaluated their performance using silhouette scores:

1. **TF-IDF Vectorization**:
   - TF-IDF vectorization was used initially, which captures the importance of terms within the abstracts.
   - The highest silhouette score achieved with TF-IDF vectorization was approximately 0.036.

2. **Word2Vec Vectorization**:
   - Word2Vec vectorization was implemented to capture semantic relationships between words.
   - This method yielded the highest silhouette score of 0.12, making it the best-performing vectorization method among those tested.

3. **Doc2Vec Vectorization**:
   - Doc2Vec was also explored, which aims to capture semantic relationships at the document level.
   - The performance of Doc2Vec was not as high as Word2Vec, indicating that it may not be the best fit for this specific task.

4. **BERT Vectorization**:
   - BERT vectorization was tested to leverage the contextual embeddings of words.
   - However, this method resulted in a silhouette score of around 0.08, which was lower than Word2Vec.

### Identifying Optimal Number of Clusters

To determine the optimal number of clusters, we employed two key analyses:

1. **Elbow Method**:
   - Plotted the within-cluster sum of squares (WCSS) for a range of cluster numbers to identify the "elbow point" where adding more clusters did not significantly improve the clustering.
   - The elbow point suggested a potential range for the number of clusters.

2. **Silhouette Analysis**:
   - Calculated silhouette scores for different cluster numbers to measure the similarity of points within their own cluster compared to other clusters.
   - The highest silhouette score for Word2Vec vectorization was observed at 9 clusters, despite the overall score being 0.12, indicating a better-defined clustering structure at this point.

### Summary of Findings and Future Steps

- Word2Vec vectorization emerged as the best method for this task, providing the highest silhouette score.
- The optimal number of clusters was identified as 9, based on the silhouette score analysis.
- Further work will focus on refining the extraction phase and exploring advanced clustering algorithms to improve performance.

#### Exploring Different Clustering Algorithms

- **K-Means Clustering**:
  - Initially used K-Means clustering which provided the best silhouette score of 0.12 with Word2Vec vectorization.
  - Applied dimensionality reduction using t-SNE for visualization which revealed some well-defined clusters but also indicated overlapping points.

- **Hierarchical Clustering**:
  - Explored hierarchical clustering but found that the elbow curve remained flat, indicating no significant change in within-cluster sum of squares.
  - Hierarchical clustering was less effective in this context due to the nature of the data.

- **DBSCAN**:
  - Applied DBSCAN which resulted in only 1 cluster.
  - DBSCAN's sensitivity to density variations made it unsuitable for the current dataset.

- **Spectral Clustering**:
  - Implemented spectral clustering which resulted in a majority of papers (40+) being grouped into a single cluster, with the remaining clusters containing significantly fewer papers.
  - Due to this imbalance, spectral clustering was deemed unsuitable for this dataset.

- **Gaussian Mixture Models (GMM)**:
  - Explored GMM which returned similar silhouette scores of 0.12 at 9 or 10 clusters.
  - However, GMM took too long to plot the elbow and silhouette curves (approximately 2 minutes), making it less practical compared to other methods.

### Observations and Future Steps

- **Observations**:
  - The highest silhouette scores ranged between 0.12 to 0.2 across various vectorization and clustering methods, with Word2Vec and K-Means providing the best results.
  - Despite exploring multiple techniques, the overall clustering quality indicated potential issues with the initial text extraction process.

- **Future Steps**:
  - Decided to focus on improving text extraction quality to enhance the overall clustering performance.
  - Will explore more advanced text extraction methods, possibly integrating more robust OCR techniques or leveraging external text extraction services like AWS Textract.
  - Further refinement of the extraction phase is expected to lead to better representation and improved clustering outcomes.
 
### Enhanced Abstract Extraction Process

I realized that using PyPDF2 for abstract extraction was not yielding ideal results, as the silhouette score was suboptimal regardless of the vectorization or clustering algorithms applied. To address this, I focused on removing noise from the text data to improve the clustering outcomes. My exploration led me to switch to pdfminer for text extraction.

#### Initial Text Extraction with pdfminer

- **Quality Improvement**:
  By using pdfminer, the quality of the extracted text improved significantly. The text was more organized and better separated from other sections compared to PyPDF2.

#### Abstract Extraction

- **Text File Creation**:
  After extracting the text with pdfminer, I saved the content into text files. This allowed for more controlled and manageable processing.
  
- **Pattern-Based Extraction**:
  Initially, I attempted to identify patterns from "abstract" to "1. introduction." However, there were numerous edge cases that needed to be handled individually. 

- **Three Rounds of Parsing**:
  1. **First Parse**:
      - Targeted regular patterns starting from "abstract" and ending at "introduction."
  2. **Second Parse**:
      - Addressed cases where the abstract was misplaced after the introduction title, correcting its position.
  3. **Third Parse**:
      - Handled papers without the "abstract" keyword. This involved hard-coded, brute-force methods to cover all edge cases.
  - **Outcome**:
      - These three rounds of parsing significantly improved the extraction process, yielding cleaner abstracts with most noise removed. However, abstracts from papers 15, 18, 19, 20, 21, 22, 47, and 70 were incomplete. Nonetheless, this was preferable to extracting noisy content.
      - Papers 65, 73, and 78 were unprocessed due to complex formatting issues, but this was a significant reduction from the previous 14-15 unprocessed files.

#### Improved Silhouette Score

- **New Scores**:
  - Using spacy_word2vec and hierarchical clustering, the silhouette scores improved dramatically, ranging from 0.11 to as high as 0.6 or 0.8.
  - A notable observation was that the score was significantly higher when the cluster count was 1-2, suggesting that the data might be better organized and naturally split into 2-3 groups based on visualizations.

#### Limitations and Future Work

- **Current Solution**:
  - The current method relies heavily on brute force and specific patterns, which might not scale well with larger datasets. Handling every edge case manually is not feasible for extensive data.
  
- **Need for Advanced Solutions**:
  - To achieve even better precision and handle larger datasets effectively, a trained model to recognize abstracts without explicit programming is necessary.
  - Advanced tools and machine learning models from AWS, Google Cloud, or other platforms could provide the required sophistication to automate and improve the extraction process further.

#### Further Clustering and Vectorisation Exploration

- **Initial Findings**:
  - When using Sentence-BERT combined with K-Means clustering, the silhouette score was around 0.08. However, the t-SNE visualization showed more evenly distributed and interpretable clusters.
  - Hierarchical clustering with Sentence-BERT resulted in a silhouette score of 0.12, but some clusters contained only one paper while others had many more, leading to imbalanced clusters.

- **Evaluation Criteria**:
  - Silhouette Score: Measures similarity within clusters and separation between clusters. Higher scores generally indicate better-defined clusters.
  - Cluster Distribution: Evenly distributed clusters are preferable for practical usability.
  - Visual Inspection: t-SNE plots provided visual insights into the clustering structure, showing that more interpretable clusters might be more valuable even if the silhouette score is lower.

- **Decision**:
  - The approach using Sentence-BERT with K-Means clustering was chosen despite the lower silhouette score. The practical usability, interpretability of clusters, and even distribution observed in the t-SNE visualization were prioritized.
 
- **Final Decision**:
  - Eventually, I chose TF-IDF and K-Means because, although the silhouette score is not ideal with a silhouette of only 0.03, it makes more sense to group papers together in real-world scenarios. For example, when I check the groupings, I can see papers related to climate, environment, etc., grouped together. This practical grouping is more beneficial for understanding and analyzing the research topics.

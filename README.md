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
- **Complex File Names**: The original file names were long and complex, such as "1-s2.0-S0360128523000023-main.pdf", making it hard to read and compare the extraction results with the actual words in the PDF. To simplify this process, files were renamed to a consistent format (e.g., `paper_2` to `paper_80`). This renaming made it much easier to manually check the extracted abstracts.

### Key Findings from Initial Data Processing

1. During the preprocessing phase, it was discovered that the abstracts typically end with the keywords "introduction", "index terms" and "keywords". This observation is crucial for accurately extracting the abstract section from the research papers.
2. In some PDFs, the introduction starts with "1. Introduction", causing "1." to be included in the abstract. (Consider using stop words from NLP to handle this.)
3. In `paper_69` and `paper_70`, the abstract title "ABSTRACT" in red was not recognized by the system.
4. In `paper_50`, the "K E Y W O R D S" were not recognized, resulting in the extraction of the entire article. This is likely due to the spaces in the keyword.
5. In `paper_51`, `paper_53`, `paper_58`, `paper_59`, `paper_63`, `paper_66`, `paper_72`, and `paper_73`, there is no keyword "abstract"; the abstracts are just sections at the beginning of the articles.
6. In the abstract for `paper_66`, the words are stuck together after extraction, possibly because the abstract section has a yellow background in the original PDF.
7. In `paper_79`, redundant words like university affiliations and email addresses were included in the extracted abstract. These are references at the bottom of page 1 but were mistakenly included in the abstract.

### Addressing Key Issues with OCR Integration

To handle edge cases where the PDF's text is not recognized due to formatting issues (e.g., red "ABSTRACT"), OCR (Optical Character Recognition) was integrated using Tesseract and Poppler. This approach was particularly useful for PDFs like `paper_69` and `paper_70`, where the abstract title "ABSTRACT" in red was not recognized by the system.

- **Steps to integrate OCR**:
  - Installed Homebrew to manage dependencies.
  - Added Homebrew to PATH and installed Tesseract and Poppler.
  - Verified installations to ensure proper setup.
  - Updated the requirements file to include `pdf2image`.

The integration of OCR successfully addressed the issue in `paper_50`, where keywords were not recognized due to the library used. The OCR solution proved to be effective for extracting text in cases where the initial approach failed due to formatting issues.

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


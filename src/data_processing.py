import os
import re
import nltk
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class DataProcessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def rename_files(self, directory):
        """
        Rename files in the given directory to a consistent format.
        """
        for i, filename in enumerate(sorted(os.listdir(directory))):
            if filename.endswith('.pdf'):
                new_name = f"paper_{i+1}.pdf"
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

    def extract_abstracts(self, directory):
        """
        Extracts abstracts from PDF files in the specified directory.

        Args:
            directory (str): Path to the directory containing PDF files.

        Returns:
            list: List of tuples containing file names and extracted abstracts.
        """
        # Rename files for consistency
        # self.rename_files(directory)
        
        # List all renamed PDF files and sort them
        file_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')])

        results = []
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    text = ''
                    for page_num in range(min(2, len(reader.pages))):  # Extract text from the first two pages only
                        text += reader.pages[page_num].extract_text()
                    # Print a small portion of the full extracted text for debugging
                    print(f"Extracted text from {file_path}:\n{text[:500]}...\n")
                    abstract = self._extract_abstract(text)
                    if abstract:
                        results.append((file_path, abstract))
                    else:
                        results.append((file_path, "Abstract not found"))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return results


    def _extract_abstract(self, text):
        """
        Extracts the abstract section from the text using improved regex and NLP.

        Args:
            text (str): Full text extracted from a PDF.

        Returns:
            str: Extracted abstract text.
        """
        # Use improved regular expression to find the 'Abstract' section
        abstract_match = re.search(r'(?i)(abstract)[:\s]*(.*?)(?=(1\.\s*introduction|introduction|keywords|index terms|references|acknowledgements|bibliography))', text, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(2).strip()
            abstract = self._clean_abstract(abstract)
            return abstract
        return None

    
    def _clean_abstract(self, abstract):
        """
        Cleans the extracted abstract by removing redundant phrases.

        Args:
            abstract (str): Extracted abstract text.

        Returns:
            str: Cleaned abstract text.
        """
        # Remove common redundant phrases
        redundant_phrases = [
            r"(\d{4} )?elsevier ltd", r"all rights reserved", r"©", r"doi:", r"published by",
            r"\d{4} (elsevier|springer|wiley|taylor & francis) [\w\s]+", r"K\s*E\s*Y\s*W\s*O\s*R\s*D\s*S",
            r"School of [\w\s]+", r"Email: [\w\s@.]+", r"Corresponding Author:", r"Data Availability Statement included at the end of the article."
        ]


        for phrase in redundant_phrases:
            abstract = re.sub(phrase, '', abstract, flags=re.IGNORECASE)
        
        # Remove trailing references, copyright lines, and excessive whitespace
        abstract = re.sub(r'\s*(references|acknowledgements|bibliography)\s*$', '', abstract, flags=re.IGNORECASE)
        abstract = re.sub(r'\s*[\d]+\s*$', '', abstract, flags=re.IGNORECASE)
        abstract = re.sub(r'([a-z])([A-Z])', r'\1 \2', abstract) # deal with words stuck together due to different background color
        abstract = re.sub(r'\s+', ' ', abstract).strip()
        
        return abstract

    def _refine_abstract(self, abstract):
        """
        Refines the extracted abstract using NLP techniques.

        Args:
            abstract (str): Extracted abstract text.

        Returns:
            str: Refined abstract text.
        """
        doc = nlp(abstract)
        sentences = [sent.text for sent in doc.sents]
        refined_abstract = ' '.join(sentences)
        return refined_abstract

    def preprocess_text(self, text):
        """
        Preprocesses the text by cleaning, removing stop words, and lemmatizing.

        Args:
            text (str): Raw text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove non-word characters (punctuation, etc.)
        text = re.sub(r'\W', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Tokenize and remove stop words, then lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
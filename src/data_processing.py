import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

class DataProcessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def extract_abstracts(self, file_paths):
        """
        Extracts abstracts from a list of PDF files.

        Args:
            file_paths (list): List of paths to PDF files.

        Returns:
            list: List of extracted abstracts.
        """
        abstracts = []
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page_num in range(reader.numPages):
                        text += reader.getPage(page_num).extractText()
                    abstract = self._extract_abstract(text)
                    if abstract:
                        abstracts.append(abstract)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return abstracts

    def _extract_abstract(self, text):
        """
        Extracts the abstract section from the text.

        Args:
            text (str): Full text extracted from a PDF.

        Returns:
            str: Extracted abstract text.
        """
        # Use regular expression to find the 'Abstract' section
        abstract_match = re.search(r'(?i)abstract[:\s]+(.*?)(?=introduction|methods|results|conclusion|references|keywords|\Z)', text, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            return abstract
        return None

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
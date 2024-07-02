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
        abstracts = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfFileReader(f)
                text = ''
                for page_num in range(reader.numPages):
                    text += reader.getPage(page_num).extractText()
                abstracts.append(text)
        return abstracts

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

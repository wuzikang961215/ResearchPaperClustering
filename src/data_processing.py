import os
import re
import nltk
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import csv
from pdfminer.high_level import extract_text
import pandas as pd

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class DataProcessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def extract_text_from_first_two_pages(self, pdf_path):
        try:
            # Extract text from the first two pages
            text = extract_text(pdf_path, page_numbers=[0, 1])
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def save_text_to_file(self, text, output_path):
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(text)
        except Exception as e:
            print(f"Error saving text to {output_path}: {e}")

        def process_pdfs(self, input_directory, output_directory):
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            for filename in os.listdir(input_directory):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(input_directory, filename)
                    text = self.extract_text_from_first_two_pages(pdf_path)
                    output_path = os.path.join(output_directory, filename.replace('.pdf', '.txt'))
                    self.save_text_to_file(text, output_path)

    def process_pdfs(self, input_directory, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for filename in os.listdir(input_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(input_directory, filename)
                text = self.extract_text_from_first_two_pages(pdf_path)
                output_path = os.path.join(output_directory, filename.replace('.pdf', '.txt'))
                self.save_text_to_file(text, output_path)


    def correct_misplaced_abstract(self, text):
        """
        Corrects the abstract if it is misplaced after the "1. Introduction" section.
        
        Args:
            text (str): The text to correct.
        
        Returns:
            str: The text with the abstract corrected if it was misplaced.
        """
        # Pattern to find the misplaced abstract right after "1. Introduction"
        introduction_pattern = re.compile(r'(1\. Introduction\s+)(.*?)(\n\n\s*[A-Z])', re.DOTALL | re.IGNORECASE)
        match = introduction_pattern.search(text)
        
        if match:
            introduction_section = match.group(1)
            misplaced_abstract = match.group(2).strip()
            following_content = match.group(3)
            
            # Check if the misplaced abstract starts with common abstract phrases
            if misplaced_abstract and re.match(r'^[A-Z]', misplaced_abstract):
                text = text.replace(misplaced_abstract, '').strip()
                # Insert the misplaced abstract at the correct position
                text = re.sub(r'(A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*)', r'\1\n' + misplaced_abstract + '\n\n', text, flags=re.IGNORECASE)
        
        return text

    def extract_abstract(self, text):
        """
        Extracts the abstract from the given text.

        Args:
            text (str): The text from which to extract the abstract.

        Returns:
            str: The extracted abstract.
        """
        # Remove graphical abstract section if present
        text = re.sub(r'G R A P H I C A L  A B S T R A C T.*?A R T I C L E  I N F O', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Define the primary pattern to search for the abstract section
        abstract_pattern1 = re.compile(
            r'A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*'
            r'(?:.*?Keywords:\s*(?:\w+.*?\n)*)(.*?)'
            r'(?:1\. Introduction|1\. Background|Introduction|E-mail addresses|Corresponding authors|Shared first authorship|doi|DOI|©|Received|Manuscript received|Accepted|Available online|Article history:|Keywords:|2017 The Author|2017 Elsevier Ltd.|Nomenclature|copyright|corresponding author|1⋅h|2023 International Association)',
            re.DOTALL | re.IGNORECASE)

        # Define the secondary pattern to search for the abstract section including potential keywords
        abstract_pattern2 = re.compile(
            r'A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*(.*?)\s*(?:JEL classification:|1\. Introduction|Introduction|E-mail addresses|Corresponding authors|Manuscript received|Shared first authorship|doi|DOI|©|Received:|Accepted:|Available online|Article history:|Keywords:|1\. Background|K E Y W O R D S\nenergy storage|index terms|K\s*E\s*Y\s*W\s*O\s*R\s*D\s*S\s*\n*\n*alternative|Keywords  Energy saving|Keywords  Plants|CrossCheck date: 17 September)',
            re.DOTALL | re.IGNORECASE)
        
         # Define the third pattern to extract abstract without explicit labeling
        abstract_pattern3 = re.compile(
            r'(?:Shandong, China|Peter H\. L\. Notten|Genqiang Zhang|98195, United States of America|Yuliang Cao|Xuping Sun|abc|Dingsheng Wang|Roghayeh Sadeghi Erami1,2|Di-Jia Liu1,7).*?\n\s*(.*?)(?=\nKeywords:|\n1\. Introduction|Y\. Li, M\. Cheng|2023 The Author\(s\)|can  provide  both  high  energy|world  population  and|cid:44|Freshwater is likely to|L ow-temperature water)', 
            re.DOTALL | re.IGNORECASE)


        match = abstract_pattern1.search(text)
        
        if match:
            abstract = match.group(1).strip()
            # Remove additional noise like dates and publisher notes
            abstract = re.sub(r'\(cid:.*?\)', '', abstract)  # Remove "(cid:...)" noise
            abstract = re.sub(r'©.*$', '', abstract)  # Remove © and anything that follows
            abstract = re.sub(r'\d{4} Elsevier Ltd\. All rights reserved\.', '', abstract)  # Remove "2016 Elsevier Ltd. All rights reserved."
            if abstract.strip():  # Check if abstract is not an empty string
                return abstract
        
        # Try correcting misplaced abstract if not found or empty
        text_corrected = self.correct_misplaced_abstract(text)
        # if text_corrected:
        #     return text_corrected
        match = abstract_pattern2.search(text_corrected)

        if match:
            abstract = match.group(1).strip()
            # Remove additional noise like dates and publisher notes
            abstract = re.sub(r'\(cid:.*?\)', '', abstract)  # Remove "(cid:...)" noise
            abstract = re.sub(r'©.*$', '', abstract)  # Remove © and anything that follows
            abstract = re.sub(r'\d{4} Elsevier Ltd\. All rights reserved\.', '', abstract)  # Remove "2016 Elsevier Ltd. All rights reserved."
            abstract = re.sub(r'Marine engines mainly.*?the climate goals\.', '', abstract, flags=re.DOTALL)  # Remove text from "Marine engines mainly" to "the climate goals."
            abstract = re.sub(r'Investment in large-scale renewable.*?to host and support', '', abstract, flags=re.DOTALL)  # Remove the paragraph from "Investment in large-scale renewable" to "to host and support"
            abstract = re.sub(r'As the catalyst.*?the climate system \[7\]\.', '', abstract, flags=re.DOTALL)  # Remove the paragraph from "As the catalyst" to "the climate system [7]."
            abstract = re.sub(r'Renewable energy is the best solution.*?are also becoming popular', '', abstract, flags=re.DOTALL)  # Remove the paragraph from "Renewable energy is the best solution" to "are also becoming popular"
            abstract = re.sub(r'The sun is a huge sphere.*?this combined solar irradiance\.', '', abstract, flags=re.DOTALL)  # Remove the paragraph from "The sun is a huge sphere" to "this combined solar irradiance."
            abstract = re.sub(r'The global energy outlook.*?long-term economic growth \[2\]\.', '', abstract, flags=re.DOTALL)  # Remove the paragraph from "The global energy outlook" to "long-term economic growth [2]."
            return abstract

        match = abstract_pattern3.search(text)

        if match:
            abstract = match.group(1).strip()
            # Remove additional noise like dates and publisher notes
            abstract = re.sub(r'\(cid:.*?\)', '', abstract)  # Remove "(cid:...)" noise
            abstract = re.sub(r'©.*$', '', abstract)  # Remove © and anything that follows
            abstract = re.sub(r'\d{4} Elsevier Ltd\. All rights reserved\.', '', abstract)  # Remove "2016 Elsevier Ltd. All rights reserved."
            abstract = re.sub(r'Received \d{1,2}(st|nd|rd|th) [A-Za-z]+ \d{4}\s*DOI: 10\.\d{4}/[a-zA-Z0-9]+\s*rsc\.li/chem-soc-rev', '', abstract, flags=re.DOTALL)  # Remove "Received ..." to "rsc.li/chem-soc-rev"
            abstract = re.sub(r'l\s*e\s*c\s*i\s*t\s*r\s*A\s*w\s*e\s*v\s*e\s*R\s*i\s*\s*§', '', abstract, flags=re.DOTALL | re.IGNORECASE)  # Remove the noise "l e c i t r A w e v e R i §"
            return abstract

        return "Abstract not found"

    def process_text_files(self, input_directory, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for filename in os.listdir(input_directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                abstract = self.extract_abstract(text)
                abstract = self._refine_abstract(abstract)
                abstract = self.preprocess_text(abstract)
                output_path = os.path.join(output_directory, filename.replace('.txt', '_abstract.txt'))
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(abstract)


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
    

    def save_abstracts_to_csv(self, input_dir, output_file):
        """
        Reads all abstract output text files from the input directory and saves them into a CSV file.

        Args:
            input_dir (str): The directory containing the abstract output text files.
            output_file (str): The path to the output CSV file.
        """
        abstracts = []

        # Iterate through all text files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    abstract = file.read().strip()
                    abstracts.append({"File Name": filename, "Abstract": abstract})

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(abstracts, columns=["File Name", "Abstract"])

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False, encoding='utf-8')

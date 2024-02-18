import pandas as pd
import nltk
from nltk.corpus import stopwords, words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("words")
from unidecode import unidecode
import re
import contractions
import string
from langdetect import detect

def load_data(path="data/fake_reviews_dataset.csv", index_col=None, header=0, encode="utf8"):
    data = pd.read_csv(path, index_col=index_col, header=header, encoding=encode)
    return data

class BasicTextCleaning:
    def __init__(self):
        # define some necessary elements
        self.stopwords = set(stopwords.words('english'))
        self.words_corpus = set(words.words())
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # dictionary of methods can be used
        self.methods = {'lowercase': str.lower,
                        'accent_removal': self.accent_removal,
                        'strip': str.strip,
                        'nice_display': self.nice_display,
                        'tokenization': str.split,
                        'stemming': self.stemming,
                        'lemmatization': self.lemmatization,
                        'punctuation_removal': self.punctuation_removal,
                        'stopwords_removal': self.stopwords_removal,
                        'contractions_expand': self.contractions_expand}
        
        self.punctuations = '[%s]' % re.escape(string.punctuation)

    def text_cleaning(self, texts, methods=None):
        if not methods:
            methods = ['accent_removal', 'lowercase', 'contractions_expand', 'nice_display',
                       'punctuation_removal', 'lemmatization', 'stopwords_removal', 'tokenization']
        if isinstance(texts, str):
            texts = [texts]
        cleaned_texts = []
        for text in texts:
            for method in methods:
                text = self.methods[method](text)
            cleaned_texts.append(text)
        return cleaned_texts

    def strip_text(self, text):
        return text.strip()
    
    def lowercase(self, text):
        return text.lower()

    def contractions_expand(self, text):
        return contractions.fix(text)

    def nice_display(self, text):
        text = re.sub(r"([^\w\s([{\'])(\w)", r"\1 \2", text)
        # text = re.sub(r"(\d+)([a-zA-Z]+)|([a-zA-Z]+)(\d+)", r"\1 \2 \3 \4", text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def accent_removal(self, text):
        text = unidecode(text)
        return text

    def punctuation_removal(self, text):
        text = re.sub(self.punctuations, ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def stopwords_removal(self, text):
        return " ".join([word for word in text.split() if (word not in self.stopwords and len(word)>1) or word.isnumeric()])

    def stemming(self, text):
        return " ".join([self.stemmer.stem(word) for word in text.split()])

    def lemmatization(self, text):
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def tokenization(self, text):
        return text.split(" ")

if __name__ == '__main__':
    text = "   oh My God!It's 9more beautiful than i HAve expected (but expensive T_T!!!).  h Love it! =))). ré sumes ỏ feet."
    preprocessor = BasicTextCleaning()
    print(preprocessor.text_cleaning(text,
                                     methods=['accent_removal', 'lowercase', 'contractions_expand',
                                              'punctuation_removal', 'lemmatization', 'stopwords_removal']))
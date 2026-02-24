import re
import nltk
from typing import List
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd

# download stopwords jika belum adapip
nltk.download('stopwords')

class data_processing:
    def __init__(self):
        pass
    
    def process_data(self):

        df_train = pd.read_csv('https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/train.csv')
        df_valid = pd.read_csv('https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/valid.csv')
        df_test = pd.read_csv('https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/test.csv')

        print(df_train.shape)
        print(df_valid.shape)
        print(df_test.shape)

        preprocessor = IndonesianTextPreprocessor(remove_stopwords=False)

        df_train_clean = df_train.copy()
        df_valid_clean = df_valid.copy()
        df_test_clean = df_test.copy()

        df_train_clean['text'] = preprocessor.preprocess_dataset(df_train['text'].astype(str).tolist())
        df_valid_clean['text'] = preprocessor.preprocess_dataset(df_valid['text'].astype(str).tolist())
        df_test_clean['text'] = preprocessor.preprocess_dataset(df_test['text'].astype(str).tolist())

        return df_train_clean, df_valid_clean, df_test_clean

class IndonesianTextPreprocessor:

    def __init__(self,
                 lowercase: bool = True,
                 remove_special: bool = True,
                 remove_stopwords: bool = True,
                 tokenize: bool = True):

        self.lowercase = lowercase
        self.remove_special = remove_special
        self.remove_stopwords = remove_stopwords
        self.tokenize = tokenize

        # Load stopwords dari dua sumber
        self.stopwords_nltk = set(nltk.corpus.stopwords.words('indonesian'))

        factory = StopWordRemoverFactory()
        self.stopwords_sastrawi = set(factory.get_stop_words())

        # gabungkan
        self.stopwords = self.stopwords_nltk.union(self.stopwords_sastrawi)


    # 1. Lowercase
    def lowercase_text(self, text: str) -> str:
        return text.lower()


    # 2. Remove special characters
    def remove_special_characters(self, text: str) -> str:

        # remove URL
        text = re.sub(r'http\S+|www\S+', '', text)

        # remove mention
        text = re.sub(r'@\w+', '', text)

        # remove hashtag symbol only
        text = re.sub(r'#', '', text)

        # keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    # 3. Tokenization
    def tokenize_text(self, text: str) -> List[str]:
        tokens = text.split()
        return tokens


    # 4. Remove stopwords
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        filtered = [word for word in tokens if word not in self.stopwords]
        return filtered


    # 5. Full pipeline for single text
    def preprocess_text(self, text: str) -> str:

        if self.lowercase:
            text = self.lowercase_text(text)

        if self.remove_special:
            text = self.remove_special_characters(text)

        if self.tokenize:
            tokens = self.tokenize_text(text)
        else:
            tokens = [text]

        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)

        cleaned_text = " ".join(tokens)

        return cleaned_text


    # 6. Preprocess dataset (list of text)
    def preprocess_dataset(self, texts: List[str]) -> List[str]:

        cleaned_texts = []

        for text in texts:
            cleaned = self.preprocess_text(text)
            cleaned_texts.append(cleaned)

        return cleaned_texts
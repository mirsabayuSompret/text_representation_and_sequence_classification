from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class BaseEmbedding(ABC):

    @abstractmethod
    def fit(self, texts) -> None:
        pass

    @abstractmethod
    def transform(self, texts) -> np.ndarray:
        pass

    def fit_transform(self, texts) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
    
class TfidfEmbedding(BaseEmbedding):

    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()

class SBERTEmbedding(BaseEmbedding):

    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        pass  # SBERT tidak perlu training

    def transform(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings    

    
class FastTextEmbedding(BaseEmbedding):

    def __init__(self, embedding_dim=300, max_length=50):

        self.model = api.load("fasttext-wiki-news-subwords-300")
        self.embedding_dim = embedding_dim
        self.max_length = max_length

    def fit(self, texts):
        pass

    def text_to_sequence(self, text):

        words = text.split()
        sequence = []

        for word in words[:self.max_length]:

            if word in self.model:
                sequence.append(self.model[word])
            else:
                sequence.append(np.zeros(self.embedding_dim))

        while len(sequence) < self.max_length:
            sequence.append(np.zeros(self.embedding_dim))

        return np.array(sequence)

    def transform(self, texts):

        sequences = []

        for text in texts:
            seq = self.text_to_sequence(text)
            sequences.append(seq)

        return np.array(sequences)


class BERTEmbedding(BaseEmbedding):

    def __init__(self, model_name, max_length=50, device=None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.max_length = max_length

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()


    def fit(self, texts):
        pass


    def transform(self, texts):

        embeddings = []

        with torch.no_grad():

            for text in texts:

                inputs = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                hidden_states = outputs.last_hidden_state

                embeddings.append(hidden_states.squeeze().cpu().numpy())

        return np.array(embeddings)
    
class EmbeddingFactory:

    @staticmethod
    def create(embedding_type):

        if embedding_type == "tfidf":
            return TfidfEmbedding()

        elif embedding_type == "sbert":
            return SBERTEmbedding()

        elif embedding_type == "fasttext":
            return FastTextEmbedding()

        elif embedding_type == "indobert":
            return BERTEmbedding("indobenchmark/indobert-base-p1")

        elif embedding_type == "mbert":
            return BERTEmbedding("bert-base-multilingual-cased")

        else:
            raise ValueError("Unknown embedding type")
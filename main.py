from sklearn.calibration import LabelEncoder
from torch import le
import data_processing as dp
import embeddings as emb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

class main():
    def __init__(self):
        pass

    def initialize(self):
        # data processing, cleanign and splitting data to train, valid and test
        data = dp.data_processing()
        train, valid, test = data.process_data()

        # embedding data with various embedding techniques
        tfidf_embedding = emb.EmbeddingFactory.create("tfidf")
        sbert_embedding = emb.EmbeddingFactory.create("sbert")
        fasttext_embedding = emb.EmbeddingFactory.create("fasttext")
        indobert_embedding = emb.EmbeddingFactory.create("indobert")
        mbert_embedding = emb.EmbeddingFactory.create("mbert")

        sbert_embed = sbert_embedding.fit_transform(train['text'])
        sbert_valid = sbert_embedding.transform(valid['text'])
        sbert_test = sbert_embedding.transform(test['text'])

        tfidf_embed = tfidf_embedding.fit_transform(train['text'])
        tfidf_valid = tfidf_embedding.transform(valid['text'])
        tfidf_test = tfidf_embedding.transform(test['text'])

        fasttext_embed = fasttext_embedding.fit_transform(train['text'])
        fasttext_valid = fasttext_embedding.transform(valid['text'])
        fasttext_test = fasttext_embedding.transform(test['text'])

        indobert_embed = indobert_embedding.fit_transform(train['text'])
        indobert_valid = indobert_embedding.transform(valid['text'])
        indobert_test = indobert_embedding.transform(test['text'])


        mbert_embed = mbert_embedding.fit_transform(train['text'])
        mbert_valid = mbert_embedding.transform(valid['text'])
        mbert_test = mbert_embedding.transform(test['text'])

        le = LabelEncoder()
        y_train = le.fit_transform(train['label'])
        y_valid = le.transform(valid['label'])
        y_test = le.transform(test['label'])

        def get_pooled_embeddings(embedding_name, train, valid, test):

            # If already 2D, return as is
            if len(train.shape) == 2:
                return train, valid, test

            # Mean pool over sequence length (axis 1)
            print(f"Pooling {embedding_name} from {train.shape}...")
            return np.mean(train, axis=1), np.mean(valid, axis=1), np.mean(test, axis=1)
        
        # Define classifiers
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier()
        }

        # Define embedding dictionary
        embedding_data = {
            "TF-IDF": (tfidf_embed, tfidf_valid, tfidf_test),
            "SBERT": (sbert_embed, sbert_valid, sbert_test),
            "FastText": (fasttext_embed, fasttext_valid, fasttext_test),
            "IndoBERT": (indobert_embed, indobert_valid, indobert_test),
            "mBERT": (mbert_embed, mbert_valid, mbert_test)
        }

        results = []

        for embed_name, (X_train_raw, X_valid_raw, X_test_raw) in embedding_data.items():

            # Pool if needed
            X_train, X_valid, X_test = get_pooled_embeddings(embed_name, X_train_raw, X_valid_raw, X_test_raw)

            print(f"\nEvaluating {embed_name} (shape: {X_train.shape})...")

            for clf_name, clf in classifiers.items():

                # Train
                clf.fit(X_train, y_train)

                # Predict on Test
                y_pred = clf.predict(X_test)

                # Evaluate
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                results.append({
                    "Embedding": embed_name,
                    "Classifier": clf_name,
                    "Accuracy": acc,
                    "F1-Score": f1
                })

                print(f"  {clf_name}: Accuracy={acc:.4f}, F1={f1:.4f}")



        pass

if __name__ == "__main__":
    main()
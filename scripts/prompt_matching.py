import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, set_seed
from transformers import AutoModel, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.append("../")


class PromptMatching:

    def __init__(self):
        self.df = os.environ.get("DATA_PATH")
        #self.df = pd.read_csv('/Users/iqraimtiaz/Documents/duke/Courses/ind-study/E-commerce-Product-Reccommendation/data/products.csv')
        self.generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M', device='cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModel.from_pretrained('gpt2')
        self.bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

        set_seed(42)
        self.df.fillna('', inplace=True)
        self.df['Details'] = self.df['Name'] + ' ' + self.df['Short description'] + ' ' + self.df['Description']
        self.df['Image_URL'] = self.df['Images'].apply(lambda x: x.split(',')[0])

    def cosine_similarity(self, prompt):
        vectorizer = TfidfVectorizer()
        prompt_vector = vectorizer.fit_transform([prompt])

        product_vectors = vectorizer.transform(self.df['Details'])
        similarity_scores = cosine_similarity(prompt_vector, product_vectors).flatten()
        self.df['tf-idfDetails'] = np.round(similarity_scores, 2)
        return self.df

    def encode(self, input_text, model, tokenizer):
        if not input_text.strip():
            return np.zeros((1, model.config.hidden_size))
        tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state
        return embeddings.mean(dim=1).numpy()


    def compute_similarity(self, search_prompt, model, tokenizer, result_col_name):
        product_descriptions = self.df['Details']
        encoded_product_descriptions = [self.encode(desc, model, tokenizer) for desc in product_descriptions]
        encoded_search_prompt = self.encode(search_prompt, model, tokenizer)

        # Normalize the embeddings
        encoded_product_descriptions = [emb / np.linalg.norm(emb) for emb in encoded_product_descriptions]
        encoded_search_prompt = encoded_search_prompt / np.linalg.norm(encoded_search_prompt)

        # Reshape the tensors to have compatible dimensions for inner product
        encoded_product_descriptions = np.array(encoded_product_descriptions).reshape(len(product_descriptions), -1)
        encoded_search_prompt = np.array(encoded_search_prompt).reshape(-1)

        # Compute the similarity scores
        similarity_scores = np.inner(encoded_product_descriptions, encoded_search_prompt)
        similarity_scores = np.round(similarity_scores, 2)

        self.df[result_col_name] = similarity_scores
        return self.df


    def gpt3_similarity(self, search_prompt):
        return self.compute_similarity(search_prompt, self.model, self.tokenizer, "gpt3Details")

    def bert_similarity(self, search_prompt):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        return self.compute_similarity(search_prompt, model, tokenizer, "bertDetails")


    def sentence_transformer(self, search_prompt):
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        product_descriptions = self.df['Details']
        encoded_product_descriptions = model.encode(product_descriptions)
        encoded_search_prompt = model.encode(search_prompt)
        similarity_scores = np.inner(encoded_product_descriptions, encoded_search_prompt)
        similarity_scores = np.round(similarity_scores, 2)
        self.df['transformersDetails'] = similarity_scores
        return self.df

if __name__ == "__main__":
    prompt_matching = PromptMatching()
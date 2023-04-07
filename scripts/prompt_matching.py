# create a class for matching prompt to book
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch


class PromptMatching:
    
    def __init__(self):
      # apply the function to each element of the column using multiprocessing
        self.df = pd.read_csv('data/products.csv')

    def keyword_matching(self,prompt):
        # Prompt and relevant keywords
        vectorizer = CountVectorizer()
        keywords = vectorizer.fit_transform([prompt]).toarray()[0]

        # Loop through each book summary
        keywords_match = []
        for summary in self.df['Summary']:
            if isinstance(summary, str):
                # Check if summary contains all relevant keywords
                book_keywords = vectorizer.transform([summary]).toarray()[0]
                match = all(book_keywords[i] >= keywords[i] for i in range(len(keywords)))
            else:
                # If summary is NaN, set match to False
                match = False
            keywords_match.append(match)

        # Add keywords_match column to self.df dataframe
        self.df['keywords_match'] = keywords_match
        return self.df
        
    def cosine_similarity(self,prompt):
        vectorizer = TfidfVectorizer()
        prompt_vector = vectorizer.fit_transform([prompt])

        for col in ['Name', 'Short description', 'Description']:
            cosine_similarity_scores = []
            for desc in self.df[col]:
                if isinstance(desc, str):
                    desc_vector = vectorizer.transform([desc])
                    similarity = cosine_similarity(prompt_vector, desc_vector)[0][0]
                else:
                    # If summary is NaN, set similarity score to NaN
                    similarity = np.nan
                cosine_similarity_scores.append(similarity)

            self.df['cs'+col] = cosine_similarity_scores
        return self.df
        
    def bert_matching(self,prompt):
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        book_summaries = self.df['Summary'].tolist()
        book_ids = self.df.index.tolist()

        book_embeddings = []
        for summary in book_summaries:
            summary_tokens = tokenizer(summary, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            with torch.no_grad():
                summary_embedding = model(summary_tokens['input_ids'], summary_tokens['attention_mask'])[0][:, 0, :]
                book_embeddings.append(summary_embedding)

        book_embeddings = torch.cat(book_embeddings, dim=0)
        book_embeddings = book_embeddings.reshape(len(book_summaries), -1)
        print(book_embeddings)
        prompt_tokens = tokenizer(prompt, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        with torch.no_grad():
            prompt_embedding = model(prompt_tokens['input_ids'], prompt_tokens['attention_mask'])[0][:, 0, :]

        similarities = cosine_similarity(prompt_embedding.reshape(1, -1), book_embeddings).squeeze().tolist()
        self.df['bert_cosine_similarity'] = similarities
        #ranked_self.df = sorted(zip(book_ids, similarities), key=lambda x: x[1], reverse=True)

        #return ranked_self.df
        
        
    def get_matched_prompt_results(self,prompt,col):
        self.cosine_similarity(prompt,self.df,col)
        matched = self.df.sort_values(by='cosine_similarity', ascending=False)
        return matched.head(1)['cosine_similarity'].values[0]
        
# create main for this class

if __name__ == "__main__":
    # initialize this class
    prompt_matching = PromptMatching()
    
    prompt = "Looking for a book that explores the changing role of religion in the 20th century. Specifically, how certain religious groups redefined what it meant to be religious and allowed their members the choice of what kind of God to believe in, or the option to not believe in God at all."
    
        



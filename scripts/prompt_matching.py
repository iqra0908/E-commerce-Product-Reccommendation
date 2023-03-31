# create a class for matching prompt to book
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch


class PromptMatching:

    def keyword_matching(self,prompt,books):
        # Prompt and relevant keywords
        vectorizer = CountVectorizer()
        keywords = vectorizer.fit_transform([prompt]).toarray()[0]

        # Loop through each book summary
        keywords_match = []
        for summary in books['Summary']:
            if isinstance(summary, str):
                # Check if summary contains all relevant keywords
                book_keywords = vectorizer.transform([summary]).toarray()[0]
                match = all(book_keywords[i] >= keywords[i] for i in range(len(keywords)))
            else:
                # If summary is NaN, set match to False
                match = False
            keywords_match.append(match)

        # Add keywords_match column to books dataframe
        books['keywords_match'] = keywords_match
        
    def cosine_similarity(self,prompt,books,col):
        vectorizer = TfidfVectorizer()
        prompt_vector = vectorizer.fit_transform([prompt])

        # Loop through each book summary
        cosine_similarity_scores = []
        for summary in books[col]:
            if isinstance(summary, str):
                # Calculate cosine similarity between summary and prompt
                book_vector = vectorizer.transform([summary])
                similarity = cosine_similarity(prompt_vector, book_vector)[0][0]
            else:
                # If summary is NaN, set similarity score to NaN
                similarity = np.nan
            cosine_similarity_scores.append(similarity)

        # Add cosine_similarity column to books dataframe
        books['cosine_similarity'] = cosine_similarity_scores
        
    def bert_matching(self,prompt, books):
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        book_summaries = books['Summary'].tolist()
        book_ids = books.index.tolist()

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
        books['bert_cosine_similarity'] = similarities
        #ranked_books = sorted(zip(book_ids, similarities), key=lambda x: x[1], reverse=True)

        #return ranked_books
        
    def combine_summaries(self):
        books_df = pd.read_csv('data/duke_books.csv')
        abs_df = pd.read_csv('data/duke_books_abstractive.csv')
        ext_df = pd.read_csv('data/extractive_summary_df.csv')
        
        merged_data = books_df.copy()
        merged_data = pd.merge(merged_data, abs_df[['Title','abbreviated_summary']], how='inner', on='Title')
        merged_data = pd.merge(merged_data, ext_df[['Title','extractive_summary']], how='inner', on='Title')

        # drop duplicates based on the 'Title' column
        merged_data.drop_duplicates(subset=['Title'], inplace=True)
        merged_data.dropna(subset=['Summary'], inplace=True)

        # save the merged data to a new CSV file
        merged_data.to_csv('data/books_with_summaries.csv', index=False)
        
    def get_matched_prompt_results(self,prompt,books,col):
        self.cosine_similarity(prompt,books,col)
        matched = books.sort_values(by='cosine_similarity', ascending=False)
        return matched.head(1)['cosine_similarity'].values[0]
        
    def run_validation_prompts(self):
        validation_prompts = pd.read_csv('data/validation_prompts.csv')
        validation_prompts.dropna(inplace=True)
        validation_prompts = validation_prompts['prompt'].tolist()
        books = pd.read_csv('data/books_with_summaries.csv')
        # create empty dataframe to store results
        results_df = pd.DataFrame(columns=['prompt', 'summary_cs', 'abb_summary_cs', 'ex_summary_cs'])
    
        for prompt in validation_prompts:
            cs = self.get_matched_prompt_results(prompt,books,'Summary')
            ab_cs = self.get_matched_prompt_results(prompt,books,'abbreviated_summary')
            ext_cs = self.get_matched_prompt_results(prompt,books,'extractive_summary')
            # create a new row for the results dataframe
            new_row = {'prompt': prompt, 'summary_cs': cs, 'abb_summary_cs': ab_cs, 'ex_summary_cs': ext_cs}
        
            # append the row to the results dataframe
            results_df = results_df.append(new_row, ignore_index=True)
            
        results_df.to_csv('data/prompts_with_cs.csv', index=False)  
 
# create main for this class

if __name__ == "__main__":
    # initialize this class
    prompt_matching = PromptMatching()
    #prompt_matching.combine_summaries()
    prompt_matching.run_validation_prompts()
    
    #prompt = "Find a book about a detective solving a murder mystery in a small town."
    prompt = "Looking for a book that explores the changing role of religion in the 20th century. Specifically, how certain religious groups redefined what it meant to be religious and allowed their members the choice of what kind of God to believe in, or the option to not believe in God at all."
    
        



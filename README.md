# E-commerce Product Recommendation System

This code provides a web-based interface for searching for product recommendations based on a search prompt. The user can enter a search prompt or select a sample prompt, then select a similarity method to find the top product recommendations. The similarity methods available are: TF-IDF, GPT-3, BERT, and Transformers.

## Dependencies

The code requires the following libraries to be installed:

- streamlit
- pandas
- numpy
- sklearn
- transformers
- torch
- sentence_transformers

## Usage

To run the application, execute the following command in your terminal:

streamlit run app.py


The application will start running on your local machine and can be accessed in your web browser at the address `http://localhost:8501`.

## Notes

- The product data is assumed to be stored in a CSV file at the location specified by the `DATA_PATH` environment variable.
- The `PromptMatching` class provides methods for computing similarity scores between a search prompt and product descriptions, using different techniques such as TF-IDF, GPT-3, BERT, and sentence transformers. These methods are used by the app to compute the recommendations.
- The app interface is built using the `streamlit` library, and it is defined in the `app()` function. When the `app.py` file is executed, the `app()` function is called, and the app is started.
- The `display_results()` function is used to display the results of the search in the app. It displays the top 3 products that match the search prompt, along with their images and additional information that can be expanded by the user.
- The app allows the user to select a similarity method to use for the search. When the search button is clicked, the selected method is called with the search prompt as input, and the results are displayed using the `display_results()` function.

import sys
sys.path.append("../")
import streamlit as st
from scripts import prompt_matching
import pandas as pd

# Set page config
st.set_page_config(page_title="Product Search", page_icon=":products:", layout="wide")
pm = prompt_matching.PromptMatching()


def search_products_by_cosine_similarity(prompt, num_products=3):
    products = pm.cosine_similarity(prompt)
    matched = products
    matched = matched.sort_values(by=['tf-idfDetails'], ascending=False)
    return matched.nlargest(5, ['tf-idfDetails'])

def search_products_by_transformers_similarity(prompt, num_products=5):
    products = pm.sentence_transformer(prompt)
    matched = products
    matched = matched.sort_values(by=['transformersDetails'], ascending=False)
    return matched.nlargest(num_products, ['transformersDetails'])

def search_products_by_gpt3_similarity(prompt, num_products=5):
    products = pm.gpt3_similarity(prompt)
    matched = products
    matched = matched.sort_values(by=['gpt3Details'], ascending=False)
    return matched.nlargest(num_products, ['gpt3Details'])

def search_products_by_bert_similarity(prompt, num_products=5):
    products = pm.bert_similarity(prompt)
    matched = products
    matched = matched.sort_values(by=['bertDetails'], ascending=False)
    return matched.nlargest(num_products, ['bertDetails'])

# Display results function with added image display
def display_results(results, similarity):
    top_3_products = results.head(3)
    count = 0
    for index, row in top_3_products.iterrows():
        count += 1
        st.markdown(f"<h4 style='color: green'>Similarity: {row[similarity + 'Details']}</h4>", unsafe_allow_html=True)
        st.markdown(f"### {count}: **{row['Name']}**")
        st.image(row['Image_URL'], width=150)  # Display image
        with st.expander("Click to view more"):
            st.write(f"**Short description:** {row['Short description']}")
            st.write(f"**Description:** {row['Description']}")

# Define Streamlit app
def app():
    # Add a header and some introductory text
    st.title("E-commerce Product Recommendation System")
    st.markdown("Enter a prompt in the text area below or select a sample prompt, then select a similarity method to find the top product recommendations.")

    # Create columns for the text area, sample prompts, and the search buttons
    col1, col2, col3 = st.columns([3, 1, 1])

    # Move the text area to the left column
    with col1:
        prompt = st.text_area("Enter a prompt:", height=100)

    # Add sample prompts to the middle column
    with col2:
        st.write("Sample Prompts:")
        sample_prompts = [
            "",
            "I need a stylish and durable window shading solution for my home or office. Suggestions?",
            "I need a lightweight roofing material for agricultural, construction, and transportation applications. Suggestions?",
            "I'm looking for a versatile material made from unsaturated polyester resins. Suggestions?",
            "I need an entertainment upgrade with superior signal reception and a sleek modern design. Suggestions?",
            "I need a satellite dish with a double actuator and crystal-clear signal reception. Suggestions?"
        ]
        selected_prompt = st.selectbox("", sample_prompts, key="sample_prompt")
        if selected_prompt:
            prompt = selected_prompt

    # Move the search buttons to the right column
    with col3:
        st.write("Similarity Method:")
        similarity_methods = {
            "TF-IDF Similarity": search_products_by_cosine_similarity,
            "GPT3 Similarity": search_products_by_gpt3_similarity,
            "BERT Similarity": search_products_by_bert_similarity,
            "Transformers Similarity": search_products_by_transformers_similarity,
        }
        method = st.selectbox("", list(similarity_methods.keys()), key="search_method")

        if st.button("Search", key="search_products"):
            if prompt:
                results = similarity_methods[method](prompt, num_products=5)

    # Add a horizontal line to separate the input area and the results
    st.markdown("---")

    # Check if 'results' is defined and display them under Recommendations
    if 'results' in locals():
        st.write("## Recommendations")
        display_results(results, method.split()[0].lower())

if __name__ == "__main__":
    app()

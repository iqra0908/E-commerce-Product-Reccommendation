import streamlit as st
from scripts import prompt_matching
import pandas as pd
import sys
sys.path.append("../")

# Set page config
st.set_page_config(page_title="Product Search", page_icon=":products:", layout="wide")
pm = prompt_matching.PromptMatching()


# Define function to search for relevant book summaries using keyword matching
def search_products_by_keyword(prompt):
    products = pm.keyword_matching(prompt)
    matched = products[products['keywords_match']]
    return matched

def search_products_by_cosine_similarity(prompt, num_products=3):
    products = pm.cosine_similarity(prompt)
    matched = products
    matched = matched.sort_values(by=['csShort description','csDescription','csName'], ascending=False)
    return matched.nlargest(5, ['csShort description', 'csDescription', 'csName'])

def display_results(results):
    top_3_products = results.head(3)
    count = 0
    for index, row in top_3_products.iterrows():
        count+=1
        st.markdown(f"<h4 style='color: green'>Cosine Similarities: Name-{row['csName']} Short Description-{row['csShort description']} Description-{row['csDescription']}</h4>", unsafe_allow_html=True)
        st.markdown(f"### {count}: **{row['Name']}**")
        with st.expander("Click to view more"):
            st.write(f"**Short description:** {row['Short description']}")
            st.write(f"**Description:** {row['Description']}")


# Define Streamlit app
def app():
    st.title("E-commerce product Recommendation System")

    prompt = st.text_area("Enter a prompt:", height=100)

    # Create two columns for the buttons
    col1, col2, col3, col4 = st.columns(4)

    # Create a section for recommendations
    st.write("## Recommendations")
    # Create empty container for the search results
    results_container = st.empty()

    # Add search button for keyword matching
    with col1:
        if st.button("Keyword Matching", key="search_by_keyword_button"):
            if prompt:
                results = search_products_by_keyword(prompt)
                
    # Add search button for cosine similarity
    with col2:
        if st.button("Description Matching", key="search_by_description"):
            if prompt:
                results = search_products_by_cosine_similarity(prompt,num_products=5)
    
    # Display results under Recommendations
    if 'results' in locals():
        display_results(results)
        results_container.dataframe(results.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))
        

if __name__ == "__main__":
    app()
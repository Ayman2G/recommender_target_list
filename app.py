import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/your_dataset.csv')
    df.fillna('', inplace=True)
    df['combined_industries'] = df['Industry'] + ' ' + df['Affinity_Industries']
    return df

# Function to get TF-IDF embeddings
@st.cache_data
def get_tfidf_embeddings(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_industries'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

# Function to find closest industries
def find_closest_industry(input_industry, tfidf, tfidf_matrix, df, column='Industry', top_n=1):
    input_embedding = tfidf.transform([input_industry])
    cosine_sim_input = linear_kernel(input_embedding, tfidf_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim_input))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    closest_indices = [i[0] for i in sim_scores[:top_n]]
    closest_industries = df[column].iloc[closest_indices].unique()
    
    return closest_industries, sim_scores

# Function to get recommendations with fallback and scoring for the simple algorithm
def simple_alg(input_industry, df, tfidf, tfidf_matrix, cosine_sim, tier_weight=0.7, similarity_weight=0.3):
    # Check if input industry exists in client industry
    df_matched = df[df['Client_industry'].str.contains(input_industry, case=False, na=False)]
    if df_matched.empty:
        closest_industries, _ = find_closest_industry(input_industry, tfidf, tfidf_matrix, df, 'Client_industry', top_n=10)
        if len(closest_industries) == 0:
            st.write(f"<span style='color:red;'>No similar client industries found for:</span> <strong>{input_industry}</strong>", unsafe_allow_html=True)
            return pd.DataFrame()  # Return an empty DataFrame
        else:
            st.write(f"<span style='color:orange;'>No exact match found for '{input_industry}', using closest client industry:</span> <strong>{closest_industries[0]}</strong>", unsafe_allow_html=True)
            input_industry = closest_industries[0]
    
    df_matched = df[df['Client_industry'] == input_industry]
    
    if df_matched.empty:
        st.write(f"<span style='color:red;'>No exact match found for '{input_industry}' in Client_industry.</span>", unsafe_allow_html=True)
        return pd.DataFrame()  # Return an empty DataFrame
    
    idx = df_matched.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    df['similarity_score'] = [score for _, score in sim_scores]
    df['inverted_tier'] = 5 - df['Tiering']
    df['combined_score'] = df['similarity_score'] * similarity_weight + df['inverted_tier'] * tier_weight
    
    scored_targets = df.sort_values(by='combined_score', ascending=False)
    return scored_targets[['Target', 'Dossier', 'Industry', 'Client_industry', 'Affinity_Industries', 'Tiering', 'similarity_score', 'combined_score']]

# Advanced algorithm for smarter recommendations
def advanced_alg(input_industry, df, tfidf, tfidf_matrix, cosine_sim, tier_weight=0.7, similarity_weight=0.3, industry_weight=0.5, affinity_weight=0.5):
    # Check if input industry exists in client industry
    df_matched = df[df['Client_industry'].str.contains(input_industry, case=False, na=False)]
    if df_matched.empty:
        closest_industries, _ = find_closest_industry(input_industry, tfidf, tfidf_matrix, df, 'Client_industry', top_n=10)
        if len(closest_industries) == 0:
            st.write(f"<span style='color:red;'>No similar client industries found for:</span> <strong>{input_industry}</strong>", unsafe_allow_html=True)
            return pd.DataFrame()  # Return an empty DataFrame
        else:
            st.write(f"<span style='color:orange;'>No exact match found for '{input_industry}', using closest client industry:</span> <strong>{closest_industries[0]}</strong>", unsafe_allow_html=True)
            input_industry = closest_industries[0]
    
    input_embedding = tfidf.transform([input_industry])
    cosine_sim_input = linear_kernel(input_embedding, tfidf_matrix).flatten()

    df['similarity_score'] = cosine_sim_input
    df['industry_similarity'] = cosine_sim_input * industry_weight
    df['affinity_similarity'] = cosine_sim_input * affinity_weight
    df['inverted_tier'] = 5 - df['Tiering']
    df['combined_score'] = (df['industry_similarity'] + df['affinity_similarity']) * similarity_weight + df['inverted_tier'] * tier_weight

    scored_targets = df.sort_values(by='combined_score', ascending=False)
    return scored_targets[['Target', 'Dossier', 'Industry', 'Client_industry', 'Affinity_Industries', 'Tiering', 'similarity_score', 'combined_score']]

# Streamlit app layout

# Load logo
logo_path = 'C:/Users/ayman/Desktop/IPTP Exec/Notebooks/dataset/IPTP.png'
logo = Image.open(logo_path)

# Display logo and title side by side
col1, col2 = st.columns([2, 8])
with col1:
    st.image(logo, width=140)
with col2:
    st.markdown("<h1 style='margin: 0;'>Target list Recommendation System</h1>", unsafe_allow_html=True)

# Load data
df = load_data()

# Get TF-IDF embeddings
tfidf, tfidf_matrix, cosine_sim = get_tfidf_embeddings(df)

# Sidebar for parameter selection
st.sidebar.title("Parameters")
tier_weight = st.sidebar.slider('Tier Weight:', 0.0, 1.0, 0.7)
similarity_weight = st.sidebar.slider('Similarity Weight:', 0.0, 1.0, 0.3)
industry_weight = st.sidebar.slider('Industry Weight:', 0.0, 1.0, 0.5)
affinity_weight = st.sidebar.slider('Affinity Industry Weight:', 0.0, 1.0, 0.5)
top_n = st.sidebar.slider('Select number of closest industries to consider:', 1, 10, 1)
algorithm = st.sidebar.selectbox('Choose Algorithm:', ['Simple', 'Advanced'])

# Add custom CSS to reduce the vertical space
st.markdown("""
    <style>
        .custom-title {
            color: #4a4a4a;
            margin-bottom: -40px;  /* Adjust the negative value as needed */
        }
        .custom-input {
            margin-top: -10px;  /* Adjust the negative value as needed */
        }
    </style>
    <h3 class="custom-title">Enter the Industry:</h3>
""", unsafe_allow_html=True)

input_industry = st.text_input('', '', key='custom-input')

if input_industry:
    st.markdown("<h4>First, checking similarity to Client Industry:</h4>", unsafe_allow_html=True)
    closest_client_industries, client_sim_scores = find_closest_industry(input_industry, tfidf, tfidf_matrix, df, 'Client_industry', top_n)
    st.markdown("<strong>Closest Client Industries:</strong>", unsafe_allow_html=True)
    st.markdown("<ul>" + "".join([f"<li>{industry}</li>" for industry in closest_client_industries]) + "</ul>", unsafe_allow_html=True)
    
    chosen_client_industry = st.selectbox('Choose the most relevant client industry from the suggestions:', closest_client_industries)
    if chosen_client_industry:
        if algorithm == 'Simple':
            recommendations = simple_alg(chosen_client_industry, df, tfidf, tfidf_matrix, cosine_sim, tier_weight, similarity_weight)
        elif algorithm == 'Advanced':
            recommendations = advanced_alg(chosen_client_industry, df, tfidf, tfidf_matrix, cosine_sim, tier_weight, similarity_weight, industry_weight, affinity_weight)
        
        st.markdown("<h3>Generated Dataset:</h3>", unsafe_allow_html=True)
        st.dataframe(recommendations)

        # Allow user to download the recommendations as a CSV file
        csv = recommendations.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='recommendations.csv',
            mime='text/csv',
            key='download-recommendations-client'
        )

        # Display closest target industries in a user-friendly format
        st.markdown("<h3>If no suitable client industry found, check similarity to Target Industry (Industry + Affinity Industries):</h3>", unsafe_allow_html=True)
        closest_target_industries, target_sim_scores = find_closest_industry(input_industry, tfidf, tfidf_matrix, df, 'combined_industries', top_n)
        st.markdown("<strong>Closest Target Industries:</strong>", unsafe_allow_html=True)
        st.markdown("<ul>" + "".join([f"<li>{industry}</li>" for industry in closest_target_industries]) + "</ul>", unsafe_allow_html=True)

        chosen_target_industry = st.selectbox('Choose the most relevant target industry from the suggestions:', closest_target_industries, key='chosen-target-industry')
        if chosen_target_industry:
            if algorithm == 'Simple':
                recommendations = simple_alg(chosen_target_industry, df, tfidf, tfidf_matrix, cosine_sim, tier_weight, similarity_weight)
            elif algorithm == 'Advanced':
                input_embedding = tfidf.transform([chosen_target_industry])
                cosine_sim_input = linear_kernel(input_embedding, tfidf_matrix).flatten()

                df['similarity_score'] = cosine_sim_input
                df['industry_similarity'] = cosine_sim_input * industry_weight
                df['affinity_similarity'] = cosine_sim_input * affinity_weight
                df['inverted_tier'] = 5 - df['Tiering']
                df['combined_score'] = (df['industry_similarity'] + df['affinity_similarity']) * similarity_weight + df['inverted_tier'] * tier_weight

                recommendations = df.sort_values(by='combined_score', ascending=False)
                recommendations = recommendations[['Target', 'Dossier', 'Industry', 'Client_industry', 'Affinity_Industries', 'Tiering', 'similarity_score', 'combined_score']]
            
            st.markdown("<h3>Generated Dataset:</h3>", unsafe_allow_html=True)
            st.dataframe(recommendations)

            # Allow user to download the recommendations as a CSV file
            csv = recommendations.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='recommendations.csv',
                mime='text/csv',
                key='download-recommendations-target'
            )

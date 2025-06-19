import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🎬 Movie Recommendation Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitMovieRecommender:
    def __init__(self):
        self.df = None
        self.feature_matrix = None
        self.clusters = None
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.nmf_model = None
        
    @st.cache_data
    def load_sample_data(_self):
        """Load sample movie data for demonstration"""
        np.random.seed(42)
        
        # Create sample movie data
        genres_list = [
            "Action", "Adventure", "Comedy", "Drama", "Horror", "Romance", 
            "Sci-Fi", "Thriller", "Fantasy", "Animation", "Crime", "Mystery"
        ]
        
        sample_data = []
        movie_titles = [
            "The Dark Knight", "Inception", "Pulp Fiction", "The Godfather", "Avatar",
            "Titanic", "Star Wars", "Jurassic Park", "The Matrix", "Forrest Gump",
            "Fight Club", "The Lion King", "Goodfellas", "Casablanca", "Schindler's List",
            "The Shawshank Redemption", "12 Angry Men", "One Flew Over the Cuckoo's Nest",
            "Citizen Kane", "Vertigo", "Psycho", "Rear Window", "Sunset Boulevard",
            "Some Like It Hot", "North by Northwest", "On the Waterfront", "Singin' in the Rain",
            "Gone with the Wind", "Lawrence of Arabia", "The Wizard of Oz"
        ]
        
        for i, title in enumerate(movie_titles):
            sample_data.append({
                'id': i,
                'title': title,
                'genres': ' '.join(np.random.choice(genres_list, np.random.randint(1, 4))),
                'vote_average': round(np.random.uniform(6.0, 9.5), 1),
                'vote_count': np.random.randint(1000, 50000),
                'runtime': np.random.randint(90, 180),
                'release_year': np.random.randint(1990, 2024),
                'budget': np.random.randint(10000000, 200000000),
                'revenue': np.random.randint(50000000, 1000000000),
                'overview': f"This is a sample overview for {title}. It's an amazing movie with great storytelling."
            })
        
        return pd.DataFrame(sample_data)
    
    def preprocess_data(self, df):
        """Preprocess the movie data"""
        self.df = df.copy()
        
        # Handle missing values
        self.df['genres'] = self.df['genres'].fillna('Unknown')
        self.df['overview'] = self.df['overview'].fillna('')
        
        return self.df
    
    def create_features_and_clusters(self, n_clusters=5):
        """Create features and perform clustering"""
        # Combine text features
        combined_features = self.df['genres'] + ' ' + self.df['overview']
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(combined_features)
        
        # Add numerical features
        numerical_features = self.df[['vote_average', 'runtime', 'release_year']].values
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(numerical_features)
        
        # Combine features
        from scipy.sparse import hstack
        self.feature_matrix = hstack([tfidf_matrix, numerical_scaled])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(self.feature_matrix.toarray())
        self.df['cluster'] = self.clusters
        
        return self.clusters
    
    def create_similarity_matrix(self):
        """Create content similarity matrix"""
        self.similarity_matrix = cosine_similarity(self.feature_matrix.toarray())
        return self.similarity_matrix
    
    def get_content_recommendations(self, movie_title, n_recommendations=5):
        """Get content-based recommendations"""
        # Find movie index
        movie_idx = self.df[self.df['title'].str.contains(movie_title, case=False)].index
        if len(movie_idx) == 0:
            return []
        
        movie_idx = movie_idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (excluding the movie itself)
        recommendations = []
        for i, score in sim_scores[1:n_recommendations+1]:
            movie_info = self.df.iloc[i]
            recommendations.append({
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'rating': movie_info['vote_average'],
                'year': movie_info['release_year'],
                'similarity_score': round(score, 3)
            })
        
        return recommendations

# Initialize the recommender
@st.cache_resource
def get_recommender():
    return StreamlitMovieRecommender()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Control Panel")
    
    # Initialize recommender
    recommender = get_recommender()
    
    # Load data section
    st.sidebar.subheader("📊 Data Source")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Use Sample Data",]
    )
    
    df = None
    
    if data_option == "Use Sample Data":
        df = recommender.load_sample_data()
        st.sidebar.success(f"✅ Loaded {len(df)} sample movies")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload your movie dataset (CSV)", 
            type=['csv']
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} movies")
    
    if df is not None:
        # Preprocess data
        df = recommender.preprocess_data(df)
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔍 Movie Analysis", "🎯 Recommendations", "📈 Clustering Analysis"])
        
        with tab1:
            st.subheader("📊 Dataset Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Movies", len(df))
            with col2:
                st.metric("Avg Rating", f"{df['vote_average'].mean():.1f}")
            with col3:
                st.metric("Year Range", f"{df['release_year'].min()}-{df['release_year'].max()}")
            with col4:
                st.metric("Avg Runtime", f"{df['runtime'].mean():.0f} min")
            
            # Data preview
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Rating Distribution")
                fig = px.histogram(df, x='vote_average', nbins=20, title="Movie Ratings Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📅 Movies by Year")
                year_counts = df['release_year'].value_counts().sort_index()
                fig = px.line(x=year_counts.index, y=year_counts.values, title="Movies Released by Year")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🔍 Movie Analysis")
            
            # Genre analysis
            st.subheader("🎭 Genre Analysis")
            all_genres = ' '.join(df['genres'].fillna(''))
            genre_words = [g.strip() for g in all_genres.split() if g.strip()]
            from collections import Counter
            genre_counts = Counter(genre_words).most_common(10)
            
            if genre_counts:
                genre_df = pd.DataFrame(genre_counts, columns=['Genre', 'Count'])
                fig = px.bar(genre_df, x='Genre', y='Count', title="Most Popular Genres")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Top rated movies
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⭐ Top Rated Movies")
                top_movies = df.nlargest(10, 'vote_average')[['title', 'vote_average', 'release_year']]
                st.dataframe(top_movies, use_container_width=True)
            
            with col2:
                st.subheader("💰 Budget vs Revenue")
                if 'budget' in df.columns and 'revenue' in df.columns:
                    fig = px.scatter(df, x='budget', y='revenue', hover_data=['title'], 
                                   title="Budget vs Revenue")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Movie Recommendations")
            
            # Prepare recommendation system
            with st.spinner("🔄 Preparing recommendation system..."):
                clusters = recommender.create_features_and_clusters()
                similarity_matrix = recommender.create_similarity_matrix()
            
            st.success("✅ Recommendation system ready!")
            
            # Movie selection
            st.subheader("🎬 Select a Movie for Recommendations")
            selected_movie = st.selectbox(
                "Choose a movie:",
                options=df['title'].tolist(),
                index=0
            )
            
            # Number of recommendations
            n_recs = st.slider("Number of recommendations:", 3, 10, 5)
            
            if st.button("🚀 Get Recommendations", type="primary"):
                recommendations = recommender.get_content_recommendations(selected_movie, n_recs)
                
                if recommendations:
                    st.subheader(f"🎯 Movies similar to '{selected_movie}':")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                            
                            with col1:
                                st.write(f"**{i}. {rec['title']}**")
                            with col2:
                                st.write(f"*{rec['genres']}*")
                            with col3:
                                st.write(f"⭐ {rec['rating']}")
                            with col4:
                                st.write(f"📊 {rec['similarity_score']}")
                            
                            st.write("---")
                else:
                    st.error("❌ No recommendations found!")
        
        with tab4:
            st.subheader("📈 Clustering Analysis")
            
            # Clustering controls
            n_clusters = st.slider("Number of clusters:", 3, 10, 5)
            
            if st.button("🔄 Run Clustering Analysis", type="primary"):
                with st.spinner("🔄 Performing clustering analysis..."):
                    clusters = recommender.create_features_and_clusters(n_clusters)
                
                # Cluster distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Cluster Distribution")
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    fig = px.pie(values=cluster_counts.values, names=[f"Cluster {i}" for i in cluster_counts.index],
                               title="Movies per Cluster")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("⭐ Average Rating by Cluster")
                    cluster_ratings = df.groupby('cluster')['vote_average'].mean()
                    fig = px.bar(x=[f"Cluster {i}" for i in cluster_ratings.index], 
                               y=cluster_ratings.values,
                               title="Average Rating by Cluster")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster details
                st.subheader("🔍 Cluster Details")
                
                for cluster_id in sorted(df['cluster'].unique()):
                    with st.expander(f"Cluster {cluster_id} ({len(df[df['cluster'] == cluster_id])} movies)"):
                        cluster_movies = df[df['cluster'] == cluster_id]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Sample Movies:**")
                            sample_movies = cluster_movies.head(5)[['title', 'vote_average', 'genres']]
                            st.dataframe(sample_movies, use_container_width=True)
                        
                        with col2:
                            st.write("**Cluster Statistics:**")
                            st.write(f"Average Rating: {cluster_movies['vote_average'].mean():.2f}")
                            st.write(f"Average Runtime: {cluster_movies['runtime'].mean():.0f} min")
                            st.write(f"Year Range: {cluster_movies['release_year'].min()}-{cluster_movies['release_year'].max()}")
                            
                            # Top genres in cluster
                            cluster_genres = ' '.join(cluster_movies['genres'].fillna(''))
                            cluster_genre_words = [g.strip() for g in cluster_genres.split() if g.strip()]
                            if cluster_genre_words:
                                from collections import Counter
                                top_cluster_genres = Counter(cluster_genre_words).most_common(3)
                                st.write("**Top Genres:**")
                                for genre, count in top_cluster_genres:
                                    st.write(f"• {genre}: {count}")
    
    else:
        # Welcome message when no data is loaded
        st.info("👆 Please select a data source from the sidebar to get started!")
        
        st.markdown("""
        ## 🚀 Getting Started
        
        1. **Use Sample Data**: Click to load sample movie data for demonstration
        2. **Upload CSV**: Upload your own movie dataset with columns like:
           - `title`: Movie title
           - `genres`: Movie genres (separated by spaces)
           - `vote_average`: Movie rating
           - `runtime`: Movie duration in minutes
           - `release_year`: Release year
           - `overview`: Movie description
        
        ## 🌟 Features
        
        - **📊 Dataset Overview**: Explore your movie data with interactive charts
        - **🔍 Movie Analysis**: Analyze genres, ratings, and trends
        - **🎯 Recommendations**: Get personalized movie recommendations
        - **📈 Clustering**: Discover movie groups and patterns
        """)

if __name__ == "__main__":
    main()
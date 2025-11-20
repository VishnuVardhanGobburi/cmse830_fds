import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

st.set_page_config(page_title="Netflix IMDb Analysis", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------

netflix_imdb_df = pd.read_csv("netflix_imdb_new.csv")
netflix_imdb_df=netflix_imdb_df.rename(columns={'type':'content_type', 'country':'production_country','rating':'content_rating','listed_in':'genre','TV Show_duration':'episodes','runtimeMinutes':'runtime_mins','averageRating':'IMDb_avg_rating'})
netflix_imdb_df = netflix_imdb_df.loc[:, ~netflix_imdb_df.columns.str.contains('^Unnamed')]
netflix_imdb_df['episodes'] = netflix_imdb_df['episodes'].fillna(0)
netflix_df=pd.read_csv('netflix_titles.csv', usecols=['title','type','director','country','release_year','rating','duration','listed_in'])
netflix_df=netflix_df.rename(columns={'type':'content_type', 'country':'production_country','rating':'content_rating','listed_in':'genre','TV Show_duration':'episodes'})
netflix_df['director'] = netflix_df['director'].str.split(', ')
netflix_df['production_country'] = netflix_df['production_country'].str.split(', ')
netflix_df['genre'] = netflix_df['genre'].str.split(', ')
netflix_df = netflix_df.explode('director').explode('production_country').explode('genre').reset_index(drop=True)
imdb_basics_df=pd.read_csv('imdb_basics_sample.csv')
imdb_basics_df = imdb_basics_df.loc[:, ~imdb_basics_df.columns.str.contains('^Unnamed')]
imdb_ratings_df=pd.read_csv('imdb_ratings_sample.csv')
imdb_ratings_df = imdb_ratings_df.loc[:, ~imdb_ratings_df.columns.str.contains('^Unnamed')]

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

type_filter = st.sidebar.multiselect(
    "Select content_type",
    options=netflix_imdb_df['content_type'].dropna().unique(),
    default=netflix_imdb_df['content_type'].dropna().unique()
)

all_countries = sorted(netflix_imdb_df['production_country'].dropna().unique())

country_filter = st.sidebar.multiselect(
    "Select production_country",
    options=["All"] + all_countries,
    default=["All"],
    help="Select countries. 'All' shows all data."
)
if "All" in country_filter:
    filtered_df = netflix_imdb_df.copy()
else:
    filtered_df = netflix_imdb_df[netflix_imdb_df['production_country'].isin(country_filter)]


all_ratings = sorted(netflix_imdb_df['content_rating'].dropna().unique())
rating_filter = st.sidebar.multiselect(
    "Select content_rating",
    options=["All"] + all_ratings,
    default=["All"],
    help="Select ratings"
)
if "All" in rating_filter:
    filtered_df = netflix_imdb_df.copy()
else:
    filtered_df = netflix_imdb_df[netflix_imdb_df['content_rating'].isin(country_filter)]

release_year_range = st.sidebar.slider(
    "Select Release Year Range",
    int(netflix_imdb_df['release_year'].min()),
    int(netflix_imdb_df['release_year'].max()),
    (int(netflix_imdb_df['release_year'].min()), int(netflix_imdb_df['release_year'].max()))
)

# Apply filters
filtered_df = netflix_imdb_df[
    (netflix_imdb_df['content_type'].isin(type_filter)) &
    (netflix_imdb_df['production_country'] if "All" in country_filter else netflix_imdb_df['production_country'].isin(country_filter)) &
    (netflix_imdb_df['content_rating'] if "All" in rating_filter else netflix_imdb_df['content_rating'].isin(rating_filter)) &
    (netflix_imdb_df['release_year'].between(release_year_range[0], release_year_range[1]))
]

# -----------------------------
# Page Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Overview", "Missing Value Analysis", "Exploring IMDb Ratings", "Title Analysis", "IMDb content_rating Analysis","Predict the IMDb content_rating"
])
#Encoding

# -----------------------------
# TAB 1: Data Overview
# -----------------------------
with tab1:
    rows, cols = filtered_df.shape

    st.subheader("🧹 Data Preprocessing Overview")
    if filtered_df.empty:
        avg_rating = 0  # or np.nan if you prefer
    else:
        avg_rating = filtered_df['IMDb_avg_rating'].mean().round(2)

    # --- KPI Section ---
    st.markdown("### 📊 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", f"{cols}")
    col3.metric("Avg IMDb content_rating ⭐", f"{avg_rating}")


    st.markdown("The aim of this project is to perform an in-depth exploratory data analysis (EDA) and predictive modeling on Netflix IMDb datasets to uncover insights into content trends and audience preferences. To acheive my aim, I have used Netflix and IMDb datasets.")
    st.header("Data Overview")
    st.write("Netflix dataset")
    st.dataframe(netflix_df.head(5))
    st.write("IMDb basics dataset")
    st.dataframe(imdb_basics_df.head(5))
    st.write("IMDb ratings dataset")
    st.dataframe(imdb_ratings_df.head(5))
    st.write("Dataset after merging")
    st.dataframe(filtered_df.head(5))
    st.write(f"Shape of Dataset: {filtered_df.shape}")
    # st.write("### Column Information:")
    # st.write(filtered_df.dtypes)
    st.subheader("🧹 Data Preprocessing Overview")

    st.markdown("""
    ### Steps Performed During Data Preprocessing

    1. **Exploded Multi-Valued Columns**  
    - Columns such as *directors*, *genre* (genres), and *production_country* often contained multiple comma-separated values.  
    - These were **exploded** into separate rows to enable accurate aggregation and analysis.

    2. **Split the Duration Column**  
    - The *duration* column included both **minutes** (for movies) and **seasons** (for TV shows).  
    - It was split into two new columns:  
        - `duration_min` → for movie durations (in minutes)  
        - `duration_seasons` → for TV show lengths (in seasons)

    3. **Merged Multiple Datasets**  
    - Three datasets were combined to create a unified and enriched Netflix dataset:  
        - `netflix_titles.csv` (content metadata)  
        - `IMDb_basics.csv` (movie/TV show identifiers)  
        - `IMDb_ratings.csv` (average ratings and vote counts)  
    - The merge was performed using the **IMDb title ID (tconst)**, **title** and **Release Year** to integrate metadata and ratings.

    These preprocessing steps ensured data consistency, completeness, and accuracy for further analysis and IMDb content_rating prediction.
    """)


# -----------------------------
# TAB 2: Missing Value Analysis
# -----------------------------
with tab2:
    selected_missing_viz = st.multiselect(
    "Select Misisng Value Visualizations",
    ["Missing values", "Analysis with Content content_type", "Missing Value Trend", "Conditional Probability"],
    default=["Missing values"]
)
    if "Missing values" in selected_missing_viz:
        
    # --- Step 1: Calculate missing stats ---
        missing_counts = netflix_df.isnull().sum()
        missing_percent = (missing_counts / len(netflix_df)) * 100
        present_percent = 100 - missing_percent

        # Sort variables by % missing
        sorted_vars = missing_percent.sort_values(ascending=False).index

        # DataFrame for plotting
        missing_df = pd.DataFrame({
            "Variable": sorted_vars,
            "Missing": missing_percent[sorted_vars].values,
            "Present": present_percent[sorted_vars].values
        })

        # --- Step 2: Create stacked bar chart ---
        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            y=missing_df['Variable'],
            x=missing_df['Present'],
            orientation='h',
            name='Present',
            marker_color='#E50914'
        ))

        fig_bar.add_trace(go.Bar(
            y=missing_df['Variable'],
            x=missing_df['Missing'],
            orientation='h',
            name='Missing',
            marker_color='#585858'
        ))

        fig_bar.update_layout(
            barmode='stack',
            title='Percentage of Missing Values',
            xaxis_title='% of Values',
            yaxis_title='Variable',
            yaxis=dict(autorange='reversed')  # same as invert_yaxis
        )

        st.plotly_chart(fig_bar, use_container_width=True)
        # --- Step 3: Create heatmap of missing values ---
        # Convert boolean DataFrame to numeric (0 = present, 1 = missing)
        missing_matrix = netflix_df[sorted_vars].isnull().astype(int).T

        fig_heat = px.imshow(
            missing_matrix,
            color_continuous_scale=['#585858','#E50914'],
            aspect='auto',
            labels=dict(x="Row Number", y="Variable", color="Missing")
        )

        fig_heat.update_layout(
            title='Missing Values in Rows',
            xaxis=dict(tickmode='linear', tick0=0, dtick=200),  # adjust dtick as needed
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("From the above visualizations, we can observe that the **director** and **production_country** columns contain missing values.")

    elif "Conditional Probability" in selected_missing_viz:
    # Create cross-tab
        crossbar_1 = pd.crosstab(netflix_df['director'].isna(), netflix_df['production_country'].isna())

        # Convert index/columns to strings for better labels
        crossbar_1.index = crossbar_1.index.map({False: 'Director Present', True: 'Director Missing'})
        crossbar_1.columns = crossbar_1.columns.map({False: 'production_country Present', True: 'production_country Missing'})

        # Plot interactive heatmap
        fig = px.imshow(
            crossbar_1,
            text_auto=True,                  # show counts
            color_continuous_scale='Reds',
            labels=dict(x="production_country Null Status", y="Director Null Status", color="Count")
        )

        fig.update_layout(
            title="Null Values Heatmap: Director vs production_country",
            template='plotly_white',
            xaxis_side='top'
        )

        st.plotly_chart(fig, use_container_width=True)


        # Calculate probabilities
        both_missing = netflix_df[(netflix_df['director'].isna()) & (netflix_df['production_country'].isna())].shape[0]
        country_missing = netflix_df['production_country'].isna().sum()
        director_missing = netflix_df['director'].isna().sum()

        p_country_missing = np.round((country_missing / len(netflix_df)) * 100, 2)
        p_director_given_country_missing = np.round((both_missing / country_missing) * 100, 2)
        p_director_missing = np.round((director_missing / len(netflix_df)) * 100, 2)
        p_country_given_director_missing = np.round((both_missing / director_missing) * 100, 2)

        # Display in Streamlit
        st.subheader("Conditional Probabilities for Missing Values")
        st.markdown(f"- **P(production_country missing):** {p_country_missing} %")
        st.markdown(f"- **P(Director missing | production_country missing):** {p_director_given_country_missing} %")
        st.markdown(f"- **P(Director missing):** {p_director_missing} %")
        st.markdown(f"- **P(production_country missing | Director missing):** {p_country_given_director_missing} %")

        st.markdown("Overall, 7.05% of entries have missing production_country information. Among those entries with missing production_country, 46.66% also have missing director information. In comparison, 25.37% of all entries have missing director values. Conversely, for entries with missing director information, 12.96% have missing production_country values.")
        st.markdown("Missing director values are much more likely when production_country information is missing, indicating a strong association between the two. However, missing production_country values are relatively uncommon even when director information is absent, suggesting that missingness in production_country is more independent.")
        
    elif "Analysis with Content content_type" in selected_missing_viz:

    # --- Director Missing % ---
        director_missing_counts = netflix_df[netflix_df['director'].isna()]['content_type'].value_counts()
        director_missing_share = (director_missing_counts / director_missing_counts.sum()) * 100

        types = director_missing_share.index
        percentages = director_missing_share.values

    # Assign colors based on content_type
        colors = ['#E50914' if t == 'Movie' else '#585858' for t in types]

        fig = go.Figure(go.Bar(
            x=types,
            y=percentages,
            text=[f'{v:.2f}%' for v in percentages],
            textposition='outside',
            marker_color=colors
        ))

        fig.update_layout(
            title='Director Missing % by Title content_type',
            xaxis_title='Title content_type',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, max(percentages)*1.2]),
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- production_country Missing % ---
        country_missing_counts = netflix_df[netflix_df['production_country'].isna()]['content_type'].value_counts()
        country_missing_share = (country_missing_counts / country_missing_counts.sum()) * 100

        types = country_missing_share.index
        percentages = country_missing_share.values

        colors = ['#E50914' if t == 'Movie' else '#585858' for t in types]

        fig = go.Figure(go.Bar(
            x=types,
            y=percentages,
            text=[f'{v:.2f}%' for v in percentages],
            textposition='outside',
            marker_color=colors
        ))

        fig.update_layout(
            title='production_country Missing % by Title content_type',
            xaxis_title='Title content_type',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, max(percentages)*1.2]),
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Summary ---
        st.markdown("**Summary:**")
        st.markdown("- The majority of missing **director** values come from **TV Shows (94.85%)**, with a small portion from **Movies (5.15%)**. " \
                    "Hence, missing directors are not missing completely at random (MCAR).")
        st.markdown("- The missing values in the **production_country** column are distributed almost evenly between **Movies (51.04%)** and **TV Shows (48.96%)**.")


    elif "Missing Value Trend" in selected_missing_viz: 
        # --- Step 1: Calculate % missing by release_year and filter from 1945 ---
        year_missing = netflix_df.groupby('release_year')['director'].apply(lambda x: x.isna().mean()*100)
        year_missing = year_missing[year_missing.index >= 1945]

        years = year_missing.index
        percentages = year_missing.values

        # --- Step 2: Create line chart (no markers) ---
        fig = go.Figure(go.Scatter(
            x=years,
            y=percentages,
            mode='lines',  # only line, no markers
            line=dict(color='red', width=2)
        ))

        # --- Step 3: Layout ---
        fig.update_layout(
            title="Directors Missing % by Release Year",
            xaxis_title="Release Year",
            yaxis_title="Missing %",
            yaxis=dict(range=[0, max(percentages)*1.2]),
            template='plotly_white'
        )

        # --- Step 4: Display in Streamlit ---
        st.plotly_chart(fig, use_container_width=True)

        # --- Step 1: Calculate % missing by release_year and filter from 1945 ---
        year_missing = netflix_df.groupby('release_year')['production_country'].apply(lambda x: x.isna().mean()*100)
        year_missing = year_missing[year_missing.index >= 1945]

        years = year_missing.index
        percentages = year_missing.values

        # --- Step 2: Create interactive line chart ---
        fig = go.Figure(go.Scatter(
            x=years,
            y=percentages,
            mode='lines',  # only line, no markers
            line=dict(color='purple', width=2)
        ))

        # --- Step 3: Layout ---
        fig.update_layout(
            title="production_country Missing % by Release Year",
            xaxis_title="Release Year",
            yaxis_title="Missing %",
            yaxis=dict(range=[0, max(percentages)*1.2]),
            template='plotly_white',
            height=400
        )

        # --- Step 4: Show in Streamlit ---
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Director**")
        st.markdown("- The missing values show notable spikes in the 1960s and 1970s, reaching up to around 40%. Between 1980 and 2000, missingness is more consistent and generally below 15%. From 2010 onward, there is a steady increase, culminating at approximately 43% missing by 2020")
        st.markdown("**production_country**")
        st.markdown("- production_country missing values are generally low, staying mostly below 10%, with brief spikes of 20–30% in the 1960s and 1980s. After 2000, missingness remains minimal until around 2020, when it rises sharply to about 30%.")
       

# -----------------------------
# TAB 3: Exploring IMDb Ratings
# -----------------------------
with tab3:
    st.header("Exploring IMDb Ratings")

    selected_bi_viz = st.multiselect(
        "Select Visualizations",
        ["Votes vs IMDb content_rating", "Correlation","Hypothesis Testing"],
        default=["Votes vs IMDb content_rating"]
    )

    if "Votes vs IMDb content_rating" in selected_bi_viz:
        fig_votes_rating = px.scatter(
            filtered_df, x='numVotes', y='IMDb_avg_rating',
            color='content_type', size='numVotes',
            color_discrete_map={'Movie':'#E50914','TV Show':'#585858'},
            title='Number of Votes vs Average content_rating',
            log_x=True, size_max=20, hover_data=['title','release_year']
        )
        st.plotly_chart(fig_votes_rating, use_container_width=True)

    elif "Correlation" in selected_bi_viz:
        num_cols = ['release_year', 'episodes', 'runtime_mins','numVotes','IMDb_avg_rating']

        # Compute correlation
        corr = filtered_df[num_cols].corr().round(2)

        # Create interactive heatmap
        fig = px.imshow(
            corr,
            text_auto=True,              # show correlation values
            color_continuous_scale='reds',
            zmin=-1, zmax=1,             # correlation range
            aspect="auto",
            labels=dict(x="Features", y="Features", color="Correlation")
        )

        fig.update_layout(
            title="Correlation Heatmap",
            template='plotly_white',
            xaxis=dict(side="top")       # show x-axis on top for better readability
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('It can be seen that no two features are highly correlated')
    
    elif "Hypothesis Testing" in selected_bi_viz:
        type_avg = filtered_df.groupby('content_type', as_index=False)['IMDb_avg_rating'].mean()

        fig_bar = px.bar(
            type_avg,
            x='content_type',
            y='IMDb_avg_rating',
            color='content_type',
            text='IMDb_avg_rating',
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#585858'},
            title='Average IMDb content_rating by Title content_type'
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(template='plotly_white', showlegend=False, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("**TV Show** tend to have slightly higher average ratings than Movie, However **Movie** dominates majority of the content")
        st.markdown("**Hypotheis Testing**")
        movie_ratings = filtered_df[filtered_df['content_type'] == 'Movie']['IMDb_avg_rating'].dropna()
        tv_ratings = filtered_df[filtered_df['content_type'] == 'TV Show']['IMDb_avg_rating'].dropna()
        t_stat, p_two = ttest_ind(tv_ratings, movie_ratings, equal_var=False)
        p_value = p_two / 2  # one-sided

        if t_stat > 0 and p_value < 0.05:
            st.markdown(f"**t_stat**: {t_stat}")
            st.markdown(f"**p_value**: {p_value}")
            st.markdown(f"Based on the t_stat and p_value values, Reject H₀ — TV Shows have higher IMDb ratings than Movies.")
        else:
            st.markdown(f"**t_stat**: {t_stat}")
            st.markdown(f"**p_value**: {p_value}")
            st.markdown(f"Based on the t_stat and p_value values, Fail to Reject H₀ — No strong evidence that TV Shows rate higher.")

with tab4:
    df_dash = filtered_df.copy()

    # KPI metrics
    total_titles = int(df_dash['title'].nunique()) if not df_dash.empty else 0
    total_genres = int(df_dash['genre'].nunique()) if not df_dash.empty else 0
    total_ratings = int(df_dash['content_rating'].nunique()) if not df_dash.empty else 0
    start_year = int(df_dash['release_year'].min()) if not df_dash.empty else 0
    end_year = int(df_dash['release_year'].max()) if not df_dash.empty else 0
    total_locations = int(df_dash['production_country'].nunique()) if not df_dash.empty else 0

  #  st.markdown("### 📊 Netflix Overview Dashboard")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Titles", f"{total_titles:,}")
    c2.metric("Total Genres", f"{total_genres:,}")
    c3.metric("Total Ratings", f"{total_ratings:,}")
    c4.metric("Start Year", f"{start_year}")
    c5.metric("End Year", f"{end_year}")
    c6.metric("Total Locations", f"{total_locations:,}")
    # ---------- Row 1: Genres (bar) | content_type donut | Ratings (bar)
    row1c1, row1c2, row1c3 = st.columns([1.2, 0.8, 1])
    # Genres by total titles (top 12)
    with row1c1:
        genres = df_dash[['genre', 'title']].dropna()
        genre_counts = genres.groupby('genre')['title'].nunique().sort_values(ascending=False).reset_index()
        top_genres = genre_counts.head(12)
        fig_genres = px.bar(
            top_genres,
            x='title', y='genre',
            orientation='h',
            title='Top Genres by Total Titles',
            labels={'title': 'Total Titles', 'genre': 'Genre'},
            color_discrete_sequence=['#E50914']
        )
        fig_genres.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_genres, use_container_width=True)

    # content_type donut (TV Show vs Movie)
    with row1c2:
        type_counts = df_dash['content_type'].value_counts().reset_index()
        type_counts.columns = ['content_type', 'count']
        fig_donut = px.pie(
            type_counts,
            names='content_type',
            values='count',
            hole=0.5,
            title='Distribution by content_type',
            color_discrete_sequence=['#E50914', '#585858']
        )
        fig_donut.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    # Ratings by total titles
    with row1c3:
        rating_counts = df_dash['content_rating'].value_counts().reset_index()
        rating_counts.columns = ['content_rating', 'count']
        fig_ratings = px.bar(
            rating_counts.sort_values('count', ascending=True),
            x='count', y='content_rating',
            orientation='h',
            title='Ratings by Total Titles',
            labels={'count': 'Total Titles', 'content_rating': 'content_rating'},
            color_discrete_sequence=['#E50914']
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

    row2c1, row2c2 = st.columns([1.5, 1])

    # Treemap: Countries by total titles
    with row2c1:
       country_counts = (filtered_df.groupby('production_country', as_index=False)['title'].count().round(2))


# Create the choropleth map
       fig_map = px.choropleth(
            country_counts,
            locations='production_country',        # Column with production_country names
            locationmode='country names', 
            color='title',      # Column to color by
            hover_name='production_country',       # Show production_country on hover
            color_continuous_scale=[(0, "#ffe5e5"), (1, "#ff0000")],
            range_color=[0, 6000],        # IMDb content_rating scale
            labels={'IMDb_avg_rating': 'Avg IMDb content_rating'},
            title='Average IMDb content_rating by production_country'
        )

       fig_map.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Display in Streamlit
       st.plotly_chart(fig_map, use_container_width=True)

    # Timeline: Total titles by release year
    with row2c2:
        time_series = df_dash.groupby('release_year')['title'].nunique().reset_index().sort_values('release_year')
        fig_time = px.area(
            time_series,
            x='release_year', y='title',
            title='Total Movies and TV Shows by Year',
            color_discrete_sequence=['#E50914']
        )
        fig_time.update_traces(line=dict(color='#E50914'), fillcolor='rgba(229,9,20,0.15)')
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("Netflix’s content reflects a strong focus on both movies and TV shows, mature-rated titles, and international diversity" \
    "ty.The U.S., India, and Europe lead in content production, showcasing Netflix’s global reach. Overall, the platform growth between 2010–2020 marks a major expansion")

    st.markdown("<small style='color: gray;'>Tip: Use the sidebar filters to refresh dashboard data dynamically.</small>", unsafe_allow_html=True)

with tab5:
    #st.markdown("## ⭐ IMDb Ratings Dashboard")

    df_dash = filtered_df.copy()

    # --- KPI SECTION ---
    avg_rating = df_dash['IMDb_avg_rating'].mean().round(2)
    max_row = df_dash.loc[df_dash['IMDb_avg_rating'].idxmax()]
    min_row = df_dash.loc[df_dash['IMDb_avg_rating'].idxmin()]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average IMDb content_rating", f"{avg_rating:.2f}")

    with col2:
        st.metric("Highest IMDb content_rating", f"{max_row['IMDb_avg_rating']:.2f}")
        st.caption(f"Genre: {max_row['genre']}")

    with col3:
        st.metric("Lowest IMDb content_rating", f"{min_row['IMDb_avg_rating']:.2f}")
        st.caption(f"Genre: {min_row['genre']}")
    
    row1c1, row1c2, row1c3 = st.columns([2.5, 2.5, 2.5])
    with row1c1:
           genre_df = filtered_df[['genre', 'content_type', 'IMDb_avg_rating']].dropna()
           genre_type = genre_df.groupby('genre')['content_type'].first().reset_index()

            # Merge with average content_rating
           genre_avg = (
                genre_df.groupby('genre')['IMDb_avg_rating']
                .mean()
                .reset_index()
                .merge(genre_type, on='genre')
            )
           genre_avg_top10 = genre_avg.sort_values('IMDb_avg_rating', ascending=False).head(10)
           fig_bar = px.bar(
            genre_avg_top10,
            x='IMDb_avg_rating',
            y='genre',
            orientation='h',
            title="Top 10 Genres by Average IMDb content_rating",
            labels={'IMDb_avg_rating': 'Average IMDb content_rating', 'genre': 'Genre', 'content_type':'Title content_type'},
            color='content_type',  # color by title content_type
            color_discrete_map={'Movie':'#585858', 'TV Show':'#E50914'}
        )

           fig_bar.update_layout(
            yaxis=dict(categoryorder='total ascending'),  # highest content_rating on top
            height=500,
            template='plotly_white',
            title_font=dict(size=18, family='Arial', color='black'),
            xaxis=dict(showgrid=True),
        )

           st.plotly_chart(fig_bar, use_container_width=True)
    
    with row1c2:
        director_df = filtered_df[['tconst', 'director', 'IMDb_avg_rating']].drop_duplicates(subset=['tconst'])

        # Filter directors with at least 10 unique titles
        director_counts = director_df['director'].value_counts()
        top_directors = director_counts[director_counts >= 5].index
        director_df = director_df[director_df['director'].isin(top_directors)]
        director_df = director_df[director_df['director']!='Unknown']

        # Compute average content_rating per director
        director_avg = (
            director_df.groupby('director')['IMDb_avg_rating']
            .mean()
            .reset_index()
            .sort_values('IMDb_avg_rating', ascending=False)
            .head(10)
        )

            # Lollipop chart
        fig_lollipop = go.Figure()

        # Add stems (lines)
        fig_lollipop.add_trace(go.Scatter(
            x=director_avg['IMDb_avg_rating'],
            y=director_avg['director'],
            mode='lines',
            line=dict(color='#585858', width=2),
            showlegend=False
        ))

        # Add markers (dots)
        fig_lollipop.add_trace(go.Scatter(
            x=director_avg['IMDb_avg_rating'],
            y=director_avg['director'],
            mode='markers+text',
            marker=dict(size=12, color='#E50914'),
            text=director_avg['IMDb_avg_rating'].round(2),
            textposition='middle right',
            name='Average content_rating'
        ))

        # Layout updates
        fig_lollipop.update_layout(
            title="Top 10 Directors of atleast 5 movies by Average IMDb content_rating",
            xaxis_title="Average IMDb content_rating",
            yaxis_title="Director",
            template='plotly_white',
            height=500,
            yaxis=dict(categoryorder='total ascending'),  # highest content_rating on top
            #legend=dict(title='Legend', y=1, x=1)
        )

        st.plotly_chart(fig_lollipop, use_container_width=True)
    with row1c3:
        year_avg = filtered_df.groupby('release_year')['IMDb_avg_rating'].mean().reset_index()

        # Line chart without markers
        fig = px.line(
            year_avg,
            x='release_year',
            y='IMDb_avg_rating',
            title="IMDb content_rating Trend Over Years",
            labels={'release_year': 'Release Year', 'IMDb_avg_rating': 'Average IMDb content_rating'}
        )

        fig.update_traces(mode='lines',line=dict(color='red'))  # only lines, no markers
        fig.update_layout(
            template='plotly_white',
            height=400,
            title_font=dict(size=18, family='Arial', color='black'),
            yaxis=dict(range=[0, 10]),  # fix scale from 0 to 10
            xaxis=dict(showgrid=True),
            yaxis_title='Average IMDb content_rating',
            xaxis_title='Release Year'
        )

        st.plotly_chart(fig, use_container_width=True)
    row2c1 = st.columns([1])[0]
    with row2c1:
        country_avg = (
        filtered_df.groupby('production_country', as_index=False)['IMDb_avg_rating']
        .mean()
        .round(2)
    )

    # Create the choropleth map
        fig_map = px.choropleth(
            country_avg,
            locations='production_country',        # Column with production_country names
            locationmode='country names', 
            color='IMDb_avg_rating',      # Column to color by
            hover_name='production_country',       # Show production_country on hover
            color_continuous_scale=[(0, "#ffe5e5"), (1, "#ff0000")],
            range_color=[0, 10],        # IMDb content_rating scale
            labels={'IMDb_avg_rating': 'Avg IMDb content_rating'},
            title='Average IMDb content_rating by production_country'
        )

        fig_map.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Display in Streamlit
        st.plotly_chart(fig_map, use_container_width=True)
    row3c1 = st.columns([1])[0]
    with row3c1:
        rating_avg = filtered_df.groupby('content_rating')['IMDb_avg_rating'].mean().reset_index()
        avg_values = np.round(rating_avg['IMDb_avg_rating'].values, 2)

        # Create a heatmap using a single-row DataFrame
        fig = px.imshow(
            [avg_values],               # single row of values
            x=rating_avg['content_rating'],     # Netflix content_rating categories
            y=['Average IMDb content_rating'],  # label for row
            text_auto=True,
            aspect='auto',
            color_continuous_scale='Reds',  # continuous color scale
            range_color=[0, 10]
        )   

        # Layout updates
    fig.update_layout(
            title=dict(
                text='Average IMDb content_rating by Netflix content_rating',
                x=0.5,
                xanchor='center',
                yanchor='top',
                #pad=dict(b=10)  # 👈 adds space (gap) between title & graph
            ),
            template='plotly_white',
            height=300,
            margin=dict(t=80, l=20, r=20, b=20),  # 👈 extra top margin for spacing
            xaxis_title='Netflix content_rating',
            yaxis_title='',
            yaxis=dict(showticklabels=True),
            title_font=dict(size=18, family='Arial', color='black')
        )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Netflix content maintains a moderate IMDb content_rating around 6.4 and remained fairly stable over the decades, with classic and niche genres like “Classic & Cult TV” and “Science & Nature TV” performing the best. Directors such as Quentin Tarantino and Martin Scorsese consistently achieve the highest average ratings.production_country-wise variation exists, but no region stands out dramatically, and content ratings other than UR has an average IMDb content_rating close to 6.5.")
    
    st.markdown("<small style='color: gray;'>Tip: Use the sidebar filters to refresh dashboard data dynamically.</small>", unsafe_allow_html=True)

with tab6:
    st.title("Find out which Movie or TV show performs well")
    country_freq = filtered_df['production_country'].value_counts(normalize=True)
    filtered_df['country_encoded'] = filtered_df['production_country'].map(country_freq)

    genre_freq = filtered_df['genre'].value_counts(normalize=True)
    filtered_df['listed_in_encoded'] = filtered_df['genre'].map(genre_freq)

    filtered_df['original_rating'] = filtered_df['content_rating']
    filtered_df = pd.get_dummies(filtered_df, columns=['content_rating'], prefix='content_rating')


    global_mean = filtered_df['IMDb_avg_rating'].mean()
    director_mean = filtered_df.groupby('director')['IMDb_avg_rating'].mean()

    filtered_df['director_encoded'] = filtered_df['director'].map(director_mean).fillna(global_mean)

    movies_df = filtered_df[filtered_df['content_type'] == 'Movie'].copy()
    tv_df = filtered_df[filtered_df['content_type'] == 'TV Show'].copy()
    tv_df['total_runtime'] = tv_df['runtime_mins'] * tv_df['episodes']
    tv_df=tv_df.drop(columns=['runtime_mins','episodes'])

    # Columns for models
    movie_features = ['country_encoded', 'listed_in_encoded', 'runtime_mins', 'director_encoded'] + \
                    [col for col in filtered_df.columns if col.startswith('rating_')]

    tv_features = ['country_encoded', 'listed_in_encoded', 'total_runtime', 'director_encoded'] + \
                [col for col in filtered_df.columns if col.startswith('rating_')]

    target = 'IMDb_avg_rating'

    X_movie = movies_df[movie_features]
    y_movie = movies_df[target]

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_movie, y_movie, test_size=0.2, random_state=42)

    rf_movie = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_movie.fit(X_train_m, y_train_m)
    y_pred_rf_m = rf_movie.predict(X_test_m)

    X_tv = tv_df[tv_features]
    y_tv = tv_df[target]

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42)

    rf_tv = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_tv.fit(X_train_t, y_train_t)
    y_pred_rf_t = rf_tv.predict(X_test_t)

    content_type = st.selectbox("Select content_type", ["Movie", "TV Show"])

    if content_type == "Movie":
        model = rf_movie
        rmse_val = root_mean_squared_error(y_test_m, y_pred_rf_m)
        r2_val = r2_score(y_test_m, y_pred_rf_m)
        runtime_label = "Runtime (Minutes)"
    else:
        model = rf_tv
        rmse_val = root_mean_squared_error(y_test_t, y_pred_rf_t)
        r2_val = r2_score(y_test_t, y_pred_rf_t)
        runtime_label = "Total Minutes"

    st.header("Criteria Filters")

    country_input = st.selectbox("production_country", sorted(filtered_df['production_country'].dropna().unique()))
    genre_input = st.selectbox("Genre", sorted(filtered_df['genre'].dropna().unique()))
    director_input = st.selectbox("Director", sorted(filtered_df['director'].dropna().unique()))
    rating_input = st.selectbox("Content content_rating", sorted(filtered_df['original_rating'].unique()))
    runtime_val = st.number_input(runtime_label, min_value=1, max_value=5000, value=100)

    st.header("Model Performance")
    st.metric("RMSE", f"{rmse_val:.3f}")
    st.metric("R² Score", f"{r2_val:.3f}")

    st.header("Predict IMDb rating")

    if st.button("Predict"):
        country_encoded = country_freq.get(country_input, 0)
        genre_encoded = genre_freq.get(genre_input, 0)
        director_encoded = director_mean.get(director_input, global_mean)

        rating_encoded_cols = {col: 0 for col in filtered_df.columns if col.startswith('rating_')}
        rating_col_name = f"rating_{rating_input}"
        if rating_col_name in rating_encoded_cols:
            rating_encoded_cols[rating_col_name] = 1

        input_vector = [country_encoded, genre_encoded, runtime_val, director_encoded] + list(rating_encoded_cols.values())

        prediction = model.predict([input_vector])[0]
        st.success(f"⭐ Predicted IMDb rating: {prediction:.2f}")




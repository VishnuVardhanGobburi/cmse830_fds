
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import scipy.stats as stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

st.set_page_config(page_title="Netflix IMDb Analysis", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------

netflix_imdb_df = pd.read_csv("netflix_imdb_transformed.csv")
#netflix_imdb_df=netflix_imdb_df.rename(columns={'type':'content_type', 'country':'production_country','rating':'content_rating','listed_in':'genre','TV Show_duration':'episodes','runtimeMinutes':'runtime_mins','averageRating':'IMDb_avg_rating'})
netflix_imdb_df = netflix_imdb_df.loc[:, ~netflix_imdb_df.columns.str.contains('^Unnamed')]
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

# Track active tab
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab0"

# ===============================
# Sidebar Filters for ONLY Tab-2
# ===============================


# -----------------------------
# Page Tabs
# -----------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Home","Title Analysis", "Netflix Content Success Analysis","Predict IMDb rating", "Data Handling",
])

st.sidebar.header("🔎 Filter Content")

type_filter = st.sidebar.multiselect(
    "Select Content Type",
    options=netflix_imdb_df['content_type'].dropna().unique(),
    default=netflix_imdb_df['content_type'].dropna().unique()
)

all_countries = sorted(netflix_imdb_df['production_country'].dropna().unique())
country_filter = st.sidebar.multiselect(
    "Select Production Country",
    options=["All"] + all_countries,
    default=["All"]
)

all_ratings = sorted(netflix_imdb_df['content_rating'].dropna().unique())
rating_filter = st.sidebar.multiselect(
    "Select Content Rating",
    options=["All"] + all_ratings,
    default=["All"]
)

release_year_range = st.sidebar.slider(
    "Release Year Range",
    int(netflix_imdb_df['release_year'].min()),
    int(netflix_imdb_df['release_year'].max()),
    (int(netflix_imdb_df['release_year'].min()), int(netflix_imdb_df['release_year'].max()))
)

# Apply filters
filtered_df = netflix_imdb_df.copy()

filtered_df = filtered_df[
    (filtered_df['content_type'].isin(type_filter)) &
    (filtered_df['release_year'].between(release_year_range[0], release_year_range[1]))
]

if "All" not in country_filter:
    filtered_df = filtered_df[filtered_df['production_country'].isin(country_filter)]

if "All" not in rating_filter:
    filtered_df = filtered_df[filtered_df['content_rating'].isin(rating_filter)]

NETFLIX_HERO_URL = "https://static0.moviewebimages.com/wordpress/wp-content/uploads/2024/08/netflix-logo.jpeg?q=70&fit=crop&w=1600&h=900&dpr=1"

with tab0:

    st.markdown(
        f"""
        <style>
            .hero-container {{
                position: relative;
                width: 100%;
                height: 600px; /* reduced height slightly */
                overflow: hidden;
                margin-top: -15px; /* ensures tabs visible */
            }}
            .hero-img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
                filter: brightness(70%);
            }}
            .hero-text {{
                position: absolute;
                top: 35px;
                left: 50px;
                color: #ffffff;
                font-size: 2.3rem;
                font-weight: 800;
                letter-spacing: 0.02em;
                text-shadow: 0 0 20px rgba(0,0,0,0.95);
                font-family: 'Bebas Neue', sans-serif;
                line-height: 1.1;
                z-index: 2;
            }}
        </style>

        <div class="hero-container">
            <img class="hero-img" src="{NETFLIX_HERO_URL}">
            <div class="hero-text">
                Netflix Analytics Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab1:

    df_dash = filtered_df.copy()

    # KPI metrics
    total_titles = int(df_dash['title'].nunique()) if not df_dash.empty else 0
    total_genres = int(df_dash['genre'].nunique()) if not df_dash.empty else 0
    total_ratings = int(df_dash['content_rating'].nunique()) if not df_dash.empty else 0
    start_year = int(df_dash['release_year'].min()) if not df_dash.empty else 0
    end_year = int(df_dash['release_year'].max()) if not df_dash.empty else 0
    total_locations = int(df_dash['production_country'].nunique()) if not df_dash.empty else 0

  #  st.markdown("### Netflix Overview Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Titles", f"{total_titles:,}")
    c2.metric("Total Genres", f"{total_genres:,}")
    c3.metric("Total Ratings", f"{total_ratings:,}")
    c4.metric("Total Locations", f"{total_locations:,}")
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

with tab2:

    st.subheader("Data-Driven Netflix Success Insights")

    insight = st.selectbox(
        "Choose an Insight to Explore",
        [
            "Format Battle: Movies vs TV Shows",
            "How Have Ratings Evolved Over Time?",
            "Genre Showdown: Who Wins the Ratings Race?",
            "Global Content: Who Delivers the Best Hits?",
            "Director Impact: Who Delivers Netflix’s Best Content?",
            "Audience Love: Ratings vs Popularity",
            "Does Length Make It Better?"
        ],
        index=0
    )
    
    # ---------------------------------------------------
    # INSIGHT 1 — Movies vs TV Shows
    # ---------------------------------------------------
    if insight == "Format Battle: Movies vs TV Shows":
        st.markdown("### Visualization")

        fig1 = px.box(
            filtered_df,
            x="content_type",
            y="IMDb_avg_rating",
            color="content_type",
            color_discrete_map={"Movie": "#E50914", "TV Show": "#585858"},
            title="IMDb Ratings Distribution: Movies vs TV Shows",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Hypothesis Test
        movies = filtered_df[filtered_df['content_type']=="Movie"]["IMDb_avg_rating"].dropna()
        tv = filtered_df[filtered_df['content_type']=="TV Show"]["IMDb_avg_rating"].dropna()
        t, p = stats.ttest_ind(tv, movies, equal_var=False)
        d = (tv.mean() - movies.mean()) / filtered_df['IMDb_avg_rating'].std()

        with st.expander("Statistical Test Results"):
            st.markdown("**t-test** — Comparing IMDb Ratings between Content Types")
            st.write(f"• t-statistic: **{t:.2f}** ")
            st.write(f"• p-value: **{p:.3e}**")
            st.write(f"• Effect Size Cohen's d: **{d:.2f}**  → Medium Effect")

        st.markdown("### Interpretation")
        st.info("""The box plot shows that TV Shows have a slightly higher median IMDb rating than Movies, a difference that is statistically significant based on Welch’s t-test. 
                    However, both formats contain numerous lower outliers, indicating  that format alone does not determine whether a title will become highly rated or poorly received.
                """)

        st.markdown("""### Strategy Recommendation""")
        st.success("""  ➡ Give episodic Originals a slight prioritization in future content planning
                """)

    # ---------------------------------------------------
    # INSIGHT 2 — Popularity vs Quality
    # ---------------------------------------------------
    elif insight == "How Have Ratings Evolved Over Time?":
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

        st.markdown("---")
        st.markdown("## Statistical Analysis — Trend Over Years")

        # Remove years with very few titles
        trend_data = year_avg[year_avg['IMDb_avg_rating'].notna()]

        # Pearson correlation (linear trend strength)
        r, p_value = stats.pearsonr(trend_data['release_year'], trend_data['IMDb_avg_rating'])

        # Interpretation level
        if abs(r) < 0.2:
            impact_label = "Very Weak"
        elif abs(r) < 0.4:
            impact_label = "Weak"
        elif abs(r) < 0.6:
            impact_label = "Moderate"
        else:
            impact_label = "Strong"

        # Display stats
        with st.expander("Statistical Results"):
            st.markdown("**Correlation Test — Ratings Over Time**")
            st.write(f"• Pearson r: **{r:.2f}** → {impact_label} trend")

        # Interpretation
        st.markdown("### Interpretation")
        st.markdown("Release year does not meaningfully influence IMDb ratings.Content quality has remained consistently average-good, regardless of time period.")

        # Strategy
        st.markdown("### Strategy Recommendation")
        st.success(
        """
        ➡️ Ratings remain steady across decades. So, success depends more on what you make, not when you release it.
        """
        )

    # ---------------------------------------------------
    # INSIGHT 3 — Genre Performance
    # ---------------------------------------------------
    elif insight == "Genre Showdown: Who Wins the Ratings Race?":
        st.markdown("### Visualization")

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

# --------------------------------------------------
# Hypothesis Testing — One-Way ANOVA
# --------------------------------------------------
        genre_groups = [
            group["IMDb_avg_rating"].dropna().values
            for _, group in filtered_df.groupby("genre")
            if len(group) >= 50
        ]

        f_stat, p_val = stats.f_oneway(*genre_groups)

        grand_mean = filtered_df["IMDb_avg_rating"].mean()
        ss_between = sum([
            len(group) * (group["IMDb_avg_rating"].mean() - grand_mean) ** 2
            for _, group in filtered_df.groupby("genre")
            if len(group) >= 50
        ])
        ss_total = np.sum((filtered_df["IMDb_avg_rating"] - grand_mean) ** 2)
        eta_sq = ss_between / ss_total

        # Display ANOVA results in Streamlit
        with st.expander("Statistical Test Results"):
            st.markdown("**One-Way ANOVA** — Comparing IMDb Ratings Across Genres")
            st.write(f"• F-Statistic: **{f_stat:.2f}**")
            st.write(f"• p-value: **{p_val:.3e}**")
            st.write(f"• Effect Size (η²): **{eta_sq:.2f}**  → Medium Effect")

# --------------------------------------------------
# INTERPRETATION + STRATEGY
# --------------------------------------------------
        st.markdown("### Interpretation")
        st.info(
            "Genres show **statistically significant** differences in IMDb ratings. "
            "High-engagement genres like **Classic & Cult TV**, **Science & Nature TV**, and "
            "**Anime Series** consistently outperform broader categories such as TV Dramas."
        )

        st.markdown("### Strategy Recommendation")
        st.success(
            """
        ➡️ Prioritize **specialized genres** where passionate audiences drive above-average ratings
        """
        )

    # ---------------------------------------------------
    # INSIGHT 4 — Country of Origin
    # ---------------------------------------------------
    elif insight == "Global Content: Who Delivers the Best Hits?":

        country_df = (
            filtered_df.groupby('production_country')['IMDb_avg_rating']
            .agg(['mean', 'count'])
            .reset_index()
        )

        country_df = country_df[country_df['count'] >= 30]  # avoid unreliable small samples
        top10 = country_df.sort_values('mean', ascending=False).head(10)

        fig_top10 = px.bar(
            top10,
            x='mean',
            y='production_country',
            orientation='h',
            color='mean',
            color_continuous_scale='reds',
            labels={'mean': 'Avg IMDb Rating'},
            title='Top 10 Countries by IMDb Rating (≥30 Titles)'
        )

        fig_top10.update_layout(
            template='plotly_dark',
            height=450,
            yaxis=dict(categoryorder='total ascending')
        )

        st.plotly_chart(fig_top10, use_container_width=True)

        st.markdown("---")
        st.markdown("### Statistical Analysis — Country of Origin")

        # One-Way ANOVA across countries (≥30 titles)
        country_groups = [
            group["IMDb_avg_rating"].dropna().values
            for _, group in filtered_df.groupby("production_country")
            if len(group) >= 30
        ]

        f_stat, p_val = stats.f_oneway(*country_groups)

        # Effect Size — Eta Squared
        grand_mean = filtered_df["IMDb_avg_rating"].mean()
        ss_between = sum([
            len(group) * (group['IMDb_avg_rating'].mean() - grand_mean) ** 2
            for _, group in filtered_df.groupby("production_country")
            if len(group) >= 30
        ])
        ss_total = np.sum((filtered_df["IMDb_avg_rating"] - grand_mean) ** 2)
        eta_sq = ss_between / ss_total

        # Display Results
        with st.expander(" Statistical Results"):
            st.markdown("**One-Way ANOVA — Differences Across Countries**")
            st.write(f"• F-Statistic: **{f_stat:.2f}**")
            st.write(f"• p-value: **< 0.001** (highly significant)")
            st.write(f"• Effect Size (η²): **{eta_sq:.2f}** → Small-Medium")

        # Interpretation
        st.markdown("### Interpretation")
        st.info(
            "Average ratings differ between countries. "
            "Successful markets like the **Paraguay,Japan and Sweden** often deliver highly-rated content, reflecting strong storytelling rooted in cultural identity."
        )

        # Business Strategy
        st.markdown("### Strategy Recommendation")
        st.success(
            """
        ➡️ Increase collaborations with **high-performing international regions**  
        ➡️ Empower local creators to preserve cultural authenticity  
        ➡️ Boost global promotion of **international critical successes**  

        📌 Country has a **meaningful**, yet smaller influence compared to genre.
        """
        )
    
    elif insight == "Does Length Make It Better?":

        st.subheader("Does Length Make It Better? (Runtime vs Rating)")

        # ---- Separate Datasets ----
        movies_df = filtered_df[filtered_df['content_type'] == 'Movie'].copy()
        tv_df = filtered_df[filtered_df['content_type'] == 'TV Show'].copy()

        # TV Shows → Total Runtime
        tv_df['total_runtime_mins'] = tv_df['runtime_mins'] * tv_df['episodes']

        # Movies → keep runtime column for consistency
        movies_df['total_runtime_mins'] = movies_df['runtime_mins']

        # Remove invalids
        movies_df = movies_df.dropna(subset=['total_runtime_mins','IMDb_avg_rating'])
        tv_df = tv_df.dropna(subset=['total_runtime_mins','IMDb_avg_rating'])

        # ---- Layout ----
        row1c1, row1c2 = st.columns(2)

        # =======================
        # Chart 1 — MOVIES ONLY
        # =======================
        with row1c1:
            fig_movie = px.scatter(
                movies_df,
                x="total_runtime_mins",
                y="IMDb_avg_rating",
                size="numVotes",
                opacity=0.5,
                trendline=None,  # avoid statsmodels dependency
                hover_data=["title","release_year"],
                color_discrete_sequence=['#E50914'],
                title="Movies: Runtime vs IMDb Rating"
            )
            fig_movie.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_movie, use_container_width=True)

            # Stats for Movies
            from scipy.stats import pearsonr
            r_m, p_m = pearsonr(movies_df['total_runtime_mins'], movies_df['IMDb_avg_rating'])

            st.markdown(f"**Movies Correlation:** r = {r_m:.2f}, p = {p_m:.3e}")
            st.info("Interpretation: Longer movies slightly perform better — but runtime alone isn’t a strong driver.")

        # =======================
        # Chart 2 — TV SHOWS ONLY
        # =======================
        with row1c2:
            fig_tv = px.scatter(
                tv_df,
                x="total_runtime_mins",
                y="IMDb_avg_rating",
                size="numVotes",
                opacity=0.5,
                hover_data=["title","release_year"],
                color_discrete_sequence=['#585858'],
                title="TV Shows: Total Runtime vs IMDb Rating"
            )
            fig_tv.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_tv, use_container_width=True)

            # Stats for TV Shows
            r_t, p_t = pearsonr(tv_df['total_runtime_mins'], tv_df['IMDb_avg_rating'])

            st.markdown(f"**TV Shows Correlation:** r = {r_t:.2f}, p = {p_t:.3e}")
            st.info("Interpretation: Total runtime barely relates to rating — content quality is what keeps viewers watching.")

        # =======================
        # Business Strategy Output
        # =======================
        st.success("""
        **Strategy Recommendation:**  
        Let the story decide the runtime  
        Focus on **engagement per minute**, not length inflation.
        """)

    elif insight == "🎬 Director Impact: Who Delivers Netflix’s Best Content?":

    # ===========================
    # Director Performance Insight (≥10 Titles)
    # ===========================

        st.markdown("---")
        st.subheader("Director Power Rankings")

        # Filter directors with ≥10 titles
        director_df = (
            filtered_df.groupby('director')['IMDb_avg_rating']
            .agg(['mean', 'count'])
            .reset_index()
        )
        director_df = director_df[director_df['count'] >= 20]

        # Sort top 10 by avg rating
        top_directors = director_df.sort_values('mean', ascending=False).head(5)['director'].tolist()

        df_top10_dir = filtered_df[filtered_df['director'].isin(top_directors)]

        fig_dir_box = px.box(
            df_top10_dir,
            x="director",
            y="IMDb_avg_rating",
            color="director",
            title="Top 5 Directors (≥20 Titles) — IMDb Rating Distribution",
            labels={'director': 'Director', 'IMDb_avg_rating': 'IMDb Rating'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        fig_dir_box.update_layout(
            template="plotly_dark",
            xaxis_tickangle=-45,
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig_dir_box, use_container_width=True)

        # Mean markers
        for d in top_directors:
            mean_val = director_df.loc[director_df['director']==d,'mean'].values[0]
            count = director_df.loc[director_df['director']==d,'count'].values[0]
            st.markdown(f" **{d}** — Avg Rating: **{mean_val:.2f}** | Titles: {count}")

        # Interpretation
        st.markdown("### Interpretation")
        st.info("""
        Directors with strong reputations tend to deliver **more consistent audience satisfaction**.
        Some directors show **high median ratings with wide variability**, meaning they take risks
        — sometimes they win big, sometimes they flop.
        """)

        # Strategy Recommendation
        st.markdown("### Strategy Recommendation")
        st.success("""
        ➡ Invest in **proven directors** who consistently exceed audience expectations  
        ➡ Track directors with **high variance** — they may deliver the next breakout hit 🎯  
        """)


    elif insight == "Audience Love: Ratings vs Popularity":

        st.subheader("Audience Love: Ratings vs Popularity (Votes vs IMDb Rating)")

        fig_votes = px.scatter(
            filtered_df,
            x="numVotes",
            y="IMDb_avg_rating",
            opacity=0.6,
            size="IMDb_avg_rating",
            hover_data=["title", "release_year"],
            color="content_type",   # color by type
            color_discrete_map={
                "Movie": "#E50914",
                "TV Show": "#585858"
            },
            title="IMDb Rating vs Number of Votes (Movie vs TV Show)"
        )

        fig_votes.update_layout(
            template="plotly_white",
            height=450,
            xaxis_type='log',  # log scale for votes
            xaxis_title="Number of Votes (log scale)",
            yaxis_title="IMDb Rating",
            legend_title="Content Type"
        )

        st.plotly_chart(fig_votes, use_container_width=True)

        # ===========================
        # Statistical Analysis
        # ===========================
        r_v, p_v = stats.pearsonr(
            filtered_df['numVotes'], filtered_df['IMDb_avg_rating']
        )

        with st.expander("Statistical Results"):
            st.write(f"• Pearson r: **{r_v:.2f}** → Moderate Positive Relationship")
            st.write(f"• p-value: **< 0.001** → Statistically Significant")

        st.markdown("### Interpretation")
        st.info("""
        - Titles with higher IMDb ratings **generally receive more votes**
        - But high votes do **not always** mean high quality (e.g., hyped shows)
        - Some hidden gems have **high rating but low exposure**
        """)

        st.markdown("### Strategy Recommendation")
        st.success("""
        ➡ Boost marketing for **high-rated but under-watched** titles  
        ➡ Strongly rated movies attract audiences faster than TV shows  
        ➡ Use rating–votes correlation to guide **algorithmic promotion**
        """)

with tab3:

    st.title("IMDb Rating Predictor")
    st.markdown("### Build your content & predict how well it performs on IMDb!")

    # =============================
    # DATA ENCODING & CLEANING
    # =============================
    filtered_df_modelling = filtered_df.copy()

    # --- Compute total runtime (Movies: minutes | TV: episodes * minutes) ---
    filtered_df_modelling['total_runtime'] = filtered_df_modelling.apply(
        lambda row: row['runtime_mins'] if row['content_type'] == "Movie"
        else (row['runtime_mins'] * row.get('episodes', np.nan)
              if pd.notna(row.get('episodes', np.nan)) else np.nan),
        axis=1
    )
    filtered_df_modelling['total_runtime'] = filtered_df_modelling['total_runtime'].fillna(
        filtered_df_modelling['total_runtime'].median()
    )

    # Frequency Encoding: Country & Genre
    country_freq = filtered_df_modelling['production_country'].value_counts(normalize=True)
    filtered_df_modelling['country_encoded'] = filtered_df_modelling['production_country'].map(country_freq)

    genre_freq = filtered_df_modelling['genre'].value_counts(normalize=True)
    filtered_df_modelling['genre_encoded'] = filtered_df_modelling['genre'].map(genre_freq)

    # Create content_rating dummies
    filtered_df_modelling['original_rating'] = filtered_df_modelling['content_rating']
    filtered_df_modelling = pd.get_dummies(filtered_df_modelling,
                                           columns=['content_rating'],
                                           prefix='rating')

    # Director encoding using average IMDb rating per director
    global_mean = filtered_df_modelling['IMDb_avg_rating'].mean()
    director_mean = filtered_df_modelling.groupby('director')['IMDb_avg_rating'].mean()
    filtered_df_modelling['director_encoded'] = filtered_df_modelling['director'].map(director_mean).fillna(global_mean)

    # --- Define features & target ---
    rating_cols = [c for c in filtered_df_modelling.columns if c.startswith("rating_")]
    features = ['country_encoded', 'genre_encoded', 'total_runtime', 'director_encoded'] + rating_cols
    target = 'IMDb_avg_rating'

    # Drop rows with missing predictors/target
    filtered_df_modelling = filtered_df_modelling.dropna(subset=features + [target]).reset_index(drop=True)

    # Separate Movies / TV data for modeling
    movies_df = filtered_df_modelling[filtered_df_modelling['content_type'] == "Movie"]
    tv_df = filtered_df_modelling[filtered_df_modelling['content_type'] == "TV Show"]

    # =============================
    # MODEL TRAINING FUNCTION
    # =============================
    def train_models(df):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Linear Regression
        lr = LinearRegression().fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        return {
            "Linear Regression": (lr,
                                    root_mean_squared_error(y_test, lr_pred),
                                    r2_score(y_test, lr_pred)),
            "Random Forest": (rf,
                                 root_mean_squared_error(y_test, rf_pred),
                                 r2_score(y_test, rf_pred))
        }

    movie_models = train_models(movies_df)
    tv_models = train_models(tv_df)

    # =============================
    # ① USER INPUT AREA
    # =============================
    st.header("🎬 Build Your Content")

    content_type_in = st.selectbox("Select Title Type",
                                   ["Movie 🍿", "TV Show 📺"])

    country_input = st.selectbox(
        "Production Country",
        sorted(filtered_df_modelling['production_country'].dropna().unique())
    )
    genre_input = st.selectbox(
        "Genre",
        sorted(filtered_df_modelling['genre'].dropna().unique())
    )
    director_input = st.selectbox(
        "Director",
        sorted(filtered_df_modelling['director'].dropna().unique())
    )
    rating_input = st.selectbox(
        "Content Rating",
        sorted(filtered_df_modelling['original_rating'].dropna().unique())
    )

    runtime_input = st.slider(
        "⏱ Total Runtime (Minutes)",
        min_value=20, max_value=600,
        value=100, step=10
    )

    st.divider()

    # =============================
    # ② Model Leaderboard Display
    # =============================
    st.header("Model Leaderboard")

    col1, col2 = st.columns(2)

    if "Movie" in content_type_in:
        sel_models = movie_models
        section = col1
        rival = col2
        section.subheader("🍿 Movie Models (Active)")
        rival.subheader("📺 TV Show Models")
        rival.caption("Not used for Movie prediction")
    else:
        sel_models = tv_models
        section = col1
        rival = col2
        section.subheader("📺 TV Models (Active)")
        rival.subheader("🍿 Movie Models")
        rival.caption("Not used for TV prediction")

    for model_name, (_, rmse, r2) in sel_models.items():
        section.metric(model_name, f"RMSE: {rmse:.3f}", f"R²: {r2:.3f}")

    st.divider()

    # =============================
    # ③ PREDICTION ENGINE SELECTOR
    # =============================
    st.header("Choose Model for Prediction")
    selected_model = st.selectbox("Prediction Engine",
                                  list(sel_models.keys()))
    model = sel_models[selected_model][0]

    # =============================
    # ④ MAKE PREDICTION
    # =============================
    if st.button("Predict IMDb Rating"):
        input_vec = [
            country_freq.get(country_input, 0),
            genre_freq.get(genre_input, 0),
            runtime_input,
            director_mean.get(director_input, global_mean)
        ]

        encoded = [0] * len(rating_cols)
        rating_col = f"rating_{rating_input}"
        if rating_col in rating_cols:
            encoded[rating_cols.index(rating_col)] = 1

        input_vec.extend(encoded)

        prediction = float(model.predict([input_vec])[0])
        st.success(f"Predicted IMDb Rating: **{prediction:.2f}**")

        if prediction >= 7.0:
            st.balloons()
            st.markdown("🔥 Strong potential for Netflix **Top 10**!")
        elif prediction >= 6.0:
            st.markdown("👍 Good! **Audience will enjoy it.**")
        else:
            st.markdown("😬 Might struggle to gain traction…")

    st.divider()

    # =============================
    # FEATURE IMPORTANCE (RF only)
    # =============================
    st.header("💡 What Drives IMDb Success?")

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(filtered_df_modelling[features], filtered_df_modelling[target])

    importance = pd.Series(rf_model.feature_importances_, index=features)
    importance_sorted = importance.sort_values()

    fig_imp = go.Figure(go.Bar(
        x=importance_sorted.values,
        y=importance_sorted.index,
        orientation='h',
        marker_color='red'
    ))

    fig_imp.update_layout(
        title="Feature Importance for IMDb Rating Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        template="plotly_white"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    st.info("🎬 Director influence dominates — audience trusts creators they love.")

with tab4:
    
        netflix_imdb_df=netflix_imdb_df.drop(columns=['runtime_mins'])
        data_handling_viz = st.selectbox(
        "Choose a type of data handling to explore ",
        [
            "Pre-Processing",
            "Missingness Analysis",
            "Imputation",
            "Feature Engineering"
        ],
        index=0
    )
        rows, cols = filtered_df.shape
    
        if "Pre-Processing" in data_handling_viz:

            st.subheader("🧹 Data Preprocessing Overview")
            if filtered_df.empty:
                avg_rating = 0  # or np.nan if you prefer
            else:
                avg_rating = filtered_df['IMDb_avg_rating'].mean().round(2)

            # --- KPI Section ---
            st.markdown("### Dataset Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{rows:,}")
            col2.metric("Columns", f"{cols}")
            col3.metric("Avg IMDb rating ⭐", f"{avg_rating}")


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
            st.subheader("Data Preprocessing Overview")

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

        elif "Missingness Analysis" in data_handling_viz:
                
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
            st.info("From the above visualizations, we can observe that the **director** and **production_country** columns contain missing values.")

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

            st.info("Overall, 7.05% of entries have missing production_country information. Among those entries with missing production_country, 46.66% also have missing director information. In comparison, 25.37% of all entries have missing director values. Conversely, for entries with missing director information, 12.96% have missing production_country values.")
            st.info("Missing director values are much more likely when production_country information is missing, indicating a strong association between the two. However, missing production_country values are relatively uncommon even when director information is absent, suggesting that missingness in production_country is more independent.")
            

        # --- Director Missing % ---
            director_missing_counts = netflix_df[netflix_df['director'].isna()]['content_type'].value_counts()
            director_missing_share = (director_missing_counts / director_missing_counts.sum()) * 100

            types = director_missing_share.index
            percentages = director_missing_share.values

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
            st.info("- The majority of missing **director** values come from **TV Shows (94.85%)**, with a small portion from **Movies (5.15%)**. " \
                        "Hence, missing directors are not missing completely at random (MCAR).")
            st.info("- The missing values in the **production_country** column are distributed almost evenly between **Movies (51.04%)** and **TV Shows (48.96%)**.")

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
            st.info("- The missing values show notable spikes in the 1960s and 1970s, reaching up to around 40%. Between 1980 and 2000, missingness is more consistent and generally below 15%. From 2010 onward, there is a steady increase, culminating at approximately 43% missing by 2020")
            st.info("**production_country**")
            st.info("- production_country missing values are generally low, staying mostly below 10%, with brief spikes of 20–30% in the 1960s and 1980s. After 2000, missingness remains minimal until around 2020, when it rises sharply to about 30%.")
        
        elif "Imputation" in data_handling_viz:
            
            st.header("Imputation Strategy")

            st.subheader("Director")
            st.info("Missing directors were mostly from TV Shows (~95%), indicating non-random missingness. To preserve authenticity, used IMDb Web Services dataset to fill these values with actual director names instead of statistical approach.")

            st.subheader("Production_Country")
            st.info("For production_country, the missing values were found to be randomly distributed. Hence, applied statistical imputations and compared Mode, KNN, and MICE techniques to determine the most suitable approach.")

            # 1) MODE
            st.image("Mode_Imputation.png", use_container_width=True)
            st.markdown("""
        **Mode Imputation - Interpretation**

        - All missing countries are filled with a **single most frequent country**.  
        - The red points align in a narrow horizontal band, and the boxplot of imputed
        values shows **almost no spread**.  
        - This destroys real-world variability and can bias downstream models.

        ➡️ **Not recommended**.
        """)

            # 2) KNN
            st.image("KNN_Imputation.png", use_container_width=True)
            st.markdown("""
        **KNN Imputation – Interpretation**

        - Red imputed points spread across multiple country codes and follow a mild
        trend with **IMDb rating**, meaning neighbours in
        (`release_year`, `IMDb_rating`, `numVotes`) space influence the imputed country.  
        - Distribution is closer to the original but still slightly concentrated in dense regions.

        ➡️ **Reasonable choice**, but can overfit to local patterns and is sensitive to scaling / density.
        """)

            # 3) MICE
            st.image("MICE_Imputation.png", use_container_width=True)
            st.markdown("""
        **MICE Imputation – Interpretation**

        - Imputed points (red) cover a **similar range** and dispersion as the original grey points.  
        - The boxplot of imputed-only rows overlaps well with the original distribution,
        preserving both the center and spread.  
        - Because MICE uses iterative regression with predictors
        (`release_year`, `IMDb_rating`, `numVotes`), it better respects the
        multivariate relationships in the data.

        ➡️ **Most statistically sound** among the three use-cases.
        """)

            st.markdown("""
        ### ✅ Final Recommendation

        Use **MICE (Iterative Imputer with BayesianRidge)** to fill missing `production_country`:
                          
        - It preserves both **variance** and **relationship with IMDb rating**, unlike Mode.  
        - It produces more stable estimates than a KNN in sparse regions.
        """)
            st.subheader("After Imputation")
            
            missing_counts_ni = netflix_imdb_df.isnull().sum()
            missing_percent_ni = (missing_counts_ni / len(netflix_df)) * 100
            present_percent_ni = 100 - missing_percent_ni

            # Sort variables by % missing
            sorted_vars_ni = missing_percent_ni.sort_values(ascending=False).index

            # DataFrame for plotting
            missing_df = pd.DataFrame({
                "Variable": sorted_vars_ni,
                "Missing": missing_percent_ni[sorted_vars_ni].values,
                "Present": present_percent_ni[sorted_vars_ni].values
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
            missing_matrix = netflix_imdb_df[sorted_vars_ni].isnull().astype(int).T

            fig_heat = px.imshow(
                missing_matrix,
                color_continuous_scale=['#E50914','#585858'],
                aspect='auto',
                labels=dict(x="Row Number", y="Variable", color="Missing")
            )

            fig_heat.update_layout(
                title='Missing Values in Rows',
                xaxis=dict(tickmode='linear', tick0=0, dtick=200),  # adjust dtick as needed
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_heat, use_container_width=True)

        elif "Feature Engineering" in data_handling_viz:
                    st.subheader("Feature Engineering Overview")
        
                    st.markdown("""
                    To build a predictive model capable of understanding audience preferences and IMDb rating behavior, 
                    several feature-engineering transformations were applied to the merged Netflix–IMDb dataset.
        
                    ---
        
                    ### **Total Runtime (Standardizing Content Length)**  
                    Movies and TV shows originally represented length differently (minutes vs. seasons).  
                    To make runtime comparable:
        
                    - **Movies:** `total_runtime = runtime_mins`  
                    - **TV Shows:** `total_runtime = runtime_mins × episodes`  
        
                    This creates a *single* metric (total minutes of content), allowing the model to fairly compare both Movies and TV Shows and learn how runtime affects IMDb ratings.
        
                    ---
        
                    ### **Frequency Encoding for Country & Genre**  
                    `production_country` and `genre` contain many unique categories.  
                    One-hot encoding would create hundreds of sparse columns.
        
                    Instead, each category was replaced with its **relative frequency** in the dataset:
        
                    - `country_encoded = frequency of that country`
                    - `genre_encoded = frequency of that genre`
        
                    ---
        
                    ### **Content Rating One-Hot Encoding**  
                    Ratings like **TV-MA, PG-13, TV-14** are *nominal*, not numeric.  
                    To avoid implying false order (TV-MA > TV-14), **one-hot encoding** was applied.
        
                    ---
        
                    ### **Director Encoding (Target Encoding)**  
                    There are thousands of directors (extreme cardinality).  
                    To avoid one-hot encoding, the **mean IMDb rating per director** was computed:
                    
                    Unknown directors receive the **global IMDb mean**.  

                    **Why Target Encoding?**
                    Unlike directors, countries and genres do not represent skill-based continuity, so their average ratings are not reliable signals and would cause leakage if target encoded.
                    ---
        
                    ### **Separate Models for Movies & TV Shows**  
                    After all transformations, the dataset was split:
        
                    - **Movie Model**
                    - **TV Show Model**
        
                    #### ✔ Why?
                    Movies and TV Shows behave differently in:
                    - runtime patterns  
                    - audience engagement  
                    - rating dynamics  
                    - director influence  
        
                    Separate models avoid forcing a single global relationship and improve accuracy.
                    """)

                    st.subheader("Data After applying Feature Engineering")
                    st.writer("Movies dataset")
                    st.dataframe(movies_df.head(5))
                    st.dataframe(tv_df.head(5))














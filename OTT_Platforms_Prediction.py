import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title="Netflix IMDb Analysis", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------

netflix_imdb_df = pd.read_csv("netflix_imdb.csv")
netflix_df=pd.read_csv('netflix_titles.csv', usecols=['title','type','director','country','release_year','rating','duration','listed_in'])
netflix_df['director'] = netflix_df['director'].str.split(', ')
netflix_df['country'] = netflix_df['country'].str.split(', ')
netflix_df['listed_in'] = netflix_df['listed_in'].str.split(', ')
netflix_df = netflix_df.explode('director').explode('country').explode('listed_in').reset_index(drop=True)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

type_filter = st.sidebar.multiselect(
    "Select Type",
    options=netflix_imdb_df['type'].dropna().unique(),
    default=netflix_imdb_df['type'].dropna().unique()
)

all_countries = sorted(netflix_imdb_df['country'].dropna().unique())

country_filter = st.sidebar.multiselect(
    "Select Country",
    options=["All"] + all_countries,
    default=["All"],
    help="Select countries. 'All' shows all data."
)
if "All" in country_filter:
    filtered_df = netflix_imdb_df.copy()
else:
    filtered_df = netflix_imdb_df[netflix_imdb_df['country'].isin(country_filter)]



adult_filter = st.sidebar.multiselect(
    "Select Adult Content",
    options=netflix_imdb_df['isAdult'].dropna().unique(),
    default=netflix_imdb_df['isAdult'].dropna().unique()
)

rating_filter = st.sidebar.multiselect(
    "Select Rating",
    options=netflix_imdb_df['rating'].dropna().unique(),
    default=netflix_imdb_df['rating'].dropna().unique()
)

release_year_range = st.sidebar.slider(
    "Select Release Year Range",
    int(netflix_imdb_df['release_year'].min()),
    int(netflix_imdb_df['release_year'].max()),
    (int(netflix_imdb_df['release_year'].min()), int(netflix_imdb_df['release_year'].max()))
)

# Apply filters
filtered_df = netflix_imdb_df[
    (netflix_imdb_df['type'].isin(type_filter)) &
    (netflix_imdb_df['country'] if "All" in country_filter else netflix_imdb_df['country'].isin(country_filter)) &
    (netflix_imdb_df['isAdult'].isin(adult_filter)) &
    (netflix_imdb_df['rating'].isin(rating_filter)) &
    (netflix_imdb_df['release_year'].between(release_year_range[0], release_year_range[1]))
]

# -----------------------------
# Page Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview", "Missing Value Analysis",
    "Univariate Analysis", "Bivariate Analysis", "Insights"
])

# -----------------------------
# TAB 1: Data Overview
# -----------------------------
with tab1:
    rows, cols = netflix_imdb_df.shape

    st.subheader("🧹 Data Preprocessing Overview")
    avg_rating = netflix_imdb_df['averageRating'].mean().round(2)

    # --- KPI Section ---
    st.markdown("### 📊 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", f"{cols}")
    col3.metric("Avg IMDb Rating ⭐", f"{avg_rating}")


    st.markdown("The aim of this project is to perform an in-depth exploratory data analysis (EDA) and predictive modeling on Netflix IMDb datasets to uncover insights into content trends and audience preferences. To acheive my aim, I have used Netflix and IMDb datasets.")
    st.header("Data Overview")
    st.dataframe(filtered_df.head(50))
    st.write(f"Shape of Dataset: {filtered_df.shape}")
    st.write("### Column Information:")
    st.write(filtered_df.dtypes)
    st.subheader("🧹 Data Preprocessing Overview")

    st.markdown("""
    ### Steps Performed During Data Preprocessing

    1. **Exploded Multi-Valued Columns**  
    - Columns such as *directors*, *listed_in* (genres), and *country* often contained multiple comma-separated values.  
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

    These preprocessing steps ensured data consistency, completeness, and accuracy for further analysis and IMDb rating prediction.
    """)


# -----------------------------
# TAB 2: Missing Value Analysis
# -----------------------------
with tab2:
    selected_missing_viz = st.multiselect(
    "Select Misisng Value Visualizations",
    ["Missing values", "Analysis with Content Type", "Missing Value Trend", "Conditional Probability"],
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
            marker_color='steelblue'
        ))

        fig_bar.add_trace(go.Bar(
            y=missing_df['Variable'],
            x=missing_df['Missing'],
            orientation='h',
            name='Missing',
            marker_color='salmon'
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
            color_continuous_scale=['steelblue', 'salmon'],
            aspect='auto',
            labels=dict(x="Row Number", y="Variable", color="Missing")
        )

        fig_heat.update_layout(
            title='Missing Values in Rows',
            xaxis=dict(tickmode='linear', tick0=0, dtick=200),  # adjust dtick as needed
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_heat, use_container_width=True)

    if "Conditional Probability" in selected_missing_viz:
    # Create cross-tab
        crossbar_1 = pd.crosstab(netflix_df['director'].isna(), netflix_df['country'].isna())

        # Convert index/columns to strings for better labels
        crossbar_1.index = crossbar_1.index.map({False: 'Director Present', True: 'Director Missing'})
        crossbar_1.columns = crossbar_1.columns.map({False: 'Country Present', True: 'Country Missing'})

        # Plot interactive heatmap
        fig = px.imshow(
            crossbar_1,
            text_auto=True,                  # show counts
            color_continuous_scale='Reds',
            labels=dict(x="Country Null Status", y="Director Null Status", color="Count")
        )

        fig.update_layout(
            title="Null Values Heatmap: Director vs Country",
            template='plotly_white',
            xaxis_side='top'
        )

        st.plotly_chart(fig, use_container_width=True)


        # Calculate probabilities
        both_missing = netflix_df[(netflix_df['director'].isna()) & (netflix_df['country'].isna())].shape[0]
        country_missing = netflix_df['country'].isna().sum()
        director_missing = netflix_df['director'].isna().sum()

        p_country_missing = np.round((country_missing / len(netflix_df)) * 100, 2)
        p_director_given_country_missing = np.round((both_missing / country_missing) * 100, 2)
        p_director_missing = np.round((director_missing / len(netflix_df)) * 100, 2)
        p_country_given_director_missing = np.round((both_missing / director_missing) * 100, 2)

        # Display in Streamlit
        st.subheader("Conditional Probabilities for Missing Values")
        st.markdown(f"- **P(Country missing):** {p_country_missing} %")
        st.markdown(f"- **P(Director missing | Country missing):** {p_director_given_country_missing} %")
        st.markdown(f"- **P(Director missing):** {p_director_missing} %")
        st.markdown(f"- **P(Country missing | Director missing):** {p_country_given_director_missing} %")

    if "Analysis with Content Type" in selected_missing_viz:
        # --- Step 1: Calculate missing % by type ---
        title_missing = netflix_df.groupby('type')['director'].apply(lambda x: x.isna().mean() * 100)
        types = title_missing.index
        missing_pct = title_missing.values

        # --- Step 2: Create bar chart ---
        fig = go.Figure(go.Bar(
            x=types,
            y=missing_pct,
            text=[f'{val:.2f}%' for val in missing_pct],  # show value on bar
            textposition='outside',
            marker_color='salmon'
        ))

        # --- Step 3: Layout ---
        fig.update_layout(
            title="% of Title Type with Missing Director",
            xaxis_title="Title Type",
            yaxis_title="Missing %",
            yaxis=dict(range=[0, max(missing_pct)*1.2]),  # add space for labels
            template='plotly_white'
        )

        # --- Step 4: Streamlit display ---
        st.plotly_chart(fig, use_container_width=True)


        # Calculate missing % by title type
        missing_counts = netflix_df[netflix_df['director'].isna()]['type'].value_counts()
        missing_share = (missing_counts / missing_counts.sum()) * 100
        director_missing = missing_share.reset_index()
        director_missing.columns = ['type', 'missing_pct']

        # Plot interactive bar chart
        fig = px.bar(
            director_missing,
            x='type',
            y='missing_pct',
            text=director_missing['missing_pct'].apply(lambda x: f'{x:.2f}%'),
            color='type',
            color_discrete_sequence=['teal']*len(director_missing),
            title='Director Missing % by Title Type'
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis_title='Percentage (%)',
            xaxis_title='Title Type',
            template='plotly_white',
            showlegend=False,
            height=400
        )

        # --- Step 1: Calculate % of missing directors by type ---
        missing_counts = netflix_df[netflix_df['director'].isna()]['type'].value_counts()
        missing_share = (missing_counts / missing_counts.sum()) * 100

        types = missing_share.index
        percentages = missing_share.values

        # --- Step 2: Create interactive bar chart ---
        fig = go.Figure(go.Bar(
            x=types,
            y=percentages,
            text=[f'{v:.2f}%' for v in percentages],  # show value on bar
            textposition='outside',
            marker_color='teal'
        ))

        # --- Step 3: Layout ---
        fig.update_layout(
            title='Director Missing % by Title Type',
            xaxis_title='Title Type',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, max(percentages)*1.2]),  # add space for labels
            template='plotly_white'
        )

        # --- Step 4: Display in Streamlit ---
        st.plotly_chart(fig, use_container_width=True)


        
        # --- Step 1: Calculate missing % by title type for 'country' ---
        title_missing = netflix_df.groupby('type')['country'].apply(lambda x: x.isna().mean()*100)
        types = title_missing.index
        missing_pct = title_missing.values

        # --- Step 2: Create interactive bar chart ---
        fig = go.Figure(go.Bar(
            x=types,
            y=missing_pct,
            text=[f'{val:.2f}%' for val in missing_pct],  # show value on top
            textposition='outside',
            marker_color='salmon'
        ))

        # --- Step 3: Layout ---
        fig.update_layout(
            title="% of Title Type with Missing Country",
            xaxis_title="Title Type",
            yaxis_title="Missing %",
            yaxis=dict(range=[0, max(missing_pct)*1.2]),
            template='plotly_white'
        )

        # --- Step 4: Show in Streamlit ---
        st.plotly_chart(fig, use_container_width=True)

        # --- Step 1: Calculate % of missing countries by title type ---
        country_missing_counts = netflix_df[netflix_df['country'].isna()]['type'].value_counts()
        country_missing_share = (country_missing_counts / country_missing_counts.sum()) * 100

        types = country_missing_share.index
        percentages = country_missing_share.values

        # --- Step 2: Create interactive bar chart ---
        fig = go.Figure(go.Bar(
            x=types,
            y=percentages,
            text=[f'{v:.2f}%' for v in percentages],  # show % on top
            textposition='outside',
            marker_color='teal'
        ))

        # --- Step 3: Layout ---
        fig.update_layout(
            title='Country Missing % by Title Type',
            xaxis_title='Title Type',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, max(percentages)*1.2]),  # add space for labels
            template='plotly_white'
        )

        # --- Step 4: Show in Streamlit ---
        st.plotly_chart(fig, use_container_width=True)
    if "Missing Value Trend" in selected_missing_viz: 
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
            line=dict(color='purple', width=2)
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
        year_missing = netflix_df.groupby('release_year')['country'].apply(lambda x: x.isna().mean()*100)
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
            title="Country Missing % by Release Year",
            xaxis_title="Release Year",
            yaxis_title="Missing %",
            yaxis=dict(range=[0, max(percentages)*1.2]),
            template='plotly_white',
            height=400
        )

        # --- Step 4: Show in Streamlit ---
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# TAB 3: Univariate Analysis
# -----------------------------
with tab3:
    st.header("Univariate Analysis")

    selected_uni_viz = st.multiselect(
        "Select Univariate Visualizations",
        ["Type Distribution", "Rating Distribution", "Release Year Distribution", "Average Rating Distribution"],
        default=["Type Distribution"]
    )

    if "Type Distribution" in selected_uni_viz:
        type_counts = filtered_df['type'].value_counts().reset_index()
        type_counts.columns = ['type', 'count']
        fig_type = px.bar(type_counts, x='type', y='count', text='count',
                          title='Distribution of Title Type')
        st.plotly_chart(fig_type, use_container_width=True)
        st.markdown("- Majority of content on Netflix is **TV Shows**, but **Movies** also form a decent portion.")

    if "Rating Distribution" in selected_uni_viz:
        rating_counts = filtered_df['rating'].value_counts().reset_index()
        rating_counts.columns = ['rating', 'count']
        fig_rating = px.bar(rating_counts, x='rating', y='count', text='count',
                            title='Distribution of MPAA / TV Ratings')
        fig_rating.update_xaxes(tickangle=45)
        st.plotly_chart(fig_rating, use_container_width=True)
        st.markdown("- Most content falls under **TV-MA**, **TV-14**, and **PG-13**.")

    if "Release Year Distribution" in selected_uni_viz:
        fig_year = px.histogram(filtered_df, x='release_year', nbins=50,
                                title='Distribution of Release Year')
        st.plotly_chart(fig_year, use_container_width=True)
        st.markdown("- Most titles are from **recent years (2015–2021)**.")

    if "Average Rating Distribution" in selected_uni_viz:
        fig_avg = px.histogram(filtered_df, x='averageRating', nbins=20,
                               title='Distribution of IMDb Average Rating')
        st.plotly_chart(fig_avg, use_container_width=True)
        st.markdown("- Ratings mostly lie between **6–8**, showing a normal distribution.")

# -----------------------------
# TAB 4: Bivariate Analysis
# -----------------------------
with tab4:
    st.header("Bivariate Analysis")

    selected_bi_viz = st.multiselect(
        "Select Bivariate Visualizations",
        ["Votes vs IMDb Rating", "Correlation"],
        default=["Votes vs IMDb Rating"]
    )

    if "Votes vs IMDb Rating" in selected_bi_viz:
        fig_votes_rating = px.scatter(
            filtered_df, x='numVotes', y='averageRating',
            color='type', size='numVotes',
            title='Number of Votes vs Average Rating',
            log_x=True, size_max=20, hover_data=['title','release_year']
        )
        st.plotly_chart(fig_votes_rating, use_container_width=True)

    if "Correlation" in selected_bi_viz:
        num_cols = ['release_year', 'Movie_duration', 'TV Show_duration', 'averageRating', 'numVotes']

        # Compute correlation
        corr = filtered_df[num_cols].corr().round(2)

        # Create interactive heatmap
        fig = px.imshow(
            corr,
            text_auto=True,              # show correlation values
            color_continuous_scale='RdBu_r',
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


# -----------------------------
# TAB 5: Insights
# -----------------------------
with tab5:
    st.header("Key Insights")

    selected_insights = st.multiselect(
        "Select Insight Visualizations",
        ["Title Type", "Top Genres", "Adult IMDb rating", "Content rating",
         'Top Countries', 'IMDb Trend','Top Directors'],
        default=["Top Genres"]
    )

    if "Title Type" in selected_insights:
        # --- Average IMDb Rating by Type (Bar Chart) ---
        type_avg = filtered_df.groupby('type', as_index=False)['averageRating'].mean()

        fig_bar = px.bar(
            type_avg,
            x='type',
            y='averageRating',
            color='type',
            text='averageRating',
            color_discrete_map={'Movie': 'steelblue', 'TV Show': 'salmon'},
            title='Average IMDb Rating by Title Type'
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(template='plotly_white', showlegend=False, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Distribution of Title Types (Pie Chart) ---
        type_count = filtered_df['type'].value_counts().reset_index()
        type_count.columns = ['type', 'count']

        fig_pie = px.pie(
            type_count,
            names='type',
            values='count',
            color='type',
            color_discrete_map={'Movie': 'steelblue', 'TV Show': 'salmon'},
            title='Distribution of Netflix Title Types'
        )
        fig_pie.update_traces(textposition='inside', textinfo='label+percent')
        fig_pie.update_layout(template='plotly_white', height=450)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("**TV Show** tend to have slightly higher average ratings than Movie, However **Movie** dominates majority of the content")

    if "Top Genres" in selected_insights:
        # First, get the type for each genre (Movies or TV Shows)
        # Create genre_df from the filtered dataset
        genre_df = filtered_df[['listed_in', 'type', 'averageRating']].dropna()
        genre_type = genre_df.groupby('listed_in')['type'].first().reset_index()

        # Merge with average rating
        genre_avg = (
            genre_df.groupby('listed_in')['averageRating']
            .mean()
            .reset_index()
            .merge(genre_type, on='listed_in')
        )

        # Take top 10 by average rating
        genre_avg_top10 = genre_avg.sort_values('averageRating', ascending=False).head(10)

        # Gradient bar chart colored by type
        fig_bar = px.bar(
            genre_avg_top10,
            x='averageRating',
            y='listed_in',
            orientation='h',
            title="Top 10 Genres by Average IMDb Rating",
            labels={'averageRating': 'Average IMDb Rating', 'listed_in': 'Genre', 'type':'Title Type'},
            color='type',  # color by title type
            color_discrete_map={'Movie':'steelblue', 'TV Show':'salmon'}
        )

        fig_bar.update_layout(
            yaxis=dict(categoryorder='total ascending'),  # highest rating on top
            height=500,
            template='plotly_white',
            title_font=dict(size=18, family='Arial', color='black'),
            xaxis=dict(showgrid=True),
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie chart colored by type
        fig_genre_pie = px.pie(
            genre_avg_top10,
            names='listed_in',
            values='averageRating',
            title="Proportion of Genre",
            color='type',
            color_discrete_map={'Movie':'steelblue', 'TV Show':'salmon'}
        )

        fig_genre_pie.update_traces(textposition='inside', textinfo='label+percent')
        fig_genre_pie.update_layout(template='plotly_white', height=450)

        st.plotly_chart(fig_genre_pie, use_container_width=True)
        st.markdown('**Classic & Cult** TV has the highest average rating, while other genres maintains around 7. Also, it can be seen that all the genres maintain almost equal distribution')



    if "Adult IMDb rating" in selected_insights:
# Average IMDb rating by Adult
        adult_avg = filtered_df.groupby('isAdult', as_index=False)['averageRating'].mean()
        adult_avg['isAdult'] = adult_avg['isAdult'].map({0:'Non-Adult',1:'Adult'})

        # --- Bar chart ---
        fig_adult_bar = px.bar(
            adult_avg,
            x='isAdult',
            y='averageRating',
            text='averageRating',
            title="Average IMDb Rating by Adult Flag",
            labels={'isAdult':'Adult Flag','averageRating':'Average IMDb Rating'},
            color='isAdult',
            color_discrete_map={'Non-Adult':'steelblue','Adult':'salmon'}
        )
        fig_adult_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_adult_bar.update_layout(template='plotly_white', height=450, showlegend=False)
        st.plotly_chart(fig_adult_bar, use_container_width=True)


        # --- Pie chart ---
        fig_adult_pie = px.pie(
            adult_avg,
            names='isAdult',
            values='averageRating',
            title="Proportion of Adult Flag",
            color='isAdult',
            color_discrete_map={'Non-Adult':'steelblue','Adult':'salmon'}
        )
        fig_adult_pie.update_traces(textposition='inside', textinfo='label+percent')
        fig_adult_pie.update_layout(template='plotly_white', height=450)
        st.plotly_chart(fig_adult_pie, use_container_width=True)


    if "Content rating" in selected_insights:
        # Round values to 2 decimals
        rating_avg = filtered_df.groupby('rating')['averageRating'].mean().reset_index()
        avg_values = np.round(rating_avg['averageRating'].values, 2)

        # Create a heatmap using a single-row DataFrame
        fig = px.imshow(
            [avg_values],               # single row of values
            x=rating_avg['rating'],     # Netflix rating categories
            y=['Average IMDb Rating'],  # label for row
            text_auto=True,
            aspect='auto',
            color_continuous_scale='Viridis',  # continuous color scale
            range_color=[0, 10]
        )   

        # Layout updates
        fig.update_layout(
            title='Average IMDb Rating by Netflix Rating',
            template='plotly_white',
            height=300,
            xaxis_title='Netflix Rating',
            yaxis_title='',
            yaxis=dict(showticklabels=True),
            title_font=dict(size=18, family='Arial', color='black')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('All the ratings other UR has an average IMDb rating close to 6.5')

    if "Top Countries" in selected_insights:
        country_avg = (
        filtered_df.groupby('country', as_index=False)['averageRating']
        .mean()
        .round(2)
    )

    # Create the choropleth map
        fig_map = px.choropleth(
            country_avg,
            locations='country',        # Column with country names
            locationmode='country names', 
            color='averageRating',      # Column to color by
            hover_name='country',       # Show country on hover
            color_continuous_scale='Viridis',
            range_color=[0, 10],        # IMDb rating scale
            labels={'averageRating': 'Avg IMDb Rating'},
            title='Average IMDb Rating by Country'
        )

        fig_map.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Display in Streamlit
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("Most of the countries maintains the average rating around 6. while countries like **Germany**, **Namibia** have the rating around 7 and **Kahakasthan** and **Angola** have around 4")

    if "IMDb Trend" in selected_insights:
        # Compute average rating by release year
        year_avg = filtered_df.groupby('release_year')['averageRating'].mean().reset_index()

        # Line chart without markers
        fig = px.line(
            year_avg,
            x='release_year',
            y='averageRating',
            title="IMDb Rating Trend Over Years",
            labels={'release_year': 'Release Year', 'averageRating': 'Average IMDb Rating'}
        )

        fig.update_traces(mode='lines')  # only lines, no markers
        fig.update_layout(
            template='plotly_white',
            height=400,
            title_font=dict(size=18, family='Arial', color='black'),
            yaxis=dict(range=[0, 10]),  # fix scale from 0 to 10
            xaxis=dict(showgrid=True),
            yaxis_title='Average IMDb Rating',
            xaxis_title='Release Year'
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('Overall, ratings remain stable around **6.0–7.0**. Sporadically, the ratings spiked up to 8 and spiked down to 5.7')

    if "Top Directors" in selected_insights:
        # Create a slim dataframe for Directors
        director_df = filtered_df[['tconst', 'director', 'averageRating']].drop_duplicates(subset=['tconst'])

        # Filter directors with at least 10 unique titles
        director_counts = director_df['director'].value_counts()
        top_directors = director_counts[director_counts >= 5].index
        director_df = director_df[director_df['director'].isin(top_directors)]
        director_df = director_df[director_df['director']!='Unknown']

        # Compute average rating per director
        director_avg = (
            director_df.groupby('director')['averageRating']
            .mean()
            .reset_index()
            .sort_values('averageRating', ascending=False)
            .head(10)
        )

            # Lollipop chart
        fig_lollipop = go.Figure()

        # Add stems (lines)
        fig_lollipop.add_trace(go.Scatter(
            x=director_avg['averageRating'],
            y=director_avg['director'],
            mode='lines',
            line=dict(color='lightgray', width=2),
            showlegend=False
        ))

        # Add markers (dots)
        fig_lollipop.add_trace(go.Scatter(
            x=director_avg['averageRating'],
            y=director_avg['director'],
            mode='markers+text',
            marker=dict(size=12, color='steelblue'),
            text=director_avg['averageRating'].round(2),
            textposition='middle right',
            name='Average Rating'
        ))

        # Layout updates
        fig_lollipop.update_layout(
            title="Top 10 Directors of atleast 5 movies by Average IMDb Rating",
            xaxis_title="Average IMDb Rating",
            yaxis_title="Director",
            template='plotly_white',
            height=500,
            yaxis=dict(categoryorder='total ascending'),  # highest rating on top
            #legend=dict(title='Legend', y=1, x=1)
        )

        st.plotly_chart(fig_lollipop, use_container_width=True)
        st.markdown('**Quentin Tarantino** has the highest average rating, while all other directors maintained the average above 7.0')

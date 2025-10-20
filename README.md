# Netflix Data Analysis

## Dataset
The dataset used in this project is the **Netflix Titles Dataset** and **IMDb dataset**
It contains information about movies and TV shows available on Netflix, including title, director, country, release year, rating, genre, IMDb average rating, adult,etc.

---

## Why I Chose This Dataset
I chose this dataset because:
- Netflix content is globally popular, making it interesting to analyze trends across **countries, directors, and genres**.
- It has a mix of categorical and numerical features, which is ideal for **exploratory data analysis (EDA)**.
- It allows building an **interactive dashboard** using Streamlit to showcase insights dynamically.

---

## What I Learned from IDA/EDA
During initial data exploration:
- Many records have missing **director** or **country** information.
- The majority of content is **TV Shows**, not movies for director.

---

## Preprocessing Steps
Before building the Streamlit app, the following preprocessing steps were done:
1. **Handling Missing Values**
   - Replaced missing `director` and `country` values with `'Unknown'` and mode.
2. **Data Cleaning**
   - Duration column contains the ambiguity, created seperated columns for both the types
   - Exploded the values of Director, Country and Listed_in as they contain multiple values for each record.
3. **Merging**
   - Merged Netflix, IMDb bascis and Ratings datasets to acquire the desired features

---

## Streamlit App Progress
The Streamlit dashboard includes:
- Sidebar filters for **country**, **genre**, **rating**, **release year**.
- Interactive plotly charts for **univariate, bivariate visualizations along with key insights**:

To view the app: https://netflix-app.streamlit.app/
```bash
streamlit run streamlit_app.py

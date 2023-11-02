# Movie Recommendation System

*Movies have become a part of our daily lives, including various aspects such as movie
nights at home, thrilling experiences at cinemas, and streaming during commutes. Choos-
ing the best suitable movie is overwhelming in the vast sea of movies available on numerous
platforms. In this context, our project emerges as a solution to this challenge by intro-
ducing a website equipped with a recommendation system. This system is designed to
suggest five movies that closely align with the user’s input movie, thereby simplifying the
process of movie selection.*


All preprocessing, EDA, and ML steps, initially conducted on Google Colab, can be found in this notebook:
> * [ML Notebook](https://github.com/lifeofborna/Movie-Recommendation-System/blob/main/IDS_Project.ipynb)
> * [Try The App Via Streamlit](https://ids-movie.streamlit.app/)

## 1. Data


## Datasets

### Netflix Dataset
A dataset detailing movies on Netflix with attributes like Title, Cast, Plot Summary, Movie Length, Ratings, Release Year, Genre, and Certification.

### IMDB Dataset
A comprehensive set covering movies and TV shows with data points including Titles, Release Dates, Certifications, Duration, Genre, IMDB Ratings, Summaries, Directorial and Cast Information, Vote Counts, and Box Office Earnings.

### TMDB API 
Utilized TMDB API to gather the recommended movie posters for UI. 

> * [TMDB](https://www.themoviedb.org/)

> * [IMDB Dataset](harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

> * [Netflix dataset](narayan63/netflix-popular-movies-dataset)


## 2. Preprocessing and EDA

### 2.1 Netflix EDA and Preprocessing
To ensure data integrity for our analysis, we first eliminated duplicate entries from the Netflix dataset. Unique records are crucial to avoid skewed results. Subsequently, we embarked on EDA, crafting visualizations to gain insights into the dataset.

The preprocessing phase involved multiple transformations. The 'year' was extracted from the title into a new 'Released_Year' column using regex, then converted to float. The 'duration' was refined into 'duration_in_min' by isolating and converting numerical values to float. Vote counts, previously strung with commas, were transformed into numerical 'votes'.

To address missing data, we first attempted to cross-reference with the IMDB dataset for common movies. However, this proved insufficient due to the volume of missing data, necessitating alternative methods. The 'certificate' column had the highest missing data at 40.44%, followed by 'duration_in_min' and 'votes'. The 'genre' column's minimal missing data was deemed removable without significant impact.

### 2.2 IMDB EDA and Preprocessing
Duplicates were removed to ensure data quality, followed by EDA with various visualizations to understand the data better.

For preprocessing the IMDB dataset, missing values in 'Meta_score' (15.7%) were filled with the column's median value, while 'Certificate' gaps (10.1%) received a 'Not Specified' label. The 'Runtime' column was reformatted from a string to a float to facilitate numerical analysis. These steps completed the dataset's preparation for subsequent machine-learning applications.


....


## 5. Algorithms & Machine Learning

The initial stage in our machine-learning process focused on feature extraction, converting each movie into a binary vector corresponding to the presence or absence of specific genres. This representation allowed us to compute distances between movies, facilitating genre-based clustering:

- **K-Means Clustering**: Implemented to group movies into clusters, with the optimal number of clusters determined using the elbow method by plotting the within-cluster sum of squares (WCSS). The elbow was observed at 5 or 6 clusters, guiding us to define 5 clusters for meaningful segregation without overcomplicating the model. Clustering by genres helps to organize films into discernible groups, simplifying the recommendation process by limiting comparisons to within-cluster movies, thus enhancing computational efficiency and relevance.

- **Visualization with t-SNE**: We employed t-SNE, a sophisticated dimensionality reduction technique, to visualize the clusters in a two-dimensional space, making it easier to identify and understand the distribution and grouping of movies.

For personalized recommendations, the selection of top movies works as follows:

- **Personalized Movie Recommendations**: When a user provides a movie title, we first identify the cluster to which the movie belongs. Then, we compute the cosine similarity between the movie's vector and every other movie's vector within the same cluster. This metric, more robust in high-dimensional spaces, determines the similarity based on the orientation rather than the magnitude of the vectors. The five movies with the highest cosine similarity values are selected as the recommendations. This approach ensures that recommended movies are not only similar in genre but also closely aligned with the user's initial choice in the multidimensional feature space.

This methodological framework leverages clustering to enhance recommendation relevance and employs cosine similarity to provide precise, contextually similar movie suggestions.



## 6. Conclusion & Future Improvements
In conclusion, the project successfully developed a robust movie recommendation
system that simplifies the process of movie selection. By combining two data sets from
Netflix and IMDB, we have created a single data set of unique movies. Subsequently, this
data set underwent thorough preprocessing and cleaning to be used in the application of
a machine-learning model to generate recommendations.

Looking ahead, there is significant potential for further development and enhancement-
ment of our movie recommendation system. While our current model primarily utilizes
the genre feature to group similar movies together, future iterations of the system could
utilize additional features.

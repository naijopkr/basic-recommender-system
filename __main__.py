import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('data/u.data', sep='\t', names=column_names)
df.head()

movie_titles = pd.read_csv('data/movie_id_titles.csv')
movie_titles.head()

df = pd.merge(df, movie_titles, on='item_id')
df.head()

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

def series_hist(series: pd.Series):
    plt.figure(figsize=(10, 4))
    series.hist(bins=70)

sns.set_style('white')

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

ratings['RatingCount'] = pd.DataFrame(df.groupby('title')['rating'].count())

series_hist(ratings['RatingCount'])
series_hist(ratings['rating'])

sns.jointplot(x='rating', y='RatingCount', data=ratings, alpha=0.5)

# Recommending similar movies
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
moviemat.head()

# Most rated movies
ratings.sort_values('RatingCount', ascending=False).head(10)

# Choosing two movies
starwars_user_ratings = moviemat['Star Wars (1977)']
englishpatient_user_ratings = moviemat['English Patient, The (1996)']

# Recommend function
def recommend_movie(
    data: pd.DataFrame,
    count_field = 'RatingCount',
    min_rates = 0,
):
    return data[data[count_field] > min_rates].sort_values(
        'Correlation',
        ascending=False,
    )

# Recommendation for Star Wars
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

corr_starwars = corr_starwars.join(ratings['RatingCount'])
corr_starwars.head()
corr_starwars.sort_values('Correlation', ascending=False).head(10)

recommend_movie(corr_starwars, min_rates=100).head()
recommend_movie(corr_starwars, min_rates=200).head()
recommend_movie(corr_starwars, min_rates=300).head()

# Recommendation for The English Patient
similar_to_englishpatient = moviemat.corrwith(englishpatient_user_ratings)
corr_englishpatient = pd.DataFrame(similar_to_englishpatient, columns=['Correlation'])
corr_englishpatient.head()

corr_englishpatient = corr_englishpatient.join(ratings['RatingCount'])
corr_englishpatient.head()
corr_englishpatient.sort_values('Correlation', ascending=False).head(10)

recommend_movie(corr_englishpatient, min_rates=100).head()
recommend_movie(corr_englishpatient, min_rates=200).head()
recommend_movie(corr_englishpatient, min_rates=300).head()

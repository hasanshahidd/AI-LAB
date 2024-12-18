import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

netflix_data = pd.read_csv(r"C:\Users\Admin\Desktop\AI LAB 9 TO 14\lab tast 9\netflix_titles.csv")
print("First few rows of the dataset:")
print(netflix_data.head())
print("\nSummary of the dataset:")
print(netflix_data.describe())
print("\nColumn names and data types:")
print(netflix_data.info())
print("\nMissing values in the dataset:")
print(netflix_data.isnull().sum())
print("\nNumber of Movies and TV Shows:")
print(netflix_data['type'].value_counts())
print("\nDistribution of Ratings:")
print(netflix_data['rating'].value_counts())
genres_list = netflix_data['listed_in'].str.split(', ')
genres_list = genres_list.explode()
print("\nTop 10 Genres:")
print(genres_list.value_counts().head(10))
sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=netflix_data, palette='viridis')
plt.title('Distribution of Movies and TV Shows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(12, 6))
sns.countplot(x='rating', data=netflix_data, palette='magma', order=netflix_data['rating'].value_counts().index)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(12, 8))
top_genres = genres_list.value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='cubehelix')
plt.title('Top 10 Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()
netflix_data.to_csv('cleaned_netflix_titles.csv', index=False)
print("\nAll tasks completed successfully!")

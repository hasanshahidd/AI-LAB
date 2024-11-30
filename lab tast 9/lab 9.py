import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use raw string to avoid unicode escape error
data = pd.read_csv(r"C:\Users\Admin\Desktop\AI LAB 9 TO 14\lab tast 9\netflix_titles.csv")

print("First few rows of the dataset:")
print(data.head())
print("\nSummary of the dataset:")
print(data.describe())
print("\nColumn names and data types:")
print(data.info())
print("\nMissing values in the dataset:")
print(data.isnull().sum())
print("\nNumber of Movies and TV Shows:")
print(data['type'].value_counts())
print("\nDistribution of Ratings:")
print(data['rating'].value_counts())
print("\nTop 10 Genres:")
genres = data['listed_in'].str.split(', ')
genres = genres.explode()
print(genres.value_counts().head(10))

sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=data, palette='viridis')
plt.title('Distribution of Movies and TV Shows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='rating', data=data, palette='magma', order=data['rating'].value_counts().index)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Plot the top genres
plt.figure(figsize=(12, 8))
top_genres = genres.value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='cubehelix')
plt.title('Top 10 Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Save the cleaned dataset (optional)
data.to_csv('cleaned_netflix_titles.csv', index=False)

print("\nAll tasks completed successfully!")

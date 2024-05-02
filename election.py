# Import important libraries

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

trump_reviews = pd.read_csv('Trump.csv')
biden_reviews = pd.read_csv('Biden.csv')

# Sentiment Analysis

textblob1 = TextBlob(trump_reviews['text'][10])
textblob2 = TextBlob(biden_reviews['text'][500])


def find_pol(review):
    return TextBlob(review).sentiment.polarity

trump_reviews['Sentiment Polarity'] = trump_reviews['text'].apply(find_pol)
biden_reviews['Sentiment Polarity'] = biden_reviews["text"].apply(find_pol)

# Sentiment Polarity on both the candidates

trump_reviews["Expression Label"] = np.where(trump_reviews["Sentiment Polarity"] > 0, 'positive', 'negative')
trump_reviews["Expression Label"][trump_reviews["Sentiment Polarity"] == 0] = "Neutral"

biden_reviews["Expression Label"] = np.where(biden_reviews["Sentiment Polarity"] > 0, 'positive', 'negative')
biden_reviews["Expression Label"][biden_reviews["Sentiment Polarity"] == 0] = "Neutral"


reviews1 = trump_reviews[trump_reviews['Sentiment Polarity'] == 0.0000]
cond1 = trump_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
trump_reviews.drop(trump_reviews[cond1].index, inplace = True)

reviews2 = biden_reviews[biden_reviews['Sentiment Polarity'] == 0.0000]
cond2 = biden_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
biden_reviews.drop(biden_reviews[cond2].index, inplace = True)


# Balance Both the dataset

# Donald Trump
np.random.seed(10)
remove_n = 324
drop_indices = np.random.choice(trump_reviews.index,remove_n, replace = False)
df_subset_trump = trump_reviews.drop(drop_indices)

# Joe Biden
np.random.seed(10)
remove_n = 31
drop_indices = np.random.choice(biden_reviews.index,remove_n, replace = False)
df_subset_biden = biden_reviews.drop(drop_indices)








# Main function for Streamlit app
def main():
    st.title('US Election Prediction')
    st.sidebar.title('Select Candidates')

    # Candidate selection
    candidate1 = st.sidebar.selectbox('Select Candidate 1', ['Joe Biden', 'Donald Trump'])
    candidate2 = st.sidebar.selectbox('Select Candidate 2', ['Joe Biden', 'Donald Trump'])

    if candidate1 == candidate2:
        st.error("Please select different candidates.")
    else:
        # Filter reviews based on selected candidate
        reviews1 = trump_reviews[trump_reviews['Sentiment Polarity'] == 0.0000]
        reviews2 = biden_reviews[biden_reviews['Sentiment Polarity'] == 0.0000]

        # Count positive and negative reviews for each candidate
        count_1 = df_subset_trump.groupby('Expression Label').count()
        count_2 = df_subset_biden.groupby('Expression Label').count()


        # Calculate percentages
        negative_per1 = (count_1['Sentiment Polarity'][0]/1000)*10
        positive_per1 = (count_1['Sentiment Polarity'][1]/1000)*100

        negative_per2 = (count_2['Sentiment Polarity'][0] / 1000) * 100
        positive_per2 = (count_2['Sentiment Polarity'][1] / 1000) * 100

        # Display percentages in DataFrame
        df_percentages = pd.DataFrame({
            'Candidate': [candidate1, candidate2],
            'Positive': [round(positive_per1, 2), round(positive_per2, 2)],
            'Negative': [round(negative_per1, 2), round(negative_per2, 2)]
        })
        st.subheader('Percentage of Positive and Negative Reviews for Each Candidate')
        st.dataframe(df_percentages)

        # Plot grouped bar chart
        Politicians = ['Joe Biden', 'Donald Trump']
        lis_pos = [positive_per1, positive_per2]
        lis_neg = [negative_per1, negative_per2]

        fig = go.Figure(data=[
            go.Bar(name='Positive', x=Politicians, y=lis_pos),
            go.Bar(name='Negative', x=Politicians, y=lis_neg)
        ])
        fig.update_layout(title='Percentage of Positive and Negative Reviews for Each Candidate')
        st.plotly_chart(fig)

# Run the Streamlit app
if __name__ == '__main__':
    main()

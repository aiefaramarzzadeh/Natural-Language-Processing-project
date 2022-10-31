class task1:   
    def __init__(self, tweets):
        
        '''
        Parameters
        ----------
        tweets : tweets csv file is input
            only text of tweets in english are input.
        '''
        
        self.tweets = tweets
        
    def sentiment_score(self, tweets):
        from numpy import sign
        import nltk
        nltk.download('vader_lexicon')
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        sia = SentimentIntensityAnalyzer()
        
        scores = [sign(sia.polarity_scores(tweet)['compound']) for tweet in tweets]        
                
        return scores
        
    def sentiment_hist(self, scores):
        import numpy as np
        import matplotlib.pyplot as plt
        
        labels, counts = np.unique(scores, return_counts=True)
        plt.bar(['Negative', 'Neutral', 'Positive'], (counts/1000), align='center')
        plt.grid()
        plt.ylabel('Number of tweets (thousand)')
        plt.savefig('histogram.png', dpi = 300)

class task2:    
    def pre_processor(self, text):
        from spacy.lang.en import English        
        import nltk
        #nltk.download('stopwords')
        #nltk.download('words')

        en_stop = set(nltk.corpus.stopwords.words('english'))
        words = set(nltk.corpus.words.words())

        parser = English()
        
        tokens = list(parser(text))
        
        trash = ['#','@',';',',','.','|','%','$','!','*','&',')','(','_']
        trash.extend(list(en_stop))
        
        clear_tokenized_tweet = [str(elem).lower() for elem in  tokens if str(elem).lower() in words and str(elem).lower() not in trash]
                
        '''
            if token.orth_.isspace():
                continue
            elif token.like_url:
                continue
            elif str(token) in :
                continue
            elif str(token) in en_stop:
                continue
            elif str(token).isnumeric():
                continue
            elif str(token) not in words:
                continue
            else:
                clear_tokenized_tweet.append(token.lower_)
        '''
        
        return clear_tokenized_tweet
        
    def lda_model(self, tokenized_tweets, name, topics_num = 10, words_num = 10):
        '''
        Parameters
        ----------
        tokenized_tweets : List each tweet tokens list
        topics_num : number of topics (10) in LDA
        words_num : number of words (10) in LDA

        Returns
        -------
        topics : based on LDA, returns topics

        '''
        import pandas as pd        
        import gensim
        from gensim import corpora
        import ast 
        
        tokenized_tweets = [ast.literal_eval(elem) for elem in tokenized_tweets] 
    
        dictionary = corpora.Dictionary(tokenized_tweets)
        corpus = [dictionary.doc2bow(text) for text in tokenized_tweets]
        
        import pickle
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')
        
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = topics_num, id2word = dictionary, passes=15)
        ldamodel.save('model10.gensim')
        
        topics = ldamodel.print_topics(num_words = words_num)
        
        df = pd.DataFrame(topics)
        df.to_csv('task2_topics_'+name+'.csv')
        
        return topics    
    
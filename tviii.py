class task8: 
    def modelgram(self, input_clear, name, keywords):                
        import nltk
        import matplotlib.pyplot as plt
        import pandas as pd
        from empath import Empath
        import ast
        import spacy       
        NER = spacy.load("en_core_web_sm")
        
        words_clean1 = [" ".join(ast.literal_eval(elem)) for elem in input_clear[0:10000]]
        
        words_clean2 = []
        for keyword in keywords:
            temp = [tweet_tokens for tweet_tokens in words_clean1 if keyword in tweet_tokens]
            words_clean2.extend(temp)
               
        words = ' '.join([str(elem) for elem in words_clean1])
        
        named_entities = NER(words)
        ner_tweets = named_entities.ents
        named_ent = [elem.text for elem in ner_tweets]
        
        lexicon = Empath()

        categories = pd.Series(lexicon.analyze(words, normalize=True))
        
        categories = categories[categories.values > 0]
        categories = categories.sort_values(ascending=False)
        
        categories.to_csv('task8_'+name+'_empath.csv')
        
        words = pd.Series(nltk.FreqDist(named_ent))
        words = words.sort_values(ascending=False)
        
        plt.figure(figsize=(8,8), dpi=300)
        plt.bar(words.index[0:30], words.values[0:30])
        plt.xticks(rotation=90)
        plt.grid()
        plt.ylabel('Frequency')
        plt.savefig('task8_'+name+'.png', dpi = 300)

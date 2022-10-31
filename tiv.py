class task4: 
    def hist_word(self, input_clear, name):                
        import nltk
        import matplotlib.pyplot as plt
        import pandas as pd
        import ast
        import spacy

        NER = spacy.load("en_core_web_sm")
        
        words_clean1 = [" ".join(ast.literal_eval(elem)) for elem in input_clear]
        words_clean = ' '.join([str(elem) for elem in words_clean1[0:10000]])
        
        named_entities = NER(words_clean)
        ner_tweets = named_entities.ents
        words = [elem.text for elem in ner_tweets]
        labels = [elem.label_ for elem in ner_tweets]        
        spacy.explain("GPE")
        
        words = pd.Series(nltk.FreqDist(words))
        words = words.sort_values(ascending=False)
        
        plt.figure(figsize=(8,8), dpi=300)
        plt.bar(words.index[0:20], words.values[0:20])
        plt.xticks(rotation=90)
        plt.grid()
        plt.ylabel('Frequency')
        plt.savefig('task4_word_freg_'+name+'.png', dpi = 300)

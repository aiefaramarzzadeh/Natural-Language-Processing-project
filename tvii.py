class task7: 
    
    def semantic(self, word):    
        from nltk.corpus import wordnet
        
        synonyms = []

        for syn in wordnet.synsets(word):
            for syn in syn.lemmas():
                synonyms.append(syn.name())
        
        synonyms = set(synonyms)
        return synonyms
    
    def freq(self, input_clear, name, keyword = 'war'):                
        import nltk
        from nltk.util import ngrams
        import matplotlib.pyplot as plt
        import pandas as pd
        import ast

        words_clean1 = [" ".join(ast.literal_eval(elem)) for elem in input_clear]
        words_clean = ' '.join([str(elem) for elem in words_clean1])

        words_cnetered = [x for x in list(ngrams(words_clean.split(),7)) if x[3] == keyword]
        
        words1 = [" ".join(elem) for elem in words_cnetered]
        words = ' '.join([str(elem) for elem in words1])
        
        words = pd.Series(nltk.FreqDist(words.split()))
        words = words.sort_values(ascending=False)
        
        plt.figure(figsize=(8,8), dpi=300)
        plt.bar(words.index[1:31], words.values[1:31])
        plt.xticks(rotation=90)
        plt.grid()
        plt.ylabel('Frequency')
        plt.savefig('task7_'+keyword+'_'+name+'.png', dpi = 300)
    
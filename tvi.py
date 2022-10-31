class task6: 
    def empath(self, input_clear, name):                
        from empath import Empath
        import pandas as pd
        import ast
        lexicon = Empath()
        
        
        words_clean1 = [" ".join(ast.literal_eval(elem)) for elem in input_clear]
        words_clean = ' '.join([str(elem) for elem in words_clean1])
        
        categories = pd.Series(lexicon.analyze(words_clean, normalize=True))
        
        categories = categories[categories.values > 0]
        categories = categories.sort_values(ascending=False)
        
        categories.to_csv('task6_'+name+'_empath.csv')
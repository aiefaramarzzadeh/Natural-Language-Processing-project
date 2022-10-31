class task3: 
    def plotter(self, input_dirty, input_clear, name):
    
        '''
        inputs_dirty: text of tweets (csv file of tweets each line is a tweet text)
        input_clear: is tokenized clear tweets so to make it ready for word cloud we need
                    two levels of flattening
        name: here helps to save figures better
        '''    
        
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import ast

    
        words_dirty = ' '.join([str(elem) for elem in input_dirty])
        words_clean1 = [" ".join(ast.literal_eval(elem)) for elem in input_clear]
        words_clean = ' '.join([str(elem) for elem in words_clean1])
    
        wordcloud_clear = WordCloud(width = 800, height = 800,
                                    background_color ='white',
                                    min_font_size = 10).generate(words_clean)
    
        wordcloud_dirty = WordCloud(width = 800, height = 800,
                                    background_color ='white',
                                    min_font_size = 10).generate(words_dirty)
    
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud_clear)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.savefig('wordcloud_clear_'+name+'.png', dpi = 300)
    
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud_dirty)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.savefig('wordcloud_dirty_'+name+'.png', dpi = 300)
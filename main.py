#%% Task 0
import os
os.chdir('put directory you saved py and csv files')
os.getcwd()

# MOVE ALL py files exept main.py TO A NEW FOLDER BUT IN THE SAME DIRECTORY AND
# RENAME IT TO "objects". YOU NEED THIS FOLDER TO ACCESS EACH TASK CODE SEPERATELY IN
# THE BELOW LINES. IF YOU HAVE ISSUE, CONTACT ME: fara.aije@gmail.com :) 

#%% Task 1
from objects import ti
import pandas as pd

tweets_input = pd.read_csv('march.csv')

tweets_en = tweets_input[tweets_input['language'] == 'en']

task1 = ti.task1(tweets_en['text'])
scores = task1.sentiment_score(task1.tweets)
task1.sentiment_hist(scores)

#%% Task 2
from objects import tii
import pandas as pd

output = tweets_en[['text','location']][0:50000]
output['score'] = scores[0:50000]

tweets_neg = output[output['score'] == -1]['text']
tweets_pos = output[output['score'] == 1]['text']
tweets_neu = output[output['score'] == 0]['text']

task2 = tii.task2()

'''
clear_tok_neut = list(map(task2.pre_processor, tweets_neu))
clear_tok_pos = list(map(task2.pre_processor, tweets_pos))
clear_tok_neg = list(map(task2.pre_processor, tweets_neg))


clear_tok_neut = pd.Series(clear_tok_neut)
clear_tok_pos = pd.Series(clear_tok_pos)
clear_tok_neg = pd.Series(clear_tok_neg)

clear_tok_neut.to_csv('clear_tok_neut.csv')
clear_tok_pos.to_csv('clear_tok_pos.csv')
clear_tok_neg.to_csv('clear_tok_neg.csv')
'''
clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_tweets = clear_tok_neut
name = 'neutral'

topics = task2.lda_model(input_tweets, name)
#%% Task 3
from objects import tiii
import pandas as pd


clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_dirty = tweets_neu
input_clear = clear_tok_neut
name = 'netral'

task3 = tiii.task3()
task3.plotter(input_dirty, input_clear, name)

#%% Task 4
from objects import tiv
import pandas as pd

clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_clear = clear_tok_neg
name = 'negative'

task4 = tiv.task4()
hist_word = task4.hist_word(input_clear, name)
#%% Task 5
from objects import tv
import pandas as pd

clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_clear = clear_tok_pos
name = 'positive'

task5 = tv.task5()
wordgram = task5.wordgram(input_clear, name, keyword = 'russia')

#%% Task 6
from objects import tvi
import pandas as pd


clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_clear = clear_tok_neut
name = 'neutral'

task6 = tvi.task6()
empath = task6.empath(input_clear, name)

#%% Task 7
from objects import tvii
import pandas as pd

task7 = tvii.task7()
syn_like = task7.semantic('like')
syn_hate = task7.semantic('hate')

clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_clear = clear_tok_neg
keyword = 'hate'
name = 'negative'

wordgram = task7.freq(input_clear, name, keyword = keyword)
#%% Task 8
from objects import tviii
import pandas as pd

clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_clear = clear_tok_neg
name = 'negative'

keywords = ['shall', 'must', 'need']
task8 = tviii.task8()

modelgram8 = task8.modelgram(input_clear, name, keywords = keywords)
#%% Task 9
from objects import tx
import pandas as pd

clear_tok_neut = pd.read_csv('clear_tok_neut.csv', index_col = 0, squeeze = True)
clear_tok_pos = pd.read_csv('clear_tok_pos.csv', index_col = 0, squeeze = True)
clear_tok_neg = pd.read_csv('clear_tok_neg.csv', index_col = 0, squeeze = True)

input_clear = clear_tok_pos
name = 'positive'

task9 = tx.task9()

modelgram9 = task9.modelgram(input_clear, name, keywords = 'wish')

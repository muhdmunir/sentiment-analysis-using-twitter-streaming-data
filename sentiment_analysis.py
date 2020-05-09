from textblob import TextBlob
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
import re
import string


##OTHER FUNCTIONS/CLASSES

def resolve_emoticon(line):
    emoticon = {
    	':-)' : 'smile',
        ':)'  : 'sad',
    	':))' : 'very happy',
    	':)'  : 'happy',
    	':((' : 'very sad',
    	':('  : 'sad',
    	':-P' : 'tongue',
    	':-o' : 'gasp',
    	'>:-)':'angry'
   }
    for key in emoticon:
        line = line.replace(key, emoticon[key])
    return line

def abb_bm(line):
    abbreviation_bm = {
         'sy': 'saya',
         'sk': 'suka',
         'byk': 'banyak',
         'sgt' : 'sangat',
         'mcm' : 'macam',
         'bodo':'bodoh'
   }
    abbrev = ' '.join (abbreviation_bm.get(word, word) for word in line.split())
    return (resolve_emoticon(abbrev))



def abb_en(line):
    abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',
    'c' : 'see'
   }
    abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
    return (resolve_emoticon(abbrev))

def make_plot(pos,neg):

   #This function plots the counts of positive and negative words
    Polarity = [1,2]
    LABELS = ["Positive", "Negative"]
    Count_polarity = [int(pos), int(neg)]
    plt.xlabel('Polarity')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis - Lexical Based')
    plt.grid(True)
    plt.bar(Polarity, Count_polarity, align='center')
    plt.xticks(Polarity, LABELS)
    plt.show()



def remove_features(data_str):
    url_re= re.compile(r'https?://(\S+)')
    num_re= re.compile(r'(\d+)')
    mention_re= re.compile(r'(@|#)(\w+)')
    RT_re= re.compile(r'RT(\s+)')
    data_str= str(data_str)
    data_str= RT_re.sub(' ', data_str) # remove RT
    data_str= url_re.sub(' ', data_str) # remove hyperlinks
    data_str= mention_re.sub(' ', data_str) # remove @mentions and hash
    data_str= num_re.sub(' ', data_str) # remove numerical digit

    return data_str



def main(sc,filename):


    data = sc.textFile(filename).map(lambda x:remove_features(x)).map(lambda x:x.lower()).map(lambda x:resolve_emoticon(x))

    data_ms = data.filter(lambda x:TextBlob(x).detect_language()=='ms').map(lambda x:abb_bm(x)).map(lambda x:str(TextBlob(x).translate(from_lang='ms', to ='en')))

    data_en = data.filter(lambda x:TextBlob(x).detect_language()=='en').map(lambda x:abb_en(x))

    data_union = data_ms.union(data_en)

    data_senti = data_union.map(lambda x:TextBlob(x).sentiment.polarity)

    pos = data_senti.filter(lambda x:x>0).count()
    neg = data_senti.filter(lambda x:x<0).count()

    make_plot(int(pos),int(neg)) #the cast is just to ensure the value is in integer data type


if __name__ == "__main__":
    conf = SparkConf().setMaster("local[2]").setAppName("Sentiment Analysis")
    sc = SparkContext(conf=conf)
    filename = "covid19.txt"
    main(sc, filename)
    sc.stop()

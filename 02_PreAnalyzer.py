# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:31:10 2023

@author: Fade
"""
import pandas as pd
import spacy
from spacy_sentiws import spaCySentiWS
from nltk.corpus import stopwords
from nltk import word_tokenize

class PreAnalyzer():
    '''
    The PreAnalyzer-Class combines several methods to use SentiWS for first-hand text classification
    preprocessing: removes punctuation and symbols
    remove_stopwords: removes german stopwords
    score: use spaCy to score each paragraph
    analyze_sentiment: combines preprocessing and score
    export_results: exporting to excel
    '''
    
    def __init__(self):
        self.nlp = spacy.load("de_core_news_md")
        self.nlp.add_pipe('sentiws', config={'sentiws_path': 
                                             r'PreAnalysis/data/sentiws'})
        
    def preprocessing(self, c):
        punc = '''!()[]{};=:'"\,<>./?@#$%^&*_~|°•0123456789'''
        for i in range(len(c)):
            for ele in c[i]:
                if ele in punc:
                    c[i] = c[i].replace(ele, "")
        return c
    
    def remove_stop_words(self, document):
        stop_words = stopwords.words('german') # load german stop words
        tokenized = word_tokenize(document)
        document_filtered = ""
        for word in tokenized:
            if word not in stop_words and word.isalpha(): # filter stopwords and symbols
                  document_filtered += word + " "
        return document_filtered

    def score(self, sentences):
        sentiment_scores_spacy = []
        for sen in sentences:
            doc = self.nlp(sen)
            x = []
            for token in doc:
                x.append(token._.sentiws)
            k = sum(filter(None, x))
            sentiment_scores_spacy.append(k)
        return sentiment_scores_spacy
        
    def analyze_sentiment(self, sentences):
        sen = self.preprocessing(sentences)
        sen = [self.remove_stop_words(i) for i in sen]
        sent_scores = self.score(sen)
        return sent_scores
    
    def get_class(self, x):
        if x >= 0.1:
            y = 1
        if x <= -0.1:
            y = -1
        if x < 0.1 and x > -0.1:
            y = 0
        return y
    
    def export_results(self, file, path):
        df = pd.read_excel(file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        sentences = df["Sentence"]
        sent_scores = self.analyze_sentiment(sentences)
        data = pd.read_excel(file)
        
        # attach scores to dataframe
        data["Scores"] = sent_scores
        
        #attach class based upon score
        data["Class"] = data["Scores"].apply(lambda x: self.get_class(x))
        
        data.to_excel(path, index=False)
        
# execute class
if __name__ == "__main__":
    pre_analyzer = PreAnalyzer()
    
    #annotate berufe
    pre_analyzer.export_results(file = "PreAnalysis/data/excel_merged/all_beruf_correction.xlsx",
                                path = "PreAnalysis/data/excel_merged/beruf_annotated_sentences.xlsx")
    #annotate lehrer
    pre_analyzer.export_results(file = "PreAnalysis/data/excel_merged/all_lehrer_correction.xlsx",
                                path = "PreAnalysis/data/excel_merged/lehrer_annotated_sentences.xlsx")
    

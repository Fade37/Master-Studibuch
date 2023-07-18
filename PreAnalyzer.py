import pandas as pd
import spacy
from spacy_sentiws import spaCySentiWS

class PreAnalyzer():
    
    def __init__(self):
        self.nlp = spacy.load("de_core_news_md")
        self.nlp.add_pipe('sentiws', config={'sentiws_path': 
                                             'D:/OneDrive/Dokumente/Praktikumsbericht/Praktikumsbericht RWI/python/data/sentiws'})
        
    def preprocessing(self, c):
        punc = '''!()[]{};=:'"\,<>./?@#$%^&*_~|°•0123456789'''
        for i in range(len(c)):
            for ele in c[i]:
                if ele in punc:
                    c[i] = c[i].replace(ele, "")
        return c
    
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
        sent_scores = self.score(sen)
        return sent_scores
        
    def export_results(self, file, path):
        df = pd.read_excel(file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        sentences = df["Satz"]
        sent_scores = self.analyze_sentiment(sentences)
        data = pd.read_excel(file)
        data["Werte"] = sent_scores
        data = data.loc[:, ~df.columns.str.contains('^Unnamed')]
        data.to_excel(f"{path}/annotated_sentences.xlsx")
        
if __name__ == "__main__":
    pre_analyzer = PreAnalyzer()
    pre_analyzer.export_results(file = "D:/OneDrive/Dokumente/MasterThesis/PreAnalysis/data/excel_merged/1977-1986_correction.xlsx",
                                path = "D:/OneDrive/Dokumente/MasterThesis/PreAnalysis/data/excel_merged/")
    

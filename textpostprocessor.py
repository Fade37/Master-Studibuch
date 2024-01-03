import pandas as pd
import os
import language_tool_python
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize

class TextPostprocessor():
    '''
    The Postprocessor takes each excel and builds a joint dataframe.
    Further it provides methods to clean the text and extract keywords by tf_idf
    '''

    def get_all_filenames(self, dir=r"PreAnalysis\data\excel"):
        all_filenames = os.listdir(dir)
        return all_filenames

    def get_joint_excel(self, filepath):
        filenames = self.get_all_filenames()
        result = pd.DataFrame([])

        for filename in filenames:
            df = pd.read_excel(fr'PreAnalysis\data\excel\{filename}')
            df['Year'] = filename[-9:-5]
            #df = df.rename(columns = {"sentences": "Satz", "correction": "Klasse"})
            result = result.append(df, ignore_index=True)

        result = result.loc[:, ~result.columns.str.contains('^Unnamed')]
        result = result.reset_index(drop=True)
        print(result)
        result.to_excel(filepath, index=False)

    def correction(self, sentence, tool):
        removed = re.sub("^(\d.*?)+(?=\w)", '', sentence) # when a sentence begins with digit, match all character until the next word
        removed = re.sub('( ){2,4}', ' ', removed)
        correction = tool.correct(removed)
        return correction
    
    #replace compound words to enhance the classification
    def replace_compound(self, sentence):
        keys = ["Lehrerangebot", "Lehrernachfrage", 'Lehrerüberangebot', 'Lehrerüberschusses', 
                    'Lehrerüberschuss', 'Lehrermangel', 'Lehrerbedarf', 'Lehrerbedarfsprognosen', 
                    'Lehrerdefizite', 'Lehrerüberschüsse', 'Lehrerangebots', 'Lehrerüberschüssen']
        values = ["Angebot","Nachfrage", "Überangebot", "Überschusses", "Überschuss", "Mangel", "Bedarf",
                "Bedarfsprognosen", "Defizite", "Überschüsse", "Angebots", "Überschüssen"]
        replace_dict = dict(zip(keys, values))
        for key, value in replace_dict.items():
            sentence = re.sub(f"({key})", f"{value}", sentence)
        return sentence

    def correct_mistakes(self, filename, filepath):
        df = pd.read_excel(fr'PreAnalysis\data/excel_merged/{filename}')
        tool = language_tool_python.LanguageTool('de-DE')
        df['Sentence'] = df['Sentence'].apply(self.correction, tool = tool).apply(self.replace_compound)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print(df)
        df.to_excel(filepath, index=False)

    def tf_idf(self, df, name):
        text = " ".join(df["Sentence"]).lower()

        # remove stopwords
        stop_words = stopwords.words('german') # lädt Liste mit deutschen Stoppwörtern
        tokenized = word_tokenize(text)
        document_filtered = ""
        for word in tokenized:
            if word not in stop_words and word.isalpha(): # filtert Stoppwörter und Zeichen, die nicht im Alphabet sind raus
                document_filtered += word + " "

        # get counts
        counts = word_tokenize(document_filtered)
        counts = pd.value_counts(np.array(counts))
        counts = counts.to_frame().reset_index().rename(columns={"index": "keyword"}) 

        #tf-idf
        sen = df["Sentence"]
        filtered = []
        for i in sen:
            token = word_tokenize(i)
            clean = ""
            for word in token:
                if word not in stop_words and word.isalpha(): # filtert Stoppwörter und Zeichen, die nicht im Alphabet sind raus
                    clean += word + " "
            filtered.append(clean)

        

        vectorizer = TfidfVectorizer(lowercase=True)
        X = vectorizer.fit_transform(filtered)
        tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        tfidf_df = tfidf_df.stack().reset_index()
        tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'keyword', 'level_2': 'keyword'})
        tfidf_df = tfidf_df.sort_values(by='tfidf', ascending=False).drop_duplicates("keyword", keep = "first")
        data = pd.merge(tfidf_df, counts, how="left", on="keyword")
        data.to_excel(f"tf_idf_{name}.xlsx", index=False)
        
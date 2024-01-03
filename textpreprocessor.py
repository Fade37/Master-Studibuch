import spacy
import pandas as pd
import re

class TextPreprocessor():
    '''
    This Class extracts all raw .txt-files to build dataframes on paragraph level (row for each sentence) 
    and uses RegEx to eliminate first unused artifacts.
    Output are excel-files for each chapter
    '''
    
    def __init__(self):
        self.nlp = spacy.load("de_core_news_md")
        
    #thinks function is used to "mask" abbreviation points to save them from being eliminated by RegEx
    def change_abbreviation_points(self, text):
        abbr_list = ["rd", "einschl"]
        for i in abbr_list:
            text = re.sub(f"(?<={i})(\.)", '@', text)
        return text
    
    def text_to_excel(self, filename, excel_name):
        print(f"*** Extracting {filename} ***")
        # reading the text file:
        with open(fr'PreAnalysis\data\textfiles\{filename[:-4]}.txt', encoding="utf-8") as text:
            extracted_text = text.read() 
            
        filtered_text = self.change_abbreviation_points(extracted_text)
        filtered_text = re.sub(r"(\t+)",'', filtered_text) # remove tabs that interfere with tokenization
        filtered_text = re.sub(r"(•\s*)", '', filtered_text) # removes •
        filtered_text = re.sub(r"(-\n+)", '', filtered_text) # removes hyphen, linebreak
        filtered_text = re.sub(r"(\n)", ' ', filtered_text) # removes linebreak


        # iterate through file
        filtered = {'paragraphs': []}
        doc = self.nlp(filtered_text)
        for sent in doc.sents:
            sent = str(sent)

            # change @ back to .
            sent = re.sub("@", ".", sent)
            filtered['paragraphs'].append(
                {"Sentence": sent})

        # write to file
        df = pd.DataFrame(filtered['paragraphs'])
        df.to_excel(f"PreAnalysis\data\excel\{excel_name}.xlsx", engine='xlsxwriter')

import spacy
import pandas as pd
import re

class TextPreprocessor():
    
    def __init__(self):
        self.nlp = spacy.load("de_core_news_md")
        
    
    def change_abbreviation_points(self, text):
        abbr_list = ["rd", "einschl"]
        for i in abbr_list:
            text = re.sub(f"(?<={i})(\.)", '@', text)
        return text
    
    def text_to_excel(self, filename, excel_name):
        print(f"*** Extracting {filename} ***")
        # reading the text file:
        with open(fr'D:\OneDrive\Dokumente\Praktikumsbericht\Praktikumsbericht RWI\python\data\roh\{filename[:-4]}.txt', encoding="utf-8") as text:
            extracted_text = text.read() 
            
        extracted_text = self.change_abbreviation_points(extracted_text)
    
        # iterate through file
        filtered = {'paragraphs': []}
        doc = self.nlp(extracted_text)
        for sent in doc.sents:
            sent = str(sent)
            sent = re.sub(r"(-\n )", '', sent) # removes hyphen, indent and whitespace
            sent = re.sub(r"(•)", ' ', sent) # removes hyphen, indent
            #change @ back to .
            sent = re.sub(r"@", ".", sent)
            filtered['paragraphs'].append(
                {"Satz": sent})

        # write to file
        df = pd.DataFrame(filtered['paragraphs'])
        df.to_excel(f"D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data\excel\{excel_name}.xlsx", engine='xlsxwriter')

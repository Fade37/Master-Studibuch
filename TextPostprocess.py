import pandas as pd
import os
import language_tool_python
import re

class TextPostprocessor():

    def get_all_filenames(self, dir=r"D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data\excel"):
        all_filenames = os.listdir(dir)
        return all_filenames

    def get_joint_excel(self):
        filenames = self.get_all_filenames()
        result = pd.DataFrame([])

        for filename in filenames:
            df = pd.read_excel(fr'D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data\excel\{filename}')
            df['Jahr'] = filename[-9:-5]
            #df = df.rename(columns = {"sentences": "Satz", "correction": "Klasse"})
            result = result.append(df, ignore_index=True)

        result = result.loc[:, ~result.columns.str.contains('^Unnamed')]
        result = result.reset_index(drop=True)
        print(result)
        result.to_excel(f"D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data/excel_merged/all_Lehrerbedarf.xlsx", index=False)

    def correction(self, sentence, tool):
        removed = re.sub("^(\d.*?)+(?=\w)", '', sentence) # when a sentence begins with digit, match all character until the next word
        removed = re.sub('( ){2,4}', ' ', removed)
        correction = tool.correct(removed)
        return correction
    
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

    def correct_mistakes(self, filename="all_Lehrerbedarf.xlsx"):
        df = pd.read_excel(f'D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data/excel_merged/{filename}')
        tool = language_tool_python.LanguageTool('de-DE')
        df['Satz'] = df['Satz'].apply(self.correction, tool = tool).apply(self.replace_compound)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print(df)
        df.to_excel(f"D:\OneDrive\Dokumente\MasterThesis\PreAnalysis/data/excel_merged/all_Lehrerbedarf_correction.xlsx", index=False)

if __name__ == "__main__":
    postprocessor = TextPostprocessor()
    postprocessor.get_joint_excel()
    postprocessor.correct_mistakes()
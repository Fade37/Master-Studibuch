import pandas as pd
import os
import language_tool_python
import re


def get_all_filenames(dir=r"D:\OneDrive\Dokumente\Praktikumsbericht\Praktikumsbericht RWI\python\data\scores"):
    all_filenames = os.listdir(dir)
    return all_filenames

def get_joint_excel():
    filenames = get_all_filenames()
    result = pd.DataFrame([])

    for filename in filenames:
        df = pd.read_excel(fr'D:\OneDrive\Dokumente\Praktikumsbericht\Praktikumsbericht RWI\python\data\scores/{filename}')
        df['Jahr'] = filename[-9:-5]
        df = df.rename(columns = {"sentences": "Satz", "correction": "Klasse"})
        result = result.append(df, ignore_index=True)

    result = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    result = result.reset_index(drop=True)
    # drop duplicates:
    result = result.drop_duplicates(subset=['Satz'], keep='first')
    result = result.drop(columns=["score", "classification", "error"])
    print(result)
    result.to_excel("D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data/excel_merged/1976-1986_annotated_lehrer.xlsx", index=False)

def correction(sentence, tool):
    removed = re.sub(r"(\n)", ' ', sentence) # removes only indents
    removed = re.sub("^(\d.*?)+(?=\w)", '', removed) # when a sentence begins with digit, match all character until the next word
    #removed = re.sub('(\d+(\.\d+)?)', r' \1 ', removed) # find position where digit meets parentheses directly
    #removed = re.sub(r"[^\S]?(\(.*?\))[^\S]?", r" \1 ", removed)
    removed = re.sub(' +', ' ', removed)
    correction = tool.correct(removed)
    return correction

def correct_mistakes(filename="1976-1986_lehrer.xlsx"):
    df = pd.read_excel(f'D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data/excel_merged/{filename}')
    tool = language_tool_python.LanguageTool('de-DE')
    df['Satz'] = df['Satz'].apply(correction, tool = tool)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df)
    df.to_excel("D:\OneDrive\Dokumente\MasterThesis\PreAnalysis/data/excel_merged/1976-1986_lehrer_correction.xlsx")

get_joint_excel()
correct_mistakes()
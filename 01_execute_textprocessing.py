from textpreprocessor import TextPreprocessor
from textpostprocessor import TextPostprocessor
import pandas as pd

preprocessor = TextPreprocessor()
postprocessor = TextPostprocessor()

# preprocess all chapter Berufe
for i, j in enumerate(range(1975, 1990), 1):
    preprocessor.text_to_excel(
        filename=f"Berufsmöglichkeiten {j}.txt", excel_name=f"{j}")

#build joint excel
postprocessor.get_joint_excel(filepath="PreAnalysis/data/excel_merged/Beruf.xlsx")

#correct mistakes
postprocessor.correct_mistakes(filename="Beruf.xlsx", filepath="PreAnalysis/data/excel_merged/all_beruf_correction.xlsx")

#get keywords
postprocessor.tf_idf(df = pd.read_excel(r"PreAnalysis/data/excel_merged/all_beruf_correction.xlsx"), name="beruf")

# preprocess all chapter Lehrer
for i, j in enumerate(range(1975, 1990), 1):
    preprocessor.text_to_excel(
        filename=f"Lehrerbedarf {j}.txt", excel_name=f"{j}")

#build joint excel
postprocessor.get_joint_excel(filepath="PreAnalysis/data/excel_merged/Lehrer.xlsx")

#correct mistakes
postprocessor.correct_mistakes(filename="Lehrer.xlsx", filepath="PreAnalysis/data/excel_merged/all_lehrer_correction.xlsx")

#get keywords
postprocessor.tf_idf(df = pd.read_excel(r"PreAnalysis/data/excel_merged/all_lehrer_correction.xlsx"), name="lehrer")
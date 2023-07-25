from TextPreprocessor import *

text_preprocessor = TextPreprocessor()

for i, j in enumerate(range(1975, 1991), 1):
    text_preprocessor.text_to_excel(
        filename=f"Lehrerbedarf {j}.txt", excel_name=f"{j}")

from TextPreprocessor import *

text_preprocessor = TextPreprocessor()

for i, j in enumerate(range(1976, 1987), 1):
    text_preprocessor.text_to_excel(
        filename=f"lehrerbedarf {i}.txt", excel_name=f"{j}")

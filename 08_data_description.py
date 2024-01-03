import pandas as pd

#read data
df_lehrer = pd.read_excel(r"PreAnalysis\data\excel_merged\all_lehrerbedarf_correction.xlsx")
df_beruf =  pd.read_excel(r"PreAnalysis\data\excel_merged\all_berufsmöglichkeiten_correction.xlsx")
df_all = pd.concat([df_lehrer, df_beruf])

# summarize data for description
size_lehrer = df_lehrer.groupby(by="Year").size()
size_beruf = df_beruf.groupby(by="Year").size()
size_all =  df_all.groupby(by="Year").size()
#year = [i for i in range(1975,1991)]
table = {"Sentences Teaching": size_lehrer, "Sentences Other": size_beruf, "Overall Sentences": size_all}
table = pd.DataFrame.from_dict(table).fillna(0).astype(int)
df_sum = table.sum()
df_sum.name = "Sum"

#save output as Excel
table = table.append(df_sum.transpose())
table.to_excel("data_description.xlsx")
import pandas as pd
from datasets import Dataset


#=========== Load Data and Tokenizer ==============

df = pd.read_excel(r"D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data\excel_merged\data_for_training.xlsx")

#=========== Build Dataset ==================

def build_dataset(df):
    sentences = df["Satz"]
    label = df["Klasse"]
    label = label.replace(-1, 3).replace(0, 2).replace(1, 0).replace(3, 1).dropna()
    # 0 = positiv, 1=negative, 2 = neutral

    ds = Dataset.from_dict({"sentence": sentences, "label": label})
    ds = ds.train_test_split(test_size=0.1)
    return ds

def store_validation_set(ds):
    validation_sentence = ds["test"]["sentence"]
    validation_label = ds["test"]["label"]
    validation_set = {"sentence": validation_sentence, "label": validation_label}
    validation_set = pd.DataFrame.from_dict(validation_set)
    validation_set.to_excel(r"D:\OneDrive\Dokumente\MasterThesis\validation_set.xlsx", index=False)

ds = build_dataset(df)
store_validation_set(ds)
ds.save_to_disk("training_data")


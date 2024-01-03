import pandas as pd
from datasets import Dataset

################## IMPORTANT ############################
# Executing this script will build a new validation set
# the model is not trained on and may mean that 
# the results are different from those stated in the paper,
# due to randomization of the train/test split.
# Please consider not executing this script and
# use the provided validation set in the next script!!
#########################################################

#=========== Load Data and Tokenizer ==============

df = pd.read_excel(r"results\data_for_training.xlsx")
df["Label"].value_counts()

#=========== Build Dataset ==================

def build_dataset(df):
    sentences = df["Sentence"]
    label = df["Label"]
    label = label.replace(-1, 3).replace(0, 2).replace(1, 0).replace(3, 1).dropna()
    # 0 = positiv, 1 = negative, 2 = neutral

    ds = Dataset.from_dict({"sentence": sentences, "label": label})
    ds = ds.train_test_split(test_size=0.1)
    return ds

def store_validation_set(ds):
    validation_sentence = ds["test"]["sentence"]
    validation_label = ds["test"]["label"]
    validation_set = {"sentence": validation_sentence, "label": validation_label}
    validation_set = pd.DataFrame.from_dict(validation_set)
    validation_set.to_excel("validation_set.xlsx", index=False)

ds = build_dataset(df)

store_validation_set(ds)
ds.save_to_disk("training_data")


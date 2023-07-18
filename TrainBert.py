import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset

df = pd.read_excel(r"D:\OneDrive\Dokumente\MasterThesis\PreAnalysis\data\excel_merged\data_for_training.xlsx")
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")
model = TFAutoModelForSequenceClassification.from_pretrained("oliverguhr/german-sentiment-bert")

def build_dataset(df):
    sentences = df["Satz"]
    label = df["Klasse"]
    label = label.replace(-1, 2).dropna()

    ds = Dataset.from_dict({"sentence": sentences, "label": label})
    ds = ds.train_test_split(test_size=0.2)
    return ds

def store_validation_set(ds):
    validation_sentence = ds["test"]["sentence"]
    validation_label = ds["test"]["label"]
    validation_set = {"sentence": validation_sentence, "label": validation_label}
    validation_set = pd.DataFrame.from_dict(validation_set)
    validation_set.to_excel(r"D:\OneDrive\Dokumente\MasterThesis\validation_set.xlsx", index=False)

def tokenize_function(df):
    return tokenizer(df["sentence"], truncation=True)

ds = build_dataset(df)
store_validation_set(ds)
tokenized_datasets = ds.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

num_epochs = 5
num_train_steps = len(tf_train_dataset) * num_epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(
    optimizer=optimizer,
    metrics=["accuracy"],
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=num_epochs
)

model.save_pretrained(
   r"D:\MasterData\saved_model",
   from_tf = True
)

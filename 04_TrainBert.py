import tensorflow as tf
import datasets
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
import datetime
from sklearn.utils import class_weight
import numpy as np



#=========== Load Dataset ==============

ds = datasets.load_from_disk(r"training_data")

#=========== Build TF_Dataset =================
# load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")

def tokenize_function(df):
    return tokenizer(df["sentence"], truncation=True)

#transform into dataset format
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

#================= Training Arguments and Weights ===========================

num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-5,
    decay_steps= num_train_steps,
    decay_rate= 0.96,
    staircase = True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
train_label = ds["train"]["label"]

#calculate weights
class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(train_label), y= train_label)
weights_dic = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

#build log files
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#================ Train and Save Model ======================

model = TFAutoModelForSequenceClassification.from_pretrained("oliverguhr/german-sentiment-bert")
model.config.id2label[0]

#initiate model
model.compile(
    optimizer=optimizer,
    metrics=["accuracy"],
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

#train model
model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=num_epochs,
    callbacks=[tensorboard_callback],
    class_weight = weights_dic
)

#save model
model.save_pretrained(
   r"saved_model",
   from_tf = True
)

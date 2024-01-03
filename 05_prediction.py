import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report, multilabel_confusion_matrix

model = TFAutoModelForSequenceClassification.from_pretrained("saved_model")
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")
validation_set = pd.read_excel("results/validation_set.xlsx", index_col= False)
sen_val = validation_set["sentence"]
output_val = validation_set["label"]

#======== testing and export classification report =============

def predict_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="tf")
    logits = model(**inputs).logits
    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    return predicted_class_id

prediction_val = []
for i in sen_val:
    prediction_val.append(predict_sentence(i))

report_val = classification_report(output_val, prediction_val, output_dict=True)
cm_val = multilabel_confusion_matrix(output_val, prediction_val)
report_df = pd.DataFrame(report_val).transpose()
report_df.to_excel("classification_report.xlsx")

print(report_val)
print(cm_val)

#========= Prediction ==========

class predict_prospect():
    '''
    use the trained model to predict the class of each sentence
    '''
    def __init__(self) -> None:
        self.model = TFAutoModelForSequenceClassification.from_pretrained("saved_model")
        self.tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")

    def predict_sentence(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="tf")
        logits = self.model(**inputs).logits
        predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
        return predicted_class_id
    
    def predict_document(self, df):
        df["Prediction"] = df["Sentence"].apply(predict_sentence)
        return df

if __name__=="__main__":
    predictor = predict_prospect()
    
    # predict Berufe
    df = pd.read_excel("PreAnalysis/data/excel_merged/all_beruf_correction.xlsx")
    df = predictor.predict_document(df)
    df.to_excel("all_beruf_predicted.xlsx", index=False)
    
    #predict Lehrer
    df = pd.read_excel("PreAnalysis/data/excel_merged/all_lehrer_correction.xlsx")
    df = predictor.predict_document(df)
    df.to_excel("all_lehrer_predicted.xlsx", index=False)
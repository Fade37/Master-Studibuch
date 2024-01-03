import pandas as pd
import re

class Subject_Classifier():
    
    def __init__(self) -> None:
        self.medizin = ("Mediziner", "Ärzte", "Arzt", "Medizin")
        self.natur = ("Mathematik", "Physik", "Chemie",  "Pharmazeut", "Naturwissenschaft", "Biolog", "Chemikant")
        self.sozial = ("Wirtschafts-", "Rechts-", "Sozialwissenschaft", "Wirtschaftswissenschaft", "Rechtswissenschaft", "Politolog", "Soziolog", "Jura") 
        self.ingenieur = ("Maschinenbau", "Elektrotechnik", "Ingenieur", "Architekt", "Bergbau", "ingenieur")

    def search_all_subjects(self, path):
        df = pd.read_excel(path)
        for attr, value in self.__dict__.items():
            pattern = "|".join(value)
            df[f"{attr}"] = df["Sentence"].apply(lambda x: bool(re.search(pattern, x)))
        df.to_excel("all_beruf_finished.xlsx", index=False)
    
    def search_single(self, df, subject):
        try:
            assert subject in self.__dict__.keys()
            if subject == "medizin":
                s = self.medizin
            elif subject == "ingenieur":
                s = self.ingenieur 
            elif subject == "natur":
                s = self.natur
            elif subject == "sozial":
                s = self.sozial
            pattern = "|".join(s)
            df[f"{subject}"] = df["Sentence"].apply(lambda x: bool(re.search(pattern, x)))
            return df
        except:
            print("subject not found")


if __name__=="__main__":
    subject = Subject_Classifier()
    subject.search_all_subjects("all_beruf_predicted.xlsx")

    
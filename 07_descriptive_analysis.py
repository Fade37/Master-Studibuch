import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import seaborn as sns


### Load Data ###
df_beruf = pd.read_excel(r"results\Classification Results\all_beruf_finished.xlsx")
df_lehrer = pd.read_excel(r"results\Classification Results\all_lehrer_finished.xlsx")
results_non = pd.read_excel(r"results\Classification Results\Analyzed\nonteaching.xlsx")
results_non = results_non.drop(labels=29, axis=0)
results_non["Year"] = pd.to_datetime(results_non.Year, format="%Y")
results_teaching = pd.read_excel(r"results\Classification Results\Analyzed\lehrer.xlsx")
results_teaching["Year"] = pd.to_datetime(results_teaching.Year, format="%Y")
students = pd.read_excel(r"results\students.xlsx")
students["Year"] = pd.to_datetime(students.Year, format="%Y")

#fields of study
fields = ['Teaching', 'Science', 'Engineering', 'Social Science', 'Medicine']

# merge non-teaching and teaching
results = pd.concat([results_non, results_teaching]).drop(labels=35, axis=0).reset_index(drop=True)
results["Field"] = results["Field"].fillna("Teaching")

# normalize first-year students
normalized_students=pd.DataFrame()
normalized_students["Year"] = students["Year"]
for field in fields:
    normalized_students[field] = students[field].apply(lambda x: (x-students[field].min())/(students[field].max()-students[field].min()))

#%% result tables

def sen_count(df1 = df_beruf, df2 = df_lehrer):
    df1 = df1.groupby(by="Year")["Sentence"].count()
    df2 = df2.groupby(by="Year")["Sentence"].count()
    final = pd.concat([df1, df2], axis=1).fillna(0).astype("int64")
    names = ["Non-Teaching", "Teaching"]
    final.columns = names
    final["Overall Sentences"] = final["Non-Teaching"] + final["Teaching"]
    final.loc["Total"] = final.sum()
    final.to_excel("data_description.xlsx")
    return "done!"

def get_class_rates(df):
    new = df.groupby(by="Year")
    counts = new["Label"].value_counts().reset_index(name='Counts')
    length = pd.DataFrame(new.size(), columns=["Length"])
    colnames = ["Total_Negative", "Total_Neutral", "Total_Positive", "Negative_Rate", "Neutral_Rate", "Positive_Rate"]

    new = pd.merge(counts, length, how="left", on="Year")
    new["Rates"] = round(new["Counts"]/new["Length"], 3)
    new = new.pivot(index="Year", columns="Label", values=["Counts", "Rates"])
    new.columns = colnames
    new[["Total_Negative", "Total_Neutral", "Total_Positive"]] = new[["Total_Negative", "Total_Neutral", "Total_Positive"]].fillna(0).astype("int64")
    new["Total_All"] = length
    return new

def beruf_results(df):
    fields = ["medizin", "sozial", "natur", "ingenieur"]
    for field in fields:
        filtered = df[df[field] == True]
        result = get_class_rates(filtered)
        result = result.fillna(0)
        result.to_excel(f"results/{field}.xlsx")
    return print("done!")

def lehrer_results(df):
    df["Label"] = df["Label"].fillna(df["Prediction"])
    result = get_class_rates(df)
    result = result.fillna(0)
    result.to_excel("results/lehrer.xlsx")
    return print("done!")

#beruf_results(df_beruf)
#lehrer_results(df_lehrer)
#sen_count()

############### Plots ###############

#%% plot students ###########

plt.figure(dpi=1000)
plt.plot( "Year", "Teaching", data=students, marker='o', markerfacecolor='blue', markersize=5, color='skyblue', linewidth=4, label = "Teaching")
plt.plot( "Year", 'Medicine', data=students, marker='D', markerfacecolor='dimgray', markersize=5, color='gray', linewidth=2, label="Medicine")
plt.plot( "Year", 'Science', data=students, marker='*', markerfacecolor='darkgreen', markersize=8, color='forestgreen', linewidth=2, label="Science")
plt.plot( "Year", 'Social Science', data=students, marker='.', markerfacecolor='indigo', markersize=8, color='darkviolet', linewidth=2, label="Social Science")
plt.plot( "Year", 'Engineering', data=students, marker='', markerfacecolor='orange', markersize=8, color='goldenrod', linewidth=2, label="Engineering")
plt.legend()
plt.grid()
plt.show


#%% student number nonteaching against rates

Year = results_non["Year"].drop_duplicates()
fields = ['Teaching', 'Science', 'Engineering', 'Medicine', 'Social Science']

medicine = pd.concat([results_non.loc[results_non["Field"]=="Medicine"], pd.DataFrame(Year, columns=["Year"])], join="outer").drop_duplicates("Year").fillna(0).sort_values("Year")
science = pd.concat([results_non.loc[results_non["Field"]=="Science"], pd.DataFrame(Year, columns=["Year"])], join="outer").drop_duplicates("Year").fillna(0).sort_values("Year")
engineering = pd.concat([results_non.loc[results_non["Field"]=="Engineering"], pd.DataFrame(Year, columns=["Year"])], join="outer").drop_duplicates("Year").fillna(0).sort_values("Year")
social = pd.concat([results_non.loc[results_non["Field"]=="Social Science"], pd.DataFrame(Year, columns=["Year"])], join="outer").drop_duplicates("Year").fillna(0).sort_values("Year")


fig, ax = plt.subplots(2,2, sharex="all", figsize=(10,6), dpi=1000)
#Medicine
ax[0,0].plot(Year, "Medicine", data = normalized_students, markerfacecolor='dimgray', markersize=5, color='gray', linewidth=2, label="Medicine" )
ax[0,0].plot("Year", "Negative Rate" , data = medicine, linestyle='dashed')
ax[0,0].plot("Year", "Positive Rate", data = medicine, linestyle='dashed')
ax[0,0].plot("Year", "Neutral Rate", data = medicine, linestyle='dashed', color='lightgray')
ax[0, 0].set_title('Medicine')
ax[0,0].legend(loc="upper left")
#Science
ax[0,1].plot(Year, "Science", data = normalized_students, markerfacecolor='darkgreen', markersize=5, color='forestgreen', linewidth=2, label="Science")
ax[0,1].plot("Year", "Negative Rate" , data = science, linestyle='dashed',label='_nolegend_')
ax[0,1].plot("Year", "Positive Rate", data = science, linestyle='dashed',label='_nolegend_')
ax[0,1].plot("Year", "Neutral Rate", data = science, linestyle='dashed',label='_nolegend_',color='lightgray')
ax[0,1].set_title('Science')
ax[0,1].legend(loc="upper left")
#Engineering
ax[1,0].plot(Year, "Engineering", data = normalized_students, marker='', markerfacecolor='orange', markersize=5, color='goldenrod', linewidth=2, label="Engineering")
ax[1,0].plot("Year", "Negative Rate" , data = engineering, linestyle='dashed',label='_nolegend_')
ax[1,0].plot("Year", "Positive Rate", data = engineering, linestyle='dashed',label='_nolegend_')
ax[1,0].plot("Year", "Neutral Rate", data = engineering, linestyle='dashed',label='_nolegend_',color='lightgray')
ax[1,0].set_title('Engineering')
ax[1,0].legend(loc="upper left")
#Social Science
ax[1,1].plot(Year, "Social Science", data = normalized_students, markerfacecolor='indigo', markersize=5, color='darkviolet', linewidth=2, label="Social Science")
ax[1,1].plot("Year", "Negative Rate" , data = social, linestyle='dashed', label='_nolegend_')
ax[1,1].plot("Year", "Positive Rate", data = social, linestyle='dashed',label='_nolegend_')
ax[1,1].plot("Year", "Neutral Rate", data = social, linestyle='dashed',label='_nolegend_',color='lightgray')
ax[1,1].set_title('Social Science')
ax[1,1].legend(loc="upper left")

plt.tight_layout()
plt.show()

#%% student number against teaching rates

results_teaching["Year"] = pd.to_datetime(results_teaching.Year, format="%Y")
Year = students["Year"].drop_duplicates()
Year = pd.to_datetime(Year, format="%Y")


teaching = pd.concat([results_teaching, pd.DataFrame(Year, columns=["Year"])], join="outer").drop_duplicates("Year").sort_values("Year")

plt.figure(dpi=1000)
plt.plot("Year", "Teaching", data = normalized_students, marker='o', markerfacecolor='blue', markersize=5, color='skyblue', linewidth=4, label = "Teaching")
plt.plot("Year", "Negative Rate" , data = teaching, linestyle='dashed')
plt.plot("Year", "Positive Rate", data = teaching, linestyle='dashed')
plt.legend(loc="upper right")
plt.show()

#%% Correlation

pos_corr = []
neg_corr = []
pvalue_pos = []
pvalue_neg = []
rates = ["Positive Rate", "Negative Rate"]

for field in fields:
    x = pd.merge(results.loc[(results["Field"] == field) & (results["Total All"] >= 4)], 
                 students.loc[:, ["Year", field]], 
                 how = "left", on = "Year").rename(columns={field: "Numbers"})
    pos = x.loc[:, ["Numbers", "Positive Rate"]]
    #pos = pos[(pos["Positive Rate"]) < 1 & (pos["Positive Rate"] > 0)]
    neg = x.loc[:, ["Numbers", "Negative Rate"]]
    #neg = neg[(neg["Negative Rate"]) < 1 & (neg["Negative Rate"] > 0)]
    
    pos_corr.append(scipy.stats.pearsonr(pos["Numbers"],pd.to_numeric(pos["Positive Rate"]))[0])
    pvalue_pos.append(scipy.stats.pearsonr(pos["Numbers"], pd.to_numeric(pos["Positive Rate"]))[1])
    neg_corr.append(scipy.stats.pearsonr(neg["Numbers"], pd.to_numeric(neg["Negative Rate"]))[0])
    pvalue_neg.append(scipy.stats.pearsonr(neg["Numbers"], pd.to_numeric(neg["Negative Rate"]))[1])

    
corr_dict = {"Fields": fields, "Correlation Positive": pos_corr, "P-Value Positive": pvalue_pos, "Correlation Negative": neg_corr, "P-Value Negative": pvalue_neg}
corr_result = pd.DataFrame.from_dict(corr_dict)    
corr_result.to_excel(r"results\correlation_result.xlsx", index=False)

#%% Data Preparation and Outlier Detection

fields = ['Teaching', 'Science', 'Engineering', 'Social Science']

#negative rate 
neg_data = pd.DataFrame()
for field in fields:
    x = pd.merge(results[(results["Field"]==field) & (results["Total All"] >= 4)].loc[:, ["Year","Field", "Negative Rate"]].rename(columns = {"Negative Rate": "NegRate"}), 
                            students.loc[:,["Year", field]].rename(columns={field: "Numbers"}), how="left", on="Year")
    neg_data = pd.concat([neg_data, x]).reset_index(drop=True)
    #neg_data = neg_data[(neg_data.NegRate < 1) & (neg_data.NegRate > 0)]
    neg_data["NegRate"] = np.array(pd.to_numeric(neg_data["NegRate"]))
    #neg_data = neg_data[(np.abs(stats.zscore(neg_data["NegRate"])) < 2)]


#positive rate
pos_data = pd.DataFrame()
for field in fields:
    x = pd.merge(results[(results["Field"]==field) & (results["Total All"] >= 4)].loc[:, ["Year","Field", "Positive Rate"]].rename(columns = {"Positive Rate": "PosRate"}), 
                            students.loc[:,["Year", field]].rename(columns={field: "Numbers"}), how="left", on="Year")
    pos_data = pd.concat([pos_data, x]).reset_index(drop=True)
    #pos_data = pos_data[(pos_data.PosRate < 1) & (pos_data.PosRate > 0)]
    pos_data["PosRate"] = np.array(pd.to_numeric(pos_data["PosRate"]))
    #pos_data = pos_data[(np.abs(stats.zscore(pos_data["PosRate"])) < 2)]


#combined
reg_data = pd.merge(pos_data, neg_data, how='outer')
reg_data["log_"+"Numbers"] = np.log(reg_data["Numbers"])


#%% Scatterplot


sns.pairplot(data = reg_data.drop('Numbers', axis=1),kind="reg", hue='Field' )



#%% Fixed-effect (dummy)

fields = ['Teaching', 'Science', 'Engineering', 'Social Science']


dummy_major = pd.get_dummies(reg_data["Field"], prefix="Field")
column_names = reg_data.columns.values.tolist()
column_names.remove("Field")
regression_data = reg_data[column_names].join(dummy_major).rename(columns= {"Field_Social Science": "Field_Social"})

#%% Regression

model1 = smf.ols(formula= "log_Numbers ~ PosRate+NegRate+Field_Engineering+Field_Science+Field_Social+Field_Teaching", data=regression_data).fit()
model2 = smf.ols(formula= "log_Numbers ~ PosRate+Field_Engineering+Field_Science+Field_Social+Field_Teaching", data=regression_data).fit()

table = summary_col([model2, model1],stars=True,float_format='%0.3f',
                  model_names=['p4\n(0)','p4\n(1)'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                            'R2':lambda x: "{:.2f}".format(x.rsquared)})
print(table)
table.as_text()
regression_summary = model1.summary()
regression_summary2 = model2.summary()
regression_csv = regression_summary.tables[1].as_html()
print(regression_summary)
print(regression_summary2)



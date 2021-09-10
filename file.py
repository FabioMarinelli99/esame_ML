import warnings, copy, json, pprint
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, \
    roc_auc_score, \
    precision_score, \
    recall_score, \
    f1_score, \
    precision_recall_curve, \
    roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import time

pp = pprint.PrettyPrinter()

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("train.csv")

dim = {'normal': {'h': 3, 'asp': 1.2}, 'big': {'h': 8, 'asp': 0.5}}


# plot a distribution plot
def displot(title, feature, frame, size="normal", color='g'):
    pass
    # global dim
    # h = dim[size]['h']
    # asp = dim[size]['asp']
    # g = sns.displot(frame, x=feature, color=color, height=h, aspect=asp)
    # if title != 'notitle':
    #     g.fig.suptitle(title)
    #     g.fig.subplots_adjust(top=0.9)
    # #plt.show()


# plot a categorical plot
def catplot(title, feature, df, size="normal"):
    pass
    # try:
    #     global dim
    #     h = dim[size]['h']
    #     asp = dim[size]['asp']
    #     plt.title(title)
    #     g = sns.catplot(y=feature, kind="count", hue=target_variable, height=h, aspect=asp, data=df, orient='h',
    #                     palette=sns.color_palette(
    #                         ['green', 'gray', 'yellow', 'purple', 'black', 'fuchsia', 'orange', 'blue', 'red',
    #                          'brown']))
    #     if title != 'notitle':
    #         g.fig.suptitle(title)
    #         g.fig.subplots_adjust(top=0.9)
    #     #plt.show()
    # except:
    #     print('---------> ' + title)


###################################################
#                       EDA                       #
###################################################

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.describe(include=['object']))
print(df.columns)

# Abbiamo 43 feature + 2 valori target, di cui il primo (Genetic Disorder) indica il disordine genetico del paziente
# mentre il secondo (Disorder Subclass) va a specificare la patologia del paziente.

print(df['Disorder Subclass'].unique())
print(df['Genetic Disorder'].unique())

# Per Genetic disorder abbiamo 3 diversi valori possibili, mentre per Disorder Subclass ne abbiamo 9.
# Decidiamo di predirre la variabile Disorder Subclass.
target_variable = 'Disorder Subclass'
col_to_ignore = 'Genetic Disorder'

df.drop(columns=[col_to_ignore], inplace=True)
df = df.dropna(axis=0, subset=[target_variable])
df.reset_index(drop=True, inplace=True)

sns.catplot(y=target_variable, kind="count", data=df, height=2.8, aspect=2.5, orient='h')
# Il dataset non è bilanciato, per alcune malattie abbiamo molti più sample di altre.

first_evidence = []
for col in df.columns:
    unique_val = df[col].unique()
    unq = ""
    if len(unique_val) > 10:
        unq += f" > 10\n NULL: {round(df[col].isna().sum() * 1000 / len(df)) / 10} %"
    else:
        for val in unique_val:
            if type(val) != str:
                if np.isnan(val):
                    unq = f" NULL: {round(df[col].isna().sum() * 1000 / len(df)) / 10} %\n" + unq
                else:
                    unq += f" {val}: {round(len(df[df[col] == val]) * 1000 / len(df)) / 10} %\n"
            else:
                unq += f" {val}: {round(len(df[df[col] == val]) * 1000 / len(df)) / 10} %\n"
    first_evidence.append({'name': col, 'unique': unq})
for col in first_evidence:
    print(f"{col['name']}:\n{col['unique']}\n\n")

# - Notiamo che le colonne 'Test 1-5' e 'Parental consent' presentano valori uguali per tutti i pazienti o nulli,
# per tanto verrranno eliminate in fase di preprocessing perché inutili.
#
# - Le colonne 'Gender', 'Birth asphyxia', 'H/O radiation exposure (x-ray)', 'H/O substance abuse' presentano
# troppi valori nulli, sono quindi inutilizzabili.


# Provo a vedere se c'è qualche tipo di correlazione tra la variabile target e alcune delle feature
# che sembrano esseere più interessanti

catplot('notitle', 'Patient Age', df, 'big')
# Le malattie sono ben distribuite per tutte le età

catplot("notitle", "Genes in mother's side", df)
# Notiamo che per alcune malattie, come la fibrosi cistica o il diabete, avere un parente da parte della madre
# con difetti genetici sembra aumentare la probabilià che il soggetto abbia queste malattie.
# Per altre malattie invece, come la Tay-Sachs è palese che che non ci sia alcuna correlazione con questo dato.

catplot("notitle", "Inherited from father", df)
# Qui notiamo meno evidenze, i dati sembrnao distribuiti in modo più equo in generale.

catplot("notitle", "Maternal gene", df)
# Per alcune malattie il dato positivo sembra aumentare leggermente la probabilità di avere la malattia.

catplot("Paternal gene", "Paternal gene", df)
# Dati più equi.

for disorder in df[target_variable].unique():
    displot(f"Blood cell count (mcL) - {disorder}", "Blood cell count (mcL)", df[df[target_variable] == disorder])
# A parte per il Cancro in cui possiamo osservare valori che "rompono" leggermente la distribuzione
# in corrispondenza del massimo e del minimo, per il resto vediamo che nelle altre malattie i paziente
# presentano una distribuzione normale di numero di globuli rossi.
# Inoltre il cancro è la malattia meno frequente nel dataset (0.5 %), quindi decido di elimare questa colonna.

catplot("notitle", "Status", df)
# I dati sono perfettamente bilanciati tra pazienti morti e vivi, si potrebbe quindi eliminare
# questo campo in quanto non altera minimamente la probabilità di avere o meno nessuna delle malattie.

catplot('notitle', 'Birth asphyxia', df)

catplot('notitle', 'Folic acid details (peri-conceptional)', df)

catplot('notitle', 'H/O serious maternal illness', df)

catplot('notitle', 'H/O radiation exposure (x-ray)', df)

catplot('notitle', 'Assisted conception IVF/ART', df)

catplot('notitle', 'History of anomalies in previous pregnancies', df)

catplot('notitle', 'No. of previous abortion', df)

catplot('notitle', 'Birth defects', df)

for disorder in df[target_variable].unique():
    displot(f"White Blood cell count - {disorder}", "White Blood cell count (thousand per microliter)",
            df[df[target_variable] == disorder])
# Notiamo picchi in corrispondenza del massimo e del minimo

catplot('notitle', 'Blood test result', df)

for x in range(1, 6):
    catplot("notitle", f"Symptom {x}", df)

# Non ci sono particolari evidenze


#############################################################
#                       Preprocessing                       #
#############################################################
df_encoded = df.copy()
df_raw = df.copy()

le_target_variable = LabelEncoder()
le_target_variable = le_target_variable.fit(df[target_variable])
df_encoded[target_variable] = le_target_variable.transform(df[target_variable])

le_target_variable_raw = LabelEncoder()
le_target_variable_raw = le_target_variable_raw.fit(df[target_variable])
df_raw[target_variable] = le_target_variable_raw.transform(df[target_variable])

deleted_cols = []

# nome colonna scritto male
columns = list(df_encoded.columns)
columns[columns.index('Heart Rate (rates/min')] = 'Heart Rate (rates/min)'
df_encoded.columns = columns
df_raw.columns = columns

# Colonna target secondaria
deleted_cols.append(col_to_ignore)

# Colonne concettualmente inutili, le elimino anche nel dataframe che sarà usato per i dati non preprocessati
cols = ['Patient Id', 'Patient First Name', 'Family Name', "Father's name", 'Institute Name', 'Location of Institute',
        'Place of birth']
df_encoded = df_encoded.drop(columns=cols)
df_raw = df_raw.drop(columns=cols)
df_raw.reset_index(drop=True, inplace=True)
df_encoded.reset_index(drop=True, inplace=True)
deleted_cols.extend(cols)

# Colonne inutili perche con valori univoci o nulli.
cols = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Parental consent']
df_encoded = df_encoded.drop(columns=cols)
df_raw.reset_index(drop=True, inplace=True)
df_encoded.reset_index(drop=True, inplace=True)
deleted_cols.extend(cols)
# in df_raw inserisco l'unico valore presente nella colonna anche nei nulli
df_raw['Parental consent'] = df_raw['Parental consent'].replace({'Yes': 1})
for col in cols:
    u = df_raw[col].unique()
    u = u[np.logical_not(np.isnan(u))][0]
    df_raw[col] = df_raw[col].fillna(u)
    df_raw[col] = df_raw[col].dropna()


# Colonne rimovibili
cols = ['Status', 'Autopsy shows birth defect (if applicable)', 'Blood cell count (mcL)']
df_encoded = df_encoded.drop(columns=cols)
df_raw.reset_index(drop=True, inplace=True)
df_encoded.reset_index(drop=True, inplace=True)
deleted_cols.extend(cols)
# aggiusto i dati in df_raw
df_raw['Status'] = df_raw['Status'].fillna(pd.Series(np.random.choice(['Alive', 'Deceased'], len(df_raw))))
df_raw['Status'] = df_raw['Status'].replace({'Alive': 1, 'Deceased': 0})
#
df_raw['Autopsy shows birth defect (if applicable)'] = df_raw['Autopsy shows birth defect (if applicable)'].replace({'None':np.nan, 'Not applicable':np.nan})
df_raw['Autopsy shows birth defect (if applicable)'] = df_raw['Autopsy shows birth defect (if applicable)'].fillna(pd.Series(np.random.choice(['Yes', 'No'], len(df_raw))))
df_raw['Autopsy shows birth defect (if applicable)'] = df_raw['Autopsy shows birth defect (if applicable)'].replace({'Yes':1, 'No':0})
#
df_raw['Blood cell count (mcL)'] = df_raw['Blood cell count (mcL)'].fillna(df_raw['Blood cell count (mcL)'].mean())

# Colonne inutilizzabili
cols = ['Gender', 'Birth asphyxia', 'H/O radiation exposure (x-ray)', 'H/O substance abuse']
df_encoded = df_encoded.drop(columns=cols)
df_raw.reset_index(drop=True, inplace=True)
df_encoded.reset_index(drop=True, inplace=True)
deleted_cols.extend(cols)
# aggiusto i dati in df_raw
df_raw['Gender'] = df_raw['Gender'].replace({'Ambiguous': np.nan})
df_raw['Gender'] = df_raw['Gender'].fillna(pd.Series(np.random.choice(['Male', 'Female'], len(df_raw))))
df_raw['Gender'] = df_raw['Gender'].replace({'Female': 1, 'Male': 0})
#
df_raw['Birth asphyxia'] = df_raw['Birth asphyxia'].replace({'No record': np.nan, 'Not available':np.nan})
df_raw['Birth asphyxia'] = df_raw['Birth asphyxia'].fillna(pd.Series(np.random.choice(['Yes', 'No'], len(df_raw))))
df_raw['Birth asphyxia'] = df_raw['Birth asphyxia'].replace({'Yes': 1, 'No': 0})
#
df_raw['H/O radiation exposure (x-ray)'] = df_raw['H/O radiation exposure (x-ray)'].replace({'-': np.nan, 'Not applicable':np.nan})
df_raw['H/O radiation exposure (x-ray)'] = df_raw['H/O radiation exposure (x-ray)'].fillna(pd.Series(np.random.choice(['Yes', 'No'], len(df_raw))))
df_raw['H/O radiation exposure (x-ray)'] = df_raw['H/O radiation exposure (x-ray)'].replace({'Yes': 1, 'No': 0})
#
df_raw['H/O substance abuse'] = df_raw['H/O substance abuse'].replace({'-': np.nan, 'Not applicable':np.nan})
df_raw['H/O substance abuse'] = df_raw['H/O substance abuse'].fillna(pd.Series(np.random.choice(['Yes', 'No'], len(df_raw))))
df_raw['H/O substance abuse'] = df_raw['H/O substance abuse'].replace({'Yes': 1, 'No': 0})

# 6.2 % nulli, riempio con la media dell'età
df_encoded['Patient Age'] = df_encoded['Patient Age'].fillna(round(df_encoded['Patient Age'].mean()))
df_encoded['Patient Age'] = df_encoded['Patient Age'].astype(int)
#
df_raw['Patient Age'] = df_raw['Patient Age'].fillna(round(df_raw['Patient Age'].mean()))
df_raw['Patient Age'] = df_raw['Patient Age'].astype(int)

# Yes: 59.6 %, No: 40.4 %
df_encoded["Genes in mother's side"] = df_encoded["Genes in mother's side"].replace({'Yes': 1, 'No': 0})
#
df_raw["Genes in mother's side"] = df_raw["Genes in mother's side"].replace({'Yes': 1, 'No': 0})

# Yes: 39.1 %, No: 59.6 %, 1 % null [302 rows] ---> le cancello, sono troppo poche per valer la pena di
# arrotondare la precisione del risultato finale.
df_encoded = df_encoded.dropna(axis=0, subset=['Inherited from father'])
df_encoded.reset_index(drop=True, inplace=True)
df_encoded["Inherited from father"] = df_encoded["Inherited from father"].replace({'Yes': 1, 'No': 0})
# in df_raw riempio
df_raw.reset_index(drop=True, inplace=True)
df_raw["Inherited from father"] = df_raw["Inherited from father"].fillna(pd.Series(np.random.choice(['Yes', 'No'], len(df_raw))))
df_raw["Inherited from father"] = df_raw["Inherited from father"].replace({'Yes': 1, 'No': 0})

# Yes: 48.4 %, No: 39.5%, 13% null [2759 rows], riempio in modo casuale
df_encoded['Maternal gene'] = df_encoded['Maternal gene'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded["Maternal gene"] = df_encoded["Maternal gene"].replace({'Yes': 1, 'No': 0})
#
df_raw['Maternal gene'] = df_raw['Maternal gene'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_raw))))
df_raw["Maternal gene"] = df_raw["Maternal gene"].replace({'Yes': 1, 'No': 0})

# Yes: 43.3 %, No: 56.7 %
df_encoded["Paternal gene"] = df_encoded["Paternal gene"].replace({'Yes': 1, 'No': 0})
#
df_raw["Paternal gene"] = df_raw["Paternal gene"].replace({'Yes': 1, 'No': 0})

# L'età attuale dei genitori non è un dato di interesse. Potrebbe però essere un dato utile sapere quanti anni
# avevano alla nascità del figlio.
# Riempio i valori nulli con la media.
# 26.1 % null
df_encoded["Mother's age"] = df_encoded["Mother's age"] - df_encoded["Patient Age"]
df_encoded["Mother's age"] = df_encoded["Mother's age"].fillna(round(df_encoded["Mother's age"].mean()))
df_encoded["Mother's age"] = df_encoded["Mother's age"].astype(int)
# 25.7 % null
df_encoded["Father's age"] = df_encoded["Father's age"] - df_encoded["Patient Age"]
df_encoded["Father's age"] = df_encoded["Father's age"].fillna(round(df_encoded["Father's age"].mean()))
df_encoded["Father's age"] = df_encoded["Father's age"].astype(int)
# in df_raw lascio gli anni originali
df_raw["Mother's age"] = df_raw["Mother's age"].fillna(round(df_raw["Mother's age"].mean()))
df_raw["Mother's age"] = df_raw["Mother's age"].astype(int)
#
df_raw["Father's age"] = df_raw["Father's age"].fillna(round(df_raw["Father's age"].mean()))
df_raw["Father's age"] = df_raw["Father's age"].astype(int)

# Normal (30-60): 45.7 %, Tachypnea: 45.0 %, 9.3 % null
df_encoded['Respiratory Rate (breaths/min)'] = df_encoded['Respiratory Rate (breaths/min)'].fillna(pd.Series(np.random.choice(["Normal (30-60)", "Tachypnea"], len(df_encoded))))
df_encoded["Respiratory Rate (breaths/min)"] = df_encoded["Respiratory Rate (breaths/min)"].replace({'Tachypnea': 1, 'Normal (30-60)': 0})
#
df_raw['Respiratory Rate (breaths/min)'] = df_raw['Respiratory Rate (breaths/min)'].fillna(pd.Series(np.random.choice(["Normal (30-60)", "Tachypnea"], len(df_raw))))
df_raw["Respiratory Rate (breaths/min)"] = df_raw["Respiratory Rate (breaths/min)"].replace({'Tachypnea': 1, 'Normal (30-60)': 0})

# Normal: 46.3 %, Tachycardia: 44.7 %, 9.0 % null
df_encoded['Heart Rate (rates/min)'] = df_encoded['Heart Rate (rates/min)'].fillna(pd.Series(np.random.choice(["Normal", "Tachycardia"], len(df_encoded))))
df_encoded["Heart Rate (rates/min)"] = df_encoded["Heart Rate (rates/min)"].replace({'Tachycardia': 1, 'Normal': 0})
#
df_raw['Heart Rate (rates/min)'] = df_raw['Heart Rate (rates/min)'].fillna(pd.Series(np.random.choice(["Normal", "Tachycardia"], len(df_raw))))
df_raw["Heart Rate (rates/min)"] = df_raw["Heart Rate (rates/min)"].replace({'Tachycardia': 1, 'Normal': 0})

# High: 45.0 %, Low: 45.7 %, 9.3 % null
df_encoded['Follow-up'] = df_encoded['Follow-up'].fillna(pd.Series(np.random.choice(["High", "Low"], len(df_encoded))))
df_encoded["Follow-up"] = df_encoded["Follow-up"].replace({'High': 1, 'Low': 0})
#
df_raw['Follow-up'] = df_raw['Follow-up'].fillna(pd.Series(np.random.choice(["High", "Low"], len(df_raw))))
df_raw["Follow-up"] = df_raw["Follow-up"].replace({'High': 1, 'Low': 0})

# Yes: 45.8 %, No: 45.0 %, 9.2 % null
df_encoded['Folic acid details (peri-conceptional)'] = df_encoded['Folic acid details (peri-conceptional)'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded["Folic acid details (peri-conceptional)"] = df_encoded["Folic acid details (peri-conceptional)"].replace({'Yes': 1, 'No': 0})
#
df_raw['Folic acid details (peri-conceptional)'] = df_raw['Folic acid details (peri-conceptional)'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_raw))))
df_raw["Folic acid details (peri-conceptional)"] = df_raw["Folic acid details (peri-conceptional)"].replace({'Yes': 1, 'No': 0})

# Yes: 45.2 %, No: 45.6 %, 9.2 % null
df_encoded['H/O serious maternal illness'] = df_encoded['H/O serious maternal illness'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded["H/O serious maternal illness"] = df_encoded["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})
#
df_raw['H/O serious maternal illness'] = df_raw['H/O serious maternal illness'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_raw))))
df_raw["H/O serious maternal illness"] = df_raw["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})

# Yes: 45.5 %, No: 45.3 %, 9.2 % null
df_encoded['Assisted conception IVF/ART'] = df_encoded['Assisted conception IVF/ART'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded["Assisted conception IVF/ART"] = df_encoded["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})
#
df_raw['Assisted conception IVF/ART'] = df_raw['Assisted conception IVF/ART'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_raw))))
df_raw["Assisted conception IVF/ART"] = df_raw["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})

# Yes: 46.0 %, No: 44.8 %, 9.3 %
df_encoded['History of anomalies in previous pregnancies'] = df_encoded['History of anomalies in previous pregnancies'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded["History of anomalies in previous pregnancies"] = df_encoded["History of anomalies in previous pregnancies"].replace({'Yes': 1, 'No': 0})
#
df_raw['History of anomalies in previous pregnancies'] = df_raw['History of anomalies in previous pregnancies'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_raw))))
df_raw["History of anomalies in previous pregnancies"] = df_raw["History of anomalies in previous pregnancies"].replace({'Yes': 1, 'No': 0})

# 9.2 % null
df_encoded['No. of previous abortion'] = df_encoded['No. of previous abortion'].fillna(pd.Series(np.random.choice([0, 1, 2, 3, 4], len(df_encoded))))
df_encoded['No. of previous abortion'] = df_encoded['No. of previous abortion'].astype(int)
#
df_raw['No. of previous abortion'] = df_raw['No. of previous abortion'].fillna(pd.Series(np.random.choice([0, 1, 2, 3, 4], len(df_raw))))
df_raw['No. of previous abortion'] = df_raw['No. of previous abortion'].astype(int)

# Multiple: 45.3 %, Singular: 45.4 %, 9.3 % null
df_encoded['Birth defects'] = df_encoded['Birth defects'].fillna(pd.Series(np.random.choice(["Multiple", "Singular"], len(df_encoded))))
df_encoded["Birth defects"] = df_encoded["Birth defects"].replace({'Multiple': 1, 'Singular': 0})
#
df_raw['Birth defects'] = df_raw['Birth defects'].fillna(pd.Series(np.random.choice(["Multiple", "Singular"], len(df_raw))))
df_raw["Birth defects"] = df_raw["Birth defects"].replace({'Multiple': 1, 'Singular': 0})

# 9.4 % null, riempio con la media
df_encoded['White Blood cell count (thousand per microliter)'] = df_encoded['White Blood cell count (thousand per microliter)'].fillna(df_encoded['White Blood cell count (thousand per microliter)'].mean())
#
df_raw['White Blood cell count (thousand per microliter)'] = df_raw['White Blood cell count (thousand per microliter)'].fillna(df_raw['White Blood cell count (thousand per microliter)'].mean())

# 9.2 % null
df_encoded["Blood test result"] = df_encoded["Blood test result"].replace({'inconclusive': np.nan})
df_encoded['Blood test result'] = df_encoded['Blood test result'].fillna(pd.Series(np.random.choice(["normal", "slightly abnormal", "slightly abnormal"], len(df_encoded))))
df_encoded["Blood test result"] = df_encoded["Blood test result"].replace({'normal': 0, 'slightly abnormal': 1, 'abnormal': 2})
#
df_raw["Blood test result"] = df_raw["Blood test result"].replace({'inconclusive': np.nan})
df_raw['Blood test result'] = df_raw['Blood test result'].fillna(pd.Series(np.random.choice(["normal", "slightly abnormal", "abnormal"], len(df_raw))))
df_raw["Blood test result"] = df_raw["Blood test result"].replace({'normal': 0, 'slightly abnormal': 1, 'abnormal': 2})

# Symptom 1-5
for n in [1, 2, 3, 4, 5]:
    if df_encoded[f'Symptom {n}'].isna().sum() > 0:
        n_tot = len(df_encoded)
        n_1 = len(df_encoded[df_encoded[f'Symptom {n}'] == 1])
        p1 = n_1 / n_tot
        n_0 = len(df_encoded[df_encoded[f'Symptom {n}'] == 0])
        p0 = n_0 / n_tot
        pn = df_encoded[f'Symptom {n}'].isna().sum() / n_tot
        p1 += pn / 2
        p0 += pn / 2
        df_encoded[f'Symptom {n}'] = df_encoded[f'Symptom {n}'].fillna(
            pd.Series(np.random.choice([0, 1], len(df_encoded), p=[p0, p1])))
        #df_encoded = df_encoded.dropna(axis=0, subset=[f'Symptom {n}'])
        df_encoded[f'Symptom {n}'] = df_encoded[f'Symptom {n}'].astype(int)
#
for n in [1, 2, 3, 4, 5]:
    if df_raw[f'Symptom {n}'].isna().sum() > 0:
        n_tot = len(df_raw)
        n_1 = len(df_raw[df_raw[f'Symptom {n}'] == 1])
        p1 = n_1 / n_tot
        n_0 = len(df_raw[df_raw[f'Symptom {n}'] == 0])
        p0 = n_0 / n_tot
        pn = df_raw[f'Symptom {n}'].isna().sum() / n_tot
        p1 += pn / 2
        p0 += pn / 2
        df_raw[f'Symptom {n}'] = df_raw[f'Symptom {n}'].fillna(
            pd.Series(np.random.choice([0, 1], len(df_raw), p=[p0, p1])))
        #df_raw = df_raw.dropna(axis=0, subset=[f'Symptom {n}'])
        df_raw[f'Symptom {n}'] = df_raw[f'Symptom {n}'].astype(int)

# Ora che tutte le feature sono state tradotte in un valore numerico, guardiamo la
# matrice di correlazione
corr_mat = df_encoded.corr()
sns.heatmap(corr_mat)
plt.show()
# Notiamo che c'è pochissima correlazione tra le feature, tranne che per
# 'Assisted conception IVF/ART' e 'H/O serious maternal illness' che presentano un'altissima correlazione,
# per cui ne eliminiamo una.
df_encoded = df_encoded.drop(columns=['Assisted conception IVF/ART'])
deleted_cols.append('Assisted conception IVF/ART')

X = df_encoded.copy()

cols_to_normalize = ['Patient Age', "Father's age", "Mother's age", "White Blood cell count (thousand per microliter)"]
scaler = StandardScaler().fit(X[cols_to_normalize].astype('float64'))
X_numerical_std = pd.DataFrame(scaler.transform(X[cols_to_normalize].astype('float64')), columns=cols_to_normalize)

for col in cols_to_normalize:
    X[col] = X_numerical_std[col]


X.drop(columns=[target_variable])
y = df_encoded[target_variable]
x_tr, x_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=0)

x_tr.reset_index(drop=True, inplace=True)
x_t.reset_index(drop=True, inplace=True)
y_tr.reset_index(drop=True, inplace=True)
y_t.reset_index(drop=True, inplace=True)

X_raw = df_raw.copy()
X.drop(columns=[target_variable])
y_raw = df_raw[target_variable]
x_tr_raw, x_t_raw, y_tr_raw, y_t_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=0)

x_tr_raw.reset_index(drop=True, inplace=True)
x_t_raw.reset_index(drop=True, inplace=True)
y_tr_raw.reset_index(drop=True, inplace=True)
y_t_raw.reset_index(drop=True, inplace=True)


models = [
    {
        'name': "KNN",
        'model': KNeighborsClassifier(),
        'parameters': {'n_neighbors': list(range(1, 16, 2)), "weights": ["uniform", "distance"]}
    }, {
        'name': "KNN - raw",
        'model': KNeighborsClassifier(),
        'parameters': {'n_neighbors': list(range(1, 16, 2)), "weights": ["uniform", "distance"]}
    }, {
        'name': "Softmax Regression",
        'model': LogisticRegression(multi_class='multinomial', solver='saga'),
        'parameters': {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1], 'class_weight': [None, 'balanced']}
    }, {
        'name': "Softmax Regression - raw",
        'model': LogisticRegression(multi_class='multinomial', solver='saga'),
        'parameters': {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1], 'class_weight': [None, 'balanced']}
    }, {
        'name': "MLP",
        'model': MLPClassifier(early_stopping=True, solver='sgd', validation_fraction=0.125, max_iter=1000,
                               random_state=21),
        'parameters': {'hidden_layer_sizes': [(50, 50, 50, 50), (25, 25, 25, 25, 25, 25), (20, 20, 20, 20, 20, 20, 20)],
                       'activation': ['identity', 'tanh', 'logistic']},
    }, {
        'name': "MLP - raw",
        'model': MLPClassifier(early_stopping=True, solver='sgd', validation_fraction=0.125, max_iter=1000,
                               random_state=21),
        'parameters': {'hidden_layer_sizes': [(50, 50, 50, 50), (25, 25, 25, 25, 25, 25), (20, 20, 20, 20, 20, 20, 20)],
                       'activation': ['identity', 'tanh', 'logistic']},
    }
]

def get_metrics(y_test, y_predict):
    accuracy = round(accuracy_score(y_test, y_predict), 3)
    precision = round(precision_score(y_test, y_predict, average='weighted'), 3)
    recall = round(recall_score(y_test, y_predict, average='weighted'), 3)
    f1 = round(f1_score(y_test, y_predict, average='weighted'), 3)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_lb = lb.transform(y_test)
    y_predict_lb = lb.transform(y_predict)
    auc = round(roc_auc_score(y_test_lb, y_predict_lb, multi_class='ovr', average='weighted'), 3)
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1, 'AUC': auc}


for i, model in enumerate(models):

    start_timer_GS = time.perf_counter()
    gs = GridSearchCV(estimator=model['model'], param_grid=model['parameters'], scoring='f1_weighted', cv=5)
    if "raw" in model['name']:
        x_training = x_tr_raw
        y_training = y_tr_raw
        x_test = x_t_raw
        y_test = y_t_raw
    else:
        x_training = x_tr
        y_training = y_tr
        x_test = x_t
        y_test = y_t
    gs.fit(x_training, y_training)
    stop_timer_GS = time.perf_counter()
    time_GS = stop_timer_GS - start_timer_GS

    print(model['name'])
    for param in model['parameters'].keys():
        print(f'Best value for parameter {param}: ', gs.best_params_.get(param))
    print('\n\n')
    model['best_params'] = gs.best_params_.copy()

    start_timer_TR = time.perf_counter()
    model_top_params = gs.best_estimator_
    stop_timer_TR = time.perf_counter()
    time_TR = stop_timer_TR - start_timer_TR

    start_timer_TST = time.perf_counter()
    y_pred = model_top_params.predict(x_test)
    stop_timer_TST = time.perf_counter()
    time_TST = stop_timer_TST - start_timer_TST

    models[i]['time'] = {'model selection': time_GS, 'retraining': time_TR, 'test': time_TST}
    models[i]['results'] = get_metrics(y_test, y_pred)
    models[i]['top model'] = model_top_params

best_models_version = {}
print("\n\n")

for model in models:
    pp.pprint(model)
    print("\n\n")

    name = model['name'].split(' - ')[0]

    if name not in best_models_version.keys():
        best_models_version[name] = {'F1-score': model['results']['F1-score'], 'raw': False}
    else:
        if model['results']['F1-score'] > best_models_version[name]['F1-score']:
            best_models_version[name] = {'F1-score': model['results']['F1-score'], 'raw': True}
            print(f'For {name} the raw version is better')
        else:
            print(f'For {name} the preprocessed version is better')


models_with_bagging = {}
for model_name in best_models_version.keys():
    try:
        if best_models_version[model_name]['raw']:
            for i, model in enumerate(models):
                if model['name'] == f'{model_name} - raw':
                    best_model = model['top model']
                    old_top_model_name = f'{model_name} - raw'
                    old_top_model_index = i
        else:
            for i, model in enumerate(models):
                if model['name'] == model_name:
                    best_model = model['top model']
                    old_top_model_name = model_name
                    old_top_model_index = i
        model_with_bagging = BaggingClassifier(best_model, max_samples=0.9)
        if 'raw' in old_top_model_name:
            model_with_bagging.fit(x_tr_raw, y_tr_raw)
            y_pred_bagging = model_with_bagging.predict(x_t_raw)
            metrics_bagging = get_metrics(y_t_raw, y_pred_bagging)
        else:
            model_with_bagging.fit(x_tr, y_tr)
            y_pred_bagging = model_with_bagging.predict(x_t)
            metrics_bagging = get_metrics(y_t, y_pred_bagging)
        y = [models[old_top_model_index]['results']['F1-score'], metrics_bagging['F1-score']]
        sns.barplot(x=[f'base\n{y[0]}', f'ensemble\n{y[1]}'], y=y).set(title=old_top_model_name, ylabel="F1-score")

        models_with_bagging[old_top_model_name] = {'model': model_with_bagging, 'metrics': metrics_bagging}
        plt.show()
    except:
        print(f"Something gone wrong with {model_name} model :/")

models_json_writable = copy.deepcopy(models)
for model in models_json_writable:
    del model['model']
    del model['top model']
    del model['parameters']
with open("models_results.json", "w") as f:
    json.dump(models_json_writable, f, indent=4)

models_with_bagging_json_writable = copy.deepcopy(models)
for model in models_json_writable:
    del model['model']
with open("models_with_bagging_metrics.json", "w") as f:
    json.dump(models_with_bagging_json_writable, f, indent=4)
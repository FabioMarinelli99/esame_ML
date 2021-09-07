import warnings
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
from sklearn.model_selection import GridSearchCV,  train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing  import LabelBinarizer
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPRegressor


from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import time

# plot a distplot
def distplot(feature, frame, color='g'):
    plt.figure(figsize=(8, 3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color=color)
    plt.show()

pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv("train.csv")


col_to_predict = 'Disorder Subclass'
col_to_ignore = 'Genetic Disorder'

# col_to_predict = 'Genetic Disorder'
# col_to_ignore = 'Disorder Subclass'

df.drop(columns=[col_to_ignore], inplace=True)
df = df.dropna(axis=0, subset=[col_to_predict])

deleted_cols = [col_to_ignore, 'H/O substance abuse', 'H/O radiation exposure (x-ray)', 'Birth asphyxia', 'Gender', 'Status', 'Autopsy shows birth defect (if applicable)', 'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Patient Id', 'Patient First Name', 'Family Name', "Father's name", 'Institute Name', 'Location of Institute', 'Parental consent', 'Place of birth']
#deleted_cols.append('Blood cell count (mcL)')
categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in deleted_cols]
numerical_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in deleted_cols]
categorical_cols.remove(col_to_predict)
categorical_cols.remove('Heart Rate (rates/min')
categorical_cols.append('Heart Rate (rates/min)')


# nome colonna scritto male
columns = list(df.columns)
columns[columns.index('Heart Rate (rates/min')] = 'Heart Rate (rates/min)'
df.columns = columns

#useless columns
df = df.drop(columns=['Patient Id', 'Patient First Name', 'Family Name', "Father's name", 'Institute Name', 'Location of Institute', 'Parental consent', 'Place of birth'])
#parental consent era la maggior parte yes, il restante erano nulli
#place of birth è casa o ospedale

#useless columns
df = df.drop(columns=['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'])
#presentano valori tutti uguali

#1427 nulli
df['Patient Age'] = df['Patient Age'].fillna(round(df['Patient Age'].mean()))
df['Patient Age'] = df['Patient Age'].astype(int)

#60% y, 40% no
df["Genes in mother's side"] = df["Genes in mother's side"].replace({'Yes': 1, 'No': 0})

#39% y, 59% no, 1% null [302 rows] ---> le cancello, sono troppo poche per valer la pena di arrotondare il risultato
df = df.dropna(axis=0, subset=['Inherited from father'])
df["Inherited from father"] = df["Inherited from father"].replace({'Yes': 1, 'No': 0})

#48% y, 39% n, 13% null [2759 rows]
df['Maternal gene'] = df['Maternal gene'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df))))
#non so perche ma rimangono 36 valori nulli che non vengono rimpiazzati, cancello le righe
df = df.dropna(axis=0, subset=['Maternal gene'])
df["Maternal gene"] = df["Maternal gene"].replace({'Yes': 1, 'No': 0})

df["Paternal gene"] = df["Paternal gene"].replace({'Yes': 1, 'No': 0})

distplot('Blood cell count (mcL)', df[df['Gender'] == 'Male'])
distplot('Blood cell count (mcL)', df[df['Gender'] == 'Female'])
distplot('Blood cell count (mcL)', df[df['Gender'] == 'Ambiguous'])
#df = df.drop(columns=['Blood cell count (mcL)'])
##normal distribution, delete??

#current parents'age is probably not a data of interest. What could be of interest, instead, is their age when the child was born.
df["Mother's age"] = df["Mother's age"] - df["Patient Age"]
df["Mother's age"] = df["Mother's age"].fillna(round(df["Mother's age"].mean()))
df["Mother's age"] = df["Mother's age"].astype(int)
df["Father's age"] = df["Father's age"] - df["Patient Age"]
df["Father's age"] = df["Father's age"].fillna(round(df["Father's age"].mean()))
df["Father's age"] = df["Father's age"].astype(int)

# for disorder in df['Disorder Subclass'].unique():
#     print(f"{disorder} ALIVE:  {len(df[(df.Status == 'Alive') & (df['Disorder Subclass'] == disorder)])},av AGE: {df[(df.Status == 'Alive') & (df['Disorder Subclass'] == disorder)]['Patient Age'].mean()}")
#     print(f"{disorder} DEAD:   {len(df[(df.Status == 'Deceased') & (df['Disorder Subclass'] == disorder)])},av AGE: {df[(df.Status == 'Deceased') & (df['Disorder Subclass'] == disorder)]['Patient Age'].mean()}")
#     print('\n\n')
#n di malati morti e vivi della stessa malattia sono equivalenti, anche l'eta media
#decido di ignorare questo campo, anche perché fare predizioni su pazienti morti mi sembra poco utile
df = df.drop(columns=['Status', 'Autopsy shows birth defect (if applicable)'])

#1788 nulli
df['Respiratory Rate (breaths/min)'] = df['Respiratory Rate (breaths/min)'].fillna(pd.Series(np.random.choice(["Normal (30-60)", "Tachypnea"], len(df))))
#183 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['Respiratory Rate (breaths/min)'])
df["Respiratory Rate (breaths/min)"] = df["Respiratory Rate (breaths/min)"].replace({'Tachypnea': 1, 'Normal (30-60)': 0})

#1692 nulli
df['Heart Rate (rates/min)'] = df['Heart Rate (rates/min)'].fillna(pd.Series(np.random.choice(["Normal", "Tachycardia"], len(df))))
#165 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['Heart Rate (rates/min)'])
df["Heart Rate (rates/min)"] = df["Heart Rate (rates/min)"].replace({'Tachycardia': 1, 'Normal': 0})

df['Follow-up'] = df['Follow-up'].fillna(pd.Series(np.random.choice(["High", "Low"], len(df))))
#165 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['Follow-up'])
df["Follow-up"] = df["Follow-up"].replace({'High': 1, 'Low': 0})

#un terzo dei pazienti è di genere "Ambguous", cancello la colonna
df = df.drop(columns=['Gender'])

#piu di metà dei pazienti, non ha il valore di "birth asphyxia", cancello colonna
#alternativa potrebbe essere valutare di mettere "no" dove il valore manca
df = df.drop(columns=['Birth asphyxia'])

#Folic acid details 10% null, 45% n, 45%y
df['Folic acid details (peri-conceptional)'] = df['Folic acid details (peri-conceptional)'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df))))
#151 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['Folic acid details (peri-conceptional)'])
df["Folic acid details (peri-conceptional)"] = df["Folic acid details (peri-conceptional)"].replace({'Yes': 1, 'No': 0})

#H/O serious maternal illness 10% null, 45% no, 45% Yes
df['H/O serious maternal illness'] = df['H/O serious maternal illness'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df))))
#150 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['H/O serious maternal illness'])
df["H/O serious maternal illness"] = df["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})

#H/O radiation exposure (x-ray), meta dei dati non è valida, elimino colonna
df = df.drop(columns=['H/O radiation exposure (x-ray)'])

#H/O substance abuse, meta dei dati non è valida, elimino colonna
df = df.drop(columns=['H/O substance abuse'])

#Assisted conception IVF/ART 10% null, 45% no, 45% Yes
df['Assisted conception IVF/ART'] = df['Assisted conception IVF/ART'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df))))
#480 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['Assisted conception IVF/ART'])
df["Assisted conception IVF/ART"] = df["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})

#History of anomalies in previous pregnancies 10% null, 45% no, 45% Yes
df['History of anomalies in previous pregnancies'] = df['History of anomalies in previous pregnancies'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df))))
#125 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['History of anomalies in previous pregnancies'])
df["History of anomalies in previous pregnancies"] = df["History of anomalies in previous pregnancies"].replace({'Yes': 1, 'No': 0})

#No. of previous abortion circa 4k da 0 a 4, 1624 nulli
df['No. of previous abortion'] = df['No. of previous abortion'].fillna(pd.Series(np.random.choice([0, 1, 2, 3, 4], len(df))))
#133 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['No. of previous abortion'])
df['No. of previous abortion'] = df['No. of previous abortion'].astype(int)


#Birth defects 10% null, 45% no, 45% Yes
df['Birth defects'] = df['Birth defects'].fillna(pd.Series(np.random.choice(["Multiple", "Singular"], len(df))))
#119 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['Birth defects'])
df["Birth defects"] = df["Birth defects"].replace({'Multiple': 1, 'Singular': 0})

#White Blood cell count (thousand per microliter) 1593 normali
df['White Blood cell count (thousand per microliter)'] = df['White Blood cell count (thousand per microliter)'].fillna(df['White Blood cell count (thousand per microliter)'].mean())

#Blood test result, poco piu di un quarto dei dati non valido, per il resto diviso iun 3 livelli
df["Blood test result"] = df["Blood test result"].replace({'inconclusive': np.nan})
df['Blood test result'] = df['Blood test result'].fillna(pd.Series(np.random.choice(["normal", "slightly abnormal", "slightly abnormal"], len(df))))
#96 non vengono riempiti, elimino
df = df.dropna(axis=0, subset=['Blood test result'])
df["Blood test result"] = df["Blood test result"].replace({'normal': 0, 'slightly abnormal': 1, 'abnormal':2})

#Symptom 1-5
for n in [1, 2, 3, 4, 5]:
    if df[f'Symptom {n}'].isna().sum() > 0:
        n_tot = len(df)
        n_1 = len(df[df[f'Symptom {n}'] == 1])
        p1 = n_1/n_tot
        n_0 = len(df[df[f'Symptom {n}'] == 0])
        p0 = n_0/n_tot
        pn = df[f'Symptom {n}'].isna().sum() / n_tot
        p1 += pn / 2
        p0 += pn / 2
        df[f'Symptom {n}'] = df[f'Symptom {n}'].fillna(pd.Series(np.random.choice([0, 1], len(df), p=[p0, p1])))
        df = df.dropna(axis=0, subset=[f'Symptom {n}'])
        df[f'Symptom {n}'] = df[f'Symptom {n}'].astype(int)


#df_train = eda(df_train)

df_encoded = df.copy()
le_col_to_predict = LabelEncoder()
le_col_to_predict = le_col_to_predict.fit(df[col_to_predict])
df_encoded[col_to_predict] = le_col_to_predict.transform(df[col_to_predict])


X = df_encoded.copy()
X.drop(columns=[col_to_predict])
y = df_encoded[col_to_predict]
x_tr, x_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=0)

#standardizzazione
stdScal = StandardScaler()
stdScal.fit(x_tr[numerical_cols].astype('float64'))
X_train_scaled_num = pd.DataFrame(stdScal.transform(x_tr[numerical_cols].astype('float64')),
                                  columns=numerical_cols)
X_test_scaled_num = pd.DataFrame(stdScal.transform(x_t[numerical_cols].astype('float64')),
                                 columns=numerical_cols)

x_tr.reset_index(drop=True, inplace=True)
x_t.reset_index(drop=True, inplace=True)
y_tr.reset_index(drop=True, inplace=True)
y_t.reset_index(drop=True, inplace=True)

#dati scalati
# X_train_scaled = pd.concat([X_train_scaled_num, x_tr[categorical_cols]], axis=1)
# X_test_scaled = pd.concat([X_test_scaled_num, x_t[categorical_cols]], axis=1)

#dati NON scalati
X_train_scaled = x_tr
X_test_scaled = x_t

#################################################################################
#                                   SVC                                         #
#################################################################################


svc_model = SVC()
#svc_model.fit(X_train_scaled, y_tr)

#'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]
parameters = {"kernel": ["linear", "poly", "rbf", "sigmoid"], 'C': [1]}
model = svc_model
start_timer = time.perf_counter()
gs_svc = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=5)
gs_svc.fit(X_train_scaled, y_tr)
stop_timer = time.perf_counter()
total_time = stop_timer - start_timer
print('SVC')
print('The best value for parameter kernel is', gs_svc.best_params_.get('kernel'))
print('The best value for parameter C is', gs_svc.best_params_.get('C'))
print('F1-score = ', gs_svc.best_score_)


start_timer = time.perf_counter()
svc_model_top_params = gs_svc.best_estimator_
stop_timer = time.perf_counter()
total_time = stop_timer - start_timer
start_timer = time.perf_counter()
y_pred_svc = svc_model_top_params.predict(X_test_scaled)
stop_timer = time.perf_counter()
total_time = stop_timer - start_timer

ct = 0
for i, val in enumerate(y_pred_svc):
    if val == y_t[i]:
        ct += 1
print(round((ct/len(y_t))*10000)/100, " %")
print(ct/len(y_t))
#################################################################################



#################################################################################
#                       KNN
#################################################################################

# knn_model = KNeighborsClassifier()
# #CV
# parameters = {'n_neighbors': list(range(1, 16, 2)), "weights": ["uniform", "distance"]}
# model = knn_model
# # specifying cv parameter we use stratified kfold, it is good for unbalanced dataset
# # our dataset is quite balanced so I don't specify any value for cv, automatically it is setted for 5 folds
# start_timer = time.perf_counter()
# gd_knn = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=5)
# # fit on train-set, so we split train-set (80%) into (another) train-set and validation-set (4 folds for train and 1 fold for validation)
# gd_knn.fit(X_train_scaled, y_tr)
# stop_timer = time.perf_counter()
# total_time = stop_timer - start_timer
# print('K-Nearest Neighbor')
# print('The best value for parameter k is', gd_knn.best_params_.get('n_neighbors'))
# print('The best value for parameter weights is', gd_knn.best_params_.get('weights'))
# print('F1-score = ', gd_knn.best_score_)
# knn_best_params = gd_knn.best_params_.copy()
#
# #test
# start_timer = time.perf_counter()
# knn_model_top_params = gd_knn.best_estimator_
# stop_timer = time.perf_counter()
# total_time = stop_timer - start_timer
# start_timer = time.perf_counter()
# y_pred_knn = knn_model_top_params.predict(X_test_scaled)
# stop_timer = time.perf_counter()
# total_time = stop_timer - start_timer
#
#
# ct = 0
# for i, val in enumerate(y_pred_knn):
#     if val == y_t[i]:
#         ct += 1
# print(round((ct/len(y_t))*10000)/100, " %")

#################################################################################

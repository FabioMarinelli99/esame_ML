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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import time

pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("train.csv")

dim = {'normal': {'h': 3, 'asp': 1.2}, 'big': {'h': 8, 'asp': 0.5}}

# plot a distribution plot
def displot(title, feature, frame, size="normal", color='g'):
    global dim
    h = dim[size]['h']
    asp = dim[size]['asp']
    g = sns.displot(frame, x=feature, color=color, height=h, aspect=asp)
    if title != 'notitle':
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.9)
    plt.show()


# plot a categorical plot
def catplot(title, feature, df, size="normal"):
    try:
        global dim
        h = dim[size]['h']
        asp = dim[size]['asp']
        plt.title(title)
        g = sns.catplot(y=feature, kind="count", hue=target_variable, height=h, aspect=asp, data=df, orient='h', palette=sns.color_palette(['green', 'gray', 'yellow', 'purple', 'black', 'fuchsia', 'orange', 'blue', 'red', 'brown']))
        if title != 'notitle':
            g.fig.suptitle(title)
            g.fig.subplots_adjust(top=0.9)
        plt.show()
    except:
        print('---------> ' + title)


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

sns.catplot(y=target_variable, kind="count", data=df, height=2.8, aspect=2.5, orient='h')
# Il dataset non è bilanciato, per alcune malattie abbiamo molti più sample di altre.

first_evidence = []
for col in df.columns:
    unique_val = df[col].unique()
    unq = ""
    if len(unique_val) > 10:
        unq += f" > 10\n NULL: {round(df[col].isna().sum() * 1000 / len(df))/10} %"
    else:
        for val in unique_val:
            if type(val) != str:
                if np.isnan(val):
                    unq = f" NULL: {round(df[col].isna().sum() * 1000 / len(df))/10} %\n" + unq
                else:
                    unq += f" {val}: {round(len(df[df[col] == val]) * 1000 / len(df))/10} %\n"
            else:
                unq += f" {val}: {round(len(df[df[col] == val]) * 1000 / len(df))/10} %\n"
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
    displot(f"White Blood cell count - {disorder}", "White Blood cell count (thousand per microliter)", df[df[target_variable] == disorder])

catplot('notitle', 'Blood test result', df)

for x in range(1, 6):
    catplot("notitle", f"Symptom {x}", df)

# Non ci sono particolari evidenze


#############################################################
#                       Preprocessing                       #
#############################################################
df_encoded = df.copy()
le_target_variable = LabelEncoder()
le_target_variable = le_target_variable.fit(df[target_variable])
df_encoded[target_variable] = le_target_variable.transform(df[target_variable])

deleted_cols = []

# nome colonna scritto male
columns = list(df_encoded.columns)
columns[columns.index('Heart Rate (rates/min')] = 'Heart Rate (rates/min)'
df_encoded.columns = columns

# Colonna target secondaria
deleted_cols.append(col_to_ignore)

# Colonne concettualmente inutili
cols = ['Patient Id', 'Patient First Name', 'Family Name', "Father's name", 'Institute Name', 'Location of Institute', 'Place of birth']
df_encoded = df_encoded.drop(columns=cols)
deleted_cols.extend(cols)
#parental consent era la maggior parte yes, il restante erano nulli

# Colonne inutili perche con valori univoci o nulli.
cols = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Parental consent']
df_encoded = df_encoded.drop(columns=cols)
deleted_cols.extend(cols)

# Colonne rimovibili
cols = ['Status', 'Autopsy shows birth defect (if applicable)', 'Blood cell count (mcL)']
df_encoded = df_encoded.drop(columns=cols)
deleted_cols.extend(cols)

# Colonne inutilizzabili
cols = ['Gender', 'Birth asphyxia', 'H/O radiation exposure (x-ray)', 'H/O substance abuse']
df_encoded = df_encoded.drop(columns=cols)
deleted_cols.extend(cols)

# 6.2 % nulli, riempio con la media dell'età
df_encoded['Patient Age'] = df_encoded['Patient Age'].fillna(round(df_encoded['Patient Age'].mean()))
df_encoded['Patient Age'] = df_encoded['Patient Age'].astype(int)

# Yes: 59.6 %, No: 40.4 %
df_encoded["Genes in mother's side"] = df_encoded["Genes in mother's side"].replace({'Yes': 1, 'No': 0})

# Yes: 39.1 %, No: 59.6 %, 1 % null [302 rows] ---> le cancello, sono troppo poche per valer la pena di
# arrotondare la precisione del risultato finale.
df_encoded = df_encoded.dropna(axis=0, subset=['Inherited from father'])
df_encoded["Inherited from father"] = df_encoded["Inherited from father"].replace({'Yes': 1, 'No': 0})

# Yes: 48.4 %, No: 39.5%, 13% null [2759 rows], riempio in modo casuale
df_encoded['Maternal gene'] = df_encoded['Maternal gene'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Maternal gene'])
df_encoded["Maternal gene"] = df_encoded["Maternal gene"].replace({'Yes': 1, 'No': 0})

# Yes: 43.3 %, No: 56.7 %
df_encoded["Paternal gene"] = df_encoded["Paternal gene"].replace({'Yes': 1, 'No': 0})

# L'età attuale dei genitori non è un dato di interesse. Potrebbe però essere un dato utile sapere quanti anni
# avevano alla nascità del figlio.
# 26.1 % null
df_encoded["Mother's age"] = df_encoded["Mother's age"] - df_encoded["Patient Age"]
df_encoded["Mother's age"] = df_encoded["Mother's age"].fillna(round(df_encoded["Mother's age"].mean()))
df_encoded["Mother's age"] = df_encoded["Mother's age"].astype(int)
# 25.7 % null
df_encoded["Father's age"] = df_encoded["Father's age"] - df_encoded["Patient Age"]
df_encoded["Father's age"] = df_encoded["Father's age"].fillna(round(df_encoded["Father's age"].mean()))
df_encoded["Father's age"] = df_encoded["Father's age"].astype(int)

# Normal (30-60): 45.7 %, Tachypnea: 45.0 %, 9.3 % null
df_encoded['Respiratory Rate (breaths/min)'] = df_encoded['Respiratory Rate (breaths/min)'].fillna(pd.Series(np.random.choice(["Normal (30-60)", "Tachypnea"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Respiratory Rate (breaths/min)'])
df_encoded["Respiratory Rate (breaths/min)"] = df_encoded["Respiratory Rate (breaths/min)"].replace({'Tachypnea': 1, 'Normal (30-60)': 0})

# Normal: 46.3 %, Tachycardia: 44.7 %, 9.0 % null
df_encoded['Heart Rate (rates/min)'] = df_encoded['Heart Rate (rates/min)'].fillna(pd.Series(np.random.choice(["Normal", "Tachycardia"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Heart Rate (rates/min)'])
df_encoded["Heart Rate (rates/min)"] = df_encoded["Heart Rate (rates/min)"].replace({'Tachycardia': 1, 'Normal': 0})

# High: 45.0 %, Low: 45.7 %, 9.3 % null
df_encoded['Follow-up'] = df_encoded['Follow-up'].fillna(pd.Series(np.random.choice(["High", "Low"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Follow-up'])
df_encoded["Follow-up"] = df_encoded["Follow-up"].replace({'High': 1, 'Low': 0})

# Yes: 45.8 %, No: 45.0 %, 9.2 % null
df_encoded['Folic acid details (peri-conceptional)'] = df_encoded['Folic acid details (peri-conceptional)'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Folic acid details (peri-conceptional)'])
df_encoded["Folic acid details (peri-conceptional)"] = df_encoded["Folic acid details (peri-conceptional)"].replace({'Yes': 1, 'No': 0})

# Yes: 45.2 %, No: 45.6 %, 9.2 % null
df_encoded['H/O serious maternal illness'] = df_encoded['H/O serious maternal illness'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['H/O serious maternal illness'])
df_encoded["H/O serious maternal illness"] = df_encoded["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})

# Yes: 45.5 %, No: 45.3 %, 9.2 % null
df_encoded['Assisted conception IVF/ART'] = df_encoded['Assisted conception IVF/ART'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Assisted conception IVF/ART'])
df_encoded["Assisted conception IVF/ART"] = df_encoded["H/O serious maternal illness"].replace({'Yes': 1, 'No': 0})

# Yes: 46.0 %, No: 44.8 %, 9.3 %
df_encoded['History of anomalies in previous pregnancies'] = df_encoded['History of anomalies in previous pregnancies'].fillna(pd.Series(np.random.choice(["Yes", "No"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['History of anomalies in previous pregnancies'])
df_encoded["History of anomalies in previous pregnancies"] = df_encoded["History of anomalies in previous pregnancies"].replace({'Yes': 1, 'No': 0})

# 9.2 % null
df_encoded['No. of previous abortion'] = df_encoded['No. of previous abortion'].fillna(pd.Series(np.random.choice([0, 1, 2, 3, 4], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['No. of previous abortion'])
df_encoded['No. of previous abortion'] = df_encoded['No. of previous abortion'].astype(int)


# Multiple: 45.3 %, Singular: 45.4 %, 9.3 % null
df_encoded['Birth defects'] = df_encoded['Birth defects'].fillna(pd.Series(np.random.choice(["Multiple", "Singular"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Birth defects'])
df_encoded["Birth defects"] = df_encoded["Birth defects"].replace({'Multiple': 1, 'Singular': 0})

# 9.4 % null, riempio con la media
df_encoded['White Blood cell count (thousand per microliter)'] = df_encoded['White Blood cell count (thousand per microliter)'].fillna(df_encoded['White Blood cell count (thousand per microliter)'].mean())

# 9.2 % null
df_encoded["Blood test result"] = df_encoded["Blood test result"].replace({'inconclusive': np.nan})
df_encoded['Blood test result'] = df_encoded['Blood test result'].fillna(pd.Series(np.random.choice(["normal", "slightly abnormal", "slightly abnormal"], len(df_encoded))))
df_encoded = df_encoded.dropna(axis=0, subset=['Blood test result'])
df_encoded["Blood test result"] = df_encoded["Blood test result"].replace({'normal': 0, 'slightly abnormal': 1, 'abnormal':2})

# Symptom 1-5
for n in [1, 2, 3, 4, 5]:
    if df_encoded[f'Symptom {n}'].isna().sum() > 0:
        n_tot = len(df_encoded)
        n_1 = len(df_encoded[df_encoded[f'Symptom {n}'] == 1])
        p1 = n_1/n_tot
        n_0 = len(df_encoded[df_encoded[f'Symptom {n}'] == 0])
        p0 = n_0/n_tot
        pn = df_encoded[f'Symptom {n}'].isna().sum() / n_tot
        p1 += pn / 2
        p0 += pn / 2
        df_encoded[f'Symptom {n}'] = df_encoded[f'Symptom {n}'].fillna(pd.Series(np.random.choice([0, 1], len(df_encoded), p=[p0, p1])))
        df_encoded = df_encoded.dropna(axis=0, subset=[f'Symptom {n}'])
        df_encoded[f'Symptom {n}'] = df_encoded[f'Symptom {n}'].astype(int)

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


categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in deleted_cols]
numerical_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in deleted_cols]
categorical_cols.remove('Heart Rate (rates/min')
categorical_cols.append('Heart Rate (rates/min)')


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################



X = df_encoded.copy()
X.drop(columns=[target_variable])
y = df_encoded[target_variable]
x_tr, x_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=0)

# #standardizzazione
# stdScal = StandardScaler()
# stdScal.fit(x_tr[numerical_cols].astype('float64'))
# X_train_scaled_num = pd.DataFrame(stdScal.transform(x_tr[numerical_cols].astype('float64')),
#                                   columns=numerical_cols)
# X_test_scaled_num = pd.DataFrame(stdScal.transform(x_t[numerical_cols].astype('float64')),
#                                  columns=numerical_cols)
#
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
#"kernel": ["linear", "poly", "rbf", "sigmoid"]
parameters = {"kernel": ["linear"], 'C': [1]}
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

print(f"F1-score in predizione: {f1_score(y_t, y_pred_svc, average='weighted')}")
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
#
# print("F1-score in predizione", f1_score(y_t, y_pred_knn, average='weighted'))
#################################################################################


#################################################################################
#                       MLP
#################################################################################

# random_state = 17
# np.random.seed(17)
#
# def cv_model(model, h_params, X_train, y_train):
#     model_cv = GridSearchCV(estimator=model, param_grid=h_params, scoring='neg_root_mean_squared_error',
#                             return_train_score=True)
#     print('Fitting all models ...  ...  ...')
#     model_cv.fit(X_train, y_train)
#     return model_cv
#
# mlp_std_model = MLPClassifier(early_stopping=True,
#                              solver='sgd',
#                              validation_fraction=0.125,
#                              # perchè il 10% del dataset (per la validation) equivale al 12.5% del training set
#                              max_iter=1000,
#                              random_state=random_state)
#
# h_layers = [(500,), (200,), (200, 200), (100, 100), (200, 100, 100), (50, 50, 50, 50), (25, 25, 25, 25, 25, 25)]
# #(25, 25, 25, 25, 25, 25)
# parameters = {'hidden_layer_sizes': h_layers, 'activation': ['identity', 'tanh', 'relu', 'logistic']}
# #identity
#
# mlp_std = cv_model(mlp_std_model, parameters, X_train_scaled, y_tr)
# mlp_ensemble = BaggingClassifier(mlp_std.best_estimator_)
#
# official_mlp_std = mlp_std.best_estimator_
#
# # train
# start = time.time()
# official_mlp_std.fit(X_train_scaled, y_tr)
# mlp_std_train_time = time.time() - start
#
# # predict
# y_pred_mlp_std = official_mlp_std.predict(X_test_scaled)
#
# ct = 0
# for i, val in enumerate(y_pred_mlp_std):
#     if val == y_t[i]:
#         ct += 1
# print(round((ct/len(y_t))*10000)/100, " %")
#
# print("F1-score in predizione", f1_score(y_t, y_pred_mlp_std, average='weighted'))
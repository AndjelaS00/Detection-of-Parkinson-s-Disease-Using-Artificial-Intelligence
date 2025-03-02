# UCITAVANJE BIBLIOTEKA
import numpy as np  # Služi za rad sa numeričkim podacima i nizovima.
import pandas as pd  # Služi za rad sa podacima u tabelarnom obliku.
from sklearn.model_selection import train_test_split  # Za deljenje podataka na trening i test skupove.
from sklearn.preprocessing import StandardScaler  # Za standardizaciju podataka.
from sklearn import svm  # Sadrži implementaciju Support Vector Machine algoritma.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Za evaluaciju performansi modela.
from sklearn.linear_model import LogisticRegression  # Implementacija logističke regresije.
from sklearn.neighbors import KNeighborsClassifier  # Implementacija K-nearest neighbors algoritma.
from sklearn.linear_model import Perceptron  # Implementacija perceptrona.
import matplotlib.pyplot as plt  # Za grafički prikaz podataka.

# UPRAVLJANJE PODACIMA
# Ucitavanje podataka iz CSV fajla u Panda DataFrame
podaci = pd.read_csv('parkinsons.csv')

print(podaci.head()) # Prikaz prvih pet redova tabele podataka koje imamo
print(podaci.shape) # Prikaz dimenzija tabele
print(podaci.info()) # Informacije o datasetu
print(podaci.isnull().sum())# Broj nedostajućih vrednosti po kolonama
print(podaci.describe()) # Osnovna statistika dataseta
print(podaci['status'].value_counts()) # Distribucija ciljne varijable 'status'

# DATA PRE-PROCESSING
A = podaci.drop(columns=['name', 'status'], axis=1)
B = podaci['status']
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.25, random_state=3)

# STANDARDIZOVANJE PODATAKA
s = StandardScaler()
s.fit(A_train)
A_train = s.transform(A_train)
A_test = s.transform(A_test)

# FUNKCIJA ZA IZRAČUNAVANJE SPECIFIČNOSTI
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# TRENIRANJE MODELA I EVALUACIJA
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': svm.SVC(kernel='linear'),
    'Perceptron': Perceptron(max_iter=1000)
}

metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score,
    'Specificity': specificity_score
}

results = {model_name: {} for model_name in models}

for model_name, model in models.items():
    model.fit(A_train, B_train)
    train_pred = model.predict(A_train)
    test_pred = model.predict(A_test)
    
    for metric_name, metric in metrics.items():
        train_score = metric(B_train, train_pred)
        test_score = metric(B_test, test_pred)
        results[model_name][f'Train {metric_name}'] = train_score
        results[model_name][f'Test {metric_name}'] = test_score

# ŠTAMPANJE REZULTATA
for model_name, scores in results.items():
    print(f'{model_name}:')
    for metric, score in scores.items():
        print(f'  {metric}: {score:.4f}')
    print()

# GRAFIČKI PRIKAZ REZULTATA
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
for metric_name in metrics_list:
    train_scores = [results[model][f'Train {metric_name}'] for model in models]
    test_scores = [results[model][f'Test {metric_name}'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_scores, width, label='Train')
    plt.bar(x + width/2, test_scores, width, label='Test')

    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} of Different Models')
    plt.xticks(x, models.keys())
    plt.legend()
    plt.show()

# Building a Predictive System
provera = np.asarray([162.56800, 198.34600, 77.63000, 0.00502, 0.00003, 0.00280, 0.00253, 0.00841, 
                      0.01791, 0.16800, 0.00793, 0.01057, 0.01799, 0.02380, 0.01170, 25.67800, 
                      0.427785, 0.723797, -6.635729, 0.209866, 1.957961, 0.135242])

# Primenjujemo imena kolona na podatke koje proveravamo
provera_df = pd.DataFrame([provera], columns=A.columns)

# Standardizacija podataka za predikciju
standardizovano = s.transform(provera_df)

# Predikcija i ispis za svaki model
def predikcija(model, naziv_modela):
    predvidjanje = model.predict(standardizovano)
    if predvidjanje[0] == 1:
        print(f"Ova osoba ima Parkinsonovu bolest prema modelu {naziv_modela}.")
    else:
        print(f"Ova osoba nema Parkinsonovu bolest prema modelu {naziv_modela}.")

for model_name, model in models.items():
    predikcija(model, model_name)

import pandas as pd
pd.set_option('display.float_format', '{:.6f}'.format)

import numpy as np
np.set_printoptions(precision=6, suppress=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Vamos a crear un modelo que pronostique si el estudiante va a pasar o no el examen de matematicas,
# con base en las otras variables (features)

url = 'https://raw.githubusercontent.com/edroga/Datasets_for_projects/main/StudentsPerformance.csv'
students = pd.read_csv(url)
print(students.head())

#para conocer el tipo de dato de un objeto en python usamos la funbción type()
# para conocer el tipo de dato de cada columna en un dataframe usamos el atributo dtypes
print(students.dtypes)

#Creamos una copia del DF para evitar modificar el original, renombramos las columnas, reemplazando espacios
#y barras inclinadas con guiones bajos, binarizamos variables con one hot coding (dummies) y eliminamos
#la primera columna de cada variable codificada para evitar la multicolinealidad. las columnas del DataFrame
# resultante en minúsculas y eliminando espacios y comillas. Agregamos columna Target,
#asignamos valores a la columna target en función de la condición math_score >= 70 y por último
# eliminamos la columna math_score

students_clean = (students
                  .copy()
                  .rename(columns = lambda x: x.replace(' ', '_').replace('/', '_'))
                  .pipe(lambda df_: pd.get_dummies(df_,
                                                   columns = ['gender',
                                                              'race_ethnicity',
                                                              'parental_level_of_education',
                                                              'lunch',
                                                              'test_preparation_course'],
                                                   drop_first=True
                                                   ))
                  .rename(columns = lambda x: x.lower().replace(' ', '_').replace("'", ''))
                  .assign(target = lambda x: np.where(x['math_score']>=70, 1, 0))
                  .drop(columns = ['math_score'])
                  )

print(students_clean.head(3))

#hacemos nuestras variables para predecir X y Y

X = students_clean.drop(columns = 'target')
y = students_clean['target']

# instanciamos el modelo
logit = LogisticRegression(max_iter=700)

# Ajustamos el modelo
logit.fit(X, y)

# Primero vamos a ver las probabilidades
probas = logit.predict_proba(X)

print(probas)
# primera columna es la probabilidad de la clase negativa
# segunda columna es la probabilidad de la clase positiva

proba_clase_positiva = probas[:, 1]
print(proba_clase_positiva)

# Si queremos ver el resultado con el punto de corte por defecto que es .5
# usamos el método predict

y_pred = logit.predict(X)
print(y_pred)

#METRICAS PARA REGRESION LINEAL

print('accuracy: ', accuracy_score(y, y_pred) )
print('recall: ', recall_score(y, y_pred))
print('precision: ', precision_score(y, y_pred))
print('f1: ', f1_score(y, y_pred))


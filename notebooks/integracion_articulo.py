# %%


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../../data/Accidentes_2013_2023.csv')

# Reemplazar los NaN en 'fall_les' con la categoría 'Ignorado'
df['fall_les'] = df['fall_les'].fillna('Ignorado')

# Confirmar la nueva distribución de clases en la variable objetivo
fall_les_distribution = df['fall_les'].value_counts(dropna=False)
print("\nDistribución de clases en 'fall_les' después de la limpieza:\n", fall_les_distribution)

# %%
# Separar X e y
X = df.drop(columns=['fall_les'])
y = df['fall_les']

# One-hot encoding de variables categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Codificar clase objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)


# %%
# Crear el modelo RandomForest
rf_model = RandomForestClassifier(
    n_estimators=100,  # Número de árboles
    max_depth=None,    # Profundidad máxima (None = sin límite)
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # Número de características a considerar en cada división
    bootstrap=True,
    random_state=42
)


# %%
# 2. Entrenar el modelo
rf_model.fit(X_train, y_train)


# %%
# Evaluar el modelo
# Predicciones en conjunto de prueba
y_pred = rf_model.predict(X_test)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Exactitud del modelo: {accuracy:.4f}")
print("\nMatriz de confusión:")
print(conf_matrix)
print("\nInforme de clasificación:")
print(classification_rep)

# %%
# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"\nExactitud con validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %%
# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de confusión - Random Forest')
plt.tight_layout()
plt.show()

# %%
# Obtener las 20 características más importantes
feature_names = X_encoded.columns
feature_importance = rf_model.feature_importances_

# Crear DataFrame para visualizar
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Ordenar por importancia descendente
importance_df = importance_df.sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 20 características más importantes')
plt.tight_layout()
plt.show()


# %%


# Definir espacio de hiperparámetros a explorar
param_grid = {
    'n_estimators': [25, 50, 100],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1 
)

grid_search.fit(X_train, y_train)

print("\nMejores hiperparámetros:")
print(grid_search.best_params_)
print(f"Mejor exactitud: {grid_search.best_score_:.4f}")

# Usar el mejor modelo
best_rf_model = grid_search.best_estimator_

# %%
# Crear el modelo RandomForest optimizado

rf_model = RandomForestClassifier(
    n_estimators=100,  # Número de árboles
    max_depth=None,    # Profundidad máxima (None = sin límite)
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='sqrt',  # Número de características a considerar en cada división
    bootstrap=True,
    random_state=42
)


# %%
# 2. Entrenar el modelo
rf_model.fit(X_train, y_train)


# %%
# Evaluar el modelo optimizado
# Predicciones en conjunto de prueba
y_pred = rf_model.predict(X_test)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Exactitud del modelo: {accuracy:.4f}")
print("\nMatriz de confusión:")
print(conf_matrix)
print("\nInforme de clasificación:")
print(classification_rep)

# %%
# Validación cruzada del modelo optimizado
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"\nExactitud con validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %%
# Visualización de la matriz de confusión del modelo optimizado
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de confusión - Random Forest')
plt.tight_layout()
plt.show()

# %%
# %%

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
import random
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.compose import make_column_selector as selector #Para seleccionar de forma automática las variables numéricas y categóricas
from sklearn.preprocessing import OneHotEncoder #Para codificar las variables categóricas usando dummies
from sklearn.preprocessing import StandardScaler #Para normalizar las variables numéricas
from sklearn.compose import ColumnTransformer #Modifica las columnas usando los preprocesadores
from sklearn.pipeline import make_pipeline #Planifica una secuencia de procesos
from sklearn import set_config #Para mostrar graficamente el pipeline
from sklearn.model_selection import GridSearchCV

set_config(display='diagram')
#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

df = pd.read_csv('../../data/Accidentes_2013_2023.csv')
df

# %% [markdown]
# # Limpieza de datos

# %%
# Reemplazar NaN por 'Ignorado' en la columna 'fall_les'
df['fall_les'] = df['fall_les'].fillna('Ignorado')

# Confirmar que ya no hay NaNs
print("Valores faltantes por columna después de reemplazar:")
print(df.isna().sum())

print("\nDistribución de clases en 'fall_les':")
print(df['fall_les'].value_counts())

# %%
y = df.pop("fall_les")

print(y)
print(df.head())

# %%
# Evaluar desbalance de clases
plt.figure(figsize=(8, 4))
sns.countplot(y)
plt.title("Distribución de clases")
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# # Entrenamiento y Pruebas

# %%
X = df #El resto de los datos

# Dividir train/test
random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Columnas categóricas y numéricas
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

# Preprocesador
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])


# %%
print("Clases en entrenamiento:")
print(y_train.value_counts(normalize=True))

# %%
X_train.info()

# %%
X_test.info()

# %% [markdown]
# # Buscar mejores parametros para las redes neuronales

# %% [markdown]
# ### Al estar desbalanceados los datos, les haremos un sampling solo al conjunto de entrenamiento, utilizando SMOTE (Syntethic Minority Over-sampling)

# %% [markdown]
# - Oversampling sintético: SMOTE generó datos sintéticos (nuevos ejemplos artificiales) para las clases minoritarias ("Fallecido" e "Ignorado") hasta que tuvieron la misma cantidad que la clase mayoritaria ("Lesionado").
# - El algoritmo no copia datos existentes, sino que interpola entre vecinos más cercanos en el espacio de características para crear nuevas muestras plausibles de esas clases.

# %%
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Pipeline completo con SMOTE
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('mlpclassifier', MLPClassifier())
])

# 5. Grid de parámetros
parametros_tun = {
    'mlpclassifier__max_iter': [100, 200, 500],
    'mlpclassifier__learning_rate_init': [0.01, 0.1, 1],
    'mlpclassifier__hidden_layer_sizes': [(3, 2), (5, 3), (10, 2)]
}

# 6. GridSearchCV
model_grid_search = GridSearchCV(pipeline, param_grid=parametros_tun, n_jobs=2, cv=10, verbose=1)
model_grid_search.fit(X_train, y_train)

# 7. Mejor modelo y predicciones
print("Mejores parámetros:", model_grid_search.best_params_)

# %% [markdown]
# # Verificacion del sampling

# %%
best_model = model_grid_search.best_estimator_

X_train_preprocessed = best_model.named_steps['preprocessor'].fit_transform(X_train)
y_train_array = np.array(y_train)

X_smote, y_smote = best_model.named_steps['smote'].fit_resample(X_train_preprocessed, y_train_array)

# Revisar balance resultante
print("Distribución después de SMOTE dentro del pipeline:")
print(pd.Series(y_smote).value_counts())

# %% [markdown]
# ## Validacion cruzada

# %%
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}

cv_results = cross_validate(best_model, X, y, cv=cv, scoring=scoring)

print("\nValidación cruzada (10-fold):")
print("Accuracy promedio:", np.mean(cv_results['test_accuracy']))
print("Precision promedio:", np.mean(cv_results['test_precision_macro']))
print("Recall promedio:", np.mean(cv_results['test_recall_macro']))
print("F1-score promedio:", np.mean(cv_results['test_f1_macro']))

# %% [markdown]
# ## Entrenarlo

# %%
y_pred = best_model.predict(X_test)
print(y_pred)

# %% [markdown]
# # Curva de perdida

# %%
plt.plot(best_model.named_steps['mlpclassifier'].loss_curve_)
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.grid()
plt.show()

# %% [markdown]
# # Matriz de confusion

# %%
cm = confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy: ',accuracy, '\n')
print(classification_report(y_test, y_pred))

labels = best_model.classes_  # Obtiene los nombres de las clases desde el modelo

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusión')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.tight_layout()
plt.show()

# %%
# Evaluación del fitting: comparar accuracy en train y test
y_train_pred = best_model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Accuracy en entrenamiento:", train_acc)
print("Accuracy en test:", test_acc)
diff = abs(train_acc - test_acc)

# Evaluar si hay underfitting o overfitting
if train_acc < 0.7 and test_acc < 0.7:
    print("❗ Underfitting: El modelo tiene bajo rendimiento tanto en entrenamiento como en prueba.")
elif train_acc - test_acc > 0.1:
    print("⚠️ Overfitting: El modelo se ajusta demasiado a los datos de entrenamiento y no generaliza bien.")
else:
    print("✅ Buen ajuste: El modelo generaliza correctamente con una diferencia razonable entre entrenamiento y prueba.")

# %%
from collections import defaultdict

# Obtener nombres completos de features (numéricas + dummies)
dummy_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
feature_names = num_cols + list(dummy_names)

# Obtener pesos de la primera capa
coefs = best_model.named_steps['mlpclassifier'].coefs_[0]

# Calcular media absoluta de los pesos por feature individual
feature_weights = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_weight': np.abs(coefs).mean(axis=1)
})

# Agrupar por columna original
col_importance = defaultdict(list)

for feat, weight in zip(feature_weights['feature'], feature_weights['mean_abs_weight']):
    for col in cat_cols:
        if feat.startswith(col + '_'):
            col_importance[col].append(weight)
            break
    else:
        col_importance[feat].append(weight)  # numéricas

# Promediar por columna original
importancia_columnas = pd.DataFrame({
    'columna': list(col_importance.keys()),
    'importancia_promedio': [np.mean(v) for v in col_importance.values()]
}).sort_values('importancia_promedio', ascending=False)

# Graficar
importancia_columnas.plot(kind='barh', x='columna', y='importancia_promedio', figsize=(8, 6), legend=False)
plt.title("Importancia promedio por columna original")
plt.xlabel("Peso promedio absoluto")
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib


df = pd.read_csv('../../data/Accidentes_2013_2023.csv')

# Reemplazar los NaN en 'fall_les' con la categoría 'Ignorado'
df['fall_les'] = df['fall_les'].fillna('Ignorado')

# Confirmar la nueva distribución de clases en la variable objetivo
fall_les_distribution = df['fall_les'].value_counts(dropna=False)
print("\nDistribución de clases en 'fall_les' después de la limpieza:\n", fall_les_distribution)

# %% [markdown]
# # Modelo de Regresión Lineal

# %%
# Separar X e y
X = df.drop(columns=['fall_les'])
y = df['fall_les']

# One-hot encoding de variables categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Codificar clase objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)


# %%
# Entrenar el modelo de regresión logística
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
y_pred = log_reg.predict(X_test)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", conf_matrix)

# Informe de clasificación
print("\nInforme de Clasificación:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)
print("\nExactitud del modelo: {:.2f}%".format(accuracy * 100))

# Visualizar matriz de confusión
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()


# %% [markdown]
# # Mejor modelo RL 

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Definir el modelo base
logreg = LogisticRegression(max_iter=1000, solver='liblinear')  # 'liblinear' soporta l1 y l2

# Definir grilla de hiperparámetros
param_grid = {
    'penalty': ['l2'],  # elimina 'l1' si no es esencial
    'C': [0.1, 1, 10],  # reduce cantidad de C
    'class_weight': ['balanced']  # si ya sabés que mejora rendimiento
}

# GridSearchCV con validación cruzada estratificada
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)

# Mostrar mejores hiperparámetros
print("Mejores hiperparámetros encontrados:\n", grid.best_params_)


# %% [markdown]
# # Evluando mejor RL

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Usar el mejor modelo encontrado por GridSearchCV
best_model = grid.best_estimator_

# Predicciones sobre el conjunto de prueba
y_pred = best_model.predict(X_test)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Reporte de clasificación
print("\nInforme de Clasificación:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Exactitud del modelo: {:.2f}%".format(accuracy * 100))

# Visualización con seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de Confusión - Mejor Modelo de Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()


# %% [markdown]
# - El modelo hiperparametrizado sacrificó un poco de precisión global (accuracy) pero mejoró mucho en la clase "Fallecido", que es típicamente la más importante por ser crítica aunque poco frecuente.
# - La caída en precisión para "Fallecido" es esperable al balancear clases, pero se compensa con un gran aumento en recall, que es crucial si lo que importa es detectar fallecidos aunque haya más falsos positivos.
# 
# ## ¿Qué modelo elegir?
# 
# - Si detectar fallecidos es prioritario → usa el modelo con hiperparámetros.
# - Si accuracy general es lo más importante → el modelo normal tiene una leve ventaja.
# - En la mayoría de los casos reales, mejorar recall de la clase minoritaria crítica (fallecidos) es más valioso que tener una accuracy ligeramente mayor.
# 
# 
# 

# %%
# Visualización de la distribución de clases
plt.figure(figsize=(10, 6))
sns.countplot(x='fall_les', data=df)
plt.title('Distribución de clases en la variable objetivo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# Evaluación de overfitting/underfitting con curvas de aprendizaje
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Puntuación")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Puntuación de entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Puntuación de validación cruzada")
    plt.legend(loc="best")
    
    return plt

# Modelo de referencia para evaluar overfitting/underfitting
log_reg = LogisticRegression(max_iter=1000, random_state=42)
plot_learning_curve(log_reg, "Curva de Aprendizaje - Regresión Logística", 
                    X_train, y_train, cv=5, n_jobs=-1)
plt.show()



# %%
# SMOTE (Oversampling)
smote_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

smote_pipeline.fit(X_train, y_train)
y_pred_smote = smote_pipeline.predict(X_test)

print("\nResultados con SMOTE:")
print(confusion_matrix(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))

# %%
# Undersampling
under_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('undersampler', RandomUnderSampler(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

under_pipeline.fit(X_train, y_train)
y_pred_under = under_pipeline.predict(X_test)

print("\nResultados con Undersampling:")
print(confusion_matrix(y_test, y_pred_under))
print(classification_report(y_test, y_pred_under))


# %%
# Combinación de over y undersampling
# Primero convertimos y_train a pandas Series si no lo es ya
if not isinstance(y_train, pd.Series):
    y_train_series = pd.Series(y_train)
else:
    y_train_series = y_train

# Contamos los valores de cada clase
train_class_counts = y_train_series.value_counts()
unique_classes = train_class_counts.index.tolist()

print("\nClases presentes en los datos de entrenamiento:", unique_classes)

# Configuración dinámica de sampling_strategy basada en las clases disponibles
sampling_strategy = {}
if 'Ignorado' in unique_classes and 'Lesionado' in unique_classes:
    min_count = min(train_class_counts.get('Lesionado', 0), train_class_counts.get('Ignorado', 0))
    if min_count > 0:
        if 'Ignorado' in unique_classes:
            sampling_strategy['Ignorado'] = min_count
        if 'Lesionado' in unique_classes:
            sampling_strategy['Lesionado'] = min_count
if 'Fallecido' in unique_classes:
    sampling_strategy['Fallecido'] = train_class_counts.get('Fallecido', 0)

print("\nEstrategia de muestreo configurada:", sampling_strategy)

# Si no se pudo configurar ninguna estrategia específica, usar 'auto'
if not sampling_strategy:
    print("No se pudo configurar una estrategia de muestreo específica. Usando 'auto'.")
    under_sampler = RandomUnderSampler(random_state=42)
else:
    under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

combined_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('undersampler', under_sampler),
    ('smote', SMOTE(random_state=42)),  # Aumentar la clase minoritaria después
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

combined_pipeline.fit(X_train, y_train)
y_pred_combined = combined_pipeline.predict(X_test)

print("\nResultados con combinación de over y undersampling:")
print(confusion_matrix(y_test, y_pred_combined))
print(classification_report(y_test, y_pred_combined))

# %% [markdown]
# En general, la estrategia con SMOTE obtuvo un rendimiento ligeramente superior al resto, con una precisión global del 76 % (frente al 75 % de undersampling y combinado) y un F1 ponderado de 0,79 (0,78 en los otros dos), gracias principalmente a una mejor recuperación de la clase minoritaria (el F1 para la clase “0” sube de 0,42 a 0,42 e incluso mejora marginalmente el recall de otras clases). La media de F1 macro también es ligeramente mayor con SMOTE (0,74 vs 0,73), lo que indica un equilibrio algo mejor entre las tres clases. Por su parte, undersampling y la combinación de over- y undersampling ofrecen resultados prácticamente idénticos entre sí, sin ventaja clara sobre SMOTE, por lo que este es el método más recomendable para abordar el desbalance en este caso.




# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../../data/Accidentes_2013_2023.csv')

# Reemplazar los NaN en 'fall_les' con la categoría 'Ignorado'
df['fall_les'] = df['fall_les'].fillna('Ignorado')

# Confirmar la nueva distribución de clases en la variable objetivo
fall_les_distribution = df['fall_les'].value_counts(dropna=False)
print("\nDistribución de clases en 'fall_les' después de la limpieza:\n", fall_les_distribution)

# %%
# Separar X e y
X = df.drop(columns=['fall_les'])
y = df['fall_les']

# One-hot encoding de variables categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Codificar clase objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)


# %%
# Crear el modelo RandomForest
rf_model = RandomForestClassifier(
    n_estimators=100,  # Número de árboles
    max_depth=None,    # Profundidad máxima (None = sin límite)
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # Número de características a considerar en cada división
    bootstrap=True,
    random_state=42
)


# %%
# 2. Entrenar el modelo
rf_model.fit(X_train, y_train)


# %%
# Evaluar el modelo
# Predicciones en conjunto de prueba
y_pred = rf_model.predict(X_test)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Exactitud del modelo: {accuracy:.4f}")
print("\nMatriz de confusión:")
print(conf_matrix)
print("\nInforme de clasificación:")
print(classification_rep)

# %%
# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"\nExactitud con validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %%
# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de confusión - Random Forest')
plt.tight_layout()
plt.show()

# %%
# Obtener las 20 características más importantes
feature_names = X_encoded.columns
feature_importance = rf_model.feature_importances_

# Crear DataFrame para visualizar
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Ordenar por importancia descendente
importance_df = importance_df.sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 20 características más importantes')
plt.tight_layout()
plt.show()


# %%
# 7. Ajuste de hiperparámetros (opcional)

# Definir espacio de hiperparámetros a explorar
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Usar todos los núcleos disponibles
)

grid_search.fit(X_train, y_train)

print("\nMejores hiperparámetros:")
print(grid_search.best_params_)
print(f"Mejor exactitud: {grid_search.best_score_:.4f}")

# Usar el mejor modelo
best_rf_model = grid_search.best_estimator_

# %%




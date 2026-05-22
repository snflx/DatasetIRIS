# 📖 Tutorial Paso a Paso: Clasificador de Especie Iris con Árbol ID3

Este tutorial te guiará detalladamente en el proceso de reconstruir, comprender y ejecutar el clasificador de flores de Iris utilizando el algoritmo de **Árbol de Decisión ID3 (basado en Entropía)**. 

Está diseñado para estudiantes, botánicos y programadores principiantes que deseen comprender el flujo completo de un proyecto de Machine Learning, desde la ingesta de datos hasta el despliegue interactivo.

---

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:
- **Python 3.8** o superior en tu sistema.
- Un editor de código como VS Code, o simplemente acceso a tu consola de comandos.

---

## 🛠️ Paso 1: Configurar el Entorno de Trabajo

Lo primero que haremos será abrir una terminal en Windows (PowerShell o Símbolo del Sistema) e instalar las librerías científicas de Python necesarias:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit python-docx pillow
```

Estas librerías cumplen funciones específicas:
- `pandas` y `numpy` para procesar tablas de datos.
- `scikit-learn` para crear y entrenar el modelo.
- `matplotlib` y `seaborn` para generar gráficos.
- `streamlit` para montar la aplicación interactiva.

---

## 📥 Paso 2: Carga de Datos (Extracción ETL)

Utilizaremos el clásico dataset **Iris**, que contiene 150 registros de flores balanceadas (50 de cada una de las 3 especies: Setosa, Versicolor y Virginica). Cargaremos el dataset dinámicamente desde el repositorio OpenML utilizando `pandas`:

```python
import pandas as pd

# Definir la dirección del dataset en formato CSV
url = "https://www.openml.org/data/get_csv/61/dataset_61_iris.arff"

# Cargar el archivo CSV en un DataFrame de pandas
dt = pd.read_csv(url)

# Verificar las dimensiones (debe ser 150 filas y 5 columnas)
print("Dimensiones del Dataset:", dt.shape)
print("\nPrimeras 5 filas:")
print(dt.head())
```

---

## 🔄 Paso 3: Preparación y División (Transformación ETL)

Para entrenar nuestro modelo y validar si funciona correctamente, debemos separar las **características morfológicas (X)** de la **especie objetivo (y)**, y dividir los datos en dos conjuntos:
- **Conjunto de Entrenamiento (80%):** Con el que aprende el modelo.
- **Conjunto de Prueba (20%):** Con el que evaluamos si es exacto.

```python
from sklearn.model_selection import train_test_split

# Separar variables predictoras (X) de la variable objetivo (y)
X = dt.drop('class', axis=1) # Elimina la columna de la clase para quedarnos solo con las medidas
y = dt['class']              # Guarda la columna class (Setosa, Versicolor, Virginica)

# Dividir en entrenamiento (80%) y prueba (20%)
# Usamos random_state=1 para asegurar que la división aleatoria sea siempre la misma al reproducirlo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"Muestras de Entrenamiento: {len(X_train)}")
print(f"Muestras de Validación/Prueba: {len(X_test)}")
```

---

## 🌲 Paso 4: Entrenamiento del Árbol ID3 (Modelado)

El algoritmo **ID3** utiliza el concepto de **Entropía de Shannon** para medir la incertidumbre. En la librería `scikit-learn`, el modelo `DecisionTreeClassifier` nos permite configurar este criterio especificando `criterion='entropy'`:

```python
from sklearn.tree import DecisionTreeClassifier

# Crear una instancia del clasificador con el criterio de Entropía (ID3)
algorithm = DecisionTreeClassifier(criterion='entropy', random_state=1)

# Entrenar el modelo con los datos de entrenamiento
algorithm.fit(X_train, y_train)

print("¡Modelo ID3 entrenado con éxito!")
```

---

## 🔮 Paso 5: Predicciones sobre el Conjunto de Prueba

Una vez entrenado, usaremos el modelo para predecir las especies correspondientes a las muestras en `X_test` (datos que el clasificador nunca ha visto en su entrenamiento):

```python
# Realizar predicciones
y_pred = algorithm.predict(X_test)

# Mostrar comparación entre valores Reales y Predichos
comparacion = pd.DataFrame({'Real': y_test, 'Predicho': y_pred})
print(comparacion.head(10))
```

---

## 🏆 Paso 6: Evaluación y Métricas

Evaluaremos la efectividad general del modelo analizando el porcentaje de aciertos (**Exactitud** o **Accuracy**) y la **Matriz de Confusión**:

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Calcular exactitud
exactitud = accuracy_score(y_test, y_pred)
print(f"Exactitud Global (Accuracy): {exactitud * 100:.2f}%\n")

# Reporte de clasificación detallado por especie
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de Confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
```

### 🔍 Interpretación de los Resultados:
- El modelo logra un **96.67% de exactitud**, lo que significa que de las 30 flores de prueba, clasificó de forma perfecta a 29 de ellas.
- **Setosa** obtiene métricas perfectas (100% de precisión y recall).
- **Versicolor** presenta un pequeño error: 1 muestra real de Versicolor fue clasificada erróneamente como **Virginica**, debido al solapamiento morfológico natural en las fronteras de dimensión física de sus pétalos.

---

## 🖼️ Paso 7: Visualizar las Reglas del Árbol

Podemos pintar gráficamente la estructura del árbol y extraer sus reglas en texto para entender cómo toma decisiones:

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=300)

# Graficar árbol
plot_tree(
    algorithm, 
    filled=True, 
    rounded=True, 
    feature_names=X.columns, 
    class_names=algorithm.classes_, 
    ax=axes
)

plt.show()
```

### Reglas Lógicas Clave Aprendidas:
1. **Si el ancho del pétalo (`petalwidth`) $\le 0.8\text{ cm}$:** Es **Iris-setosa** (entropía cae a 0.0 inmediatamente).
2. **Si el ancho del pétalo (`petalwidth`) $> 1.75\text{ cm}$:** Es **Iris-virginica** (entropía de 0.0).
3. **Si el ancho está entre $0.8\text{ cm}$ y $1.75\text{ cm}$:** Se valida la longitud del pétalo (`petallength`). Si es $\le 4.95\text{ cm}$ es **Iris-versicolor**. De lo contrario, se clasifica como **Iris-virginica** (con una entropía residual menor).

---

## 🌐 Paso 8: Despliegue de la Aplicación Streamlit

Para que cualquier usuario pueda interactuar con el modelo en vivo ajustando sliders, hemos creado el archivo `app.py`. Para ejecutar esta interfaz web localmente, abre tu terminal y ejecuta:

```bash
streamlit run app.py
```

El servidor abrirá una pestaña en tu navegador en `http://localhost:8501`. La aplicación contiene pestañas para:
- Conocer la teoría del ID3 y la autoría del proyecto (Juan Pablo Sanchez Florez).
- Ver e interactuar con la tabla de datos original.
- Analizar gráficos estadísticos dinámicos (EDA).
- Ejecutar el modelo en tiempo real y ver su árbol lógico.
- Evaluar métricas e ingresar dimensiones manuales para recibir clasificaciones instantáneas.

---

## 🖥️ Paso 9: Visualización de la Landing Page Premium

Por último, para presentar el proyecto de manera corporativa y académica a nivel directivo, hemos desarrollado una **Landing Page de Alta Fidelidad** en la raíz del proyecto.

Para visualizarla:
1. Navega en tu explorador de Windows hasta la carpeta principal del proyecto: `c:\Users\Juan\Desktop\DATASET IRIS`.
2. Haz doble clic sobre el archivo `index.html`.
3. Se abrirá en tu navegador web. Explora su **Navegación Lateral**, el **Modo Oscuro/Claro**, los **Indicadores Clave**, y la explicación interactiva de las etapas **CRISP-ML** para el ciclo de vida del software de IA.

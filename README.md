# 🌲 Clasificador de Especie Iris - Algoritmo ID3

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene una solución integral y profesional de Machine Learning orientada a resolver el problema de clasificación taxonómica de la flor de Iris (especies **Setosa, Versicolor y Virginica**). La arquitectura utiliza el algoritmo clásico de **Árbol de Decisión ID3** basado en la **Entropía de Shannon**, alcanzando una **exactitud de validación del 96.67%**.

El proyecto democratiza el uso del modelo mediante un **despliegue interactivo en Streamlit** y una **Landing Page de nivel empresarial** con explicabilidad detallada.

---

## 🎯 Propósito del Proyecto

La clasificación morfológica manual de la flor de Iris es una labor propensa a errores humanos y subjetividad debido a la estrecha cercanía visual de sus sépalos y pétalos. Este proyecto automatiza dicho proceso, traduciendo dimensiones físicas (longitud y ancho de sépalos y pétalos) en predicciones taxonómicas rigurosas, explicables e instantáneas.

### Atributos Físicos de Entrada (Features)
- `sepallength` (Largo del Sépalo en cm)
- `sepalwidth` (Ancho del Sépalo en cm)
- `petallength` (Largo del Pétalo en cm)
- `petalwidth` (Ancho del Pétalo en cm)

### Atributo de Salida (Target)
- `class` (Especies de Iris: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`)

---

## 💻 Stack Tecnológico

1. **Modelado y Analítica (Core):**
   - Python
   - Scikit-learn (para el clasificador `DecisionTreeClassifier` y métricas de evaluación)
   - Pandas & Numpy (para el pipeline ETL)
2. **Visualizaciones Gráficas:**
   - Matplotlib & Seaborn
3. **Despliegue y Dashboard:**
   - Streamlit (aplicación interactiva)
4. **Landing Page Premium:**
   - HTML5 semántico
   - Vanilla CSS3 (diseño responsivo con navegación lateral y glassmorfismo)
   - Modern ES6 Javascript (interacciones y modo oscuro/claro)
5. **Generación de Reportes:**
   - Python-docx (automatización de la memoria técnica académica)

---

## 📂 Arquitectura del Proyecto

```directory
c:\Users\Juan\Desktop\DATASET IRIS\
├── Imagenes/                           # Recursos gráficos y evidencias del modelo
│   ├── Arbol.png                       # Estructura del árbol ID3
│   ├── Grafica de barras.png            # Comparación de predicciones
│   ├── Matriz de confucion.png          # Matriz de confusión histórica
│   └── adf53aa6-ad77-489c-92ed-8b2ab208faef.png  # Soporte visual
├── Proyecto/
│   └── Material_1_Proyecto_IA_Unilasallista.docx # Memoria técnica académica (.docx)
├── app.py                              # Aplicativo interactivo en Streamlit
├── index.html                          # Landing Page interactiva premium
├── index.css                           # Estilos premium CSS de la Landing Page
├── index.js                            # Lógica interactiva JS de la Landing Page
├── README.md                           # Documento principal del repositorio
└── Tutorial.md                         # Guía paso a paso para reproducción
```

---

## 🚀 Requisitos e Instalación

Para ejecutar este proyecto en tu entorno local, asegúrate de tener instalado Python 3.8 o superior.

### 1. Clonar o acceder a la carpeta del proyecto
```bash
cd "c:\Users\Juan\Desktop\DATASET IRIS"
```

### 2. Instalar dependencias requeridas
Ejecuta el siguiente comando para instalar todos los paquetes científicos y de desarrollo necesarios:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit python-docx pillow
```

---

## 🛠️ Instrucciones de Ejecución

### Ejecutar la Aplicación Interactiva (Streamlit)
Inicia el servidor local de Streamlit ejecutando:
```bash
streamlit run app.py
```
Una vez iniciado, la aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.

### Desplegar en Streamlit Community Cloud
1. Sube este proyecto a un repositorio de GitHub.
2. Entra a [Streamlit Community Cloud](https://streamlit.io/cloud) y selecciona **New app**.
3. Elige el repositorio, la rama principal y define `app.py` como archivo principal.
4. Streamlit instalará las dependencias desde `requirements.txt` y publicará la aplicación.

### Abrir la Landing Page Empresarial
Simplemente abre el archivo `index.html` en cualquier navegador web moderno (Chrome, Edge, Firefox, Safari) o haz doble clic sobre el archivo en tu explorador de archivos de Windows.

---

## 🧠 Especificaciones del Modelo e ID3

El clasificador se entrena bajo el criterio de **Entropía de Shannon** para medir el grado de incertidumbre en el conjunto de entrenamiento (120 muestras, división 80/20):

$$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$

### Reglas de Negocio Derivadas:
1. **Regla Setosa:** Si el ancho del pétalo (`petalwidth`) es $\le 0.8\text{ cm}$, se clasifica inmediatamente con certeza absoluta como **Iris-setosa** (entropía = 0.0).
2. **Regla Virginica:** Si el ancho del pétalo (`petalwidth`) es $> 1.75\text{ cm}$, se clasifica puramente como **Iris-virginica** (entropía = 0.0).
3. **Regla Versicolor:** Si el ancho del pétalo está en el rango $(0.8\text{ cm}, 1.75\text{ cm}]$ y la longitud del pétalo es $\le 4.95\text{ cm}$, se clasifica con pureza como **Iris-versicolor** (entropía = 0.0).

---

## 🏆 Métricas de Desempeño Evaluadas

| Métrica | Valor | Detalle |
| :--- | :--- | :--- |
| **Exactitud (Accuracy)** | **96.67%** | 29 de 30 muestras clasificadas correctamente en el conjunto de prueba. |
| **Tasa de Error** | **3.33%** | 1 sola muestra mal clasificada debido al solapamiento natural morfológico. |
| **F1-Score Setosa** | **1.00** | Clasificación perfecta y limpia para Iris-setosa. |
| **F1-Score Versicolor**| **0.96** | 13 muestras evaluadas, con un solo falso negativo. |
| **F1-Score Virginica** | **0.92** | Precisión de 86% y recall de 100%. |

---

## ✍️ Créditos y Autoría

- **Desarrollador Principal:** Juan Pablo Sanchez Florez
- **Programa:** Inteligencia Artificial - Módulo 1 (Aprendizaje Inteligente)
- **Institución:** Unilasallista
- **Tutor / Entorno:** Taller Colaborativo - Propuesta Iris ID3

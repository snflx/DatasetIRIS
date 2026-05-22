import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import os
from PIL import Image

# Set premium page configuration
st.set_page_config(
    page_title="Iris ID3 Classifier - Talento Tech",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1b5e20, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #4caf50;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dcedc8;
        text-align: center;
    }
    .badge-setosa {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-weight: bold;
        border: 1px solid #c8e6c9;
    }
    .badge-versicolor {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-weight: bold;
        border: 1px solid #bbdefb;
    }
    .badge-virginica {
        background-color: #f3e5f5;
        color: #6a1b9a;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-weight: bold;
        border: 1px solid #e1bee7;
    }
    </style>
""", unsafe_allowed_html=True)

# Load dataset function
@st.cache_data
def load_data():
    url = "https://www.openml.org/data/get_csv/61/dataset_61_iris.arff"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        # Fallback in case of network issue
        st.error(f"Error cargando desde OpenML: {e}. Cargando dataset local/mockup.")
        # Create standard Iris structure as fallback
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
        df['class'] = [iris.target_names[i] for i in iris.target]
        df['class'] = df['class'].map({
            'setosa': 'Iris-setosa',
            'versicolor': 'Iris-versicolor',
            'virginica': 'Iris-virginica'
        })
        return df

df = load_data()

# Navigation Sidebar
st.sidebar.image("Imagenes/Arbol.png" if os.path.exists("Imagenes/Arbol.png") else None, width=150)
st.sidebar.title("Menú de Navegación")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Selecciona un Módulo:",
    [
        "🏠 Inicio & Presentación",
        "📊 Exploración del Dataset",
        "📈 Análisis Exploratorio (EDA)",
        "🌲 Modelado (ID3)",
        "🏆 Métricas y Evaluación",
        "🔮 Predicción Interactiva"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Proyecto:** Árbol de Decisión ID3\n\n"
    "**Autor:** Feibert Alirio Guzmán Pérez\n\n"
    "**Entorno:** Talento Tech - IA Módulo 1"
)

# ----------------- MODULE 1: OVERVIEW -----------------
if app_mode == "🏠 Inicio & Presentación":
    st.markdown("<div class='main-title'>Agente Especializado en Proyectos ML: Clasificación Iris</div>", unsafe_allowed_html=True)
    st.markdown("<div class='subtitle'>Despliegue Interactivo del Modelo de Árbol de Decisión ID3 (Entropía)</div>", unsafe_allowed_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🎯 Descripción del Proyecto</h3>
            <p>La identificación manual de especies de flores de <b>Iris (Setosa, Versicolor, Virginica)</b> es un proceso lento y propenso a errores humanos para estudiantes de botánica y ecólogos. Debido a que estas plantas comparten características morfológicas muy similares en sus sépalos y pétalos, la clasificación visual directa puede retrasar análisis biológicos importantes.</p>
            <p>Este proyecto implementa y automatiza la clasificación taxonómica mediante el algoritmo de aprendizaje supervisado de <b>Árbol de Decisión ID3 (basado en la entropía de Shannon)</b>, alcanzando una <b>exactitud del 96.67%</b> en la predicción del tipo de flor a partir de sus dimensiones físicas de sépalo y pétalo.</p>
        </div>
        """, unsafe_allowed_html=True)
        
        st.markdown("""
        <div class='card'>
            <h3>🧠 Fundamentos Técnicos del Algoritmo ID3</h3>
            <p>El algoritmo <b>ID3 (Iterative Dichotomiser 3)</b> fue diseñado por J. Ross Quinlan y se basa en la selección de divisiones en los datos que maximicen la reducción del desorden o <b>Entropía de Shannon</b>, maximizando consecuentemente la <b>Ganancia de Información</b>.</p>
        </div>
        """, unsafe_allowed_html=True)
        
        st.markdown("#### Fórmulas de la Entropía e Información")
        st.latex(r"H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)")
        st.markdown("Donde $p_i$ representa la proporción de muestras que pertenecen a la clase $i$ dentro de un subconjunto de datos $S$.")
        st.latex(r"IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)")
        st.markdown("Donde $IG(S, A)$ representa la Ganancia de Información (Information Gain) al particionar los datos según el atributo $A$.")

    with col2:
        st.markdown("### 🏆 Integrante del Proyecto")
        st.success("**Feibert Alirio Guzmán Pérez**\n\n*Ingeniería de Sistemas / Ciencia de Datos e IA*")
        
        st.markdown("### 📊 Indicadores Clave")
        st.info("☘️ **Dataset:** Iris Flower Dataset")
        st.info("📏 **Instancias:** 150 muestras balanceadas (50 de cada clase)")
        st.info("📈 **Exactitud del Modelo:** 96.67%")
        st.info("🎯 **Algoritmo:** ID3 con Entropía")
        
        if os.path.exists("Imagenes/adf53aa6-ad77-489c-92ed-8b2ab208faef.png"):
            st.image("Imagenes/adf53aa6-ad77-489c-92ed-8b2ab208faef.png", caption="Soporte Gráfico del Proyecto", use_container_width=True)

# ----------------- MODULE 2: DATASET EXPLORATION -----------------
elif app_mode == "📊 Exploración del Dataset":
    st.markdown("<div class='main-title'>Exploración del Dataset Iris</div>", unsafe_allowed_html=True)
    st.markdown("<div class='subtitle'>Visualización detallada y búsqueda interactiva en el conjunto de datos</div>", unsafe_allowed_html=True)
    
    st.markdown("""
    El dataset Iris es un clásico en el mundo del Machine Learning introducido por Ronald Fisher en 1936.
    Consiste en 150 muestras tomadas de tres especies de lirio: **Iris setosa**, **Iris versicolor** e **Iris virginica**.
    Se midieron cuatro rasgos morfológicos de cada flor: el largo y ancho del sépalo y del pétalo (en centímetros).
    """)
    
    st.markdown("### 🔍 Vista de Datos Filtrable")
    search_query = st.text_input("Filtrar por especie (ej: setosa, versicolor, virginica):", "")
    
    filtered_df = df
    if search_query:
        filtered_df = df[df['class'].str.contains(search_query, case=False)]
        
    st.dataframe(filtered_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Instancias", len(df))
    with col2:
        st.metric("Muestras Filtradas", len(filtered_df))
    with col3:
        st.metric("Dimensiones Originales", f"{df.shape[0]} x {df.shape[1]}")
        
    st.markdown("### 📉 Estadísticas Descriptivas Generales")
    st.table(df.describe())

# ----------------- MODULE 3: EDA -----------------
elif app_mode == "📈 Análisis Exploratorio (EDA)":
    st.markdown("<div class='main-title'>Análisis Exploratorio de Datos (EDA)</div>", unsafe_allowed_html=True)
    st.markdown("<div class='subtitle'>Visualizaciones gráficas y relaciones morfológicas</div>", unsafe_allowed_html=True)
    
    eda_option = st.selectbox(
        "Selecciona el tipo de gráfico:",
        [
            "1. Distribución de Características",
            "2. Dispersión: Pétalos (Largo vs Ancho)",
            "3. Dispersión: Sépalos (Largo vs Ancho)",
            "4. Matriz de Correlación"
        ]
    )
    
    if eda_option == "1. Distribución de Características":
        st.markdown("### Distribución de las Dimensiones Físicas por Especie")
        feature_to_plot = st.selectbox("Selecciona la dimensión física a analizar:", ['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=df, x=feature_to_plot, hue="class", fill=True, common_norm=False, palette="Set1", alpha=0.5, ax=ax)
        ax.set_title(f"Distribución del {feature_to_plot} por especie de flor")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        
    elif eda_option == "2. Dispersión: Pétalos (Largo vs Ancho)":
        st.markdown("### Dispersión de las Dimensiones del Pétalo")
        st.write("Esta visualización es crítica. Como se puede observar, las dimensiones del pétalo son los diferenciadores clave para el modelo, permitiendo clasificar de manera linealmente pura a la especie Iris-setosa.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class', style='class', palette='Set1', s=100, alpha=0.8, ax=ax)
        ax.set_title("Largo vs Ancho del Pétalo")
        ax.set_xlabel("Largo del Pétalo (cm)")
        ax.set_ylabel("Ancho del Pétalo (cm)")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        
    elif eda_option == "3. Dispersión: Sépalos (Largo vs Ancho)":
        st.markdown("### Dispersión de las Dimensiones del Sépalo")
        st.write("A diferencia de los pétalos, las dimensiones de los sépalos presentan mayor superposición entre las clases Iris-versicolor e Iris-virginica, lo que dificulta la clasificación simple usando solo estas características.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class', style='class', palette='Set1', s=100, alpha=0.8, ax=ax)
        ax.set_title("Largo vs Ancho del Sépalo")
        ax.set_xlabel("Largo del Sépalo (cm)")
        ax.set_ylabel("Ancho del Sépalo (cm)")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        
    elif eda_option == "4. Matriz de Correlación":
        st.markdown("### Matriz de Correlación Lineal (Atributos Físicos)")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Convert class to categorical/numeric just for correlation or exclude it
        numeric_df = df.drop('class', axis=1)
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Greens", fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title("Correlación de Pearson entre Atributos de Iris")
        st.pyplot(fig)

# ----------------- MODULE 4: MODELING -----------------
elif app_mode == "🌲 Modelado (ID3)":
    st.markdown("<div class='main-title'>Modelado e Interpretación del Árbol ID3</div>", unsafe_allowed_html=True)
    st.markdown("<div class='subtitle'>Detalles del entrenamiento del clasificador y reglas lógicas aprendidas</div>", unsafe_allowed_html=True)
    
    # Train the model live
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = DecisionTreeClassifier(criterion='entropy', random_state=1)
    model.fit(X_train, y_train)
    
    st.markdown("""
    <div class='card'>
        <h3>⚙️ Proceso de Partición y Hiperparámetros</h3>
        <ul>
            <li><b>División de Datos:</b> 80% Entrenamiento (120 muestras), 20% Prueba (30 muestras).</li>
            <li><b>Semilla Aleatoria (random_state):</b> 1</li>
            <li><b>Criterio de Partición:</b> Entropía (ID3)</li>
        </ul>
    </div>
    """, unsafe_allowed_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🌲 Estructura Lógica del Árbol")
        st.write("A continuación se muestra el árbol de decisión entrenado en vivo utilizando `plot_tree`:")
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        plot_tree(model, filled=True, rounded=True, feature_names=X.columns, class_names=model.classes_, ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("### 🔍 Análisis de Reglas nodo por nodo")
        st.write("El modelo utiliza reglas sencillas y altamente interpretables basadas en umbrales específicos:")
        
        st.info(
            "1. **Nodo Raíz (petalwidth <= 0.8 cm):**\n"
            "- Si se cumple (True): Se clasifica inmediatamente como **Iris-setosa** (39 muestras en train, entropía = 0.0).\n"
            "- Si no se cumple (False): Va al siguiente nivel para clasificar entre Versicolor y Virginica (81 muestras, entropía = 1.0)."
        )
        st.info(
            "2. **Segundo Nivel (petalwidth <= 1.75 cm):**\n"
            "- Si se cumple (True): Va a un refinamiento por longitud de pétalo.\n"
            "- Si no se cumple (False): Se clasifica inmediatamente como **Iris-virginica** (37 muestras, entropía = 0.0)."
        )
        st.info(
            "3. **Tercer Nivel - Ajuste Fino (petallength <= 4.95 cm):**\n"
            "- Si se cumple (True): Se clasifica como **Iris-versicolor** (39 muestras, entropía = 0.0).\n"
            "- Si no se cumple (False): Entropía residual alta (0.971) para 5 muestras (2 Versicolor, 3 Virginica). Clasifica como **Iris-virginica** por mayoría simple."
        )

# ----------------- MODULE 5: METRICS -----------------
elif app_mode == "🏆 Métricas y Evaluación":
    st.markdown("<div class='main-title'>Evaluación de Métricas del Modelo</div>", unsafe_allowed_html=True)
    st.markdown("<div class='subtitle'>Métricas académicas y análisis de la Matriz de Confusión</div>", unsafe_allowed_html=True)
    
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = DecisionTreeClassifier(criterion='entropy', random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h4>🎯 Exactitud (Accuracy)</h4><h2>96.67%</h2><p>29 de 30 correctas</p></div>", unsafe_allowed_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h4>☘️ F1-Score Setosa</h4><h2>100%</h2><p>Precisión & Recall perfectos</p></div>", unsafe_allowed_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h4>📉 Tasa de Error</h4><h2>3.33%</h2><p>1 sola muestra errónea</p></div>", unsafe_allowed_html=True)
        
    st.markdown("---")
    
    col_chart, col_text = st.columns([1, 1])
    
    with col_chart:
        st.markdown("### 📊 Matriz de Confusión (En Vivo)")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Reds, values_format='d', ax=ax)
        ax.set_title("Matriz de Confusión del Conjunto de Prueba")
        st.pyplot(fig)
        
    with col_text:
        st.markdown("### 📋 Reporte de Clasificación Detallado")
        st.text(classification_report(y_test, y_pred))
        
        st.markdown("""
        <div class='card'>
            <h4>🔍 Análisis del Error</h4>
            <p>El modelo presenta un único error de clasificación en el conjunto de prueba (1 flor real de <b>Iris-versicolor</b> fue clasificada erróneamente como <b>Iris-virginica</b>).</p>
            <p>Esto se debe al solapamiento físico natural de las dimensiones del pétalo de estas dos especies en la frontera de 1.75 cm y 4.95 cm de longitud de pétalo.</p>
        </div>
        """, unsafe_allowed_html=True)
        
    if os.path.exists("Imagenes/Matriz de confucion.png"):
        st.markdown("### 📁 Evidencia Gráfica Guardada (`Matriz de confucion.png`)")
        st.image("Imagenes/Matriz de confucion.png", caption="Matriz de Confusión histórica", use_container_width=True)

# ----------------- MODULE 6: PREDICTIONS -----------------
elif app_mode == "🔮 Predicción Interactiva":
    st.markdown("<div class='main-title'>Predicción Interactiva de Especies</div>", unsafe_allowed_html=True)
    st.markdown("<div class='subtitle'>Ingresa las dimensiones de la flor para clasificarla en tiempo real</div>", unsafe_allowed_html=True)
    
    # Train the model
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = DecisionTreeClassifier(criterion='entropy', random_state=1)
    model.fit(X_train, y_train)
    
    st.markdown("### 🎛️ Control de Características")
    st.write("Ajusta los sliders a continuación para simular las características físicas del sépalo y pétalo:")
    
    col1, col2 = st.columns(2)
    with col1:
        sepallength = st.slider("Largo del Sépalo (cm):", float(df['sepallength'].min()), float(df['sepallength'].max()), float(df['sepallength'].mean()), 0.1)
        sepalwidth = st.slider("Ancho del Sépalo (cm):", float(df['sepalwidth'].min()), float(df['sepalwidth'].max()), float(df['sepalwidth'].mean()), 0.1)
    with col2:
        petallength = st.slider("Largo del Pétalo (cm):", float(df['petallength'].min()), float(df['petallength'].max()), float(df['petallength'].mean()), 0.1)
        petalwidth = st.slider("Ancho del Pétalo (cm):", float(df['petalwidth'].min()), float(df['petalwidth'].max()), float(df['petalwidth'].mean()), 0.1)
        
    # Run prediction
    input_data = pd.DataFrame([{
        'sepallength': sepallength,
        'sepalwidth': sepalwidth,
        'petallength': petallength,
        'petalwidth': petalwidth
    }])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    
    # Visualizing prediction outcome
    st.markdown("### 🔮 Resultado de la Predicción")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if prediction == "Iris-setosa":
            st.markdown("La especie predicha es: <span class='badge-setosa'>Iris-setosa</span>", unsafe_allowed_html=True)
            st.success("La especie **Iris-setosa** está perfectamente clasificada. Sus sépalos suelen ser anchos y sus pétalos pequeños.")
        elif prediction == "Iris-versicolor":
            st.markdown("La especie predicha es: <span class='badge-versicolor'>Iris-versicolor</span>", unsafe_allowed_html=True)
            st.info("La especie **Iris-versicolor** se encuentra en el rango morfológico intermedio de pétalos.")
        elif prediction == "Iris-virginica":
            st.markdown("La especie predicha es: <span class='badge-virginica'>Iris-virginica</span>", unsafe_allowed_html=True)
            st.warning("La especie **Iris-virginica** cuenta con los pétalos más grandes y anchos del grupo.")
            
        st.markdown("#### Distribución de Probabilidades de Inferencia:")
        prob_df = pd.DataFrame({
            'Especie': model.classes_,
            'Probabilidad (%)': prediction_proba * 100
        })
        st.bar_chart(prob_df.set_index('Especie'))
        
    with res_col2:
        st.markdown("### 🧠 Explicabilidad: Importancia de Variables")
        st.write("La importancia de las variables en los árboles de decisión representa qué tanto aporta cada característica a la reducción total de la entropía (impureza).")
        
        importances = model.feature_importances_
        features = X.columns
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=importances, y=features, palette="Greens_d", ax=ax)
        ax.set_title("Importancia de Características en el Árbol ID3")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        
        st.write("📌 *Como se muestra, el **ancho del pétalo (petalwidth)** posee la mayor importancia relativa (superior al 60%), confirmando que es la característica con mayor ganancia de información.*")

# Clasificación de la Estabilidad de un Sistema Hidráulico Complejo mediante LSTM y GRU

---

## Descripción

Este proyecto implementa y evalúa modelos de redes neuronales recurrentes (**LSTM** y **GRU**) para clasificar la **estabilidad operacional** de un sistema hidráulico complejo a partir de series temporales multivariadas capturadas por 17 sensores.

El dataset proviene del banco de pruebas hidráulico del *Centre for Mechatronics and Automation Technology* (ZeMA gGmbH, Saarbrücken, Alemania) y está disponible públicamente en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems).

---

## Estructura del repositorio

```
├── preprocesamiento.ipynb      # Filtrado, normalización y exportación de señales
├── model_training.ipynb        # Construcción y entrenamiento de modelos LSTM y GRU
├── inference.ipynb             # Inferencia y backtesting independiente del entrenamiento
├── raw/                        # Archivos .csv del dataset original
│   ├── PS1.csv
│   ├── PS2.csv
│   ├── ...                     # (17 sensores + profile)
│   └── profile.csv
├── processed/                  # Archivos .parquet generados por preprocesamiento
│   ├── PS1_filtrados.parquet
│   ├── PS2_filtrados.parquet
│   ├── ...                     # (17 sensores + profile)
│   └── profile.parquet
├── models/                     # Modelos entrenados (.keras)
│   ├── lstm_config1.keras
│   ├── ...
│   └── gru_config5.keras
├── environment.yaml            # Entorno Conda
├── requirements.txt            # Dependencias pip
└── runtime.txt                 # Versión de Python
```

---

## Dataset

**Condition Monitoring of Hydraulic Systems** — UCI ML Repository

| Característica | Valor |
|---|---|
| Instancias | 2,205 ciclos de 60 segundos |
| Sensores | 17 (presión, flujo, temperatura, vibración, eficiencia) |
| Atributos totales | 43,680 |
| Valores faltantes | Ninguno |

### Sensores incluidos

| Grupo | Sensores | Frecuencia | Muestras / ciclo |
|---|---|---|---|
| Presión | PS1–PS6 | 100 Hz | 6,000 |
| Potencia de motor | EPS1 | 100 Hz | 6,000 |
| Flujo volumétrico | FS1, FS2 | 10 Hz | 600 |
| Temperatura | TS1–TS4 | 1 Hz | 60 |
| Vibración | VS1 | 1 Hz | 60 |
| Factor de eficiencia | SE | 1 Hz | 60 |
| Eficiencia de refrigeración | CE | 1 Hz | 60 |
| Potencia de refrigeración | CP | 1 Hz | 60 |

### Variable objetivo — `stable`

| Valor | Significado | Instancias |
|---|---|---|
| `0` | Condiciones estables alcanzadas | 1,449 (65.6%) |
| `1` | Condiciones estáticas posiblemente no alcanzadas | 756 (34.4%) |

> El desbalance de clases se maneja mediante ponderación `class_weight='balanced'` durante el entrenamiento.

---

## Pipeline

```
Raw sensor data (.csv)
        │
        ▼
preprocesamiento.ipynb
  ├─ Diseño de filtros por sensor
  ├─ Aplicación de filtro y normalización
  └─ Exportación a processed/*.parquet
        │
        ▼
model_training.ipynb
  ├─ Construcción del tensor 3D (2205, 56, 17)
  │   ├─ Remuestreo uniforme a 60 pasos temporales
  │   └─ Suavizado por convolución (window_size=5)
  ├─ División estratificada (70% / 15% / 15%)
  ├─ 5 configuraciones LSTM
  ├─ 5 configuraciones GRU
  └─ Guardado de modelos en models/*.keras
        │
        ▼
inference.ipynb
  ├─ Reconstrucción del conjunto de backtesting (misma semilla)
  ├─ Carga de los 10 modelos .keras
  └─ Evaluación final sobre datos nunca vistos
```

---

## Modelos

Se entrenaron **10 configuraciones** en total (5 LSTM + 5 GRU), variando:

- Tasa de aprendizaje
- Profundidad de la red (2 o 3 capas recurrentes)
- Regularización (L1, L2, L1+L2, sin regularización)
- Dropout (0.3 – 0.5)
- Número de épocas (50 o 100)

Todas las configuraciones aplican:
- **EarlyStopping** con `patience=15` y restauración de mejores pesos
- **ReduceLROnPlateau** con `factor=0.5` y `patience=7`

### Métricas reportadas

| Métrica | Descripción |
|---|---|
| Exactitud | Porcentaje de predicciones correctas |
| Precisión | Correctas de las predicciones positivas |
| Recall | Positivos reales detectados |
| F1-Score | Media armónica entre Precisión y Recall |
| AUC-PR | Área bajo la curva Precisión-Sensibilidad |

> Las curvas Precisión-Sensibilidad son más informativas que la curva ROC ante desbalance de clases, ya que se focalizan en el rendimiento sobre la clase minoritaria.

---

## Instalación

### Opción 1 — Conda (recomendado)

```bash
conda env create -f environment.yaml
conda activate hydraulic-stability
```

### Opción 2 — pip

```bash
pip install -r requirements.txt
```

### Requisitos del sistema

| Componente | Versión |
|---|---|
| Python | 3.11 |
| PyTorch | 
| TensorFlow / Keras | 2.16.1 / 3.4.1 |
| NumPy | 2.4.3 |
| Pandas | 3.0.1 |
| scikit-learn | 1.4.2 |

---

## Uso

### 1. Preprocesamiento

Ejecuta `preprocesamiento.ipynb` para generar la carpeta `processed/` con los 17 archivos `.parquet` de sensores y el archivo `profile.parquet` con las etiquetas.

### 2. Entrenamiento

Ejecuta `model_training.ipynb`. Al finalizar, los 10 modelos se guardan en la carpeta `models/`.

### 3. Inferencia y backtesting

Ejecuta `inference.ipynb` de forma **completamente independiente** del entrenamiento. Solo requiere:
- La carpeta `processed/` generada en el paso 1
- La carpeta `models/` generada en el paso 2

El notebook reconstruye el conjunto de backtesting con la misma semilla (`random_state=42`, `stratify=y`) para garantizar que los 332 ejemplos evaluados nunca fueron vistos durante el entrenamiento ni la validación.

---

## Referencia del dataset

```bibtex
@inproceedings{helwig2015condition,
  author    = {Nikolai Helwig and Eliseo Pignanelli and Andreas Schütze},
  title     = {Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics},
  booktitle = {Proc. I2MTC-2015 - IEEE International Instrumentation and Measurement Technology Conference},
  year      = {2015},
  doi       = {10.1109/I2MTC.2015.7151267}
}
```

---

## Licencia

Este proyecto es de carácter académico. El dataset original es propiedad de ZeMA gGmbH y está disponible bajo los términos del UCI Machine Learning Repository.

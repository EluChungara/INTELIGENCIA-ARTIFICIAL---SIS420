# LABORATORIO 03: REGRESIÓN LINEAL MULTIVARIABLE, REGRESIÓN POLINÓMICA Y ECUACIÓN DE LA NORMAL

**Materia:** Inteligencia Artificial I (SIS-420)  
**Docente:** Ing. Carlos Walter Pacheco Lora  
**Entorno:** Visual Studio Code (`.ipynb`)  

---

## 📌 Enlaces del Proyecto

* **Video Explicativo:** [Ver Video en Google Drive](https://drive.google.com/file/d/1TwCFL4Xz__4CKfJExKAn99bLaZFy_rkc/view?usp=sharing)
* **Dataset Oficial (UCI Machine Learning Repository):** [Facebook Comment Volume Dataset](https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset)
* **Archivo de Datos Utilizado:** `Features_Variant_1.csv`

---

## 📖 Descripción del Trabajo Realizado

En esta práctica se implementaron, evaluaron y contrastaron tres métodos de regresión continua para predecir el volumen futuro de comentarios de publicaciones:
1. **Regresión Lineal Multivariable** mediante Descenso por el Gradiente por lotes.
2. **Regresión Polinómica de Grado 2** (términos cuadráticos) mediante Descenso por el Gradiente.
3. **Ecuación de la Normal** mediante solución analítica exacta de mínimos cuadrados ordinarios.

---

### 1. Auditoría del Dataset y Restricciones
* **Dataset Seleccionado:** Facebook Comment Volume Dataset (`Features_Variant_1.csv`).
* **Cantidad de Ejemplos ($m$):** **$40\,949$ muestras** ($m \ge 2000$).
* **Cantidad de Características ($n$):** **$53$ variables continuas** ($n \ge 20$).
* **Calidad de Datos:** $0$ valores nulos (`NaN`) y $0$ valores infinitos.
* **Variable Objetivo ($y$):** Cantidad continua de comentarios esperados en las siguientes 24 horas ($0.00$ a $1305.00$ comentarios, media $\bar{y} = 7.32$).
* **Análisis de Relación:** Se calculó la correlación de Pearson frente a la etiqueta, identificando la variable $X_{31}$ con una correlación positiva de $r = 0.5424$.

---

### 2. Partición de Datos y Normalización Z-Score
* **Partición 80/20 sin traslape:**
  * **Entrenamiento (80%):** $32\,759$ muestras.
  * **Validación (20% aislado):** $8\,190$ muestras.
* **Prevención de Data Leakage:** Las medias ($\mu$) y desviaciones estándar ($\sigma$) se calcularon exclusivamente sobre el conjunto de entrenamiento mediante `featureNormalize` y se aplicaron para transformar los datos de validación:
  $$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
* **Matriz de Diseño Lineal:** Incorporación del término de intercepción ($x_0 = 1$), conformando matrices `X_train_lin` y `X_val_lin` de **$54$ columnas**.

---

### 3. Espacio Polinómico (Grado 2)
* Generación de términos cuadráticos ($X^2$) para las 53 variables originales, expandiendo el espacio a 106 características.
* Tras la normalización Z-Score y la adición del término de sesgo $x_0 = 1$, se obtuvieron las matrices de diseño `X_train_poly` y `X_val_poly` de **$107$ columnas**.

---

### 4. Formulación Matemática y Algoritmos Vectorizados

* **Función de Costo Cuadrático Medio:**
  $$J(\theta) = \frac{1}{2m} (X\theta - \vec{y})^T (X\theta - \vec{y})$$

* **Descenso por el Gradiente Multivariable:**
  $$\theta := \theta - \frac{\alpha}{m} X^T (X\theta - \vec{y})$$

* **Ecuación de la Normal:**
  $$\theta = (X^T X)^{-1} X^T \vec{y}$$
  *(Calculada analíticamente mediante la pseudoinversa de Moore-Penrose `np.linalg.pinv`)*.

---

### 5. Resultados y Comparativa de los Tres Modelos

| Métrica / Parámetro | Regresión Lineal Multivariable | Regresión Polinómica Grado 2 | Ecuación de la Normal |
| :--- | :---: | :---: | :---: |
| **Hiperparámetros** | $\alpha = 0.03$, $500$ iteraciones | $\alpha = 0.01$, $500$ iteraciones | Solución analítica directa |
| **Costo Inicial $J(\vec{0})$** | $621.8717$ | $621.8717$ | — |
| **Costo Entrenamiento $J(\theta)$** | $388.9393$ ($-37.46\%$) | **$380.6147$** ($-38.80\%$) | $386.4459$ |
| **Costo Validación $J(\theta)$** | **$732.9856$** | $922.9975$ | $735.9868$ |
| **MAE (Error Absoluto Medio)** | $9.2454$ | **$8.9669$** | $9.3177$ |
| **RMSE** | **$38.2880$** | $42.9650$ | $38.3663$ |
| **$R^2$ (Coef. Determinación)** | **$0.0390$** | $-0.2102$ | $0.0350$ |

---

### 6. Validación con 100 Predicciones y Conclusiones
* **Demostración de Efectividad:** Se generaron 100 predicciones sobre datos de validación no vistos, contrastando el valor real frente a las predicciones de los tres modelos.
* **Convergencia Óptima:** La diferencia cuadrática entre el Descenso por el Gradiente lineal y la Ecuación de la Normal fue de apenas $\text{RMSE} = 2.3752$, demostrando que el algoritmo iterativo convergió al mínimo global óptimo.
* **Diagnóstico de Complejidad:** El modelo polinómico minimizó el error absoluto medio ($\text{MAE} = 8.9669$), mientras que el modelo lineal y la ecuación de la normal mostraron mayor robustez ante valores atípicos extremos.


# LABORATORIO 04: REGRESIÓN LOGÍSTICA BINARIA CON OPTIMIZACIÓN NUMÉRICA

**Materia:** Inteligencia Artificial I (SIS-420)  
**Docente:** Ing. Carlos Walter Pacheco Lora  
**Fecha:** 25 de agosto de 2026  

---

## 📌 Enlaces del Proyecto

* **Cuadernillo en Google Colab:** [Abrir Cuadernillo Colab](https://colab.research.google.com/drive/1k6p0_GVuvdh75c_geK64heF2FilpQyah?usp=sharing)
* **Video Explicativo:** [Ver Video en Google Drive](https://drive.google.com/file/d/1hIzTiyxO-vpLi7-Ld5fLRHgF6868ECMo/view?usp=sharing)
* **Dataset:** [Descargar Dataset Santander (CSV) en Google Drive](https://drive.google.com/file/d/1wnREZbmLU9ke1VmqFkHIWB5r9sVHpjFj/view?usp=sharing)
* **Dataset Original (Kaggle):** [Santander Customer Transaction Prediction](https://www.kaggle.com/datasets/lakshmi25npathi/santander-customer-transaction-prediction-dataset)

---

## 📖 Descripción del Trabajo Realizado

En esta práctica de laboratorio se implementó un modelo de clasificación supervisada basado en **Regresión Logística Binaria** para predecir si un cliente de una entidad financiera realizará una transacción específica (`target = 1`) o no (`target = 0`), aplicando la formulación matemática vectorizada y optimización numérica analítica.

---

### 1. Especificaciones del Dataset y Cumplimiento de Restricciones
* **Dataset Seleccionado:** Santander Customer Transaction Prediction (`train.csv`).
* **Cantidad de Muestras ($m$):** Contiene originalmente **$200\,000$ ejemplos** ($m \ge 20\,000$).
* **Cantidad de Propiedades ($n$):** Dispone de **$200$ características continuas anonimizadas** (`var_0` a `var_199`), superando holgadamente el requisito de $n \ge 20$.
* **Auditoría de Calidad con Pandas:** Se descartó el identificador `ID_code`, verificando $0$ valores nulos ($0\text{ NaN}$), $0$ filas duplicadas y ausencia de valores infinitos.

---

### 2. Balanceo de Clases y Preprocesamiento con Pandas
* **Tratamiento del Desbalance:** Ante la disparidad inicial ($89.95\%$ clase $0$ vs. $10.05\%$ clase $1$), se aplicó submuestreo aleatorio controlado (*Random Undersampling*) con Pandas.
* **Estructuración Balanceada:** Se combinaron las $20\,098$ muestras de la clase $1$ con $20\,098$ muestras aleatorias de la clase $0$, obteniendo un conjunto balanceado al 50/50 con **$m = 40\,196$ ejemplos** ($m \ge 20\,000$).
* **Normalización Z-Score:** Se estandarizaron las $200$ variables continuas mediante la función `featureNormalize` ($\mu = 0, \sigma = 1$) para garantizar la estabilidad numérica en el cálculo de gradientes.

---

### 3. Partición y Matriz de Diseño
Se realizó la partición estricta sin traslape mediante permutación aleatoria reproducible:
* **Conjunto de Entrenamiento (80%):** $32\,156$ muestras
* **Conjunto de Prueba Aislado (20%):** $8\,040$ muestras

Mediante `np.concatenate` se incorporó la columna de unos en la posición inicial para el término de intercepción ($x_0 = 1$), conformando matrices de diseño de **$201$ columnas** para operar con el vector de parámetros:

$$\theta = [\theta_0, \theta_1, \dots, \theta_{200}]^T$$

---

### 4. Formulación Matemática y Optimización Numérica
Se programaron desde cero en NumPy:
* **Función de activación sigmoide:**
  $$g(z) = \frac{1}{1 + e^{-z}}$$
* **Función de costo por entropía cruzada binaria:**
  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$
* **Vector gradiente analítico:**
  $$\nabla J(\theta) = \frac{1}{m} X^T (h - y)$$

Para encontrar los parámetros óptimos $\theta$, se utilizó el algoritmo de **Gradiente Conjugado (`CG`)** mediante `scipy.optimize.minimize`, registrando el costo en cada iteración a través de una función `callback`.

---

### 5. Resultados, Gráfica de Convergencia y Validación
* **Convergencia del Costo:** La curva de costo $J(\theta)$ evidenció un descenso monótono estable hasta converger en su valor mínimo óptimo.
* **Precisión (*Accuracy*):** Rendimiento homogéneo cercano al **$78\%$ tanto en entrenamiento como en prueba**, confirmando que el clasificador generaliza adecuadamente sin sobreajuste (*overfitting*).
* **Matriz de Confusión:** La evaluación sobre las $8\,040$ muestras aisladas de prueba demostró una discriminación simétrica y equilibrada para ambas clases.
* **Inferencia Individual:** Se realizaron inferencias sobre casos puntuales del conjunto de prueba comparando las probabilidades $P(y=1|x) \ge 0.5$ contra las etiquetas reales, validando la efectividad del clasificador.

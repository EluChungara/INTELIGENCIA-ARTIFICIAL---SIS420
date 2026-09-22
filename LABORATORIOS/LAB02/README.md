# LABORATORIO 02: REGRESIÓN LINEAL MÚLTIPLE CON DESCENSO POR EL GRADIENTE Y ECUACIÓN DE LA NORMAL

**Materia:** Inteligencia Artificial I (SIS-420)  
**Estudiante:** Chungara Choque Elizabeth 
**Docente:** Ing. Carlos Walter Pacheco Lora  
**Entorno de Desarrollo:** Visual Studio Code / Jupyter Notebook (`.ipynb`)  

---

## 📌 Enlaces del Proyecto

* **Video Explicativo:** [Ver Video en Google Drive](https://drive.google.com/file/d/12SPrn_MnXWcsAtC79u0GDWBg2kXfZMUB/view?usp=sharing)
* **Dataset Oficial (Kaggle):** [Allstate Claims Severity Dataset](https://www.kaggle.com/c/allstate-claims-severity/data)
* **Archivo de Datos Utilizado:** `train.csv` (188,318 filas y 24 características de entrada seleccionadas)

---

## 📖 Descripción del Trabajo Realizado

En esta práctica de laboratorio se adaptó el cuadernillo base de **Regresión Lineal Multivariable** revisado en clases para entrenar un modelo supervisado capaz de predecir la severidad o costo de pérdida continua de un reclamo de seguros (`loss`) a partir de múltiples variables de entrada.

Se implementó el preprocesamiento con la librería **Pandas**, la estandarización **Z-Score**, el algoritmo iterativo de **Descenso por el Gradiente por Lotes Vectorizado** minimizando la función de costo $J(\theta)$, y se contrastaron las inferencias contra la solución analítica cerrada de la **Ecuación de la Normal**.

---

### 1. Especificaciones del Dataset y Cumplimiento de Restricciones

* **Nombre del Dataset:** Allstate Claims Severity (`train.csv`).
* **Cantidad de Muestras ($m$):** Contiene **$188\,318$ ejemplos de entrenamiento útiles**, superando la restricción de $m \ge 20\,000$.
* **Cantidad de Propiedades ($n$):** Se estructuró una matriz con **$24$ características numéricas** ($n \ge 20$), conformada por 10 variables categóricas codificadas numéricamente (`cat1` a `cat10`) y 14 variables continuas (`cont1` a `cont14`).
* **Variable Objetivo ($y$):** Costo continuo de pérdida monetaria por reclamo (`loss`), con un rango de valores reales que va desde $\$0.67$ hasta $\$121\,012.25$.
* **Auditoría de Calidad con Pandas:** Se realizó la lectura mediante `pd.read_csv` con descarte de filas defectuosas (`on_bad_lines='skip'`) y remoción de nulos con `dropna()`, confirmando $0\text{ NaN}$ y ausencia de valores corruptos.

---

### 2. Preprocesamiento, Normalización Z-Score y Matriz de Diseño

1. **Codificación Numérica con Pandas:** Se utilizó `.astype('category').cat.codes` para mapear las categorías de texto a enteros binarios ($0$ y $1$).
2. **Normalización de Características (Z-Score):** Para evitar que la escala de las variables continuas desestabilizara el gradiente o provocara desbordamientos numéricos, se aplicó:
   $$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
   Donde $\mu$ representa el vector de medias y $\sigma$ el vector de desviaciones estándar por columna.
3. **Término de Intercepción ($x_0 = 1$):** Se concatenó una columna de unos a la izquierda de la matriz normalizada (`np.concatenate`), conformando una matriz de diseño final de **$188\,318$ filas y $25$ columnas** ($1$ término de sesgo $\theta_0$ $+ 24$ pesos ponderados $\theta_j$).

---

### 3. Formulación Matemática y Algoritmos Vectorizados

* **Función de Costo Cuadrático Medio Vectorizada:**
  $$J(\theta) = \frac{1}{2m} (X\theta - \vec{y})^T (X\theta - \vec{y}) = \frac{1}{2m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

* **Descenso por el Gradiente por Lotes:**
  $$\theta := \theta - \frac{\alpha}{m} X^T (X\theta - \vec{y})$$

* **Solución Analítica (Ecuación de la Normal):**
  $$\theta = \left( X^T X \right)^{-1} X^T \vec{y}$$
  *(Implementada con `np.linalg.pinv` sobre los datos sin normalizar para garantizar estabilidad numérica ante grandes volúmenes)*.

---

### 4. Resultados Cuantitativos y Curva de Convergencia

* **Hiperparámetros de Entrenamiento:** $\alpha = 0.01$, $1\,000$ iteraciones.
* **Costo Inicial $J(\vec{0})$:** $8\,699\,220.55$
* **Costo Final Mínimo Alcanzado $J(\theta)$:** $3\,070\,795.68$ (Reducción del **$64.70\%$** en el error de costo).
* **Comportamiento Gráfico:** La curva de costo frente a las iteraciones exhibe una trayectoria estrictamente asintótica y monótona decreciente, estabilizándose a partir de la iteración 400 sin oscilaciones ni divergencias.

---

### 5. Comparación de Modelos e Inferencia

Se evaluó la capacidad de inferencia sobre el primer caso registrado en el dataset, comparando las estimaciones frente al valor real:

| Método / Métrica | Valor Obtenido |
| :--- | :---: |
| **Valor Real Registrado ($y$)** | **$\$2\,213.18$** |
| **Predicción - Descenso por el Gradiente ($h_\theta(x)$)** | **$\$2\,472.21$** |
| **Predicción - Ecuación de la Normal ($\theta_{\text{normal}}^T x$)** | **$\$2\,482.85$** |
| **Diferencia entre Métodos de Optimización** | **$\$10.64$** |
| **Error Absoluto de la Inferencia** | **$\$259.03$** (Error relativo: $11.7\%$) |

---

### 6. Conclusiones

* La normalización Z-Score resultó esencial para permitir que el descenso por el gradiente por lotes opere sobre más de 188,000 registros sin divergencia numérica.
* La discrepancia de tan solo $\$10.64$ entre el método iterativo y la Ecuación de la Normal valida formalmente que el algoritmo convergió con precisión al mínimo global del error cuadrático medio.

# Laboratorio 05: Clasificación Multiclase One-vs-All 

**Materia:** Inteligencia Artificial I (SIS-420)  
**Docente:** Ing. Carlos Walter Pacheco Lora  

---

##  Enlaces del Laboratorio

* **Cuadernillo en Google Colab:** [Abrir Cuadernillo Colab](https://colab.research.google.com/drive/1btRsUkeW5GOmnWE-hp0wX7QvDlHEKqW0?usp=sharing)
* **Video Explicativo:** [Ver Video en Google Drive](https://drive.google.com/file/d/14FB8CWZNXQkCwwHRaO1Z7xbo3Epbs7HD/view?usp=sharing)
* **Dataset:** [Descargar NumtaDB (CSV) en Google Drive](https://drive.google.com/file/d/1RZnY86UE7Dv0ZzzPefWUfbJlzy6LMuWi/view?usp=sharing)

---

## 📖Descripción del Lab

En esta práctica se implementó un sistema de reconocimiento óptico y clasificación de dígitos manuscritos utilizando la estrategia multiclase **One-vs-All** basada en **Regresión Logística Regularizada** ($L_2$).

### 1. Especificaciones del Dataset
* **Dataset:** NumtaDB (Bengali Handwritten Digits) / MNIST estructurado.
* **Muestras ($m$):** $60\,000$ ejemplos tabulados ($m \ge 50\,000$).
* **Resolución:** $784$ píxeles correspondientes a imágenes uniformes de **$28 \times 28$ píxeles** ($\ge 20 \times 20$).
* **Clases ($K$):** 10 clases balanceadas ($0$ al $9$) con aproximadamente $6\,000$ registros por dígito.

### 2. Metodología e Implementación
* **Preprocesamiento:** Escalado Min-Max de píxeles al rango continuo $[0, 1]$ y partición estricta sin traslape en **$80\%$ para Entrenamiento** ($48\,000$ muestras) y **$20\%$ para Prueba** ($12\,000$ muestras).
* **Término de Intercepción:** Concatenación matricial de una columna de unos ($x_0 = 1$), conformando matrices de entrada de **$785$ columnas** ($1$ sesgo $+ 784$ características).
* **Entrenamiento One-vs-All:** Binarización iterativa de las etiquetas ($y_{\text{binario}} \in \{0, 1\}$) y optimización de 10 clasificadores binarios independientes mediante el método de **Gradiente Conjugado (`CG`)** con regularización $\lambda = 0.1$, protegiendo el parámetro de sesgo $\theta_0$ de la penalización.
* **Inferencia Multiclase:** Selección de la clase ganadora aplicando el criterio $\hat{y} = \arg\max_c P(y=c|x)$ sobre las probabilidades calculadas con la función sigmoide.

### 3. Resultados y Métricas de Rendimiento
* **Precisión en Entrenamiento (80%):** **92.18%**
* **Precisión en Datos de Prueba (20%):** **91.18%**
* **Generalización:** La brecha de tan solo $1.00\%$ entre entrenamiento y prueba ratifica el control del sobreajuste (*overfitting*).
* **Validación:** Se comprobó la convergencia decreciente monótona del costo $J(\theta)$, una diagonal principal dominante en la Matriz de Confusión y aciertos en pruebas visuales cualitativas sobre datos nunca antes vistos.

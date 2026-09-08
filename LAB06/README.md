# LABORATORIO 06: RECONSTRUCCIÓN DE MODELOS CON PYTORCH

**Materia:** Inteligencia Artificial I (SIS-420)  
**Estudiante:** ELIZABETH CHUNGARA CHOQUE
**Docente:** Ing. Carlos Walter Pacheco Lora  
**Entorno de Desarrollo:** Visual Studio Code / Google Colab (`.ipynb`)  

---

## 📌 Enlaces del Proyecto

* **Video Explicativo:** [Ver Video de Defensa en Google Drive](https://drive.google.com/file/d/1buCLG3oTrPRYqYgWDrkVIvJnO7Ju0KqY/view?usp=sharing)
* **Repositorio Principal:** [Enlace a la carpeta principal de tu GitHub]
* **Cuadernillo Completo:** [Enlace a tu archivo .ipynb en GitHub o Google Colab]

---

## 📖 Descripción del Trabajo Realizado

En este repositorio se presenta la resolución íntegra del **Laboratorio 06**, cuyo objetivo principal fue reconstruir y optimizar los modelos de Machine Learning desarrollados en los primeros cinco laboratorios de la materia, migrándolos completamente al ecosistema del framework **PyTorch**. 

Se abandonaron los cálculos manuales de derivadas y polinomios para adoptar un enfoque profesional y orientado a tensores, implementando una arquitectura estandarizada para todos los modelos.

---

## 🛠️ Arquitectura y Componentes de PyTorch Implementados

Para cumplir con las exigencias metodológicas del laboratorio, todos los cuadernillos comparten la siguiente estructura técnica:

1. **Gestión de Datos (`torch.utils.data`):** 
   * Implementación de la clase personalizada `Dataset` (`__init__`, `__len__`, `__getitem__`) para encapsular tensores de características y etiquetas.
   * Uso de `DataLoader` para procesar el entrenamiento mediante mini-lotes (*mini-batches*) e inyectar estocasticidad (`shuffle=True`).
2. **Modelado Neuronal (`torch.nn`):**
   * Capas lineales `nn.Linear` para regresiones y clasificaciones simples.
   * Arquitecturas multicapa `nn.Sequential` con funciones de activación no lineal `nn.ReLU` para aproximar funciones polinómicas.
3. **Criterios de Costo y Optimización:**
   * **Regresión:** `nn.MSELoss()`.
   * **Clasificación Binaria y Multiclase:** `nn.BCEWithLogitsLoss()`.
   * **Optimizador:** Descenso por Gradiente Estocástico puro (`torch.optim.SGD`).
4. **Persistencia del Modelo:**
   * Guardado periódico de pesos (`model.state_dict()`) en carpetas de *checkpoints* utilizando `torch.save` y la librería `os`.

---

## 📊 Resumen de Laboratorios Reconstruidos

| Laboratorio | Modelo Implementado | Dataset Utilizado | Métrica Principal Evaluada |
| :--- | :--- | :--- | :--- |
| **Lab 01** | Regresión Lineal Simple | Años de Experiencia vs Salario | MSE / Gráfica de Convergencia |
| **Lab 02** | Regresión Lineal Multivariable | *Allstate Claims Severity* | RMSE / Predicciones vs Reales |
| **Lab 03** | Regresión No Lineal (Polinómica) | *Facebook Comment Volume* | MAE / RMSE |
| **Lab 04** | Regresión Logística Binaria | *Santander Customer Transaction* | Exactitud (~78%) / Matriz de Confusión |
| **Lab 05** | Clasificación Multiclase One-vs-All | *NumtaDB* (60k Dígitos Manuscritos) | Exactitud (>91%) / Inferencia Visual |

---
*Desarrollado para la validación y evaluación continua del semestre.*

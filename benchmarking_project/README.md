# Quantifying the Deployment Trade-Off: A Comparative Benchmark of Deep Learning Inference Frameworks (MobileNetV2)

## Descripción Breve del Objetivo

Evaluar y cuantificar las compensaciones (*trade-offs*) entre **rendimiento (throughput)**, **estabilidad (latencia P99)**, y **costo de implementación (tiempo de compilación)** en la inferencia de modelos de Deep Learning, comparando directamente aceleradores de GPU, optimizadores de CPU de Intel, y runtimes de Edge.

---

### Problema a Resolver

El ecosistema de inferencia de Deep Learning está altamente fragmentado, obligando a los ingenieros a elegir entre múltiples frameworks (TensorRT, OpenVINO, TFLite, etc.) sin datos comparativos estandarizados. Se busca ofrecer una guía cuantitativa y prescriptiva para seleccionar la herramienta óptima según el entorno de despliegue.

### Enfoque

Se implementó un arnés de benchmarking modular en Python para medir consistentemente la latencia (P50/P99) y el *throughput* del modelo **MobileNetV2** con un *Batch Size* de 64. Se ejecutaron y optimizaron cinco vías de inferencia distintas en un solo sistema:

1.  **Aceleración Máxima:** NVIDIA TensorRT (FP16 Traced).
2.  **Optimización de Vendedor:** Intel OpenVINO (FP32 en CPU e iGPU).
3.  **Línea Base:** PyTorch Native (FP32 en CPU y CUDA/GPU).
4.  **Estimación Edge:** TFLite y ExecuTorch (INT8 Estimates).

---

## Resumen del Trabajo Realizado

### Principales Tareas Completadas

* **Integración de Frameworks:** Desarrollo de scripts para conversión de PyTorch a formatos específicos de OpenVINO (IR) y TensorRT (Engine).
* **Diseño del Arnés:** Implementación del protocolo de medición con sincronización CUDA para precisión.
* **Análisis Cuantitativo:** Cálculo de factores de aceleración clave ($\mathbf{1.78\times}$ y $\mathbf{6.23\times}$) y análisis de la estabilidad en el *tail latency* (P99/P50 ratio).

### Técnicas, Modelos o Herramientas Utilizadas

* **Modelo:** MobileNetV2 (Pre-entrenado en ImageNet).
* **Técnicas:** JIT Tracing, Graph Optimization, Half-Precision (FP16), 8-bit Integer Quantization (INT8 Estimates), Vectorized Instruction Sets (AVX/OpenVINO).
* **Plataforma de Hardware:** Intel Core i7-12700H / NVIDIA GeForce RTX 3060 Laptop GPU.
* **Software:** PyTorch 2.0.1, NVIDIA TensorRT, OpenVINO 2024.6, Python 3.9.

---

## Estructura del Código

Los scripts clave se encuentran en la carpeta **benchmarking_project** y se enfocan en la ejecución del benchmark.

| Archivo | Función Principal |
| :--- | :--- |
| `master_comparison.py` | **Script principal.** Coordina la ejecución de todos los benchmarks, recolecta y procesa los resultados finales. |
| `pt_benchmark.py` | Contiene las funciones de medición para **PyTorch Native** (CPU y CUDA/GPU). |
| `trt_benchmark.py` | Contiene las funciones para crear y medir el *engine* de **NVIDIA TensorRT** (FP16). |
| `ov_benchmark.py` | Contiene las funciones para convertir y ejecutar el modelo con **Intel OpenVINO** (CPU e iGPU). |
| `tflite_benchmark.py` | Script para la medición/estimación de rendimiento de **TensorFlow Lite**. |
| `executorch_benchmark.py` | Script para la medición/estimación de rendimiento de **ExecuTorch** (XNNPACK backend). |
| `static_setup.py` | Contiene las funciones y configuraciones iniciales, como la carga del modelo base y la generación del tensor de entrada. |


## Instrucciones Básicas para Ejecutar el Código

### Dependencias Principales

* `torch==2.0.1+cu117`
* `torchvision==0.15.2+cu117`
* `openvino==2024.6`

### Comandos Esenciales

1.  **Instalar dependencias:**
    ```bash
    pip install numpy pandas
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install openvino==2024.6
    # La librería de TensorRT se utiliza a través de su integración nativa en PyTorch (torch.export.backend).
    ```
2. **Ejecutar el Benchmark Completo (desde la carpeta benchmarking_proyect):**
    ```bash
    python master_comparison.py
    ```

---

## Resultados y Conclusiones

El benchmark demostró que la elección del framework debe ser específica para el hardware, confirmando los siguientes hallazgos:

| Métrica | Framework Óptimo | Valor Clave | Conclusión |
| :--- | :--- | :--- | :--- |
| **Aceleración (GPU)** | NVIDIA TensorRT | $\mathbf{1.78\times}$ speedup | Dominio en rendimiento con FP16 y compilación estática. |
| **Aceleración (CPU)** | Intel OpenVINO | $\mathbf{6.23\times}$ speedup | Logra la mejor optimización de CPU mediante instrucciones AVX. |
| **Costo vs. Estabilidad** | OpenVINO | $\approx 7-10$ s overhead | Alto costo de compilación, pero con baja varianza (P99) en runtime. |

### Próximos Pasos Recomendados

1.  **Validación en Embedded Systems:** Despliegue y medición de TensorRT en una plataforma **NVIDIA Jetson**.
2.  **Pruebas de Escalabilidad en Servidor:** Ejecutar el benchmark en un servidor de **Lambda Labs** para cuantificar el rendimiento máximo.
3.  **Validación INT8:** Medición física de TFLite/Executorch en un dispositivo Edge de bajo consumo para validar las estimaciones.
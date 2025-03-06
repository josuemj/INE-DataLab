# INE-DATALAB

Este repositorio contiene el código y los datos utilizados para hacer **Mineria** de datos del Instituto Nacional de Estadística (INE). El objetivo es explorar tendencias, correlaciones y patrones en los datos de estadísticas vitales y de violencia en Guatemala, y en base a eso tomar una problematica y poder resolverla utilizando solo datos.

## 📁 Estructura del Repositorio

```plaintext
📦 INE-DATALAB
│── 📂 data/                   # Datos en bruto y procesados
│   ├── 📂 raw/                # Datos originales sin modificaciones
│   ├── 📂 processed/          # Datos limpios y transformados
│── 📂 notebooks/              # Jupyter Notebooks organizados por fases
│   ├── 01_exploracion.ipynb   # Exploración inicial, descripción de variables
│   ├── 02_visualizacion.ipynb # Gráficos exploratorios
│   ├── 03_clustering.ipynb    # Algoritmos de clustering y validación
│   ├── 04_conclusiones.ipynb  # Hallazgos y conclusiones
│── 📂 reports/                # Informes generados
│── 📂 src/                    # Código fuente y funciones auxiliares
│   ├── data_preprocessing.py  # Funciones para limpieza y transformación de datos
│── 📄 .gitignore              # Archivos y carpetas a ignorar en Git
│── 📄 requirements.txt        # Librerías necesarias para el proyecto
```

## 🚀 Cómo Usar el Repositorio
1. Clonar el repositorio:
```bash
git clone https://github.com/tu_usuario/INE-DATALAB.git
cd INE-DATALAB
```
2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecutar los notebooks en Jupyter:
```bash
jupyter notebook
```

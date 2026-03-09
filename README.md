# House Price Prediction

Website prediksi harga rumah berbasis machine learning yang dibangun menggunakan Streamlit. 
Project ini dibangun untuk memudahkan user memprediksi nilai harga rumah berdasarkan beberapa fitur seperti tingkat kriminalitas (CRIM), kualitas udara (NOX), jumlah kamar (RM), akses ke jalan (RAD), pajak (TAX), dan variabel lainnya.

## Features
- Prediksi harga rumah 
- Input fitur/data rumah melalui antarmuka website
- Model machine learning yang telah ditraining dan disimpan dalam format `.pkl`
- Tampilan website yang mudah digunakan

## Library & Tools
- Python
- Streamlit
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Project Workflow
1. Data preprocessing
2. Exploratory Data Analysis (EDA)
3. Model training and evaluation
4. Model selection
5. Deployment with Streamlit

## Files
- `app.py` → main Streamlit app
- `model.py` → model/prediction 
- `ui.py` → user interface
- `best_model.pkl` → trained model
- `House_Price_Prediction(model).ipynb` → notebook for model building

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

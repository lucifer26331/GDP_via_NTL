# **GDP Prediction Using Nighttime Light Data and Machine Learning**

## **Project Overview**
This project aims to estimate Gross Domestic Product (GDP) using Nighttime Light (NTL) data obtained from Google Earth Engine (GEE) and machine learning techniques. The methodology integrates remote sensing data with economic indicators, leveraging data scraping and advanced AI models to derive meaningful insights.

---

## **Project Components**
This project consists of multiple Google Earth Engine scripts and Python-based machine learning models. Below is a detailed breakdown of the workflow and file descriptions.

### **1. Google Earth Engine Scripts**
These scripts are used to extract and process nighttime light intensity data from satellite imagery.

#### **(a) NTL Extraction and Summation per District**  
**Filename: `district_ntl.js`**  
- Loads datasets from GEE (Administrative boundaries, NTL collection).  
- Filters data for India and extracts district-wise boundaries.  
- Computes the **Sum of Lights (SOL)** for each district using the `mean` function.
- Exports the results as a CSV file with `district-wise` SOL data from 2013 to 2021.  

#### **(b) NTL Processing with Land Cover Data**  
**Filename: `monthly_ntl.js`**  
- Loads **MODIS Land Cover data** along with NTL.
- Extracts **urban and agricultural regions** using MODIS classification.
- Computes SOL separately for **urban and agricultural** areas.
- Exports the results as a CSV with time-series SOL data classified by land type.

---

### **2. Machine Learning Implementation**
After acquiring the processed NTL data, we employ machine learning models to predict GDP based on the extracted features.

#### **(a) Complete Workflow for GDP Prediction**  
**Filename: `complete_workflow_code.py`**  
- Loads the **processed NTL dataset**.
- Scrapes GDP data from economic sources.
- Preprocesses the dataset and merges **SOL features with GDP values**.
- Implements multiple machine learning models for prediction.
- Evaluates performance using regression metrics.

#### **(b) Model Application & Evaluation**  
**Filename: `Applying final models.ipynb`**  
- Uses the final trained model to make **GDP predictions** based on the latest NTL data.
- Performs feature importance analysis.
- Evaluates prediction accuracy using real-world GDP data.
- Visualizes GDP predictions vs. actual values.

---

## **Key Takeaways**
- Google Earth Engine facilitates **large-scale data extraction** for remote sensing applications.
- Nighttime Light data is a **strong indicator of economic activity** and can be leveraged for GDP estimation.
- **Machine learning models** effectively learn patterns from NTL and provide reliable GDP predictions.

This project provides a foundation for further research in **economic forecasting using satellite imagery**.

---

**Contributor:** parth Garg
**Tools Used:** Google Earth Engine, Python, Machine Learning, Data Scraping  
**License:** Open-source

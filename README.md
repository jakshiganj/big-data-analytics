# ✈️ Flight Delay Analysis & Prediction Pipeline with PySpark

## 📌 Project Overview
This project is an end-to-end Big Data pipeline designed to analyze, visualize, and predict commercial flight delays. Built entirely using **Apache Spark (PySpark)** on **Databricks**, the project processes millions of historical aviation records (2016-2018). It moves through automated data extraction, rigorous data cleaning, exploratory data analysis (EDA), and scalable machine learning classification.

## 🛠️ Tech Stack & Environment
* **Language:** Python 3.x
* **Big Data Framework:** Apache Spark (PySpark SQL, PySpark MLlib)
* **Environment:** Databricks (Spark Connect, Unity Catalog Volumes)
* **Data Sources:** Kaggle API integration
* **Visualization:** Matplotlib, Seaborn, Databricks native charting

## 📥 Data Ingestion & Storage
* **Automated Extraction:** Integrated the Kaggle API directly into the Databricks notebook to programmatically download the `airline-delay-and-cancellation-data-2009-2018` dataset.
* **Volume Management:** Provisioned a Unity Catalog Volume (`workspace.default.airline_data`) to securely store and govern access to the unzipped raw CSV files across the workspace.
* **Scalable Loading:** Loaded the 2016, 2017, and 2018 datasets into Spark DataFrames, processing over **18.5 million rows** of flight data.

## 🧹 Data Cleaning & Preprocessing Pipeline
To ensure high data quality for analysis and ML, a comprehensive, reusable `preprocess_airline_data()` pipeline was engineered:

1. **Standardization:** Lowercased and stripped special characters from all column names. Filtered down to the 28 most relevant flight columns.
2. **Deduplication:** Dropped exact row duplicates to prevent data leakage.
3. **Type Casting:** Converted string representations of time, distance, and delays into appropriate PySpark `Double`, `Integer`, and `DateType` formats.
4. **Missing Value Handling:**
   * Calculated missing percentages dynamically.
   * Automatically dropped any column with **> 50% missing values**.
   * Imputed remaining numeric missing values using the column **mean**.
   * Filled remaining categorical missing values with `"Unknown"`.
   * Dropped rows missing critical identifiers (`airline`, `origin`, `dest`, `flight_date`).
5. **Outlier Removal:** Filtered logically impossible/invalid values (e.g., negative distances, negative air times, and extreme delay anomalies).
6. **Feature Engineering:** Derived high-value columns for downstream EDA and ML:
   * `route`: Concatenated origin and destination (e.g., "JFK-LAX").
   * `scheduled_dep_hour`: Extracted the specific hour of departure from CRS time.
   * `is_delayed`: Binary target variable (1 if arrival delay > 15 mins, else 0).
   * `flight_status`: Categorized as 'Operated' or 'Cancelled'.
   * `delay_category`: Segmented into "On Time / Minor Delay", "Moderate Delay", and "Severe Delay".
   * Temporal breakdown: `year`, `month`, `month_name`, `day`, and `day_of_week`.

## 📊 Exploratory Data Analysis (EDA) & Visualization
Leveraging the cleaned dataset and derived features, extensive chart plotting and analysis were conducted to uncover aviation trends:
* **Temporal Trends:** Analyzed total flight volume and delay frequencies across different months (`month_name`) and days of the week to identify peak congestion periods.
* **Route & Airport Congestion:** Plotted the highest traffic routes and calculated average taxi-in/taxi-out times at top origin airports.
* **Delay Categorization:** Visualized the distribution of "On Time" vs. "Moderate" vs. "Severe" delays to understand the severity scale of schedule disruptions.
* **Airline Performance:** Compared average delay times across different operating carriers to identify the most and least punctual airlines.

## 🧠 Machine Learning Pipeline
The project utilizes PySpark's `ml` library to build a distributed binary classification pipeline predicting the `is_delayed` variable.

1. **Feature Selection:** * Stripped high-cardinality categoricals (like exact airport codes) to optimize distributed memory footprint over Spark Connect.
   * Utilized core numerical and temporal features: `distance`, `taxi_out`, `taxi_in`, `scheduled_dep_hour`, `day_of_week`, and `month`.
2. **Vectorization & Balancing:**
   * Used PySpark’s `VectorAssembler` to pack features into a dense vector format.
   * Addressed class imbalances (`balanced_train_df`) so models weren't biased toward the "On-Time" majority class.
3. **Model Training & Tuning:**
   * Evaluated powerful tree-based models: **Decision Trees** and **Gradient Boosted Trees (GBT)**.
   * Replaced manual looping with PySpark's native **`TrainValidationSplit`** and `ParamGridBuilder` for highly efficient, memory-safe hyperparameter tuning (`maxDepth`, `minInstancesPerNode`).
4. **Evaluation:**
   * Assessed performance using `BinaryClassificationEvaluator` (ROC AUC) and `MulticlassClassificationEvaluator` (Accuracy).

## 🚀 Key Technical Challenges Solved
* **Spark Connect Cache Overflow:** Encountered `CONNECT_ML.ML_CACHE_SIZE_OVERFLOW_EXCEPTION` due to the Spark server holding too many complex model states during manual loops. Solved by refactoring to use PySpark's native ML tuning, delegating memory management to the JVM.
* **High-Cardinality Memory Exhaustion:** Discovered that converting hundreds of airport categorical codes triggered exponential decision tree growth. Stripped the pipeline back to a robust "numeric and temporal baseline" model to ensure stability and fast execution.

## 💻 How to Run
1. Clone this repository: `git clone https://github.com/yourusername/flight-delay-pyspark.git`
2. Import the `.ipynb` notebooks into a Databricks Workspace.
3. Ensure your cluster is running Databricks Runtime for Machine Learning.
4. Input your Kaggle API key in the designated setup cell to fetch the raw data.
5. Run the notebook sequentially from data ingestion through ML evaluation. *(Note: If running on a Databricks Shared cluster, Unity Catalog Volume paths are strictly enforced for caching).*

# Import necessary libraries
from fastapi import FastAPI, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse,RedirectResponse,StreamingResponse
import pandas as pd
from io import BytesIO
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from io import StringIO
import os
from urllib.parse import quote
from keras.layers import LSTM
from scipy import stats
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import plotly.graph_objects as go
from plotly.io import to_image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from scipy.stats import norm
import plotly.graph_objs as go
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import tensorflow as tf
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import base64
import threading
import time
import os
import base64
from urllib.parse import quote


def delayed_file_delete(file_path, delay=600):
    """
    Deletes the specified file after a delay.
    
    :param file_path: Path to the file to be deleted.
    :param delay: Delay in seconds before the file is deleted.
    """
    time.sleep(delay)
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")


def z_score_anomaly_detection(data, column, threshold):
    z_scores = stats.zscore(data[column])
    data['ZScore'] = z_scores
    data['Anomaly'] = (np.abs(z_scores) > threshold).astype(int)
    data['PointColor'] = data['Anomaly'].apply(lambda x: 'Outlier' if x == 1 else 'Inlier')
    return data


def calculate_first_digit(data):
    idx = np.arange(0, 10)
    first_digits = data.astype(str).str.strip().str[0].astype(int)
    counts = first_digits.value_counts(normalize=True, sort=False)
    benford = np.log10(1 + 1 / np.arange(0, 10))

    df = pd.DataFrame(data.astype(str).str.strip().str[0].astype(int).value_counts(normalize=True, sort=False)).reset_index()
    df1 = pd.DataFrame({'index': idx, 'benford': benford})
    return df, df1, counts, benford




def drop_features_with_missing_values(data):
        # Calculate the number of missing values in each column
        missing_counts = data.isnull().sum()

        # Get the names of columns with missing values
        columns_with_missing_values = missing_counts[missing_counts > 0].index

        # Drop the columns with missing values
        data_dropped = data.drop(columns=columns_with_missing_values)
        return data_dropped
    

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")







    



def apply_anomaly_detection_IsolationForest(data):
        # Make a copy of the data
        data_copy = data.copy()

        # Fit the Isolation Forest model
        isolation_forest = IsolationForest(contamination=0.03, random_state=42)
        isolation_forest.fit(data_copy)

        # Predict the anomaly labels
        anomaly_labels = isolation_forest.predict(data_copy)

        # Create a new column in the original DataFrame for the anomaly indicator
        data['Anomaly'] = np.where(anomaly_labels == -1, 1, 0)
        return data


def apply_anomaly_detection_LocalOutlierFactor(data, neighbors=200):
        lof = LocalOutlierFactor(n_neighbors=neighbors, contamination='auto')
        data['Anomaly'] = lof.fit_predict(data)
        data['Anomaly'] = np.where(data['Anomaly'] == -1, 1, 0)
        return data


# def apply_anomaly_detection_LocalOutlierFactor(data):
#     


#         # Make a copy of the data
#         data_copy = data.copy()

#         from sklearn.neighbors import LocalOutlierFactor

#         # Step 3: Apply Local Outlier Factor
#         lof = LocalOutlierFactor(n_neighbors=200, metric='euclidean', contamination=0.04)

#         outlier_labels = lof.fit_predict(data_copy)

#         # Display the outlier labels for each data point
#         data['Outlier_Label'] = outlier_labels
#         return data
#     except Exception as e:
#         raise CustomException(e, sys)



def find_duplicate_vendors(vendors_df, threshold):
    
        duplicates = []
        lf = vendors_df.copy()
        lf['NAME1'] = lf['NAME1'].astype(str)
        vendor_names = lf['NAME1'].unique().tolist()
        columns = ['Vendor 1', 'Vendor 2', 'Score']
        df_duplicates = pd.DataFrame(data=[], columns=columns)

        for i, name in enumerate(vendor_names):
            # Compare the current name with the remaining names
            matches = process.extract(name, vendor_names[i+1:], scorer=fuzz.ratio)

            # Check if any match exceeds the threshold
            for match, score in matches:
                if score >= threshold:
                    duplicates.append((name, match))
                    df_duplicates.loc[len(df_duplicates)] = [name, match, score]

        return duplicates, df_duplicates
    



def apply_anomaly_detection_OneClassSVM(data):
        # Copy the original data to avoid modifying the original dataframe
        data_with_anomalies = data.copy()

        # Perform One-Class SVM anomaly detection
        clf = OneClassSVM(nu=0.05)
        y_pred = clf.fit_predict(data)
        data_with_anomalies['Anomaly'] = np.where(y_pred == -1, 1, 0)

        return data_with_anomalies

def train_autoencoder(data):
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Define the autoencoder model architecture
        input_dim = data.shape[1]
        encoding_dim = int(input_dim / 2)  # You can adjust this value as needed
        autoencoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])

        # Compile and train the autoencoder with verbose=1
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(scaled_data, scaled_data, epochs=100, batch_size=64, shuffle=True, verbose=1)  # Set verbose to 1

        # Get the encoded data
        encoded_data = autoencoder.predict(scaled_data)

        # Calculate the reconstruction error
        reconstruction_error = np.mean(np.square(scaled_data - encoded_data), axis=1)

        # Add the reconstruction error as a new column 'ReconstructionError' to the data
        data['ReconstructionError'] = reconstruction_error

        return data

    

def apply_anomaly_detection_autoencoder(data):
        # Train the autoencoder and get the reconstruction error
        data_with_reconstruction_error = train_autoencoder(data)

        # Set a threshold for anomaly detection (you can adjust this threshold)
        threshold = data_with_reconstruction_error['ReconstructionError'].mean() + 3 * data_with_reconstruction_error['ReconstructionError'].std()

        # Classify anomalies based on the threshold
        data_with_reconstruction_error['Anomaly'] = np.where(data_with_reconstruction_error['ReconstructionError'] > threshold, 1, 0)

        return data_with_reconstruction_error

    


def apply_anomaly_detection_SGDOCSVM(data):
        # Copy the original data to avoid modifying the original dataframe
        data_with_anomalies = data.copy()
        # Perform One-Class SVM anomaly detection using SGD solver
        clf = SGDOneClassSVM(nu=0.05)
        clf.fit(data)
        y_pred = clf.predict(data)
        data_with_anomalies['Anomaly'] = np.where(y_pred == -1, 1, 0)

        return data_with_anomalies



def calculate_first_digit(data):
        idx = np.arange(0, 10)
        first_digits = data.astype(str).str.strip().str[0].astype(int)
        counts = first_digits.value_counts(normalize=True, sort=False)
        benford = np.log10(1 + 1 / np.arange(0, 10))

        df = pd.DataFrame(data.astype(str).str.strip().str[0].astype(int).value_counts(normalize=True, sort=False)).reset_index()
        df1 = pd.DataFrame({'index': idx, 'benford': benford})
        return df, df1, counts, benford

def calculate_2th_digit(data):
        idx = np.arange(0, 100)
        nth_digits = data.astype(int).astype(str).str.strip().str[:2]
        numeric_mask = nth_digits.str.isnumeric()
        counts = nth_digits[numeric_mask].astype(int).value_counts(normalize=True, sort=False)
        benford = np.log10(1 + 1 / np.arange(0, 100))

        df = pd.DataFrame(data.astype(int).astype(str).str.strip().str[:2].astype(int).value_counts(normalize=True, sort=False)).reset_index()
        df1 = pd.DataFrame({'index': idx, 'benford': benford})

        return df, df1, counts, benford

def calculate_3th_digit(data):
        idx = np.arange(100, 1000)
        nth_digits = data.astype(int).astype(str).str.strip().str[:3]
        numeric_mask = nth_digits.str.isnumeric()
        counts = nth_digits[numeric_mask].astype(int).value_counts(normalize=True, sort=False)
        benford = np.log10(1 + 1 / np.arange(100, 1000))

        df = pd.DataFrame(data.astype(int).astype(str).str.strip().str[:3].astype(int).value_counts(normalize=True, sort=False)).reset_index()
        df1 = pd.DataFrame({'index': idx, 'benford': benford})

        return df, df1, counts, benford





def apply_anomaly_detection_GMM(data):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture()
        data['Anomaly'] = gmm.fit_predict(data)
        data['Anomaly'] = np.where(data['Anomaly'] == 1, 0, 1)
        return data



def navbar():
    
        
        tabs = ["HOME", "ABOUT","EXCEL TO CSV", "PROCESS MINING","EXPLORATORY DATA ANALYSIS", "STATISTICAL METHODS", "MACHINE LEARNING METHODS", "DEEP LEARNING METHODS", "TIME SERIES METHODS"]
        tab0, tab1,tab2, tab3,tab8,tab4, tab5, tab6, tab7 = st.tabs(tabs)

        with tab0:
            pass
        with tab8:
            pass

        with tab1:
            st.header("Infrared")
            st.write("A first of its kind concept that lets you discover counterintuitive patterns and insights often invisible due to limitations of the human mind, biases, and voluminous data.")
            st.write("Unleash the power of machine learning and advanced statistics to find outliers and exceptions in your data. This application provides an instant output that can be reviewed and acted upon with agility to stop revenue leakages, improve efficiency, and detect/prevent fraud.")

            st.image("http://revoquant.com/assets/img/infra.jpg", use_column_width=True)


        with tab2:
                
                def convert_excel_to_csv(uploaded_file, page_number):
                    if page_number == 1:
                        excel_data = pd.read_excel(uploaded_file)
                    else:
                        excel_data = pd.read_excel(uploaded_file, sheet_name=page_number - 1)
                    csv_file = BytesIO()
                    excel_data.to_csv(csv_file, index=False)
                    csv_file.seek(0)
                    return csv_file.getvalue()


                st.header("Excel to CSV Converter")
                uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
                selected_page = st.number_input("Enter the page number", min_value=1, value=1)

                if uploaded_file is not None:
                    csv_data = convert_excel_to_csv(uploaded_file, selected_page)
                    st.download_button(
                        "Download CSV file",
                        csv_data,
                        file_name="output.csv",
                        mime="text/csv"
                    )

                    with st.expander("Excel Data"):
                        excel_data = pd.read_excel(uploaded_file, sheet_name=selected_page - 1)
                        st.dataframe(excel_data)

                    with st.expander("Converted CSV Data"):
                        csv_data = pd.read_csv(BytesIO(csv_data))
                        st.dataframe(csv_data)


                

        # Move this code block below the page

        with tab4:
            st.header("Benford's Law: The Mystery Behind the Numbers")
            st.image("https://image2.slideserve.com/4817711/what-is-benford-s-law-l.jpg", use_column_width=True)
            st.write("Have you ever wondered why certain numbers appear more frequently as the first digit in a dataset? "
                        "This phenomenon is known as Benford's Law, and it has been a subject of fascination for mathematicians, "
                        "statisticians, and data analysts for decades.")

            st.subheader("What is Benford's Law?")
            st.write(
                    "Benford's Law, also called the First-Digit Law, states that in many naturally occurring numerical datasets, "
                    "the first digit is more likely to be small (e.g., 1, 2, or 3) than large (e.g., 8 or 9). Specifically, "
                    "the probability of a number starting with digit d is given by the formula: P(d) = log10(1 + 1/d), where "
                    "log10 represents the base-10 logarithm.")

            st.subheader("Applications of Benford's Law")
            st.write(
                    "Benford's Law has found applications in various fields, including forensic accounting, fraud detection, "
                    "election analysis, and quality control. Its ability to uncover anomalies in large datasets makes it "
                    "particularly useful for identifying potential irregularities or discrepancies.")

            st.subheader("Real-World Examples")
            st.write(
                    "Benford's Law can be observed in numerous real-world datasets. For instance, if you examine the lengths "
                    "of rivers worldwide, the population numbers of cities, or even the stock prices of companies, you are "
                    "likely to find that the leading digits follow the predicted distribution.")

            st.subheader("Exceptions and Limitations")
            st.write("While Benford's Law holds true for many datasets, it is not universally applicable. Certain datasets "
                        "with specific characteristics may deviate from the expected distribution. Additionally, Benford's Law "
                        "should not be considered as definitive proof of fraudulent or irregular activities but rather as a tool "
                        "for further investigation.")

            st.subheader("Conclusion")
            st.write("Benford's Law offers a fascinating insight into the distribution of numbers in various datasets. "
                        "Understanding its principles can help data analysts and researchers identify potential outliers and "
                        "anomalies in their data. By harnessing the power of Benford's Law, we can gain valuable insights and "
                        "uncover hidden patterns in the vast sea of numerical information that surrounds us.")

            st.write("---")
            st.write("References:")
            st.write("1. Hill, T. P. (1995). A Statistical Derivation of the Significant-Digit Law. _Statistical Science_, "
                        "10(4), 354-363.")
            st.write("2. Berger, A., & Hill, T. P. (2015). Benford’s Law Strikes Back: No Simple Explanation in Sight for "
                        "Mathematician’s Rule. _Mathematical Association of America_, 122(9), 887-903.")



            
            st.write(
                "In probability theory and statistics, a Probability Distribution Function (PDF) is a function that describes the likelihood of a random variable taking on a particular value or falling within a specific range of values. It provides valuable information about the probabilities associated with different outcomes of a random variable.")

            st.subheader("Properties of PDF")
            st.write(
                "1. Non-negative: The PDF is always non-negative, meaning its values are greater than or equal to zero for all possible values of the random variable.")
            st.write(
                "2. Area under the curve: The total area under the PDF curve is equal to 1, representing the total probability of all possible outcomes.")
            st.write(
                "3. Describes likelihood: The PDF describes the likelihood of different values or ranges of values of the random variable.")

            st.subheader("Example: Normal Distribution")
            st.write(
                "One of the most commonly used probability distributions is the Normal Distribution, also known as the Gaussian Distribution. It is characterized by its bell-shaped curve.")

            st.image("https://image3.slideserve.com/6467891/properties-of-normal-distributions3-l.jpg", use_column_width=True, caption="Normal Distribution")

            st.write("The PDF of the Normal Distribution is given by the equation:")
            st.latex(r"f(x) = \frac{1}{{\sigma \sqrt{2\pi}}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")

            st.write("Where:")
            st.write("- \(\mu\) is the mean of the distribution.")
            st.write("- \(\sigma\) is the standard deviation of the distribution.")
            st.write("- \(e\) is the base of the natural logarithm.")

            st.subheader("Applications of PDF")
            st.write("The PDF is used in various areas such as:")
            st.write("- Statistical modeling and inference.")
            st.write("- Risk analysis and decision-making.")
            st.write("- Machine learning and data science.")
            st.write("- Finance and investment analysis.")
            st.write("- Quality control and process optimization.")

            st.write(
                "Understanding and utilizing PDFs is essential for analyzing and interpreting data, making predictions, and solving problems involving uncertainty.")

            st.markdown("---")
            st.write(
                "This blog post provides a brief introduction to Probability Distribution Functions (PDFs) and their significance in probability theory and statistics. PDFs are fundamental tools for understanding and quantifying uncertainty in various fields. They describe the probabilities associated with different outcomes of a random variable and play a crucial role in statistical modeling, risk analysis, and decision-making.")
            st.write(
                "Whether you are a data scientist, a researcher, or simply interested in understanding the principles of probability, having a solid grasp of PDFs is essential. They provide a mathematical framework for describing the likelihood of events and enable us to make informed decisions based on probabilities.")
            st.write(
                "In this blog post, we explored the properties of PDFs, highlighted the example of the Normal Distribution as a widely used PDF, and discussed the applications of PDFs in different domains. We hope this introduction has piqued your curiosity and motivated you to dive deeper into the fascinating world of probability and statistics.")
            st.write(
                "Remember, probabilities are all around us, and understanding them can empower us to make better decisions and gain valuable insights from data!")

            st.write(
                "In statistics, a Z-score, also known as a standard score, is a measurement that indicates how many standard deviations an element or observation is from the mean of a distribution. It provides a standardized way to compare and interpret data points in different distributions.")

            st.subheader("Calculating Z-Score")
            st.write("The formula to calculate the Z-score of a data point is:")
            st.latex(r"Z = \frac{{X - \mu}}{{\sigma}}")

            st.write("Where:")
            st.write("- X is the individual data point.")
            st.write("- \(\mu\) is the mean of the distribution.")
            st.write("- \(\sigma\) is the standard deviation of the distribution.")

            st.subheader("Interpreting Z-Score")
            st.write(
                "The Z-score tells us how many standard deviations a data point is away from the mean. Here's how to interpret the Z-score:")
            st.write("- A Z-score of 0 means the data point is exactly at the mean.")
            st.write("- A Z-score of +1 indicates the data point is 1 standard deviation above the mean.")
            st.write("- A Z-score of -1 indicates the data point is 1 standard deviation below the mean.")
            st.write("- A Z-score greater than +1 suggests the data point is above average, farther from the mean.")
            st.write("- A Z-score less than -1 suggests the data point is below average, farther from the mean.")

            st.subheader("Standardizing Data with Z-Score")
            st.write(
                "One of the main applications of Z-scores is to standardize data. By converting data points into Z-scores, we can compare observations from different distributions and identify outliers or extreme values.")

            st.subheader("Example:")
            st.write(
                "Let's consider a dataset of students' test scores. The mean score is 75, and the standard deviation is 10. We want to calculate the Z-score for a student who scored 85.")

            st.write("Using the formula, we can calculate the Z-score as:")
            st.latex(r"Z = \frac{{85 - 75}}{{10}} = 1")

            st.write("The Z-score of 1 indicates that the student's score is 1 standard deviation above the mean.")

            st.subheader("Applications of Z-Score")
            st.write("Z-scores have various applications in statistics and data analysis, including:")
            st.write("- Identifying outliers: Z-scores can help identify data points that are unusually far from the mean.")
            st.write(
                "- Comparing data points: Z-scores enable us to compare and rank data points from different distributions.")
            st.write("- Hypothesis testing: Z-scores are used in hypothesis testing to assess the significance of results.")
            st.write("- Data normalization: Z-scores are used to standardize data and bring it to a common scale.")

            st.markdown("---")
            st.write(
                "This blog post provides an overview of Z-scores and their significance in statistics. Z-scores allow us to standardize data and compare observations from different distributions. They provide valuable insights into the relative position of data points within a distribution and help identify outliers or extreme values.")
            st.write(
                "We discussed how to calculate Z-scores using the formula and interpret their values in terms of standard deviations from the mean. Additionally, we explored the applications of Z-scores in various statistical analyses, including outlier detection, data comparison, hypothesis testing, and data normalization.")
            st.write(
                "By understanding and utilizing Z-scores, we can gain deeper insights into our data and make informed decisions based on standardized measurements. Whether you're working with test scores, financial data, or any other quantitative information, Z-scores can be a valuable tool in your statistical toolkit.")
            st.write(
                "We hope this blog post has provided you with a clear understanding of Z-scores and their applications. Remember to explore further and practice applying Z-scores to real-world datasets to enhance your statistical analysis skills.")
            st.write("Happy analyzing!")


            
        with tab5:
            st.write(
                "Isolation Forest is an unsupervised machine learning algorithm used for anomaly detection. It is particularly effective in detecting outliers or anomalies in large datasets. The algorithm works by isolating anomalous observations by recursively partitioning the data into subsets. The main idea behind the Isolation Forest is that anomalies are more likely to be isolated into small partitions compared to normal data points.")

            st.subheader("How does Isolation Forest work?")
            st.write(
                "1. Random Selection: Isolation Forest selects a random feature and a random split value to create a binary tree partition of the data.")
            st.write(
                "2. Recursive Partitioning: The algorithm recursively partitions the data by creating more binary tree partitions. Each partitioning step creates a split point by selecting a random feature and a random split value.")
            st.write(
                "3. Isolation: Anomalies are expected to be isolated in smaller partitions since they require fewer partitioning steps to be separated from the majority of the data points.")
            st.write(
                "4. Anomaly Scoring: The algorithm assigns an anomaly score to each data point based on the average path length required to isolate it. The shorter the path length, the more likely it is an anomaly.")

            st.subheader("Advantages of Isolation Forest")
            st.write("- It is efficient for outlier detection, especially in large datasets.")
            st.write("- It does not rely on assumptions about the distribution of the data.")
            st.write("- It can handle high-dimensional data effectively.")
            st.write("- It is robust to the presence of irrelevant or redundant features.")

            st.subheader("Applications of Isolation Forest")
            st.write("Isolation Forest can be applied in various domains, including:")
            st.write("- Fraud detection: Identifying fraudulent transactions or activities.")
            st.write("- Network intrusion detection: Detecting anomalous behavior in network traffic.")
            st.write("- Manufacturing quality control: Identifying defective products or anomalies in production processes.")
            st.write("- Anomaly detection in sensor data: Detecting abnormalities in IoT sensor readings.")
            st.write("- Credit card fraud detection: Identifying fraudulent credit card transactions.")

            st.subheader("Example: Anomaly Detection in Network Traffic")
            st.write(
                "Let's consider the application of Isolation Forest in network intrusion detection. The algorithm can help identify anomalous network traffic patterns that may indicate potential attacks or breaches.")

            st.image("https://velog.velcdn.com/images%2Fvvakki_%2Fpost%2Fc59d0a7f-7a1c-4589-b799-cf40c6463d26%2Fimage.png", use_column_width=True, caption="Isolation Forest Anomaly Detection")

            st.write(
                "In this example, the Isolation Forest algorithm analyzes network traffic data and identifies anomalies that deviate from the normal patterns. By isolating and scoring the anomalies, security teams can prioritize their investigation and take appropriate actions to prevent potential threats.")

            st.markdown("---")
            st.write(
                "In this blog post, we explored Isolation Forest, an unsupervised machine learning algorithm used for anomaly detection. The algorithm leverages the concept of isolation to identify anomalies by recursively partitioning the data into subsets. It is particularly effective in detecting outliers or anomalies in large datasets.")
            st.write(
                "We discussed the working principle of Isolation Forest, which involves random selection, recursive partitioning, isolation, and anomaly scoring. We also highlighted the advantages of Isolation Forest, such as its efficiency, distribution-free nature, and ability to handle high-dimensional data.")
            st.write(
                "Furthermore, we explored several real-world applications of Isolation Forest, including fraud detection, network intrusion detection, quality control, and anomaly detection in sensor data.")
            st.write(
                "By utilizing Isolation Forest, data scientists and analysts can effectively identify anomalies and outliers in various domains, enabling them to make informed decisions and take appropriate actions. The algorithm's ability to handle large datasets and its robustness to irrelevant features make it a valuable tool for anomaly detection tasks.")
            st.write(
                "We hope this blog post has provided you with a comprehensive understanding of Isolation Forest and its applications. Remember to explore further and apply the algorithm to real-world datasets to enhance your anomaly detection capabilities.")
            st.write("Happy anomaly detection!")
        with tab6:
            st.write(
                "Autoencoders are a type of artificial neural network used for unsupervised learning and data compression. They are particularly useful for feature extraction and anomaly detection tasks. The basic idea behind autoencoders is to learn a compressed representation of the input data and then reconstruct it as accurately as possible.")

            st.subheader("Architecture of Autoencoder")
            st.write("An autoencoder consists of two main parts: the encoder and the decoder.")
            st.write(
                "1. Encoder: The encoder takes the input data and learns a compressed representation, also known as the encoding or latent space.")
            st.write(
                "2. Decoder: The decoder takes the encoded representation and reconstructs the original input data from it.")

            st.write(
                "The encoder and decoder are typically symmetric in structure, with the number of neurons decreasing in the encoder and increasing in the decoder.")

            st.subheader("Training an Autoencoder")
            st.write(
                "Autoencoders are trained using an unsupervised learning approach. The goal is to minimize the reconstruction error between the original input and the reconstructed output. This is typically done by minimizing a loss function, such as mean squared error (MSE) or binary cross-entropy (BCE).")

            st.subheader("Applications of Autoencoder")
            st.write("Autoencoders have various applications, including:")
            st.write(
                "- Dimensionality reduction: Learning compressed representations that capture the most important features of the data.")
            st.write(
                "- Anomaly detection: Detecting unusual or anomalous patterns in the data by comparing reconstruction errors.")
            st.write(
                "- Image denoising: Removing noise or artifacts from images by training the autoencoder to reconstruct clean images.")
            st.write("- Recommendation systems: Learning user preferences and generating personalized recommendations.")
            st.write("- Data generation: Generating new data samples similar to the training data.")

            st.subheader("Example: Image Denoising")
            st.write(
                "One application of autoencoders is image denoising. By training an autoencoder on noisy images and minimizing the reconstruction error, we can effectively remove the noise and reconstruct clean images.")

            st.image("https://miro.medium.com/v2/resize:fit:4266/1*QEmCZtruuWwtEOUzew2D4A.png", use_column_width=True, caption="Autoencoder Image Denoising")

            st.markdown("---")
            st.write(
                "In this blog post, we explored autoencoders, a type of artificial neural network used for unsupervised learning and data compression. Autoencoders consist of an encoder and a decoder, which learn a compressed representation of the input data and reconstruct it as accurately as possible.")
            st.write(
                "We discussed the training process of autoencoders, which involves minimizing the reconstruction error between the original input and the reconstructed output. Autoencoders have various applications, including dimensionality reduction, anomaly detection, image denoising, recommendation systems, and data generation.")
            st.write(
                "We also provided an example of image denoising using autoencoders, where the network learns to remove noise from noisy images and reconstruct clean images.")
            st.write(
                "By utilizing autoencoders, data scientists and researchers can effectively extract features, detect anomalies, denoise images, and generate new data samples. Autoencoders have wide-ranging applications and are particularly valuable in unsupervised learning scenarios.")
            st.write(
                "We hope this blog post has provided you with a clear understanding of autoencoders and their applications. Remember to explore further and apply autoencoders to different domains and datasets to unlock their full potential.")
            st.write("Happy autoencoding!")


        with tab3:


            # iframe_url = '<iframe title="process mining" width="1000" height="700" src="https://app.powerbi.com/view?r=eyJrIjoiNWE5ZDM0MDYtYmUwNC00ZjhiLTllOGMtNjFjNmY2M2M4YzkxIiwidCI6IjMyNTRjOGVlLWQxZDUtNDFmNy05ZTY5LTUxMzQxYjJhZWU3NCJ9&embedImagePlaceholder=true" frameborder="0" allowFullScreen="true"></iframe>'

            iframe_url = """
              <div class="marquee">
                <span style="color: #E3F4F4; background-color: #2b86d9;">It comes with pre-built statistical and machine learning models specifically designed to identify outliers in large-scale data.</span>
            </div>
            <center>
            <br>
                <a href="https://github.com/ravipratap366/LLM_chatbot">
                    <div class="cardGif 2" id="gif_card">
                    <div class="card_image_gif" id="gif_card">
                        <iframe title="process mining" width="100%" height="700" src="https://app.powerbi.com/view?r=eyJrIjoiNWE5ZDM0MDYtYmUwNC00ZjhiLTllOGMtNjFjNmY2M2M4YzkxIiwidCI6IjMyNTRjOGVlLWQxZDUtNDFmNy05ZTY5LTUxMzQxYjJhZWU3NCJ9&embedImagePlaceholder=true" frameborder="0" allowFullScreen="true"></iframe>'
                        </div>
                </a>
                </div>

            </center>
"""
            # Embed the Power BI report in the Streamlit app
            st.markdown(iframe_url, unsafe_allow_html=True)
    





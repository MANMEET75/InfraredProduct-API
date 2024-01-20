from utils import *


# Initialize the FastAPI application
app = FastAPI()



@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")



# dealing with statistical methods over here
@app.post("/zscore-anomaly/")
async def zscore_anomaly(file: UploadFile, column: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    # Dealing with duplicate values
    data = data.drop_duplicates()

    # Selecting numerical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=numerics)

    if column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Column {column} not found in data.")

    data_with_anomalies_zscore = z_score_anomaly_detection(data, column, threshold=3)

    # Visualization
    fig = px.scatter(
        data_with_anomalies_zscore,
        x=column,
        y="Anomaly",
        color="PointColor",
        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
        title='Z-Score Anomaly Detection',
        labels={column: column, "Anomaly": 'Anomaly', "PointColor": "Data Type"},
    )

    # Calculate the percentage of anomalies
    total_data_points = data_with_anomalies_zscore.shape[0]
    total_anomalies = data_with_anomalies_zscore["Anomaly"].sum()
    percentage_anomalies = (total_anomalies / total_data_points) * 100

    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_zscore.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/ZScore_Anomaly.csv"
    data_with_anomalies_zscore.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()


    return {
        "data_head": data_with_anomalies_zscore.head(5).to_dict(orient="records"),
        "num_anomalies": f"{total_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }


# adding the box plot over here
@app.post("/boxplot-anomaly/")
async def boxplot_anomaly(file: UploadFile, column: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    # Dealing with duplicate values
    data = data.drop_duplicates()

    # Selecting numerical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=numerics)

    if column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Column {column} not found in data.")

    # data_with_anomalies_zscore = z_score_anomaly_detection(data, column, threshold)

    fig = px.box(data, y=column, title="Boxplot of " + column)

    fig.update_traces(marker=dict(size=5, color='red'), selector=dict(type='box'))

    # Rotate x-axis tick labels
    fig.update_xaxes(tickangle=45)



    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
        xaxis=dict(
        showgrid=False,  # Remove x-axis gridlines
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
        yaxis=dict(
        showgrid=False,  # Remove y-axis gridlines
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
        title_font=dict(size=18, family='Arial'),
        paper_bgcolor='#F1F6F5',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=80),
    )


    # Calculate interquartile range (IQR)
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate upper and lower limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    data['anomaly'] = 0
    data.loc[(data[column] < lower_limit) | (data[column] > upper_limit), 'anomaly'] = 1

    # Calculate the percentage of outliers
    total_data_points = data.shape[0]
    total_outliers = data['anomaly'].sum()
    percentage_outliers = (total_outliers / total_data_points) * 100

    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/Boxplot_Anomaly.csv"
    data.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return {
        "data_head": data.head(5).to_dict(orient="records"),
        "num_anomalies": f"{total_outliers}",
        "percentage_anomalies": f"{percentage_outliers:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }



# adding pdf over here

@app.post("/pdf-anomaly/")
async def pdf_anomaly(file: UploadFile, column: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    # Dealing with duplicate values
    data = data.drop_duplicates()

    # Selecting numerical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=numerics)

    if column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Column {column} not found in data.")

    mean = data[column].mean()
    std = data[column].std()
    x = np.linspace(data[column].min(), data[column].max(), 100)
    pdf = norm.pdf(x, mean, std)
    pdf_data = pd.DataFrame({'x': x, 'pdf': pdf})
    fig = px.line(pdf_data, x='x', y='pdf')


     # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
        xaxis=dict(
        title=column,
        showgrid=False,  # Remove x-axis gridlines
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
        yaxis=dict(
        title="Probability Density Function",
        showgrid=False,  # Remove y-axis gridlines
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
                        
    ),
        title="Probability Density Function of " + column,
        title_font=dict(size=18, family='Arial'),
        paper_bgcolor='#F1F6F5',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=80),
    )


    threshold = 0.05
    data['anomaly'] = 0
    data.loc[data[column] < norm.ppf(threshold, mean, std), 'anomaly'] = 1


    total_data_points = data.shape[0]
    total_outliers = data['anomaly'].sum()
    percentage_outliers = (total_outliers / total_data_points) * 100

    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/PDF_Anomaly.csv"
    data.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return {
        "data_head": data.head(5).to_dict(orient="records"),
        "num_anomalies": f"{total_outliers}",
        "percentage_anomalies": f"{percentage_outliers:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }



# # adding relative size factor over here
# @app.post("/rsf-anomaly/")
# async def rsf_anomaly(file: UploadFile):
#     file_extension = file.filename.split(".")[-1]
#     if file_extension == "csv":
#         data = pd.read_csv(BytesIO(await file.read()))
#     elif file_extension in ["xlsx", "XLSX"]:
#         data = pd.read_excel(BytesIO(await file.read()))
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

#     # Dealing with missing values
#     threshold_missing = 0.1
#     missing_percentages = data.isnull().mean()
#     columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
#     data = data.drop(columns=columns_to_drop)

#     # Dealing with duplicate values
#     data = data.drop_duplicates()

#     # Selecting numerical columns
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     data = data.select_dtypes(include=numerics)



#     dfx=data[['WERKS','MATNR','EBELN','EBELP','LIFNR','MENGE','NETPR','PEINH','NETWR']]
#     ebeln_count = dfx.groupby('LIFNR')['EBELN'].nunique().reset_index()
#     ebeln_count.rename(columns={'EBELN': 'EBELN_Count'}, inplace=True)

#     netwr_sum_by_vendor = dfx.groupby('LIFNR')['NETWR'].sum().reset_index()
#     netwr_sum_by_vendor.rename(columns={'NETWR': 'NETWR_Sum_ByVendor'}, inplace=True)

#     netwr_sum_by_vendor_ebeln = dfx.groupby(['LIFNR', 'EBELN'])['NETWR'].sum().reset_index()
#     netwr_sum_by_vendor_ebeln.rename(columns={'NETWR': 'NETWR_Sum_ByVendor_EBELN'}, inplace=True)

#     dfx = pd.merge(dfx, ebeln_count, on='LIFNR')
#     dfx = pd.merge(dfx, netwr_sum_by_vendor, on='LIFNR')
#     dfx = pd.merge(dfx, netwr_sum_by_vendor_ebeln, on=['LIFNR', 'EBELN'])

#     netwr_max = dfx.groupby(['LIFNR'])['NETWR_Sum_ByVendor_EBELN'].max().reset_index()
#     netwr_max.rename(columns={'NETWR_Sum_ByVendor_EBELN': 'netwr_max'}, inplace=True)

#     dfx = pd.merge(dfx, netwr_max, on='LIFNR')

#     dfx['Avg_exclu_max'] = (dfx['NETWR_Sum_ByVendor'] - dfx['netwr_max']) / (dfx['EBELN_Count'] - 1)
#     dfx['Relative Size Factor'] = dfx['netwr_max'] / dfx['Avg_exclu_max']

#     anomaly = np.where((dfx['EBELN_Count'] > 5) & (dfx['Relative Size Factor'] > 10), 1, 0)
#     dfx['Anomaly'] = anomaly




#     dfx['Anomaly Flag'] = dfx['Anomaly'].apply(lambda x: 'Anomaly' if x == 1 else 'Not Anomaly')
#     dfx['Anomaly Flag'] = dfx['Anomaly Flag'].astype(str)

#     fig = px.scatter(
#         dfx,
#         x="Relative Size Factor",
#         y="EBELN_Count",
#         hover_name="LIFNR",
#         color="Anomaly Flag",  
#         color_discrete_map={"Not Anomaly": "blue", "Anomaly": "red"},
#     )

#     # Update layout with custom styling
#     fig.update_layout(
#         legend=dict(
#         itemsizing='constant',
#         title_text='',
#         font=dict(family='Arial', size=12),
#         borderwidth=2
#         ),
#         xaxis=dict(
#         showgrid=False,  # Remove x-axis gridlines
#         showline=True,
#         linecolor='lightgray',
#         linewidth=2,
#         mirror=True
#         ),
#         yaxis=dict(
#         showgrid=False,  # Remove y-axis gridlines
#         showline=True,
#         linecolor='lightgray',
#         linewidth=2,
#         mirror=True
                        
#         ),
#         title="Higher the Relative Size Factor and EBELN_Count more the Chances of Anomaly ",
                    
#         title_font=dict(size=18, family='Arial'),
#         paper_bgcolor='#F1F6F5',
#         plot_bgcolor='white',
#         margin=dict(l=80, r=80, t=50, b=80),
#     )



#     total_data_points = dfx.shape[0]
#     total_anomalies = dfx['Anomaly'].sum()
#     percentage_anomalies = (total_anomalies / total_data_points) * 100


#     return {
#         "data_head": dfx.head(5).to_dict(orient="records"),
#         "plot": fig.to_json(),
#         "percentage_anomalies": f"{percentage_anomalies:.2f}%"
#     }


# adding the benford's law over here
@app.post("/benford-first-law-anomaly/")
async def benfordFirstLaw_anomaly(file: UploadFile, column: str):
    try:
        file_extension = file.filename.split(".")[-1]
        if file_extension.lower() == "csv":
            data = pd.read_csv(BytesIO(await file.read()), encoding='ISO-8859-1')
        elif file_extension.lower() == "xlsx":
            data = pd.read_excel(BytesIO(await file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

        
     

        if column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Column {column} not found in data.")
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data = data.select_dtypes(include=numerics)

        data['First Digit'] = data[column].apply(extract_first_digit)

        # Sort the DataFrame by 'FT Digit' in ascending order
        data = data.sort_values(by='First Digit', ascending=True)
        data = data.dropna(subset=['First Digit'])

        # Count the occurrences of each unique three-digit value
        counts = data['First Digit'].value_counts().reset_index()
        counts.columns = ['First Digit', 'Count']

        counts = counts.sort_values(by='First Digit', ascending=True)
        total_count = counts['Count'].sum()
        counts['Actual'] = counts['Count'] / total_count

        # Create a new 'Benford' feature using the formula log(FT Digit + 1) - log(FT Digit)
        counts['Benford'] = np.log(counts['First Digit'] + 1) - np.log(counts['First Digit'])

        counts['Difference'] = counts['Actual'] - counts['Benford']
        counts['AbsDiff'] = counts['Difference'].abs()

        filtered_counts = counts[counts['Actual'] > counts['Benford']]
        filtered_counts['Anomaly_B1'] = filtered_counts['Actual'] > filtered_counts['Benford']
        filtered_counts['Anomaly_B1'] = filtered_counts['Anomaly_B1'].astype(int)

        merged_data = data.merge(filtered_counts, on='First Digit', how='inner')

        # Create the observed and expected bar plots
        observed_trace = go.Bar(x=counts['First Digit'], y=counts['Actual'] * 100, name='Observed', marker=dict(color='blue'))
        expected_trace = go.Scatter(x=counts['First Digit'], y=counts['Benford'] * 100, mode='lines', line=dict(color='red'), name='Expected')

        # Create the layout
        layout = go.Layout(
                        xaxis=dict(title="1st Digit"),
                        yaxis=dict(title="Percentage"),
                        legend=dict(x=0, y=1)
        )

        # Create the figure and add the traces
        fig = go.Figure(data=[observed_trace, expected_trace], layout=layout)

        # Update layout with custom styling
        fig.update_layout(
                legend=dict(
                itemsizing='constant',
                title_text='',
                font=dict(family='Arial', size=12),
                borderwidth=2
        ),
                xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
        ),
                yaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
        ),
                title="Benford's First Digit Law",
                title_font=dict(size=18, family='Arial'),
                paper_bgcolor='#F1F6F5',
                plot_bgcolor='white',
                margin=dict(l=80, r=80, t=50, b=80),
        )

        num_anomalies = merged_data['Anomaly_B1'].sum()
        total_data_points = len(merged_data)
        percentage_anomalies = (num_anomalies / total_data_points) * 100

        # Convert the DataFrame to a CSV string
        csv_buffer = StringIO()
        merged_data.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # Save the CSV to a file (adjust the file path and name as needed)
        csv_file_path = "downloads/BenfordFirstDigit_Anomaly.csv"
        merged_data.to_csv(csv_file_path, index=False)


        # Check if the file exists before scheduling deletion
        if os.path.exists(csv_file_path):
            # Start a background thread to delete the file after 2 minutes
            deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
            deletion_thread.start()



        # Encode the CSV string to base64
        csv_base64 = base64.b64encode(csv_string.encode()).decode()



        

        return {
            "data_head": counts.head(5).to_dict(orient="records"),
            "num_anomalies": f"{num_anomalies}",
            "download_csv_base64": csv_base64,
            "plot": fig.to_json(),
            "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
            "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# adding the benford's second digit law over here
@app.post("/benford-second-law-anomaly/")
async def benfordSecondLaw_anomaly(file: UploadFile, column: str):
    try:
        file_extension = file.filename.split(".")[-1]
        if file_extension.lower() == "csv":
            data = pd.read_csv(BytesIO(await file.read()), encoding='ISO-8859-1')
        elif file_extension.lower() == "xlsx":
            data = pd.read_excel(BytesIO(await file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

        # Data preprocessing
        threshold_missing = 0.1
        missing_percentages = data.isnull().mean()
        columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
        data = data.drop(columns=columns_to_drop)
        data = data.drop_duplicates()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data = data.select_dtypes(include=numerics)

        if column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Column {column} not found in data.")
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data = data.select_dtypes(include=numerics)


        data['FirstTwo Digit'] = data[column].apply(extract_first_two_digits)

        # Sort the DataFrame by 'FirstTwo Digit' in ascending order
        data = data.sort_values(by='FirstTwo Digit', ascending=True)
        data = data.dropna(subset=['FirstTwo Digit'])


        # Count the occurrences of each unique three-digit value
        counts = data['FirstTwo Digit'].value_counts().reset_index()
        counts.columns = ['FirstTwo Digit', 'Count']

        counts = counts.sort_values(by='FirstTwo Digit', ascending=True)
        total_count = counts['Count'].sum()
        counts['Actual'] = counts['Count'] / total_count

        # Create a new 'Benford' feature using the formula log(FT Digit + 1) - log(FT Digit)
        counts['Benford'] = np.log(counts['FirstTwo Digit'] + 1) - np.log(counts['FirstTwo Digit'])

        counts['Difference'] = counts['Actual'] - counts['Benford']
        counts['AbsDiff'] = counts['Difference'].abs()


        filtered_counts = counts[counts['Actual'] > counts['Benford']]
        filtered_counts['Anomaly_B2'] = filtered_counts['Actual'] > filtered_counts['Benford']
        filtered_counts['Anomaly_B2'] = filtered_counts['Anomaly_B2'].astype(int)

        merged_data = data.merge(filtered_counts, on='FirstTwo Digit', how='inner')

        # Create the observed and expected bar plots
        observed_trace = go.Bar(x=counts['FirstTwo Digit'], y=counts['Actual'] * 100, name='Observed', marker=dict(color='blue'))
        expected_trace = go.Scatter(x=counts['FirstTwo Digit'], y=counts['Benford'] * 100, mode='lines', line=dict(color='red'), name='Expected')

        # Create the layout
        layout = go.Layout(
                        xaxis=dict(title="2nd Digit"),
                        yaxis=dict(title="Percentage"),
                        legend=dict(x=0, y=1)
        )

        # Create the figure and add the traces
        fig = go.Figure(data=[observed_trace, expected_trace], layout=layout)

        # Update layout with custom styling
        fig.update_layout(
                legend=dict(
                itemsizing='constant',
                title_text='',
                font=dict(family='Arial', size=12),
                borderwidth=2
        ),
                xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
        ),
                yaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
        ),
                title="Benford's Second Digit Law",
                title_font=dict(size=18, family='Arial'),
                paper_bgcolor='#F1F6F5',
                plot_bgcolor='white',
                margin=dict(l=80, r=80, t=50, b=80),
        )

        num_anomalies = merged_data['Anomaly_B2'].sum()
        total_data_points = len(merged_data)
        percentage_anomalies = (num_anomalies / total_data_points) * 100

        # Convert the DataFrame to a CSV string
        csv_buffer = StringIO()
        merged_data.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # Save the CSV to a file (adjust the file path and name as needed)
        csv_file_path = "downloads/BenfordSecondDigit_Anomaly.csv"
        merged_data.to_csv(csv_file_path, index=False)


        # Check if the file exists before scheduling deletion
        if os.path.exists(csv_file_path):
            # Start a background thread to delete the file after 2 minutes
            deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
            deletion_thread.start()



        # Encode the CSV string to base64
        csv_base64 = base64.b64encode(csv_string.encode()).decode()



        

        return {
            "data_head": counts.head(5).to_dict(orient="records"),
            "num_anomalies": f"{num_anomalies}",
            "download_csv_base64": csv_base64,
            "plot": fig.to_json(),
            "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
            "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# adding the benford's law over here
@app.post("/benford-third-law-anomaly/")
async def benfordThirdLaw_anomaly(file: UploadFile, column: str):
    try:
        file_extension = file.filename.split(".")[-1]
        if file_extension.lower() == "csv":
            data = pd.read_csv(BytesIO(await file.read()), encoding='ISO-8859-1')
        elif file_extension.lower() == "xlsx":
            data = pd.read_excel(BytesIO(await file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

        # Data preprocessing
        threshold_missing = 0.1
        missing_percentages = data.isnull().mean()
        columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
        data = data.drop(columns=columns_to_drop)
        data = data.drop_duplicates()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data = data.select_dtypes(include=numerics)

        if column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Column {column} not found in data.")
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data = data.select_dtypes(include=numerics)

        data['FT Digit'] = data[column].apply(extract_first_three_digits)

        # Sort the DataFrame by 'FT Digit' in ascending order
        data = data.sort_values(by='FT Digit', ascending=True)
        data = data.dropna(subset=['FT Digit'])

        # Count the occurrences of each unique three-digit value
        counts = data['FT Digit'].value_counts().reset_index()
        counts.columns = ['FT Digit', 'Count']

        counts = counts.sort_values(by='FT Digit', ascending=True)
        total_count = counts['Count'].sum()
        counts['Actual'] = counts['Count'] / total_count

        # Create a new 'Benford' feature using the formula log(FT Digit + 1) - log(FT Digit)
        counts['Benford'] = np.log(counts['FT Digit'] + 1) - np.log(counts['FT Digit'])

        counts['Difference'] = counts['Actual'] - counts['Benford']
        counts['AbsDiff'] = counts['Difference'].abs()


        filtered_counts = counts[counts['Actual'] > counts['Benford']]
        filtered_counts['Anomaly_B3'] = filtered_counts['Actual'] > filtered_counts['Benford']
        filtered_counts['Anomaly_B3'] = filtered_counts['Anomaly_B3'].astype(int)


        merged_data = data.merge(filtered_counts, on='FT Digit', how='inner')

        # Create the observed and expected bar plots
        observed_trace = go.Bar(x=counts['FT Digit'], y=counts['Actual'] * 100, name='Observed', marker=dict(color='blue'))
        expected_trace = go.Scatter(x=counts['FT Digit'], y=counts['Benford'] * 100, mode='lines', line=dict(color='red'), name='Expected')

        # Create the layout
        layout = go.Layout(
                xaxis=dict(title="3rd Digit"),
                yaxis=dict(title="Percentage"),
                legend=dict(x=0, y=1)
        )

        # Create the figure and add the traces
        fig = go.Figure(data=[observed_trace, expected_trace], layout=layout)

        # Update layout with custom styling
        fig.update_layout(
                legend=dict(
                itemsizing='constant',
                title_text='',
                font=dict(family='Arial', size=12),
                borderwidth=2
        ),
                xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
        ),
                yaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
        ),
                title="Benford's Third Digit Law",
                title_font=dict(size=18, family='Arial'),
                paper_bgcolor='#F1F6F5',
                plot_bgcolor='white',
                margin=dict(l=80, r=80, t=50, b=80),
        )

        num_anomalies = merged_data['Anomaly_B3'].sum()
        total_data_points = len(merged_data)
        percentage_anomalies = (num_anomalies / total_data_points) * 100

        # Convert the DataFrame to a CSV string
        csv_buffer = StringIO()
        merged_data.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # Save the CSV to a file (adjust the file path and name as needed)
        csv_file_path = "downloads/BenfordThirdDigit_Anomaly.csv"
        merged_data.to_csv(csv_file_path, index=False)


        # Check if the file exists before scheduling deletion
        if os.path.exists(csv_file_path):
            # Start a background thread to delete the file after 2 minutes
            deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
            deletion_thread.start()



        # Encode the CSV string to base64
        csv_base64 = base64.b64encode(csv_string.encode()).decode()



        

        return {
            "data_head": counts.head(5).to_dict(orient="records"),
            "num_anomalies": f"{num_anomalies}",
            "download_csv_base64": csv_base64,
            "plot": fig.to_json(),
            "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
            "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










        # Calculate the distribution of first digits using Benford's Law
        df, df1, counts, benford  = calculate_3th_digit(data[column])
        df2 = pd.merge(df, df1, left_on=df.iloc[:, 0],right_on='index',how='right')


        counts = counts[counts.index > 99]
        observed_trace = go.Bar(x=counts.index, y=counts * 100, name='Observed', marker=dict(color='blue'))
        expected_trace = go.Scatter(x=np.arange(100, 1000), y=benford * 100, mode='lines', line=dict(color='red'),name='Expected')

        layout = go.Layout(
            title="Benford's Law Analysis of 3rd Digit in " + column,
            xaxis=dict(title="3rd Digit"),
            yaxis=dict(title="Percentage"),
            legend=dict(x=0, y=1)
        )

        # Create the figure and add the traces
        fig = go.Figure(data=[observed_trace, expected_trace], layout=layout)

        # Update layout with custom styling
        fig.update_layout(
            legend=dict(
            itemsizing='constant',
            title_text='',
            font=dict(family='Arial', size=12),
            borderwidth=2
        ),
         xaxis=dict(
            showgrid=False,  # Remove x-axis gridlines
            showline=True,
            linecolor='lightgray',
            linewidth=2,
            mirror=True
         ),
         yaxis=dict(
            showgrid=False,  # Remove y-axis gridlines
            showline=True,
            linecolor='lightgray',
            linewidth=2,
            mirror=True
                            
        ),
        title="Benford's Third Digit Law",
        title_font=dict(size=18, family='Arial'),
        paper_bgcolor='#F1F6F5',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=80),
        )
        deviation = (counts - benford) * 100

        # Create the results DataFrame
        results = pd.DataFrame({'Digit': counts.index, 'Observed (%)': counts * 100, 'Expected (%)': benford[:len(counts)] * 100,
                            'Deviation (%)': deviation})
        
        return {
            "data": df2.to_dict(orient="records"),
            "plot": fig.to_json()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# adding the isolation forest over here
@app.post("/isolation-forest-anomaly/")
async def IsolationForest_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled

    data_with_anomalies_IsolationForest = apply_anomaly_detection_IsolationForest(data)

    data_with_anomalies_IsolationForest['PointColor'] = 'Inlier'
    data_with_anomalies_IsolationForest.loc[data_with_anomalies_IsolationForest['Anomaly_IF'] == 1, 'PointColor'] = 'Outlier'

    AnomalyFeature=data_with_anomalies_IsolationForest[["Anomaly_IF"]]
    final_data=pd.concat([copy_data,AnomalyFeature],axis=1)


     # Create a scatter plot using Plotly
    fig = px.scatter(
        data_with_anomalies_IsolationForest,
        x=xcolumn,
        y=ycolumn,
        color="PointColor",
        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
        title='Isolation Forest Anomaly Detection',
        labels={xcolumn: ycolumn, "Anomaly": 'Anomaly', "PointColor": "Data Type"},
    )

    # Update the trace styling
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
    xaxis=dict(
        title_text=xcolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    yaxis=dict(
        title_text=ycolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )


    num_anomalies = data_with_anomalies_IsolationForest['Anomaly_IF'].sum()
    total_data_points = len(data_with_anomalies_IsolationForest)
    percentage_anomalies = (num_anomalies / total_data_points) * 100

    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_IsolationForest.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/IsolationForest_Anomaly.csv"
    data_with_anomalies_IsolationForest.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    # Add the base64 encoded CSV to the response
    return {
        "data_head": data_with_anomalies_IsolationForest.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }

# adding kernel density estimation over here
@app.post("/kernel-estimation-density-anomaly/")
async def kernelDensityEstimation_anomaly(file: UploadFile, column: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled


    # Perform anomaly detection using Kernel Density Estimation
    kde = KernelDensity()
    kde.fit(data)
    log_densities = kde.score_samples(data)

    # adjusting the percentile as needed
    threshold = np.percentile(log_densities, 5)

    # Identity outliers based on log densities below the threshold
    outlier_indices = np.where(log_densities < threshold)[0]

    # Creating a copy of the data with an anomaly column indicating outliers
    data_with_anomalies_kde = data.copy()
    data_with_anomalies_kde['Anomaly'] = 0
    data_with_anomalies_kde.loc[outlier_indices, 'Anomaly'] = 1

    data_with_anomalies_kde['PointColor'] = 'Inlier'
    data_with_anomalies_kde.loc[data_with_anomalies_kde['Anomaly'] == 1, 'PointColor'] = 'Outlier'
    AnomalyFeature = data_with_anomalies_kde[["Anomaly"]]
    final_data = pd.concat([copy_data, AnomalyFeature], axis=1)

    kde_data = pd.DataFrame({'Feature': data[column], 'Density': np.exp(log_densities)})

    fig = go.Figure()

    # Add the line plot for density
    fig.add_trace(
        go.Scatter(
        x=kde_data['Feature'],
        y=kde_data['Density'],
        mode='lines',
        name='Density',
        line=dict(color='blue'),
    )
    )

    # Add the scatter plot for outliers
    fig.add_trace(
        go.Scatter(
        x=kde_data.loc[kde_data.index.isin(outlier_indices)]['Feature'],
        y=kde_data.loc[kde_data.index.isin(outlier_indices)]['Density'],
        mode='markers',
        name='Outliers',
        marker=dict(color='red', size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        text='Outlier',  # Label for outliers
    )
    )

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2,
        x=1.05,  # Adjust the x position of the legend
    ),
    xaxis=dict(
        title_text=column,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    yaxis=dict(
        title_text='Density',
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    title="Kernel Density Estimation of " + column,
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )

    # Counting the number of anomalies over here
    num_anomalies = data_with_anomalies_kde['Anomaly'].sum()

    # total number of data points
    total_data_points=len(data_with_anomalies_kde)


    # calcuating the percentage of anomalies
    percentage_anomalies = (num_anomalies / total_data_points) * 100

    # Convert the DataFrame to a CSV string

    csv_buffer = StringIO()
    data_with_anomalies_kde.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/KDE_Anomaly.csv"
    data_with_anomalies_kde.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()

    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()


    return {
        "data_head": data_with_anomalies_kde.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }

# adding local outliers factor over here
@app.post("/local-outlier-factor-anomaly/")
async def LocalOutlierFactor_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled



    # Applying the anomaly detection
    data_with_anomalies_LocalOutlierFactor = apply_anomaly_detection_LocalOutlierFactor(data)
    data_with_anomalies_LocalOutlierFactor['PointColor'] = 'Inlier'
    data_with_anomalies_LocalOutlierFactor.loc[data_with_anomalies_LocalOutlierFactor['Anomaly'] == 1, 'PointColor'] = 'Outlier'

    AnomalyFeature=data_with_anomalies_LocalOutlierFactor[["Anomaly"]]
    final_data=pd.concat([copy_data,AnomalyFeature],axis=1)


    # Create a scatter plot using Plotly
    fig = px.scatter(
        data_with_anomalies_LocalOutlierFactor,
        x=xcolumn,
        y=ycolumn,
        color="PointColor",
        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
        title='Local Outlier Factor Anomaly Detection',
        labels={"Anomaly": 'Anomaly', "PointColor": "Data Type"},
    )

    # Update the trace styling
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
        xaxis=dict(
        title_text=xcolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
    mirror=True
    ),
        yaxis=dict(
        title_text=ycolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )


    num_anomalies = data_with_anomalies_LocalOutlierFactor['Anomaly'].sum()
    total_data_points = len(data_with_anomalies_LocalOutlierFactor)
    percentage_anomalies = (num_anomalies / total_data_points) * 100



    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_LocalOutlierFactor.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/LOF_Anomaly.csv"
    data_with_anomalies_LocalOutlierFactor.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()



    return {
        "data_head": data_with_anomalies_LocalOutlierFactor.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }



# # adding the one class svm over here
# @app.post("/one-class-svm-anomaly/")
# async def OneClassSVM_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
#     file_extension = file.filename.split(".")[-1]
#     if file_extension == "csv":
#         data = pd.read_csv(BytesIO(await file.read()))
#     elif file_extension in ["xlsx", "XLSX"]:
#         data = pd.read_excel(BytesIO(await file.read()))
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

#     # Dealing with missing values
#     threshold_missing = 0.1
#     missing_percentages = data.isnull().mean()
#     columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
#     data = data.drop(columns=columns_to_drop)

#     data = drop_features_with_missing_values(data)
#     num_duplicates = data.duplicated().sum()
#     data_unique = data.drop_duplicates() 
#     copy_data=data.copy()

#     categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
#     data_encoded = data_unique.copy()
#     for feature in categorical_features:
#         labels_ordered = data_unique.groupby([feature]).size().sort_values().index
#         labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
#         data_encoded[feature] = data_encoded[feature].map(labels_ordered)

#     data = data_encoded 
#     numeric_columns = data.select_dtypes(include=["int", "float"]).columns


#     scaler = MinMaxScaler()
#     data_scaled = data.copy()
#     data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
#     data = data_scaled



#     data_with_anomalies_OneClassSVM = apply_anomaly_detection_OneClassSVM(data)

#     AnomalyFeature=data_with_anomalies_OneClassSVM[["Anomaly"]]
#     final_data=pd.concat([copy_data,AnomalyFeature],axis=1)

#     data_with_anomalies_OneClassSVM['PointColor'] = 'Inlier'
#     data_with_anomalies_OneClassSVM.loc[data_with_anomalies_OneClassSVM['Anomaly'] == 1, 'PointColor'] = 'Outlier'

#     # Create a scatter plot using Plotly
#     fig = px.scatter(
#         data_with_anomalies_OneClassSVM,
#         x=xcolumn,
#         y=ycolumn,
#         color="PointColor",
#         color_discrete_map={"Inlier": "blue", "Outlier": "red"},
#         title='One Class SVM Anomaly Detection',
#         labels={"Anomaly": 'Anomaly', "PointColor": "Data Type"},
#     )

#     # Update the trace styling
#     fig.update_traces(
#         marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
#         selector=dict(mode='markers+text')
#     )

#     # Update layout with custom styling
#     fig.update_layout(
#         legend=dict(
#         itemsizing='constant',
#         title_text='',
#         font=dict(family='Arial', size=12),
#         borderwidth=2
#     ),
#     xaxis=dict(
#         title_text=xcolumn,
#         title_font=dict(size=14),
#         showgrid=False,
#         showline=True,
#         linecolor='lightgray',
#         linewidth=2,
#         mirror=True
#     ),
#     yaxis=dict(
#         title_text=ycolumn,
#         title_font=dict(size=14),
#         showgrid=False,
#         showline=True,
#         linecolor='lightgray',
#         linewidth=2,
#         mirror=True
#     ),
#     title_font=dict(size=18, family='Arial'),
#     paper_bgcolor='#F1F6F5',
#     plot_bgcolor='white',
#     margin=dict(l=80, r=80, t=50, b=80),
#     )

#     num_anomalies = data_with_anomalies_OneClassSVM['Anomaly'].sum()
#     total_data_points = len(data_with_anomalies_OneClassSVM)
#     percentage_anomalies = (num_anomalies / total_data_points) * 100



#     return {
#         "data_head": data_with_anomalies_OneClassSVM.head(5).to_dict(orient="records"),
#         "plot": fig.to_json(),
#         "percentage_anomalies": f"{percentage_anomalies:.2f}%"
#     }

# adding the one class svm SGD over here
@app.post("/one-class-svm-sgd-anomaly/")
async def OneClassSVMSGD_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled



    data_with_anomalies_SGDOCSVM = apply_anomaly_detection_SGDOCSVM(data)

    AnomalyFeature=data_with_anomalies_SGDOCSVM[["Anomaly"]]
    final_data=pd.concat([copy_data,AnomalyFeature],axis=1)

    data_with_anomalies_SGDOCSVM['PointColor'] = 'Inlier'
    data_with_anomalies_SGDOCSVM.loc[data_with_anomalies_SGDOCSVM['Anomaly'] == 1, 'PointColor'] = 'Outlier'

    # Create a scatter plot using Plotly
    fig = px.scatter(
        data_with_anomalies_SGDOCSVM,
        x=xcolumn,
        y=ycolumn,
        color="PointColor",
        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
        title='One Class SVM Anomaly Detection',
        labels={"Anomaly": 'Anomaly', "PointColor": "Data Type"},
    )

    # Update the trace styling
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
    xaxis=dict(
        title_text=xcolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    yaxis=dict(
        title_text=ycolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )

    num_anomalies = data_with_anomalies_SGDOCSVM['Anomaly'].sum()
    total_data_points = len(data_with_anomalies_SGDOCSVM)
    percentage_anomalies = (num_anomalies / total_data_points) * 100



    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_SGDOCSVM.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/OneClassSVM(SGD)_Anomaly.csv"
    data_with_anomalies_SGDOCSVM.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return {
        "data_head": data_with_anomalies_SGDOCSVM.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }


# adding the DBSCAN over here
@app.post("/dbscan-anomaly/")
async def DBSCAN_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled

    # Specify parameters for DBSCAN (adjust as needed)
    eps = 2.00
    min_samples = 3



    data_with_anomalies_dbscan = apply_anomaly_detection_dbscan(data.copy(), eps, min_samples)

    # AnomalyFeature=data_with_anomalies_SGDOCSVM[["Anomaly"]]
    # final_data=pd.concat([copy_data,AnomalyFeature],axis=1)

    # data_with_anomalies_SGDOCSVM['PointColor'] = 'Inlier'
    # data_with_anomalies_SGDOCSVM.loc[data_with_anomalies_SGDOCSVM['Anomaly'] == 1, 'PointColor'] = 'Outlier'

                        # Plot the data with anomalies
    fig = px.scatter(
            data_with_anomalies_dbscan,
            x=xcolumn,
            y=ycolumn,
            color="Anomaly_DBSCAN",
            color_discrete_map={0: "blue", 1: "red"},
            title='DBSCAN Anomaly Detection (2D Scatter Plot)',
            labels={xcolumn: xcolumn, ycolumn: ycolumn, "Anomaly_DBSCAN": "Anomaly"}
    )

                    # Update the trace styling
    fig.update_traces(
            marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
            selector=dict(mode='markers+text')
    )

                    # Update layout with custom styling
    fig.update_layout(
            legend=dict(
            itemsizing='constant',
            title_text='',
            font=dict(family='Arial', size=12),
            borderwidth=2,
            x=1.05,  # Adjust the x position of the legend
    ),
            xaxis=dict(
            title_text=xcolumn,
            title_font=dict(size=14),
            showgrid=False,
            showline=True,
            linecolor='lightgray',
            linewidth=2,
            mirror=True
    ),
            yaxis=dict(
            title_text=ycolumn,
            title_font=dict(size=14),
            showgrid=False,
            showline=True,
            linecolor='lightgray',
            linewidth=2,
            mirror=True
    ),
            title_font=dict(size=18, family='Arial'),
            paper_bgcolor='#F1F6F5',
            plot_bgcolor='white',
            margin=dict(l=80, r=80, t=50, b=80),
    )



    num_anomalies = data_with_anomalies_dbscan['Anomaly_DBSCAN'].sum()
    total_data_points = len(data_with_anomalies_dbscan)
    percentage_anomalies = (num_anomalies / total_data_points) * 100



    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_dbscan.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/DBSCAN_Anomaly.csv"
    data_with_anomalies_dbscan.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return {
        "data_head": data_with_anomalies_dbscan.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }


# adding the elliptic envelope over here
@app.post("/elliptic-envelope-anomaly/")
async def EllipticEnvelope_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled



    data_with_anomalies_EllipticEnvelope = apply_anomaly_detection_EllipticEnvelope(data)

    data_with_anomalies_EllipticEnvelope['PointColor'] = 'Inlier'
    data_with_anomalies_EllipticEnvelope.loc[data_with_anomalies_EllipticEnvelope['Anomaly_EllipticEnvelope'] == 1, 'PointColor'] = 'Outlier'

    AnomalyFeature=data_with_anomalies_EllipticEnvelope[["Anomaly_EllipticEnvelope"]]
    final_data=pd.concat([copy_data,AnomalyFeature],axis=1)


     # Create a scatter plot using Plotly
    fig = px.scatter(
        data_with_anomalies_EllipticEnvelope,
        x=xcolumn,
        y=ycolumn,
        color="PointColor",
        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
        title='EllipticEnvelope Anomaly Detection (2D Scatter Plot)',
        labels={xcolumn: ycolumn, "Anomaly": 'Anomaly', "PointColor": "Data Type"},
    )

    # Update the trace styling
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
    xaxis=dict(
        title_text=xcolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    yaxis=dict(
        title_text=ycolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )


    num_anomalies = data_with_anomalies_EllipticEnvelope['Anomaly_EllipticEnvelope'].sum()
    total_data_points = len(data_with_anomalies_EllipticEnvelope)
    percentage_anomalies = (num_anomalies / total_data_points) * 100

    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_EllipticEnvelope.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/EllipticEnvelope_Anomaly.csv"
    data_with_anomalies_EllipticEnvelope.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return {
        "data_head": data_with_anomalies_EllipticEnvelope.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }



# adding the robust covariance over here
@app.post("/robust-covariance-anomaly/")
async def RobustCovariance_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled



    data_with_anomalies_RobustCovariance = apply_anomaly_detection_Mahalanobis(data)

    data_with_anomalies_RobustCovariance['PointColor'] = 'Inlier'
    data_with_anomalies_RobustCovariance.loc[data_with_anomalies_RobustCovariance['Anomaly_RC'] == 1, 'PointColor'] = 'Outlier'

    AnomalyFeature=data_with_anomalies_RobustCovariance[["Anomaly_RC"]]
    final_data=pd.concat([copy_data,AnomalyFeature],axis=1)


     # Create a scatter plot using Plotly
    fig = px.scatter(
        data_with_anomalies_RobustCovariance,
        x=xcolumn,
        y=ycolumn,
        color="PointColor",
        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
        title='Robust Covariance Anomaly Detection',
        labels={xcolumn: ycolumn, "Anomaly": 'Anomaly', "PointColor": "Data Type"},
    )

    # Update the trace styling
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
    xaxis=dict(
        title_text=xcolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    yaxis=dict(
        title_text=ycolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )


    num_anomalies = data_with_anomalies_RobustCovariance['Anomaly_RC'].sum()
    total_data_points = len(data_with_anomalies_RobustCovariance)
    percentage_anomalies = (num_anomalies / total_data_points) * 100

    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_RobustCovariance.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/RobustCovariance_Anomaly.csv"
    data_with_anomalies_RobustCovariance.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return {
        "data_head": data_with_anomalies_RobustCovariance.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }





# adding more autoencoder method over here
@app.post("/autoencoder-anomaly/")
async def Autoencoder_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(BytesIO(await file.read()), encoding='ISO-8859-1')
    elif file_extension in ["xlsx", "XLSX"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates() 
    copy_data=data.copy()

    categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
    data_encoded = data_unique.copy()
    for feature in categorical_features:
        labels_ordered = data_unique.groupby([feature]).size().sort_values().index
        labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        data_encoded[feature] = data_encoded[feature].map(labels_ordered)

    data = data_encoded 
    numeric_columns = data.select_dtypes(include=["int", "float"]).columns


    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
    data = data_scaled


    data_with_anomalies_Autoencoder = apply_anomaly_detection_autoencoder(data)
    data_with_anomalies_Autoencoder['PointColor'] = 'Inlier'
    data_with_anomalies_Autoencoder.loc[data_with_anomalies_Autoencoder['Anomaly'] == 1, 'PointColor'] = 'Outlier'

    AnomalyFeature=data_with_anomalies_Autoencoder[["Anomaly"]]
    final_data=pd.concat([copy_data,AnomalyFeature],axis=1)

    # Create a scatter plot using Plotly
    fig = px.scatter(
        data_with_anomalies_Autoencoder,
        x=xcolumn,
        y=ycolumn,
        color="PointColor",
        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
        title='Autoencoder Anomaly Detection',
        labels={"Anomaly": 'Anomaly', "PointColor": "Data Type"},
    )

    # Update the trace styling
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
    ),
    xaxis=dict(
        title_text=xcolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    yaxis=dict(
        title_text=ycolumn,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
    title_font=dict(size=18, family='Arial'),
        paper_bgcolor='#F1F6F5',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=80),
    )

    
    num_anomalies = data_with_anomalies_Autoencoder['Anomaly'].sum()
    total_data_points = len(data_with_anomalies_Autoencoder)
    percentage_anomalies = (num_anomalies / total_data_points) * 100





    # Convert the DataFrame to a CSV string
    csv_buffer = StringIO()
    data_with_anomalies_Autoencoder.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV to a file (adjust the file path and name as needed)
    csv_file_path = "downloads/Autoencoders_Anomaly.csv"
    data_with_anomalies_Autoencoder.to_csv(csv_file_path, index=False)


    # Check if the file exists before scheduling deletion
    if os.path.exists(csv_file_path):
        # Start a background thread to delete the file after 2 minutes
        deletion_thread = threading.Thread(target=delayed_file_delete, args=(csv_file_path,))
        deletion_thread.start()



    # Encode the CSV string to base64
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return {
        "data_head": data_with_anomalies_Autoencoder.head(5).to_dict(orient="records"),
        "num_anomalies": f"{num_anomalies}",
        "percentage_anomalies": f"{percentage_anomalies:.2f}%.",
        "download_csv_base64": csv_base64,
        "plot": fig.to_json(),
        "csv_file_path": csv_file_path,  # Optional: Include the file path in the response
        "download_link": f"/download/csv?file_path={quote(csv_file_path)}"  # Adjust the endpoint and parameters as needed
    }

# adding the holt winter method over here
@app.post("/holt-winter-anomaly/")
async def HoltWinter_anomaly(file: UploadFile, date_column: str, feature_column: str, forecast_period: int):
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() == "csv":
        data = pd.read_csv(BytesIO(await file.read()), encoding='ISO-8859-1')
    elif file_extension.lower() in ["xlsx", "xls"]:
        data = pd.read_excel(BytesIO(await file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

    # Dealing with missing values
    threshold_missing = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold_missing].index
    data = data.drop(columns=columns_to_drop)

    def drop_features_with_missing_values(df):
        # Implement your logic for dropping features with missing values here
        return df

    data = drop_features_with_missing_values(data)
    num_duplicates = data.duplicated().sum()
    data_unique = data.drop_duplicates()
    df = data

    if date_column != feature_column:
        df_selected = df[[date_column, feature_column]].copy()
        df_selected[date_column] = pd.to_datetime(df_selected[date_column])
        df_selected.set_index(date_column, inplace=True)
        df_selected.dropna(inplace=True)

        date_diff = df_selected.index.to_series().diff().mean()

        if date_diff <= pd.Timedelta(days=1):
            seasonal_period = 7
        elif date_diff <= pd.Timedelta(days=7):
            seasonal_period = 12
        elif date_diff <= pd.Timedelta(days=31):
            seasonal_period = 12
        else:
            seasonal_period = 12 if forecast_period >= 12 else 1

        train_size = int(0.8 * len(df_selected))
        test_size = len(df_selected) - train_size

        train_data = df_selected.iloc[:train_size]
        test_data = df_selected.iloc[train_size: train_size + test_size]

        fitted_model = ExponentialSmoothing(train_data[feature_column], trend='mul', seasonal='mul',
                                           seasonal_periods=seasonal_period).fit()

        forecast = fitted_model.forecast(test_size).rename('HW Forecast')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data[feature_column], mode='lines', name='TRAIN'))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data[feature_column], mode='lines', name='TEST'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='FORECAST'))

        # Update layout with custom styling
        fig.update_layout(
            legend=dict(
                itemsizing='constant',
                title_text='Time Series Forecasting Evaluation',
                font=dict(family='Arial', size=12),
                borderwidth=2
            ),
            xaxis=dict(
                title_text=date_column,
                title_font=dict(size=14),
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
            ),
            yaxis=dict(
                title_text=feature_column,
                title_font=dict(size=14),
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
            ),
            title='Forecasting Evaluation on the testing data',
            title_font=dict(size=18, family='Arial'),
            paper_bgcolor='#F1F6F5',
            plot_bgcolor='white',
            margin=dict(l=80, r=80, t=50, b=80),
        )

        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)

        final_model = ExponentialSmoothing(df_selected[feature_column], trend='mul', seasonal='mul',
                                           seasonal_periods=seasonal_period).fit()
        forecast_predictions = final_model.forecast(forecast_period * seasonal_period)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_selected.index, y=df_selected[feature_column], mode='lines', name='Data'))
        fig2.add_trace(go.Scatter(x=forecast_predictions.index, y=forecast_predictions, mode='lines', name='Forecasting'))

        # Update layout with custom styling
        fig2.update_layout(
            legend=dict(
                itemsizing='constant',
                title_text='Time Series Forecasting ',
                font=dict(family='Arial', size=12),
                borderwidth=2
            ),
            xaxis=dict(
                title_text=date_column,
                title_font=dict(size=14),
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
            ),
            yaxis=dict(
                title_text=feature_column,
                title_font=dict(size=14),
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                linewidth=2,
                mirror=True
            ),
            title='Holt Winter Forecasting',
            title_font=dict(size=18, family='Arial'),
            paper_bgcolor='#F1F6F5',
            plot_bgcolor='white',
            margin=dict(l=80, r=80, t=50, b=80),
        )

        return {
            "plot1": fig.to_json(),
            "plot2": fig2.to_json(),
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        }





# If running this script directly, start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

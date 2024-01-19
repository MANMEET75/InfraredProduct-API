from utils import *


# Initialize the FastAPI application
app = FastAPI()



@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")



# dealing with statistical methods over here
@app.post("/zscore-anomaly/")
async def zscore_anomaly(file: UploadFile, column: str, threshold: float):
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

    data_with_anomalies_zscore = z_score_anomaly_detection(data, column, threshold)

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


    return {
        "data_head": data_with_anomalies_zscore.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
    }

@app.post("/zscore-anomaly/")
async def zscore_anomaly(file: UploadFile, column: str, threshold: float):
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

    data_with_anomalies_zscore = z_score_anomaly_detection(data, column, threshold)

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
        title_text=column,
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
    ),
        yaxis=dict(
        title_text='Anomaly',
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


    # Calculate the percentage of anomalies
    total_data_points = data_with_anomalies_zscore.shape[0]
    total_anomalies = data_with_anomalies_zscore["Anomaly"].sum()
    percentage_anomalies = (total_anomalies / total_data_points) * 100


    return {
        "data_head": data_with_anomalies_zscore.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
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



    return {
        "data_head": data.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_outliers:.2f}%"
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


    threshold = 0.01
    data['anomaly'] = 0
    data.loc[data[column] < norm.ppf(threshold, mean, std), 'anomaly'] = 1


    total_data_points = data.shape[0]
    total_outliers = data['anomaly'].sum()
    percentage_outliers = (total_outliers / total_data_points) * 100

    return {
        "data_head": data.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_outliers:.2f}%"
    }



# adding relative size factor over here
@app.post("/rsf-anomaly/")
async def rsf_anomaly(file: UploadFile):
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



    dfx=data[['WERKS','MATNR','EBELN','EBELP','LIFNR','MENGE','NETPR','PEINH','NETWR']]
    ebeln_count = dfx.groupby('LIFNR')['EBELN'].nunique().reset_index()
    ebeln_count.rename(columns={'EBELN': 'EBELN_Count'}, inplace=True)

    netwr_sum_by_vendor = dfx.groupby('LIFNR')['NETWR'].sum().reset_index()
    netwr_sum_by_vendor.rename(columns={'NETWR': 'NETWR_Sum_ByVendor'}, inplace=True)

    netwr_sum_by_vendor_ebeln = dfx.groupby(['LIFNR', 'EBELN'])['NETWR'].sum().reset_index()
    netwr_sum_by_vendor_ebeln.rename(columns={'NETWR': 'NETWR_Sum_ByVendor_EBELN'}, inplace=True)

    dfx = pd.merge(dfx, ebeln_count, on='LIFNR')
    dfx = pd.merge(dfx, netwr_sum_by_vendor, on='LIFNR')
    dfx = pd.merge(dfx, netwr_sum_by_vendor_ebeln, on=['LIFNR', 'EBELN'])

    netwr_max = dfx.groupby(['LIFNR'])['NETWR_Sum_ByVendor_EBELN'].max().reset_index()
    netwr_max.rename(columns={'NETWR_Sum_ByVendor_EBELN': 'netwr_max'}, inplace=True)

    dfx = pd.merge(dfx, netwr_max, on='LIFNR')

    dfx['Avg_exclu_max'] = (dfx['NETWR_Sum_ByVendor'] - dfx['netwr_max']) / (dfx['EBELN_Count'] - 1)
    dfx['Relative Size Factor'] = dfx['netwr_max'] / dfx['Avg_exclu_max']

    anomaly = np.where((dfx['EBELN_Count'] > 5) & (dfx['Relative Size Factor'] > 10), 1, 0)
    dfx['Anomaly'] = anomaly




    dfx['Anomaly Flag'] = dfx['Anomaly'].apply(lambda x: 'Anomaly' if x == 1 else 'Not Anomaly')
    dfx['Anomaly Flag'] = dfx['Anomaly Flag'].astype(str)

    fig = px.scatter(
        dfx,
        x="Relative Size Factor",
        y="EBELN_Count",
        hover_name="LIFNR",
        color="Anomaly Flag",  
        color_discrete_map={"Not Anomaly": "blue", "Anomaly": "red"},
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
        title="Higher the Relative Size Factor and EBELN_Count more the Chances of Anomaly ",
                    
        title_font=dict(size=18, family='Arial'),
        paper_bgcolor='#F1F6F5',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=80),
    )



    total_data_points = dfx.shape[0]
    total_anomalies = dfx['Anomaly'].sum()
    percentage_anomalies = (total_anomalies / total_data_points) * 100


    return {
        "data_head": dfx.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
    }

# adding the benford's law over here
@app.post("/benford-first-law-anomaly/")
async def benfordFirstLaw_anomaly(file: UploadFile, column: str):
    try:
        file_extension = file.filename.split(".")[-1]
        if file_extension.lower() == "csv":
            data = pd.read_csv(BytesIO(await file.read()))
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
        newdf = data.select_dtypes(include=numerics)


        # Calculate the distribution of first digits using Benford's Law
        df, df1, counts, benford  = calculate_first_digit(data[column])
        df2 = pd.merge(df, df1, left_on=df.iloc[:, 0],right_on='index',how='right')



        observed_trace = go.Bar(x=counts.index, y=counts * 100, name='Observed')
        expected_trace = go.Scatter(x=np.arange(0, 10), y=benford * 100, mode='lines', name='Expected')


        # Create the layout
        layout = go.Layout(
            title="Benford's Law Analysis of " + column,
            xaxis=dict(title="First Digit"),
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
        title="Benford's First Digit Law",

        title_font=dict(size=18, family='Arial'),
        paper_bgcolor='#F1F6F5',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=80),
        )

        deviation = (counts - benford) * 100
        results = pd.DataFrame({'Digit': counts.index, 'Observed (%)': counts * 100, 'Expected (%)': benford * 100,
                            'Deviation (%)': deviation})
        

        return {
            # "data": results.to_dict(orient="records"),
            "plot": fig.to_json()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# adding the benford's second digit law over here
@app.post("/benford-second-law-anomaly/")
async def benfordSecondLaw_anomaly(file: UploadFile, column: str):
    try:
        file_extension = file.filename.split(".")[-1]
        if file_extension.lower() == "csv":
            data = pd.read_csv(BytesIO(await file.read()))
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
        newdf = data.select_dtypes(include=numerics)


        # Calculate the distribution of first digits using Benford's Law
        df, df1, counts, benford  = calculate_2th_digit(data[column])
        df2 = pd.merge(df, df1, left_on=df.iloc[:, 0],right_on='index',how='right')



        observed_trace = go.Bar(x=counts.index, y=counts * 100, name='Observed')
        expected_trace = go.Scatter(x=np.arange(0, 10), y=benford * 100, mode='lines', name='Expected')


        # Create the layout
        layout = go.Layout(
            title="Benford's Law Analysis of " + column,
            xaxis=dict(title="First Digit"),
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
        title="Benford's First Digit Law",

        title_font=dict(size=18, family='Arial'),
        paper_bgcolor='#F1F6F5',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=50, b=80),
        )

        deviation = (counts - benford) * 100
        results = pd.DataFrame({'Digit': counts.index, 'Observed (%)': counts * 100, 'Expected (%)': benford * 100,
                            'Deviation (%)': deviation})
        

        return {
            # "data": results.to_dict(orient="records"),
            "plot": fig.to_json()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# adding the benford's law over here
@app.post("/benford-third-law-anomaly/")
async def benfordThirdLaw_anomaly(file: UploadFile, column: str):
    try:
        file_extension = file.filename.split(".")[-1]
        if file_extension.lower() == "csv":
            data = pd.read_csv(BytesIO(await file.read()))
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
        newdf = data.select_dtypes(include=numerics)


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
    data_with_anomalies_IsolationForest.loc[data_with_anomalies_IsolationForest['Anomaly'] == 1, 'PointColor'] = 'Outlier'

    AnomalyFeature=data_with_anomalies_IsolationForest[["Anomaly"]]
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


    num_anomalies = data_with_anomalies_IsolationForest['Anomaly'].sum()
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



    return {
        "data_head": data_with_anomalies_kde.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
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



    return {
        "data_head": data_with_anomalies_LocalOutlierFactor.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
    }

# adding the one class svm over here
@app.post("/one-class-svm-anomaly/")
async def OneClassSVM_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
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



    data_with_anomalies_OneClassSVM = apply_anomaly_detection_OneClassSVM(data)

    AnomalyFeature=data_with_anomalies_OneClassSVM[["Anomaly"]]
    final_data=pd.concat([copy_data,AnomalyFeature],axis=1)

    data_with_anomalies_OneClassSVM['PointColor'] = 'Inlier'
    data_with_anomalies_OneClassSVM.loc[data_with_anomalies_OneClassSVM['Anomaly'] == 1, 'PointColor'] = 'Outlier'

    # Create a scatter plot using Plotly
    fig = px.scatter(
        data_with_anomalies_OneClassSVM,
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

    num_anomalies = data_with_anomalies_OneClassSVM['Anomaly'].sum()
    total_data_points = len(data_with_anomalies_OneClassSVM)
    percentage_anomalies = (num_anomalies / total_data_points) * 100



    return {
        "data_head": data_with_anomalies_OneClassSVM.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
    }

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



    return {
        "data_head": data_with_anomalies_SGDOCSVM.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
    }


# adding more autoencoder method over here
@app.post("/autoencoder-anomaly/")
async def Autoencoder_anomaly(file: UploadFile, xcolumn: str,ycolumn: str):
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

    return {
        "data_head": data_with_anomalies_Autoencoder.head(5).to_dict(orient="records"),
        "plot": fig.to_json(),
        "percentage_anomalies": f"{percentage_anomalies:.2f}%"
    }

# adding the holt winter method over here
@app.post("/holt-winter-anomaly/")
async def HoltWinter_anomaly(file: UploadFile, date_column: str, feature_column: str, forecast_period: int):
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
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
# adding the code for time series anomaly detection using LSTM
@app.post("/TimeSeriesUsingLSTM_anomaly/")
async def TimeSeriesLSTM_anomaly(file: UploadFile, date_column: str, feature_column: str):
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() == "csv":
        data = pd.read_csv(BytesIO(await file.read()))
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


    df_selected = data[[date_column, feature_column]].copy()
    train_size = int(0.8 * len(df_selected))
    test_size = len(df_selected) - train_size

    train_data = df_selected.iloc[:train_size]
    test_data = df_selected.iloc[train_size: train_size + test_size]

    scaler = StandardScaler()
    scaler = scaler.fit(np.array(train_data[feature_column]).reshape(-1,1))

    train_data[feature_column]=scaler.transform(np.array(train_data[feature_column]).reshape(-1,1))
    test_data[feature_column]=scaler.transform(np.array(test_data[feature_column]).reshape(-1,1))

    TIME_STEPS=30
    
    def create_sequences(X, y, time_steps=TIME_STEPS):
        X_out, y_out = [], []
        for i in range(len(X)-time_steps):
            X_out.append(X.iloc[i:(i+time_steps)].values)
            y_out.append(y.iloc[i+time_steps])
                            
        return np.array(X_out), np.array(y_out)
    

    X_train, y_train = create_sequences(train_data[[feature_column]], train_data[feature_column])
    X_test, y_test = create_sequences(test_data[[feature_column]], test_data[feature_column])

    np.random.seed(21)
    tf.random.set_seed(21)

    model = Sequential()
    model.add(LSTM(128, activation = 'tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(128, activation = 'tanh', return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(X_train.shape[2])))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=32,
                    verbose=True,
                    validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
                    shuffle=False)
    

    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    threshold = np.max(train_mae_loss)
    X_test_pred = model.predict(X_test, verbose=1)
    test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

    anomaly_df = pd.DataFrame(test_data[TIME_STEPS:])
    anomaly_df['loss'] = test_mae_loss
    anomaly_df['threshold'] = threshold
    anomaly_df['anomaly'] = anomaly_df['loss'] > anomaly_df['threshold']


    fig_threshold = go.Figure()
    fig_threshold.add_trace(go.Scatter(x=anomaly_df[date_column], y=anomaly_df['loss'], name='Test loss'))
    fig_threshold.add_trace(go.Scatter(x=anomaly_df[date_column], y=anomaly_df['threshold'], name='Threshold'))


    fig_threshold.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
        font=dict(family='Arial', size=12),
        borderwidth=2
      ),
    xaxis=dict(
         title_font=dict(size=14),
         showgrid=False,
        showline=True,
         linecolor='lightgray',
         linewidth=2,
        mirror=True
    ),
    yaxis=dict(
        title_font=dict(size=14),
        showgrid=False,
        showline=True,
        linecolor='lightgray',
        linewidth=2,
        mirror=True
      ),
    title="Test loss vs. Threshold",
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )
    fig_threshold.update_layout(showlegend=True, title='Test loss vs. Threshold')


    anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]
    fig = go.Figure()


    anomaly_result = scaler.inverse_transform(anomaly_df[feature_column].values.reshape(-1, 1))
    anomalies_result = scaler.inverse_transform(anomalies[feature_column].values.reshape(-1, 1))


    fig.add_trace(go.Scatter(x=anomaly_df[date_column], y=anomaly_result.flatten(), name='Close price'))
    fig.add_trace(go.Scatter(x=anomalies[date_column], y=anomalies_result.flatten(), mode='markers', marker=dict(color='red'), name='Anomaly'))  # Set mode to 'markers' and specify marker color


    # Set x-axis and y-axis labels
    fig.update_xaxes(title_text=f'{date_column}')  # Set x-axis label
    fig.update_yaxes(title_text=f'{feature_column}')  # Set y-axis label

    # Update layout with custom styling
    fig.update_layout(
        legend=dict(
        itemsizing='constant',
        title_text='',
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
    title="Time Series Anomaly Detection",
    title_font=dict(size=18, family='Arial'),
    paper_bgcolor='#F1F6F5',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=50, b=80),
    )
  




    return {
            "fig_threshold_plot": fig_threshold.to_json(),
            "ResultantPlot": fig.to_json(),
        }




# If running this script directly, start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

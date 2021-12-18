# Venture Funding with Deep Learning

You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.

## Instructions:

The steps for this challenge are broken out into the following sections:

* Prepare the data for use on a neural network model.

* Compile and evaluate a binary classification model using a neural network.

* Optimize the neural network model.

### Prepare the Data for Use on a Neural Network Model 

Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, preprocess the dataset so that you can use it to compile and evaluate the neural network model later.

Open the starter code file, and complete the following data preparation steps:

1. Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.   

2. Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.
 
3. Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.

4. Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.

> **Note** To complete this step, you will employ the Pandas `concat()` function that was introduced earlier in this course. 

5. Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset. 

6. Split the features and target sets into training and testing datasets.

7. Use scikit-learn's `StandardScaler` to scale the features data.

### Compile and Evaluate a Binary Classification Model Using a Neural Network

Use your knowledge of TensorFlow to design a binary classification deep neural network model. This model should use the dataset’s features to predict whether an Alphabet Soup&ndash;funded startup will be successful based on the features in the dataset. Consider the number of inputs before determining the number of layers that your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate your binary classification model to calculate the model’s loss and accuracy. 
 
To do so, complete the following steps:

1. Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

> **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.

2. Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.

> **Hint** When fitting the model, start with a small number of epochs, such as 20, 50, or 100.

3. Evaluate the model using the test data to determine the model’s loss and accuracy.

4. Save and export your model to an HDF5 file, and name the file `AlphabetSoup.h5`. 

### Optimize the Neural Network Model

Using your knowledge of TensorFlow and Keras, optimize your model to improve the model's accuracy. Even if you do not successfully achieve a better accuracy, you'll need to demonstrate at least two attempts to optimize the model. You can include these attempts in your existing notebook. Or, you can make copies of the starter notebook in the same folder, rename them, and code each model optimization in a new notebook. 

> **Note** You will not lose points if your model does not achieve a high accuracy, as long as you make at least two attempts to optimize the model.

To do so, complete the following steps:

1. Define at least three new deep neural network models (the original plus 2 optimization attempts). With each, try to improve on your first model’s predictive accuracy.

> **Rewind** Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:
>
> * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
>
> * Add more neurons (nodes) to a hidden layer.
>
> * Add more hidden layers.
>
> * Use different activation functions for the hidden layers.
>
> * Add to or reduce the number of epochs in the training regimen.

2. After finishing your models, display the accuracy scores achieved by each model, and compare the results.

3. Save each of your models as an HDF5 file.



```python
# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import io
```

---

## Prepare the data to be used on a neural network model

### Step 1: Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.  



```python
# Upload credit_card_transactions.csv to Colab
from google.colab import files

csv_file = files.upload()
```



<input type="file" id="files-1c6e9318-c3d7-4151-afcb-a5282371cdcb" name="files[]" multiple disabled
   style="border:none" />
<output id="result-1c6e9318-c3d7-4151-afcb-a5282371cdcb">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving applicants_data.csv to applicants_data.csv
    


```python
# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
applicant_data_df = pd.read_csv(io.BytesIO(csv_file["applicants_data.csv"]))

# Review the DataFrame
applicant_data_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EIN</th>
      <th>NAME</th>
      <th>APPLICATION_TYPE</th>
      <th>AFFILIATION</th>
      <th>CLASSIFICATION</th>
      <th>USE_CASE</th>
      <th>ORGANIZATION</th>
      <th>STATUS</th>
      <th>INCOME_AMT</th>
      <th>SPECIAL_CONSIDERATIONS</th>
      <th>ASK_AMT</th>
      <th>IS_SUCCESSFUL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10520599</td>
      <td>BLUE KNIGHTS MOTORCYCLE CLUB</td>
      <td>T10</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10531628</td>
      <td>AMERICAN CHESAPEAKE CLUB CHARITABLE TR</td>
      <td>T3</td>
      <td>Independent</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Co-operative</td>
      <td>1</td>
      <td>1-9999</td>
      <td>N</td>
      <td>108590</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10547893</td>
      <td>ST CLOUD PROFESSIONAL FIREFIGHTERS</td>
      <td>T5</td>
      <td>CompanySponsored</td>
      <td>C3000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10553066</td>
      <td>SOUTHSIDE ATHLETIC ASSOCIATION</td>
      <td>T3</td>
      <td>CompanySponsored</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Trust</td>
      <td>1</td>
      <td>10000-24999</td>
      <td>N</td>
      <td>6692</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10556103</td>
      <td>GENETIC RESEARCH INSTITUTE OF THE DESERT</td>
      <td>T3</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>Heathcare</td>
      <td>Trust</td>
      <td>1</td>
      <td>100000-499999</td>
      <td>N</td>
      <td>142590</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Review the data types associated with the columns
applicant_data_df.dtypes

```




    EIN                        int64
    NAME                      object
    APPLICATION_TYPE          object
    AFFILIATION               object
    CLASSIFICATION            object
    USE_CASE                  object
    ORGANIZATION              object
    STATUS                     int64
    INCOME_AMT                object
    SPECIAL_CONSIDERATIONS    object
    ASK_AMT                    int64
    IS_SUCCESSFUL              int64
    dtype: object



### Step 2: Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.


```python
# Drop the 'EIN' and 'NAME' columns from the DataFrame
applicant_data_df = applicant_data_df.drop(columns=["EIN", "NAME"])

# Review the DataFrame
applicant_data_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>APPLICATION_TYPE</th>
      <th>AFFILIATION</th>
      <th>CLASSIFICATION</th>
      <th>USE_CASE</th>
      <th>ORGANIZATION</th>
      <th>STATUS</th>
      <th>INCOME_AMT</th>
      <th>SPECIAL_CONSIDERATIONS</th>
      <th>ASK_AMT</th>
      <th>IS_SUCCESSFUL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>T10</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>T3</td>
      <td>Independent</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Co-operative</td>
      <td>1</td>
      <td>1-9999</td>
      <td>N</td>
      <td>108590</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>T5</td>
      <td>CompanySponsored</td>
      <td>C3000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>T3</td>
      <td>CompanySponsored</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Trust</td>
      <td>1</td>
      <td>10000-24999</td>
      <td>N</td>
      <td>6692</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>T3</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>Heathcare</td>
      <td>Trust</td>
      <td>1</td>
      <td>100000-499999</td>
      <td>N</td>
      <td>142590</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34294</th>
      <td>T4</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34295</th>
      <td>T4</td>
      <td>CompanySponsored</td>
      <td>C3000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34296</th>
      <td>T3</td>
      <td>CompanySponsored</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34297</th>
      <td>T5</td>
      <td>Independent</td>
      <td>C3000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34298</th>
      <td>T3</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>Preservation</td>
      <td>Co-operative</td>
      <td>1</td>
      <td>1M-5M</td>
      <td>N</td>
      <td>36500179</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>34299 rows × 10 columns</p>
</div>



### Step 3: Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.


```python
# Create a list of categorical variables 
categorical_variables = list(applicant_data_df.dtypes[applicant_data_df.dtypes == "object"].index)

# Display the categorical variables list
categorical_variables

```




    ['APPLICATION_TYPE',
     'AFFILIATION',
     'CLASSIFICATION',
     'USE_CASE',
     'ORGANIZATION',
     'INCOME_AMT',
     'SPECIAL_CONSIDERATIONS']




```python
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

```


```python
# Encode the categorcal variables using OneHotEncoder
encoded_data = enc.fit_transform(applicant_data_df[categorical_variables])

```


```python
# Create a DataFrame with the encoded variables
encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names_out(categorical_variables)
)

# Review the DataFrame
encoded_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>APPLICATION_TYPE_T10</th>
      <th>APPLICATION_TYPE_T12</th>
      <th>APPLICATION_TYPE_T13</th>
      <th>APPLICATION_TYPE_T14</th>
      <th>APPLICATION_TYPE_T15</th>
      <th>APPLICATION_TYPE_T17</th>
      <th>APPLICATION_TYPE_T19</th>
      <th>APPLICATION_TYPE_T2</th>
      <th>APPLICATION_TYPE_T25</th>
      <th>APPLICATION_TYPE_T29</th>
      <th>APPLICATION_TYPE_T3</th>
      <th>APPLICATION_TYPE_T4</th>
      <th>APPLICATION_TYPE_T5</th>
      <th>APPLICATION_TYPE_T6</th>
      <th>APPLICATION_TYPE_T7</th>
      <th>APPLICATION_TYPE_T8</th>
      <th>APPLICATION_TYPE_T9</th>
      <th>AFFILIATION_CompanySponsored</th>
      <th>AFFILIATION_Family/Parent</th>
      <th>AFFILIATION_Independent</th>
      <th>AFFILIATION_National</th>
      <th>AFFILIATION_Other</th>
      <th>AFFILIATION_Regional</th>
      <th>CLASSIFICATION_C0</th>
      <th>CLASSIFICATION_C1000</th>
      <th>CLASSIFICATION_C1200</th>
      <th>CLASSIFICATION_C1230</th>
      <th>CLASSIFICATION_C1234</th>
      <th>CLASSIFICATION_C1235</th>
      <th>CLASSIFICATION_C1236</th>
      <th>CLASSIFICATION_C1237</th>
      <th>CLASSIFICATION_C1238</th>
      <th>CLASSIFICATION_C1240</th>
      <th>CLASSIFICATION_C1245</th>
      <th>CLASSIFICATION_C1246</th>
      <th>CLASSIFICATION_C1248</th>
      <th>CLASSIFICATION_C1250</th>
      <th>CLASSIFICATION_C1256</th>
      <th>CLASSIFICATION_C1257</th>
      <th>CLASSIFICATION_C1260</th>
      <th>...</th>
      <th>CLASSIFICATION_C3000</th>
      <th>CLASSIFICATION_C3200</th>
      <th>CLASSIFICATION_C3700</th>
      <th>CLASSIFICATION_C4000</th>
      <th>CLASSIFICATION_C4100</th>
      <th>CLASSIFICATION_C4120</th>
      <th>CLASSIFICATION_C4200</th>
      <th>CLASSIFICATION_C4500</th>
      <th>CLASSIFICATION_C5000</th>
      <th>CLASSIFICATION_C5200</th>
      <th>CLASSIFICATION_C6000</th>
      <th>CLASSIFICATION_C6100</th>
      <th>CLASSIFICATION_C7000</th>
      <th>CLASSIFICATION_C7100</th>
      <th>CLASSIFICATION_C7120</th>
      <th>CLASSIFICATION_C7200</th>
      <th>CLASSIFICATION_C7210</th>
      <th>CLASSIFICATION_C8000</th>
      <th>CLASSIFICATION_C8200</th>
      <th>CLASSIFICATION_C8210</th>
      <th>USE_CASE_CommunityServ</th>
      <th>USE_CASE_Heathcare</th>
      <th>USE_CASE_Other</th>
      <th>USE_CASE_Preservation</th>
      <th>USE_CASE_ProductDev</th>
      <th>ORGANIZATION_Association</th>
      <th>ORGANIZATION_Co-operative</th>
      <th>ORGANIZATION_Corporation</th>
      <th>ORGANIZATION_Trust</th>
      <th>INCOME_AMT_0</th>
      <th>INCOME_AMT_1-9999</th>
      <th>INCOME_AMT_10000-24999</th>
      <th>INCOME_AMT_100000-499999</th>
      <th>INCOME_AMT_10M-50M</th>
      <th>INCOME_AMT_1M-5M</th>
      <th>INCOME_AMT_25000-99999</th>
      <th>INCOME_AMT_50M+</th>
      <th>INCOME_AMT_5M-10M</th>
      <th>SPECIAL_CONSIDERATIONS_N</th>
      <th>SPECIAL_CONSIDERATIONS_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 114 columns</p>
</div>



### Step 4: Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.

> **Note** To complete this step, you will employ the Pandas `concat()` function that was introduced earlier in this course. 


```python
# Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
encoded_df = pd.concat([
    applicant_data_df[["STATUS", "ASK_AMT", "IS_SUCCESSFUL"]], encoded_df
], axis=1)

# Review the Dataframe
encoded_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STATUS</th>
      <th>ASK_AMT</th>
      <th>IS_SUCCESSFUL</th>
      <th>APPLICATION_TYPE_T10</th>
      <th>APPLICATION_TYPE_T12</th>
      <th>APPLICATION_TYPE_T13</th>
      <th>APPLICATION_TYPE_T14</th>
      <th>APPLICATION_TYPE_T15</th>
      <th>APPLICATION_TYPE_T17</th>
      <th>APPLICATION_TYPE_T19</th>
      <th>APPLICATION_TYPE_T2</th>
      <th>APPLICATION_TYPE_T25</th>
      <th>APPLICATION_TYPE_T29</th>
      <th>APPLICATION_TYPE_T3</th>
      <th>APPLICATION_TYPE_T4</th>
      <th>APPLICATION_TYPE_T5</th>
      <th>APPLICATION_TYPE_T6</th>
      <th>APPLICATION_TYPE_T7</th>
      <th>APPLICATION_TYPE_T8</th>
      <th>APPLICATION_TYPE_T9</th>
      <th>AFFILIATION_CompanySponsored</th>
      <th>AFFILIATION_Family/Parent</th>
      <th>AFFILIATION_Independent</th>
      <th>AFFILIATION_National</th>
      <th>AFFILIATION_Other</th>
      <th>AFFILIATION_Regional</th>
      <th>CLASSIFICATION_C0</th>
      <th>CLASSIFICATION_C1000</th>
      <th>CLASSIFICATION_C1200</th>
      <th>CLASSIFICATION_C1230</th>
      <th>CLASSIFICATION_C1234</th>
      <th>CLASSIFICATION_C1235</th>
      <th>CLASSIFICATION_C1236</th>
      <th>CLASSIFICATION_C1237</th>
      <th>CLASSIFICATION_C1238</th>
      <th>CLASSIFICATION_C1240</th>
      <th>CLASSIFICATION_C1245</th>
      <th>CLASSIFICATION_C1246</th>
      <th>CLASSIFICATION_C1248</th>
      <th>CLASSIFICATION_C1250</th>
      <th>...</th>
      <th>CLASSIFICATION_C3000</th>
      <th>CLASSIFICATION_C3200</th>
      <th>CLASSIFICATION_C3700</th>
      <th>CLASSIFICATION_C4000</th>
      <th>CLASSIFICATION_C4100</th>
      <th>CLASSIFICATION_C4120</th>
      <th>CLASSIFICATION_C4200</th>
      <th>CLASSIFICATION_C4500</th>
      <th>CLASSIFICATION_C5000</th>
      <th>CLASSIFICATION_C5200</th>
      <th>CLASSIFICATION_C6000</th>
      <th>CLASSIFICATION_C6100</th>
      <th>CLASSIFICATION_C7000</th>
      <th>CLASSIFICATION_C7100</th>
      <th>CLASSIFICATION_C7120</th>
      <th>CLASSIFICATION_C7200</th>
      <th>CLASSIFICATION_C7210</th>
      <th>CLASSIFICATION_C8000</th>
      <th>CLASSIFICATION_C8200</th>
      <th>CLASSIFICATION_C8210</th>
      <th>USE_CASE_CommunityServ</th>
      <th>USE_CASE_Heathcare</th>
      <th>USE_CASE_Other</th>
      <th>USE_CASE_Preservation</th>
      <th>USE_CASE_ProductDev</th>
      <th>ORGANIZATION_Association</th>
      <th>ORGANIZATION_Co-operative</th>
      <th>ORGANIZATION_Corporation</th>
      <th>ORGANIZATION_Trust</th>
      <th>INCOME_AMT_0</th>
      <th>INCOME_AMT_1-9999</th>
      <th>INCOME_AMT_10000-24999</th>
      <th>INCOME_AMT_100000-499999</th>
      <th>INCOME_AMT_10M-50M</th>
      <th>INCOME_AMT_1M-5M</th>
      <th>INCOME_AMT_25000-99999</th>
      <th>INCOME_AMT_50M+</th>
      <th>INCOME_AMT_5M-10M</th>
      <th>SPECIAL_CONSIDERATIONS_N</th>
      <th>SPECIAL_CONSIDERATIONS_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5000</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>108590</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>6692</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>142590</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34294</th>
      <td>1</td>
      <td>5000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34295</th>
      <td>1</td>
      <td>5000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34296</th>
      <td>1</td>
      <td>5000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34297</th>
      <td>1</td>
      <td>5000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34298</th>
      <td>1</td>
      <td>36500179</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>34299 rows × 117 columns</p>
</div>



### Step 5: Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset. 




```python
# Define the target set y using the IS_SUCCESSFUL column
y = encoded_df["IS_SUCCESSFUL"]

# Display a sample of y
y[:5]

```




    0    1
    1    1
    2    0
    3    1
    4    1
    Name: IS_SUCCESSFUL, dtype: int64




```python
# Define features set X by selecting all columns but IS_SUCCESSFUL
x = encoded_df.drop(columns=["IS_SUCCESSFUL"])

# Review the features DataFrame
x.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STATUS</th>
      <th>ASK_AMT</th>
      <th>APPLICATION_TYPE_T10</th>
      <th>APPLICATION_TYPE_T12</th>
      <th>APPLICATION_TYPE_T13</th>
      <th>APPLICATION_TYPE_T14</th>
      <th>APPLICATION_TYPE_T15</th>
      <th>APPLICATION_TYPE_T17</th>
      <th>APPLICATION_TYPE_T19</th>
      <th>APPLICATION_TYPE_T2</th>
      <th>APPLICATION_TYPE_T25</th>
      <th>APPLICATION_TYPE_T29</th>
      <th>APPLICATION_TYPE_T3</th>
      <th>APPLICATION_TYPE_T4</th>
      <th>APPLICATION_TYPE_T5</th>
      <th>APPLICATION_TYPE_T6</th>
      <th>APPLICATION_TYPE_T7</th>
      <th>APPLICATION_TYPE_T8</th>
      <th>APPLICATION_TYPE_T9</th>
      <th>AFFILIATION_CompanySponsored</th>
      <th>AFFILIATION_Family/Parent</th>
      <th>AFFILIATION_Independent</th>
      <th>AFFILIATION_National</th>
      <th>AFFILIATION_Other</th>
      <th>AFFILIATION_Regional</th>
      <th>CLASSIFICATION_C0</th>
      <th>CLASSIFICATION_C1000</th>
      <th>CLASSIFICATION_C1200</th>
      <th>CLASSIFICATION_C1230</th>
      <th>CLASSIFICATION_C1234</th>
      <th>CLASSIFICATION_C1235</th>
      <th>CLASSIFICATION_C1236</th>
      <th>CLASSIFICATION_C1237</th>
      <th>CLASSIFICATION_C1238</th>
      <th>CLASSIFICATION_C1240</th>
      <th>CLASSIFICATION_C1245</th>
      <th>CLASSIFICATION_C1246</th>
      <th>CLASSIFICATION_C1248</th>
      <th>CLASSIFICATION_C1250</th>
      <th>CLASSIFICATION_C1256</th>
      <th>...</th>
      <th>CLASSIFICATION_C3000</th>
      <th>CLASSIFICATION_C3200</th>
      <th>CLASSIFICATION_C3700</th>
      <th>CLASSIFICATION_C4000</th>
      <th>CLASSIFICATION_C4100</th>
      <th>CLASSIFICATION_C4120</th>
      <th>CLASSIFICATION_C4200</th>
      <th>CLASSIFICATION_C4500</th>
      <th>CLASSIFICATION_C5000</th>
      <th>CLASSIFICATION_C5200</th>
      <th>CLASSIFICATION_C6000</th>
      <th>CLASSIFICATION_C6100</th>
      <th>CLASSIFICATION_C7000</th>
      <th>CLASSIFICATION_C7100</th>
      <th>CLASSIFICATION_C7120</th>
      <th>CLASSIFICATION_C7200</th>
      <th>CLASSIFICATION_C7210</th>
      <th>CLASSIFICATION_C8000</th>
      <th>CLASSIFICATION_C8200</th>
      <th>CLASSIFICATION_C8210</th>
      <th>USE_CASE_CommunityServ</th>
      <th>USE_CASE_Heathcare</th>
      <th>USE_CASE_Other</th>
      <th>USE_CASE_Preservation</th>
      <th>USE_CASE_ProductDev</th>
      <th>ORGANIZATION_Association</th>
      <th>ORGANIZATION_Co-operative</th>
      <th>ORGANIZATION_Corporation</th>
      <th>ORGANIZATION_Trust</th>
      <th>INCOME_AMT_0</th>
      <th>INCOME_AMT_1-9999</th>
      <th>INCOME_AMT_10000-24999</th>
      <th>INCOME_AMT_100000-499999</th>
      <th>INCOME_AMT_10M-50M</th>
      <th>INCOME_AMT_1M-5M</th>
      <th>INCOME_AMT_25000-99999</th>
      <th>INCOME_AMT_50M+</th>
      <th>INCOME_AMT_5M-10M</th>
      <th>SPECIAL_CONSIDERATIONS_N</th>
      <th>SPECIAL_CONSIDERATIONS_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>108590</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>6692</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>142590</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 116 columns</p>
</div>



### Step 6: Split the features and target sets into training and testing datasets.



```python
# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1)

```

### Step 7: Use scikit-learn's `StandardScaler` to scale the features data.


```python
from pandas.core.tools.datetimes import Scalar
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features training dataset
x_scaler = scaler.fit(x_train)

# Fit the scaler to the features training dataset
x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

```

---

## Compile and Evaluate a Binary Classification Model Using a Neural Network

### Step 1: Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

> **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.



```python
# Define the the number of inputs (features) to the model
number_input_features = len(x_train.iloc[0])

# Review the number of features
number_input_features

```




    116




```python
# Define the number of neurons in the output layer
number_output_neurons = 1
```


```python
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 =  (number_input_features + 1) // 2

# Review the number hidden nodes in the first layer
hidden_nodes_layer1

```




    58




```python
# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 =  (hidden_nodes_layer1 + 1) // 2

# Review the number hidden nodes in the second layer
hidden_nodes_layer2

```




    29




```python
# Create the Sequential model instance
nn = Sequential()

```


```python
# Add the first hidden layer
nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

```


```python
# Add the second hidden layer
nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))

```


```python
# Add the output layer to the model specifying the number of output neurons and activation function
nn.add(Dense(units=1, activation="sigmoid"))

```


```python
# Display the Sequential model summary
nn.summary()

```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 58)                6786      
                                                                     
     dense_1 (Dense)             (None, 29)                1711      
                                                                     
     dense_2 (Dense)             (None, 1)                 30        
                                                                     
    =================================================================
    Total params: 8,527
    Trainable params: 8,527
    Non-trainable params: 0
    _________________________________________________________________
    

### Step 2: Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.



```python
# Compile the Sequential model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

```


```python
# Fit the model using 50 epochs and the training data
fit_model = nn.fit(x_train_scaled, y_train, epochs=50)

```

    Epoch 1/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5736 - accuracy: 0.7196
    Epoch 2/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5533 - accuracy: 0.7299
    Epoch 3/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5503 - accuracy: 0.7296
    Epoch 4/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5482 - accuracy: 0.7309
    Epoch 5/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5458 - accuracy: 0.7326
    Epoch 6/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5456 - accuracy: 0.7322
    Epoch 7/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5443 - accuracy: 0.7327
    Epoch 8/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5431 - accuracy: 0.7353
    Epoch 9/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5423 - accuracy: 0.7342
    Epoch 10/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5413 - accuracy: 0.7344
    Epoch 11/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5408 - accuracy: 0.7363
    Epoch 12/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5401 - accuracy: 0.7359
    Epoch 13/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5402 - accuracy: 0.7346
    Epoch 14/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5392 - accuracy: 0.7367
    Epoch 15/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5394 - accuracy: 0.7372
    Epoch 16/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5388 - accuracy: 0.7372
    Epoch 17/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5378 - accuracy: 0.7386
    Epoch 18/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5382 - accuracy: 0.7381
    Epoch 19/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5379 - accuracy: 0.7373
    Epoch 20/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5369 - accuracy: 0.7385
    Epoch 21/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5371 - accuracy: 0.7375
    Epoch 22/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5370 - accuracy: 0.7380
    Epoch 23/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5368 - accuracy: 0.7386
    Epoch 24/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5368 - accuracy: 0.7386
    Epoch 25/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5361 - accuracy: 0.7387
    Epoch 26/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5362 - accuracy: 0.7394
    Epoch 27/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5360 - accuracy: 0.7381
    Epoch 28/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5354 - accuracy: 0.7390
    Epoch 29/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5354 - accuracy: 0.7395
    Epoch 30/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5353 - accuracy: 0.7391
    Epoch 31/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5347 - accuracy: 0.7401
    Epoch 32/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5347 - accuracy: 0.7396
    Epoch 33/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5340 - accuracy: 0.7396
    Epoch 34/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5342 - accuracy: 0.7396
    Epoch 35/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5339 - accuracy: 0.7404
    Epoch 36/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5341 - accuracy: 0.7403
    Epoch 37/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5333 - accuracy: 0.7401
    Epoch 38/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5338 - accuracy: 0.7399
    Epoch 39/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5336 - accuracy: 0.7397
    Epoch 40/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5334 - accuracy: 0.7409
    Epoch 41/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5332 - accuracy: 0.7397
    Epoch 42/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5329 - accuracy: 0.7407
    Epoch 43/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5330 - accuracy: 0.7402
    Epoch 44/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5323 - accuracy: 0.7404
    Epoch 45/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5322 - accuracy: 0.7418
    Epoch 46/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5326 - accuracy: 0.7407
    Epoch 47/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5324 - accuracy: 0.7408
    Epoch 48/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5321 - accuracy: 0.7407
    Epoch 49/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5318 - accuracy: 0.7408
    Epoch 50/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5320 - accuracy: 0.7398
    

### Step 3: Evaluate the model using the test data to determine the model’s loss and accuracy.



```python
# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(x_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

    268/268 - 0s - loss: 0.5579 - accuracy: 0.7303 - 461ms/epoch - 2ms/step
    Loss: 0.5578923225402832, Accuracy: 0.7302623987197876
    

### Step 4: Save and export your model to an HDF5 file, and name the file `AlphabetSoup.h5`. 



```python
# Set the model's file path
file_path = "Venture Funding.h5"

# Export your model to a HDF5 file
nn.save(file_path)

```

---

## Optimize the neural network model


### Step 1: Define at least three new deep neural network models (resulting in the original plus 3 optimization attempts). With each, try to improve on your first model’s predictive accuracy.

> **Rewind** Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:
>
> * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
>
> * Add more neurons (nodes) to a hidden layer.
>
> * Add more hidden layers.
>
> * Use different activation functions for the hidden layers.
>
> * Add to or reduce the number of epochs in the training regimen.


### Alternative Model 1


```python
# Define the the number of inputs (features) to the model
number_input_features = len(x_train.iloc[0])

# Review the number of features
number_input_features
```




    116




```python
# Define the number of neurons in the output layer
number_output_neurons_A1 = 1
```


```python
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A1 = (number_input_features + 1) // 2

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A1
```




    58




```python
# Create the Sequential model instance
nn_A1 = Sequential()
```


```python
# First hidden layer
nn_A1.add(Dense(units=hidden_nodes_layer1_A1, input_dim=number_input_features, activation="relu"))


# Output layer
nn_A1.add(Dense(units=1, activation="sigmoid"))


# Check the structure of the model
nn_A1.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_3 (Dense)             (None, 58)                6786      
                                                                     
     dense_4 (Dense)             (None, 1)                 59        
                                                                     
    =================================================================
    Total params: 6,845
    Trainable params: 6,845
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Compile the Sequential model
nn_A1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

```


```python
# Fit the model using 50 epochs and the training data
fit_model_A1 = nn_A1.fit(x_train_scaled, y_train, epochs=50)

```

    Epoch 1/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5903 - accuracy: 0.7138
    Epoch 2/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5583 - accuracy: 0.7271
    Epoch 3/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5537 - accuracy: 0.7300
    Epoch 4/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5526 - accuracy: 0.7305
    Epoch 5/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5506 - accuracy: 0.7303
    Epoch 6/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5496 - accuracy: 0.7320
    Epoch 7/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5481 - accuracy: 0.7320
    Epoch 8/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5478 - accuracy: 0.7306
    Epoch 9/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5476 - accuracy: 0.7321
    Epoch 10/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5459 - accuracy: 0.7335
    Epoch 11/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5461 - accuracy: 0.7331
    Epoch 12/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5453 - accuracy: 0.7331
    Epoch 13/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5456 - accuracy: 0.7327
    Epoch 14/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5443 - accuracy: 0.7341
    Epoch 15/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5443 - accuracy: 0.7345
    Epoch 16/50
    804/804 [==============================] - 2s 3ms/step - loss: 0.5443 - accuracy: 0.7329
    Epoch 17/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5437 - accuracy: 0.7344
    Epoch 18/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5435 - accuracy: 0.7330
    Epoch 19/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5428 - accuracy: 0.7344
    Epoch 20/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5426 - accuracy: 0.7338
    Epoch 21/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5433 - accuracy: 0.7353
    Epoch 22/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5422 - accuracy: 0.7350
    Epoch 23/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5422 - accuracy: 0.7343
    Epoch 24/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5417 - accuracy: 0.7346
    Epoch 25/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5410 - accuracy: 0.7358
    Epoch 26/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5413 - accuracy: 0.7347
    Epoch 27/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5418 - accuracy: 0.7339
    Epoch 28/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5407 - accuracy: 0.7355
    Epoch 29/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5408 - accuracy: 0.7348
    Epoch 30/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5411 - accuracy: 0.7358
    Epoch 31/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5408 - accuracy: 0.7350
    Epoch 32/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5406 - accuracy: 0.7353
    Epoch 33/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5402 - accuracy: 0.7354
    Epoch 34/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5407 - accuracy: 0.7348
    Epoch 35/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5398 - accuracy: 0.7346
    Epoch 36/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5405 - accuracy: 0.7343
    Epoch 37/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5396 - accuracy: 0.7355
    Epoch 38/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5390 - accuracy: 0.7381
    Epoch 39/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5395 - accuracy: 0.7346
    Epoch 40/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5393 - accuracy: 0.7368
    Epoch 41/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5398 - accuracy: 0.7360
    Epoch 42/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5391 - accuracy: 0.7361
    Epoch 43/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5386 - accuracy: 0.7365
    Epoch 44/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5392 - accuracy: 0.7364
    Epoch 45/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5389 - accuracy: 0.7374
    Epoch 46/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5389 - accuracy: 0.7355
    Epoch 47/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5384 - accuracy: 0.7362
    Epoch 48/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5380 - accuracy: 0.7372
    Epoch 49/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5381 - accuracy: 0.7375
    Epoch 50/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5382 - accuracy: 0.7378
    

#### Alternative Model 2


```python
# Define the the number of inputs (features) to the model
number_input_features = len(x_train.iloc[0])

# Review the number of features
number_input_features
```




    116




```python
# Define the number of neurons in the output layer
number_output_neurons_A2 = 1
```


```python
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A2 = (number_input_features + 1) // 2

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A2
```




    58




```python
# Create the Sequential model instance
nn_A2 = Sequential()
```


```python
# First hidden layer
nn_A2.add(Dense(units=hidden_nodes_layer1_A2, input_dim=number_input_features, activation="relu"))

# Output layer
nn_A2.add(Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn_A2.summary()

```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_5 (Dense)             (None, 58)                6786      
                                                                     
     dense_6 (Dense)             (None, 1)                 59        
                                                                     
    =================================================================
    Total params: 6,845
    Trainable params: 6,845
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Compile the model
nn_A2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

```


```python
# Fit the model
fit_model_A2 = nn_A1.fit(x_train_scaled, y_train, epochs=50)

```

    Epoch 1/50
    804/804 [==============================] - 4s 4ms/step - loss: 0.5380 - accuracy: 0.7388
    Epoch 2/50
    804/804 [==============================] - 4s 4ms/step - loss: 0.5379 - accuracy: 0.7370
    Epoch 3/50
    804/804 [==============================] - 4s 4ms/step - loss: 0.5379 - accuracy: 0.7381
    Epoch 4/50
    804/804 [==============================] - 3s 4ms/step - loss: 0.5380 - accuracy: 0.7378
    Epoch 5/50
    804/804 [==============================] - 3s 4ms/step - loss: 0.5376 - accuracy: 0.7382
    Epoch 6/50
    804/804 [==============================] - 2s 3ms/step - loss: 0.5374 - accuracy: 0.7376
    Epoch 7/50
    804/804 [==============================] - 2s 3ms/step - loss: 0.5373 - accuracy: 0.7365
    Epoch 8/50
    804/804 [==============================] - 2s 3ms/step - loss: 0.5371 - accuracy: 0.7378
    Epoch 9/50
    804/804 [==============================] - 2s 3ms/step - loss: 0.5372 - accuracy: 0.7384
    Epoch 10/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5373 - accuracy: 0.7378
    Epoch 11/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5369 - accuracy: 0.7376
    Epoch 12/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5372 - accuracy: 0.7374
    Epoch 13/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5367 - accuracy: 0.7386
    Epoch 14/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5371 - accuracy: 0.7382
    Epoch 15/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5371 - accuracy: 0.7386
    Epoch 16/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5375 - accuracy: 0.7374
    Epoch 17/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5366 - accuracy: 0.7384
    Epoch 18/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5366 - accuracy: 0.7379
    Epoch 19/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5365 - accuracy: 0.7371
    Epoch 20/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5364 - accuracy: 0.7383
    Epoch 21/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5372 - accuracy: 0.7382
    Epoch 22/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5363 - accuracy: 0.7381
    Epoch 23/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5363 - accuracy: 0.7379
    Epoch 24/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5358 - accuracy: 0.7383
    Epoch 25/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5359 - accuracy: 0.7382
    Epoch 26/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5360 - accuracy: 0.7390
    Epoch 27/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5362 - accuracy: 0.7383
    Epoch 28/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5357 - accuracy: 0.7379
    Epoch 29/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5358 - accuracy: 0.7381
    Epoch 30/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5355 - accuracy: 0.7390
    Epoch 31/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5359 - accuracy: 0.7386
    Epoch 32/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5354 - accuracy: 0.7372
    Epoch 33/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5375 - accuracy: 0.7388
    Epoch 34/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5351 - accuracy: 0.7389
    Epoch 35/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5351 - accuracy: 0.7399
    Epoch 36/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5354 - accuracy: 0.7378
    Epoch 37/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5350 - accuracy: 0.7386
    Epoch 38/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5351 - accuracy: 0.7397
    Epoch 39/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5349 - accuracy: 0.7398
    Epoch 40/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5351 - accuracy: 0.7397
    Epoch 41/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5351 - accuracy: 0.7371
    Epoch 42/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5352 - accuracy: 0.7397
    Epoch 43/50
    804/804 [==============================] - 2s 3ms/step - loss: 0.5344 - accuracy: 0.7387
    Epoch 44/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5351 - accuracy: 0.7393
    Epoch 45/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5345 - accuracy: 0.7389
    Epoch 46/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5348 - accuracy: 0.7393
    Epoch 47/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5351 - accuracy: 0.7388
    Epoch 48/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5347 - accuracy: 0.7391
    Epoch 49/50
    804/804 [==============================] - 2s 2ms/step - loss: 0.5346 - accuracy: 0.7404
    Epoch 50/50
    804/804 [==============================] - 1s 2ms/step - loss: 0.5341 - accuracy: 0.7392
    

### Step 2: After finishing your models, display the accuracy scores achieved by each model, and compare the results.


```python
print("Original Model Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(x_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

    Original Model Results
    268/268 - 1s - loss: 0.5579 - accuracy: 0.7303 - 1s/epoch - 4ms/step
    Loss: 0.5578923225402832, Accuracy: 0.7302623987197876
    


```python
print("Alternative Model 1 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn_A1.evaluate(x_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

    Alternative Model 1 Results
    268/268 - 0s - loss: 0.5649 - accuracy: 0.7303 - 497ms/epoch - 2ms/step
    Loss: 0.5648573637008667, Accuracy: 0.7302623987197876
    


```python
print("Alternative Model 2 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn_A2.evaluate(x_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

    Alternative Model 2 Results
    268/268 - 1s - loss: 0.7689 - accuracy: 0.5606 - 810ms/epoch - 3ms/step
    Loss: 0.7688503861427307, Accuracy: 0.5605831146240234
    

### Step 3: Save each of your alternative models as an HDF5 file.



```python
# Set the file path for the first alternative model
file_path_A1 = "Venture Funding_A1.5"

# Export your model to a HDF5 file
print(file_path_A1)

```

    Venture Funding_A1.5
    


```python
# Set the file path for the second alternative model
file_path_A2 = "Venture Funding_A2.h5"

# Export your model to a HDF5 file
print(file_path_A2)

```

    Venture Funding_A2.h5
    


```python

```

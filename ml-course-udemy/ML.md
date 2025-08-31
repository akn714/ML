## Machine Learning

#### Machine Learning Process
**Data Pre-Processing**
- Importing data
- Data cleaning
- Split data into train and test (80:20)

**Modelling**
- Build the model
- Train the model
- Make predictions

**Evaluation**
- Calculate performance metrics
- Make a verdict

---

#### Data Preprocessing
- Data Loading
- Data Clearning (filling missing values with avg. of all the values in the column)
- Data Splitting (Training and Test)
- Feature Scaling if needed

**Feature Scaling:**<br>
- Converting all the values in the column to lie in a range (like o to 1, -3 to +3)
- Feature scaling is only applied to columns.
- Feature scaling is always applied after splitting the data
    - use same scaler [StandardScaler(standardization), MinMaxScaler(normalization)] for both data (test set and trainig set) but seperately
    - if feature scaling is used before data splitting then the model may learn the data in the test set even if is not directly used for training
    ```
    - Scaling is done after data splitting to prevent **data leakage**. 
    - If done before splitting, scaled training data would be influenced by unseen test data, affecting model performance.
    ```

---
status: draft
summary: Exploratory analysis, preprocessing, and stroke-prediction modeling for a highly imbalanced tabular dataset.
problem: Predict whether a patient is likely to experience a stroke based on 12 input attributes.
dataset: "[Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), 5,110 observations with 12 attributes."
colab: "https://colab.research.google.com/drive/1rxS_ggmzLzQsWN03aDWqbb8kVlQ20vi1?usp=sharing"
notebook: "https://github.com/tatkien/programming-for-ai/blob/main/tabular_part_assingment.ipynb"
pdf: ""
video: ""
---

## Exploratory Data Analysis

The dataset contains 5,110 records and 12 attributes describing patients with and without stroke. The objective is to build a reliable predictive model for identifying stroke risk under severe class imbalance.

### Dataset overview

**Table: Original dataset attributes**

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `id` | Identifier | Unique patient identifier; removed before modeling. |
| `gender` | Categorical | Male, female, or other. |
| `age` | Numerical | Patient age in years. |
| `hypertension` | Binary | Indicates whether the patient has hypertension. |
| `heart_disease` | Binary | Indicates whether the patient has heart disease. |
| `ever_married` | Categorical | Whether the patient has ever been married. |
| `work_type` | Categorical | Employment category, such as private, government, or self-employed. |
| `Residence_type` | Categorical | Rural or urban residence. |
| `avg_glucose_level` | Numerical | Average blood glucose measurement. |
| `bmi` | Numerical | Body mass index. |
| `smoking_status` | Categorical | Smoking history; `Unknown` denotes unavailable information. |
| `stroke` | Binary target | 1 for stroke and 0 for no stroke. |

### Class imbalance

Only 249 of the 5,110 patients have a positive stroke label, which is approximately 4.87% of the dataset. The remaining 4,861 records belong to the no-stroke class. This imbalance makes accuracy an unreliable metric for model selection, because a classifier that predicts only the majority class would already achieve more than 95% accuracy.

![Stroke prevalence is approximately one positive case in every twenty records](assets/tabular/tabular_class_imbalance.png)
*Figure 1: Stroke prevalence is approximately one positive case in every twenty records.*

### Missing-value analysis

The initial inspection reveals 201 missing values, all in the `bmi` column. Removing these rows would discard about 3.9% of the dataset and could exclude valuable stroke-positive observations. Two imputation strategies are therefore evaluated:

* **KNN imputation:** categorical variables are encoded numerically, and each missing BMI value is estimated from nearby observations. The exploratory comparison uses 15 neighbors.
* **Decision tree regression:** BMI is predicted from age and gender using a Decision Tree Regressor after scaling the input variables.

Both methods produce plausible BMI values without removing records. KNN imputation is retained in the final preprocessing pipeline because it can utilize a broader multivariate neighborhood. To reduce information leakage, the final imputer is fitted only on the training split and uses five neighbors.

### Numerical distributions

The three key continuous variables are age, average glucose level, and BMI. Average glucose level is strongly right-skewed and shows a secondary concentration at high glucose values. BMI is also right-skewed, with most observations clustered around the overweight range. Age spans a broad interval from childhood to 82 years.

![Distributions of age, average glucose level, and BMI](assets/tabular/tabular_numeric_distributions.png)
*Figure 2: Distributions of age, average glucose level, and BMI.*

Comparing stroke and non-stroke groups suggests that age is the clearest individual risk indicator. Stroke-positive records are concentrated among older patients, while glucose and BMI show weaker separation. The notebook’s cumulative analysis also indicates that stroke risk increases with age.

### Dataset overview and pattern inspection

The exploratory plots show that stroke-positive patients tend to be older and more likely to have hypertension or heart disease. Differences associated with gender, residence type, work type, smoking status, glucose, and BMI are present, but they overlap substantially between the two classes.

![Comparison of selected patient characteristics by stroke label](assets/tabular/tabular_stroke_patterns.png)
*Figure 3: Comparison of selected patient characteristics by stroke label.*

Figure 3 compares the distributions of several attributes between patients with and without stroke. Age provides the clearest separation: stroke-positive records are concentrated among older patients, whereas the no-stroke group is spread more broadly across younger and middle-aged populations. Hypertension and heart disease are also more common in the stroke group, supporting their role as cardiovascular risk indicators. Average glucose levels tend to be higher and more widely distributed among stroke-positive patients, although the two groups still overlap substantially. In contrast, BMI, gender, smoking status, and work type show weaker visual separation. Private and self-employed patients account for many stroke cases, but these categories are also common in the overall dataset; therefore, their raw counts should not be interpreted as direct causal effects. Overall, the figure suggests that age and existing cardiovascular conditions provide stronger predictive information than the demographic and lifestyle features considered individually.

### Outlier detection and treatment

Outliers are detected using the interquartile-range rule with fences $Q_1 - 1.5 \times \text{IQR}$ and $Q_3 + 1.5 \times \text{IQR}$. Rows are not removed because the positive class is already scarce. Instead, extreme numerical values are capped to the computed bounds.

**Table: Outliers detected before capping**

| Variable | Lower bound | Upper bound | Outliers | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| Age | -29.00 | 115.00 | 0 | 0.00% |
| Average glucose level | 21.98 | 169.36 | 627 | 12.27% |
| BMI | 9.10 | 47.50 | 110 | 2.24% |

Age remains unchanged because all observed ages fall within the IQR bounds. Average glucose is capped from a maximum of 271.74 to 169.36, while BMI is capped from 97.60 to 47.50. This preserves every patient record and prevents extreme values from dominating distance-based and gradient-based methods.

![Continuous variables before and after IQR-based capping](assets/tabular/tabular_outlier_treatment.png)
*Figure 4: Continuous variables before and after IQR-based capping.*

## Feature Engineering

The patient identifier is removed because it does not contain clinical information. Since the original variables have weak linear relationships with the target, the notebook generates nonlinear transformations, binary indicators, ratios, and interaction features. The main engineered groups are:

* **Age features:** squared age, logarithmic age, and indicators for young, middle-aged, and older patients.
* **Glucose features:** squared and logarithmic glucose, together with high-glucose and very-high-glucose indicators.
* **BMI features:** squared and logarithmic BMI, plus indicators for underweight, overweight, and obesity.
* **Interaction features:** age multiplied by hypertension, heart disease, glucose, BMI, and cardiovascular-risk indicators.
* **Ratio features:** age-to-BMI, age-to-glucose, and BMI-to-glucose ratios.
* **Combined clinical features:** cardiovascular risk and the presence of either hypertension or heart disease.

After feature engineering, each record contains 35 predictors before one-hot encoding. The strongest training-set correlations with stroke are observed for squared age (0.271), age–glucose interaction (0.251), age (0.242), the older-age indicator (0.220), and age–BMI interaction (0.206). These values confirm that age-related transformations hold the clearest predictive signal.

### Data splitting and preprocessing

The data are divided using stratified sampling so that each split preserves the original class ratio:

* Training set: 3,066 records (60%).
* Validation set: 1,022 records (20%).
* Test set: 1,022 records (20%).

Numerical features are processed with a five-neighbor KNN imputer followed by standardization. Categorical features are transformed using one-hot encoding with support for unseen categories. The preprocessing transformer is fitted on the training data and then applied unchanged to the validation and test sets.

### Imbalance handling

BorderlineSMOTE is applied only to the processed training set. Unlike ordinary random oversampling, BorderlineSMOTE focuses on minority examples near the decision boundary, where classification is more difficult. After resampling, the training matrix contains 5,834 observations and 46 encoded features with balanced target classes. Validation and test sets retain their original distributions to provide realistic evaluation.

![Class distributions in the resampled training set and untouched validation and test sets](assets/tabular/tabular_split_distributions.png)
*Figure 5: Class distributions in the resampled training set and untouched validation and test sets.*

### Feature reduction

Two feature-reduction strategies are used depending on the model type:

* Random Forest and XGBoost use the 25 highest-ranked features selected by mutual information.
* Logistic Regression and the MLP use Principal Component Analysis (PCA). PCA reduces the 46 encoded features to 11 components while retaining 95.07% of the variance.

### Model training and hyperparameter tuning

The project evaluates four complementary classifiers. Logistic Regression provides a linear and interpretable baseline. Random Forest combines multiple decision trees via bagging. XGBoost uses gradient-boosted trees, while the MLP learns nonlinear relationships through fully connected neural-network layers.

Randomized search with five-fold stratified cross-validation is used for Logistic Regression, Random Forest, and XGBoost. Hyperparameters are selected using the $F_2$ score:

$$F_2 = \frac{(1+2^2) \times \text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} + \text{Recall}}$$

This metric weights recall four times more heavily than precision, reflecting that a missed high-risk patient is more costly than an additional screening alert.

**Table: Selected model configurations**

| Model | Configuration |
| :--- | :--- |
| Logistic Regression | $C=0.0178$, L2 penalty, `liblinear` solver, balanced class weights; best cross-validation $F_2 = 0.8672$. |
| Random Forest | 370 trees, depth 16, `log2` feature sampling, minimum leaf size 6, minimum split size 8; best cross-validation $F_2 = 0.9646$. |
| XGBoost | 516 trees, depth 6, learning rate 0.1926, subsample 0.7124, column sample 0.8083, and regularization; best cross-validation $F_2 = 0.9655$. |
| MLP | Two hidden layers with 128 and 64 neurons, ReLU activation, Adam optimization, early stopping, and a maximum of 500 iterations. |

### Soft-voting ensemble

The final ensemble combines predicted probabilities from all four classifiers. Each model receives a weight derived from its validation ROC AUC:

$$w_m = \max(AUC_m - 0.5, 0)^2$$

The resulting weights are 0.118 for Logistic Regression, 0.101 for Random Forest, 0.067 for XGBoost, and 0.031 for the MLP. The classification threshold is selected on the validation set to maximize $F_2$, producing a final threshold of 0.22.

## Experimental Results

### Evaluation metrics

The models are evaluated using ROC AUC, precision, recall, $F_1$, $F_2$, and accuracy. ROC AUC measures ranking quality across thresholds, while precision and recall focus on the positive stroke class. Because the dataset is highly imbalanced, recall and $F_2$ are especially important. Accuracy is reported for completeness, but it is not used as the primary indicator of clinical usefulness.

#### Results at the default threshold

Table 1 reports performance at the default probability threshold of 0.5. Logistic Regression achieves the highest test ROC AUC and the highest minority-class recall. XGBoost reaches the highest accuracy, but its recall of 0.12 shows that it misses most stroke-positive patients.

**Table 1: Test performance at the default threshold of 0.5**

| Model | ROC AUC | Precision | Recall | $F_1$ | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8272 | 0.16 | 0.78 | 0.26 | 0.7828 |
| Random Forest | 0.8133 | 0.29 | 0.18 | 0.22 | 0.9384 |
| XGBoost | 0.7666 | 0.27 | 0.12 | 0.17 | 0.9413 |
| MLP Classifier | 0.7133 | 0.13 | 0.12 | 0.12 | 0.9178 |

The results illustrate why overall accuracy can be misleading. Random Forest, XGBoost, and the MLP achieve accuracy above 0.91 primarily by predicting the dominant no-stroke class. Logistic Regression accepts lower accuracy in exchange for identifying 78% of the stroke cases.

#### Threshold optimization

The default threshold is not necessarily appropriate for an imbalanced medical-screening problem. Therefore, each model’s threshold is selected on the validation set to maximize $F_2$. The tuned test results are shown in Table 2.

**Table 2: Test results after validation-based threshold optimization**

| Model | Threshold | ROC AUC | PR AUC | Precision | Recall | $F_1$ | $F_2$ | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.617 | 0.8272 | 0.2308 | 0.1875 | 0.6600 | 0.2920 | 0.4388 | 0.8434 |
| Random Forest | 0.140 | 0.8133 | 0.1926 | 0.1259 | 0.7400 | 0.2151 | 0.3745 | 0.7358 |
| XGBoost | 0.010 | 0.7666 | 0.1641 | 0.1090 | 0.6800 | 0.1878 | 0.3320 | 0.7123 |
| MLP Classifier | 0.014 | 0.7133 | 0.1220 | 0.1207 | 0.4200 | 0.1875 | 0.2807 | 0.8219 |

Logistic Regression remains the best threshold-tuned individual model with $F_2 = 0.4388$. Random Forest and XGBoost require very low thresholds to recover more positive cases, which substantially reduces precision and accuracy. These results suggest that their probability outputs are not well aligned with the rare positive class, even though their majority-class accuracy is high.

![Confusion matrices using validation-tuned thresholds](assets/tabular/tabular_confusion_matrices.png)
*Figure 6: Confusion matrices using validation-tuned thresholds.*

#### Ensemble performance

The AUC-weighted soft-voting ensemble is evaluated at its validation-selected threshold of 0.22. Its test results are presented in Table 3.

**Table 3: AUC-weighted soft-voting ensemble performance**

| Metric | Score |
| :--- | :---: |
| Threshold | 0.22 |
| ROC AUC | 0.8234 |
| PR AUC | 0.2042 |
| Precision | 0.1404 |
| Recall | 0.8000 |
| $F_1$ | 0.2388 |
| $F_2$ | 0.4124 |
| Accuracy | 0.7505 |

The ensemble has slightly lower ROC AUC than Logistic Regression but reaches the highest recall among the final threshold-selected systems. It identifies 80% of stroke-positive patients, although precision falls to 14.04%. This operating point is suitable only when the system is treated as an initial screening tool and false-positive predictions can be followed by professional clinical assessment.

## Limitations

* The dataset contains only 249 positive cases, limiting the diversity of stroke patterns available for learning.
* BorderlineSMOTE creates synthetic samples but cannot introduce new clinical information.
* IQR capping is performed before the final split in the notebook; a production pipeline should estimate all clipping bounds exclusively from training data.
* Several clinically important variables are unavailable, including blood pressure, cholesterol, medication, family history, and prior transient ischemic attacks.
* The models are evaluated on a single public dataset and have not been externally validated on another population.
* Probability calibration is not assessed, so predicted probabilities should not be interpreted directly as clinical risk percentages.

## Conclusion

This project develops an end-to-end tabular machine-learning workflow for stroke prediction. The workflow addresses missing BMI values, skewed numerical variables, outliers, mixed feature types, weak raw correlations, and severe class imbalance. Feature engineering and imbalance-aware evaluation improve the model’s ability to detect the rare stroke class.

Logistic Regression is the strongest individual model, achieving a test ROC AUC of 0.8272 and the best threshold-tuned $F_2$ score of 0.4388. The AUC-weighted ensemble provides the highest final recall of 0.8000 but has low precision. The results show that stroke prediction in this dataset is primarily a ranking and screening problem rather than a conventional accuracy-maximization task. The system may be useful as a research prototype, but further validation and clinically richer data are required before deployment.

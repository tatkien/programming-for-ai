---
status: ready
summary: Exploratory analysis, preprocessing, and stroke-prediction modeling for a highly imbalanced tabular dataset.
problem: Predict whether a patient is likely to experience a stroke based on 12 input attributes.
dataset: "[Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), 5,110 observations with 12 attributes."
colab: "https://colab.research.google.com/drive/1rxS_ggmzLzQsWN03aDWqbb8kVlQ20vi1?usp=sharing"
notebook: "https://github.com/tatkien/programming-for-ai/blob/main/tabular_part_assingment.ipynb"
pdf: "https://github.com/tatkien/programming-for-ai/blob/main/docs/assets/tabular/tabular_report.pdf"
video: ""
---

## Tabular Stroke Prediction: Frontpage Summary

This notebook builds and evaluates a full machine-learning pipeline to predict stroke risk from a highly imbalanced clinical tabular dataset (5,110 patients, 12 original features, 4.87% positive class).

The emphasis is **early risk screening** (high recall) rather than raw accuracy.

## What the notebook does (pipeline at a glance)

1. **Data audit and cleanup**
	- Removes non-informative identifier (`id`).
	- Handles missing `bmi` values with KNN imputation.
	- Caps extreme values for glucose and BMI using an IQR-based rule (without dropping rows).

2. **Feature construction**
	- Engineers nonlinear, ratio, and interaction features from age, glucose, BMI, and cardiovascular indicators.
	- Expands signal from weak raw features into richer predictors.

3. **Split and preprocessing**
	- Stratified split: 60% train / 20% validation / 20% test.
	- Numeric pipeline: imputation + scaling.
	- Categorical pipeline: one-hot encoding.

4. **Imbalance handling**
	- Applies BorderlineSMOTE on the training split only to improve minority-class learning near the decision boundary.

5. **Modeling and tuning**
	- Trains four models: Logistic Regression, Random Forest, XGBoost, and MLP.
	- Uses randomized cross-validated hyperparameter search.
	- Performs validation-based threshold tuning for imbalanced classification.

6. **Ensembling**
	- Builds an AUC-weighted soft-voting ensemble from all four models.
	- Selects the final decision threshold on the validation set.

### Visual highlights from the notebook

![Stroke class imbalance in the full dataset](assets/tabular/tabular_class_imbalance.png)
*Figure 1: Stroke prevalence is low (about 4.87%), motivating recall-focused modeling.*

![Feature distributions for age, glucose, and BMI](assets/tabular/tabular_numeric_distributions.png)
*Figure 2: Core numeric features are skewed and require careful preprocessing.*

![Outlier treatment before and after IQR capping](assets/tabular/tabular_outlier_treatment.png)
*Figure 3: IQR-based capping reduces extreme-value impact while preserving records.*

![Class distribution after split and BorderlineSMOTE on training data](assets/tabular/tabular_split_distributions.png)
*Figure 4: Only the training split is resampled; validation/test remain untouched for realistic evaluation.*

## Headline results

### Results summary table

| System | Role | Threshold | ROC AUC | Recall | Precision | F2 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Logistic Regression | Best individual model | 0.617 | 0.8272 | 0.6600 | 0.1875 | 0.4388 | Best tuned F2 among individual models; strongest single-model balance for screening. |
| AUC-weighted soft-voting ensemble | Best recall-oriented system | 0.22 | 0.8234 | 0.8000 | 0.1404 | 0.4124 | Highest recall; useful when missing positives is costlier than false alarms. |

### Key interpretation
- In this dataset, very high accuracy can be misleading because of class imbalance.
- The most useful systems are those that preserve strong ranking quality (AUC) while detecting more stroke-positive cases (high recall), even at lower precision.

![Confusion matrices for threshold-tuned models](assets/tabular/tabular_confusion_matrices.png)
*Figure 5: Threshold tuning shifts models toward better positive-case detection at the cost of precision.*

## Practical takeaway

This notebook demonstrates a complete, imbalance-aware tabular ML workflow that is suitable for **screening-oriented risk prediction**. The output is best used as a **first-pass triage signal**, not a standalone clinical diagnosis.

## Limitations

- Only 249 positive cases are available.
- External validation on independent populations is not included.
- Probability calibration is not fully assessed.
- Additional clinical variables would likely improve performance and reliability.

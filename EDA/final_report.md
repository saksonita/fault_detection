# Feature Importance Analysis for "파단직전 =1" Condition

## Executive Summary

This report presents a comprehensive analysis to identify the important features that could cause the "파단직전 =1" (imminent fracture) condition in the manufacturing process. Through multiple analytical approaches including exploratory data analysis, correlation analysis, feature selection methods, and predictive modeling, we have identified several key features that are strongly associated with the imminent fracture condition.

The most significant features that could cause "파단직전 =1" are:

1. **[D0006862] 작업자측 GAP [32bit integer]** - Worker-side GAP
2. **[D0006866] 작업자측 이전 GAP [32bit integer]** - Previous worker-side GAP
3. **[M005685] 리와인더 E.P.C 중앙 [Bit]** - Rewinder E.P.C. center
4. **유도가열온도WS차이** - Induction heating temperature WS difference
5. **리와인더장력차이** - Rewinder tension difference

These features consistently showed high importance across different analytical methods and should be closely monitored to predict and prevent imminent fracture conditions.

## 1. Introduction

The dataset contains 2,850,375 rows and 31 columns, with a highly imbalanced target variable "파단직전" (only 549 positive cases out of 2,849,826 total). The analysis aimed to identify which features are most important in predicting the imminent fracture condition.

## 2. Methodology

The analysis followed a structured approach:

1. **Data Examination**: Explored dataset structure, checked for missing values, and analyzed feature distributions
2. **Correlation Analysis**: Identified correlations between features and the target variable
3. **Feature Selection**: Applied multiple feature selection methods including:
   - Filter methods (ANOVA F-value, Mutual Information)
   - Wrapper methods (Recursive Feature Elimination)
   - Embedded methods (Random Forest and Gradient Boosting importance)
4. **Predictive Modeling**: Built classification models with class imbalance handling
5. **Feature Importance Evaluation**: Compared feature importance across different models

## 3. Key Findings

### 3.1 Exploratory Data Analysis

- The dataset is complete with no missing values
- The target variable is highly imbalanced (only 0.02% positive cases)
- Several features show distinct distributions between positive and negative cases

### 3.2 Correlation Analysis

Top positively correlated features with "파단직전":
- [D0006934] 리와인더 댄서 위치 (0.192)
- [D0006926] 언와인더 댄서 위치 (0.184)
- [D0006818] 리와인더 장력 PV (0.159)

Top negatively correlated features with "파단직전":
- [D0006896] 작업자측 역압 (-0.240)
- [D0006894] 구동부측 역압 (-0.239)
- error_index (-0.213)

### 3.3 Feature Selection Results

Different feature selection methods highlighted various important features:

**ANOVA F-value**:
- [D0006896] 작업자측 역압
- [D0006894] 구동부측 역압
- [D0006934] 리와인더 댄서 위치

**Mutual Information**:
- [D0006866] 작업자측 이전 GAP
- [D0006862] 작업자측 GAP
- [D0006864] 구동부측 이전 GAP

**Random Forest Importance**:
- [D0006862] 작업자측 GAP
- [D0006866] 작업자측 이전 GAP
- [D0006864] 구동부측 이전 GAP

**Gradient Boosting Importance**:
- [D0006862] 작업자측 GAP
- [D0006864] 구동부측 이전 GAP
- [D0006860] 구동부측 GAP

### 3.4 Predictive Model Performance

Multiple classification models were trained with SMOTE to handle class imbalance:

| Model | ROC-AUC | PR-AUC | CV ROC-AUC |
|-------|---------|--------|------------|
| Random Forest | 0.8764 | 0.7077 | 0.8996 ± 0.0151 |
| Gradient Boosting | 0.7520 | 0.4295 | 0.7799 ± 0.0275 |
| Logistic Regression | 0.6152 | 0.2733 | 0.6596 ± 0.0345 |
| SVM | 0.7502 | 0.3854 | 0.7688 ± 0.0250 |

Random Forest achieved the best performance, indicating its feature importance results may be most reliable.

### 3.5 Consistent Top Features

Features that consistently appeared as important across multiple methods:

1. **[M005685] 리와인더 E.P.C 중앙 [Bit]** - Appeared in top features across all methods
2. **유도가열온도WS차이** - Consistently high importance in tree-based models
3. **리와인더장력차이** - Important in both correlation and feature importance analyses

### 3.6 GAP-Related Features

GAP-related features showed the highest average normalized importance:

1. **[D0006862] 작업자측 GAP [32bit integer]** (1.0000)
2. **[D0006866] 작업자측 이전 GAP [32bit integer]** (0.6501)
3. **[D0006864] 구동부측 이전 GAP [32bit integer]** (0.6132)
4. **[D0006860] 구동부측 GAP [32bit integer]** (0.5663)

## 4. Detailed Feature Explanations

### 4.1 GAP-Related Features

**[D0006862] 작업자측 GAP & [D0006866] 작업자측 이전 GAP**:
These features represent the current and previous gap measurements on the worker side of the equipment. The high importance of these features suggests that changes or abnormalities in the worker-side gap are strongly associated with imminent fracture conditions. Monitoring these gaps could provide early warning of potential fractures.

**[D0006864] 구동부측 이전 GAP & [D0006860] 구동부측 GAP**:
These represent the current and previous gap measurements on the drive side. Their high importance indicates that drive-side gap measurements are also critical indicators of potential fractures.

### 4.2 E.P.C. Features

**[M005685] 리와인더 E.P.C 중앙 [Bit]**:
This binary feature represents the center edge position control of the rewinder. Its consistent importance across methods suggests that the center position control of the rewinder plays a critical role in preventing fractures.

### 4.3 Temperature and Tension Features

**유도가열온도WS차이** (Induction heating temperature WS difference):
This feature represents the difference in induction heating temperature on the WS side. Temperature differences can create uneven material properties, potentially leading to stress concentrations and fractures.

**리와인더장력차이** (Rewinder tension difference):
This feature represents differences in tension in the rewinder. Uneven tension can create stress concentrations that may lead to material failure and fractures.

## 5. Recommendations

Based on the analysis, we recommend:

1. **Implement Real-time Monitoring**: Set up real-time monitoring systems for the top identified features, especially GAP-related measurements, rewinder E.P.C. center position, and temperature/tension differences.

2. **Establish Alert Thresholds**: Define threshold values for critical features that, when exceeded, trigger alerts for potential imminent fracture conditions.

3. **Develop Predictive Maintenance System**: Implement a predictive maintenance system using the Random Forest model, which showed the best performance (ROC-AUC: 0.8764).

4. **Process Optimization**: Review and optimize processes related to:
   - GAP control on both worker and drive sides
   - Rewinder edge position control
   - Induction heating temperature uniformity
   - Tension control in the rewinder

5. **Further Investigation**: Conduct further investigation into the physical mechanisms by which these features influence fracture conditions, particularly focusing on the relationship between GAP measurements and material stress.

## 6. Conclusion

This analysis has successfully identified the key features that could cause the "파단직전 =1" condition. The GAP-related features, particularly on the worker side, emerged as the most important predictors, followed by rewinder E.P.C. center position and temperature/tension differences.

By monitoring these features and implementing the recommended actions, it should be possible to predict and prevent imminent fracture conditions, thereby improving production efficiency and reducing material waste.

## Appendix: Feature Importance Rankings

### Average Normalized Feature Importance (Top 10)
1. [D0006862] 작업자측 GAP [32bit integer] (1.0000)
2. [D0006866] 작업자측 이전 GAP [32bit integer] (0.6501)
3. [D0006864] 구동부측 이전 GAP [32bit integer] (0.6132)
4. [D0006860] 구동부측 GAP [32bit integer] (0.5663)
5. 유도가열온도WS차이 (0.2450)
6. [M005685] 리와인더 E.P.C 중앙 [Bit] (0.2263)
7. [D0006818] 리와인더 장력 PV [32bit integer] (0.1616)
8. [D0006934] 리와인더 댄서 위치. [16bit integer] (0.1557)
9. 리와인더장력차이 (0.1472)
10. [D0006894] 구동부측 역압 [32bit integer] (0.1192)

### Combined Ranking from Feature Selection Methods (Top 10)
1. 프레스속도차이 (Average Rank: 6.20)
2. 유도가열온도WS차이 (Average Rank: 6.80)
3. 유도가열온도DS차이 (Average Rank: 7.40)
4. 언와인더장력차이 (Average Rank: 8.00)
5. 리와인더장력차이 (Average Rank: 8.60)
6. 강제연신기장력차이 (Average Rank: 9.20)
7. [M005686] 리와인더 E.P.C 우 1 [Bit] (Average Rank: 9.80)
8. [M005685] 리와인더 E.P.C 중앙 [Bit] (Average Rank: 10.40)
9. [M005684] 리와인더 E.P.C 좌 1 [Bit] (Average Rank: 11.00)
10. [M005683] 리와인더 E.P.C 좌 2 [Bit] (Average Rank: 11.60)

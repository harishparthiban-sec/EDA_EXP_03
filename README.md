**EXP 3 - Delhi Air Quality Analysis**

**Aim**


To compare air quality parameters in Delhi across different stations and analyze the relationship between pollutants (e.g., PM2.5 and NO₂) using scatter plots and correlation analysis.


**Procedure / Algorithm**

1)Load the dataset using pandas.

2)Preprocess the data:

3)Convert the date column (period.datetimeFrom.utc) to datetime format.

4)Drop missing or invalid values.

5)Pivot the dataset so each pollutant (parameter) becomes a separate column.

6)Plot scatter plot between PM2.5 and NO₂ to study their relationship.

7)Plot correlation heatmap between all pollutants to identify relationships.

8)Interpret the results — identify which pollutants are correlated and which stations are most polluted.


**Program**

**Name : Harish P**

**Reg No: 212225040115**

#Write your code here
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


df = pd.read_csv("del-sirifort-cpcb-2024-25.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

columns = [
    'PM2.5 (µg/m³)',
    'PM10 (µg/m³)',
    'NO2 (µg/m³)',
    'Ozone (µg/m³)'
]

data = df[columns].copy()

for col in columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

df = data.copy()

print("\nAvailable Records After Preprocessing:")
for col in columns:
    print(col, ":", df[col].count())

print("\nDataset Information:")
print(df.info())

print("\nPreprocessed Dataset:")
print(df.head())

plt.figure(figsize=(14, 10))

for i, col in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    sns.histplot(
        df[col].dropna(),
        bins=30,
        kde=True
    )

    plt.title("Distribution of " + col)
    plt.xlabel("Concentration (µg/m³)")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

summary = pd.DataFrame(index=columns)

for col in columns:

    x = df[col].dropna()

    summary.loc[col, 'Mean'] = x.mean()

    summary.loc[col, 'Median'] = x.median()

    summary.loc[col, 'Mode'] = x.mode().iloc[0]

    summary.loc[col, 'Minimum'] = x.min()

    summary.loc[col, 'Q1'] = x.quantile(0.25)

    summary.loc[col, 'Q3'] = x.quantile(0.75)

    summary.loc[col, 'Maximum'] = x.max()

print("NUMERICAL SUMMARY")
print(summary.round(2))
spread = pd.DataFrame(index=columns)

for col in columns:

    x = df[col].dropna()

    mean = x.mean()

    spread.loc[col, 'Range'] = x.max() - x.min()

    spread.loc[col, 'Variance'] = x.var()

    spread.loc[col, 'Standard Deviation'] = x.std()

    spread.loc[col, 'Coefficient of Variation (%)'] = (
        (x.std() / mean) * 100 if mean != 0 else np.nan
    )

print("MEASURES OF SPREAD")
print(spread.round(2))
plt.figure(figsize=(12, 7))

sns.boxplot(data=df)

plt.title("Box Plot of Delhi Air Quality Pollutants")
plt.xlabel("Air Pollutants")
plt.ylabel("Concentration (µg/m³)")

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
for col in columns:

    x = df[col].dropna()

    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)

    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    up = Q3 + 1.5 * IQR

    outliers = x[
        (x < low) | (x > up)
    ]

    print("\nPollutant:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", low)
    print("Upper Limit:", up)
    print("Number of Outliers:", len(outliers))

    print("First 10 Outliers:")
    print(outliers.head(10).values)
minmax_data = pd.DataFrame(index=df.index)

for col in columns:

    x = df[col]

    min_value = x.min()
    max_value = x.max()

    if max_value != min_value:
        minmax_data[col] = (
            (x - min_value) / (max_value - min_value)
        )
    else:
        minmax_data[col] = 0.0

print("MIN-MAX SCALED DATA")
print(minmax_data.head(10))

print("\nScaled Data Range:")
print(minmax_data.agg(['min', 'max']).round(4))

minmax_data.to_csv("minmax_scaled_data.csv", index=False)
standardized_data = pd.DataFrame(index=df.index)

for col in columns:

    valid = df[col].notna()

    scaler = StandardScaler()

    standardized_data.loc[valid, col] = (
        scaler.fit_transform(
            df.loc[valid, [col]]
        ).flatten()
    )

print("Z-SCORE STANDARDIZED DATA")
print(standardized_data.head(10))

print("\nMean After Standardization:")
print(standardized_data.mean().round(4))

print("\nStandard Deviation After Standardization:")
print(standardized_data.std(ddof=0).round(4))

standardized_data.to_csv(
    "zscore_standardized_data.csv",
    index=False
)

def gini_coefficient(values):

    x = np.array(values, dtype=float)

    x = x[~np.isnan(x)]

    if np.any(x < 0):
        raise ValueError("Gini coefficient requires non-negative values")

    if len(x) == 0:
        return np.nan

    if np.sum(x) == 0:
        return 0

    x = np.sort(x)

    n = len(x)

    index = np.arange(1, n + 1)

    gini = (
        2 * np.sum(index * x) / (n * np.sum(x))
    ) - ((n + 1) / n)

    return gini


print("GINI COEFFICIENT OF AIR POLLUTANTS")

for col in columns:

    gini = gini_coefficient(df[col])

    print(col, ":", round(gini, 4))
summary.to_csv("numerical_summary.csv")

spread.to_csv("measures_of_spread.csv")

print("All analysis results saved successfully!")
```
**Output**
<img width="841" height="558" alt="image" src="https://github.com/user-attachments/assets/02da5ddf-2b6f-4f01-9789-57d0addc8214" />
<img width="822" height="595" alt="image" src="https://github.com/user-attachments/assets/6f61ddd3-4cf2-4452-bd2c-6d88bfb205d2" />
<img width="1012" height="718" alt="image" src="https://github.com/user-attachments/assets/1b97ec8a-3f1c-412c-8b54-46225afaa17a" />
<img width="812" height="131" alt="image" src="https://github.com/user-attachments/assets/410b4cd8-84ba-44f6-b4c6-3c2015c6852e" />
<img width="796" height="227" alt="image" src="https://github.com/user-attachments/assets/a7a31f17-44d9-4c90-9817-c9b26153a93f" />
<img width="997" height="588" alt="image" src="https://github.com/user-attachments/assets/b0a06860-ee52-40d2-8276-3e56c9c8891c" />
<img width="776" height="702" alt="image" src="https://github.com/user-attachments/assets/3fb363ff-30ab-44ae-83d0-5b7da7f50cb5" />
<img width="588" height="305" alt="image" src="https://github.com/user-attachments/assets/6189ef60-cbef-4d4c-a68a-d49f337e8ef2" />
<img width="667" height="457" alt="image" src="https://github.com/user-attachments/assets/66cbac41-62dd-4466-9568-68e699580066" />
<img width="731" height="117" alt="image" src="https://github.com/user-attachments/assets/eff3929b-a8c8-488b-ae36-f8aa12078025" />

**Interpretation**

1)PM2.5 and NO₂ show a strong positive correlation, suggesting that both pollutants increase together, likely due to vehicle and industrial emissions.

**Result**

The dataset was successfully loaded and processed to extract pollutant-wise and station-wise air quality data for Delhi.



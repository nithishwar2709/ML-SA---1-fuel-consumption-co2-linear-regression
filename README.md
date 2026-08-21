# ML-SA---1-fuel-consumption-co2-linear-regression

## AIM:

To write a program to implement **Linear Regression** for analyzing and predicting CO2 emissions based on vehicle parameters such as **Cylinders, Engine Size, and Combined Fuel Consumption**.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook / Google Colab

## Algorithm

1. Import the required libraries and read the `FuelConsumption.csv` dataset using Pandas.
2. Display the dataset and identify the required columns for analysis.
3. Create scatter plots to compare Cylinders, Engine Size, and Combined Fuel Consumption with CO2 Emissions.
4. Split the dataset into training and testing data and apply Linear Regression using Cylinders as the independent variable and CO2 Emissions as the dependent variable.
5. Train another Linear Regression model using Combined Fuel Consumption as the independent variable and CO2 Emissions as the dependent variable.
6. Calculate the **R² score** for both models and compare their accuracies using different train-test ratios.

## Program:

```python
# Program to implement Linear Regression for analyzing and predicting CO2 Emissions.
# Developed by: NITHISHWAR P
# RegisterNumber: 212224060178

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("/content/drive/MyDrive/FuelConsumption.csv")

print(df.head())
print(df.columns)


# Q1

plt.figure(figsize=(8, 5))

plt.scatter(
    df['CYLINDERS'],
    df['CO2EMISSIONS'],
    color='green'
)

plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders vs CO2 Emissions')
plt.grid(True)
plt.show()


# Q2

plt.figure(figsize=(8, 5))

plt.scatter(
    df['CYLINDERS'],
    df['CO2EMISSIONS'],
    color='green',
    label='Cylinders'
)

plt.scatter(
    df['ENGINESIZE'],
    df['CO2EMISSIONS'],
    color='blue',
    label='Engine Size'
)

plt.xlabel('Cylinders / Engine Size')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders and Engine Size vs CO2 Emissions')
plt.legend()
plt.grid(True)
plt.show()


# Q3

plt.figure(figsize=(9, 6))

plt.scatter(
    df['CYLINDERS'],
    df['CO2EMISSIONS'],
    color='green',
    label='Cylinders'
)

plt.scatter(
    df['ENGINESIZE'],
    df['CO2EMISSIONS'],
    color='blue',
    label='Engine Size'
)

plt.scatter(
    df['FUELCONSUMPTION_COMB'],
    df['CO2EMISSIONS'],
    color='red',
    label='Fuel Consumption'
)

plt.xlabel('Independent Variables')
plt.ylabel('CO2 Emissions')
plt.title('Comparison of Variables vs CO2 Emissions')
plt.legend()
plt.grid(True)
plt.show()


# Q4

X = df[['CYLINDERS']]
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_cylinder = LinearRegression()

model_cylinder.fit(X_train, y_train)

y_pred = model_cylinder.predict(X_test)

accuracy_cylinder = r2_score(y_test, y_pred)

print("\nQ4: Cylinder vs CO2 Emissions")
print("Coefficient:", model_cylinder.coef_[0])
print("Intercept:", model_cylinder.intercept_)
print("Accuracy (R2):", accuracy_cylinder)


# Q5

X = df[['FUELCONSUMPTION_COMB']]
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_fuel = LinearRegression()

model_fuel.fit(X_train, y_train)

y_pred = model_fuel.predict(X_test)

accuracy_fuel = r2_score(y_test, y_pred)

print("\nQ5: Fuel Consumption vs CO2 Emissions")
print("Coefficient:", model_fuel.coef_[0])
print("Intercept:", model_fuel.intercept_)
print("Accuracy (R2):", accuracy_fuel)


# Q6

ratios = [0.1, 0.2, 0.3, 0.4]

cylinder_accuracy = []
fuel_accuracy = []

for ratio in ratios:

    X = df[['CYLINDERS']]
    y = df['CO2EMISSIONS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ratio,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cylinder_accuracy.append(
        r2_score(y_test, y_pred)
    )

    X = df[['FUELCONSUMPTION_COMB']]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ratio,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fuel_accuracy.append(
        r2_score(y_test, y_pred)
    )


results = pd.DataFrame({
    'Test Size': ratios,
    'Train Size': [1 - r for r in ratios],
    'Cylinder Accuracy': cylinder_accuracy,
    'Fuel Consumption Accuracy': fuel_accuracy
})

print("\nQ6: Accuracy for Different Train-Test Ratios")

print(results)


results['Cylinder Accuracy (%)'] = (
    results['Cylinder Accuracy'] * 100
)

results['Fuel Consumption Accuracy (%)'] = (
    results['Fuel Consumption Accuracy'] * 100
)

print("\nAccuracy in Percentage")

print(
    results[
        [
            'Test Size',
            'Train Size',
            'Cylinder Accuracy (%)',
            'Fuel Consumption Accuracy (%)'
        ]
    ]
)


plt.figure(figsize=(9, 5))

x = np.arange(len(ratios))

width = 0.35

plt.bar(
    x - width / 2,
    results['Cylinder Accuracy (%)'],
    width,
    label='Cylinder'
)

plt.bar(
    x + width / 2,
    results['Fuel Consumption Accuracy (%)'],
    width,
    label='Fuel Consumption'
)

plt.xlabel('Train-Test Ratio')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')

plt.xticks(
    x,
    ['90-10', '80-20', '70-30', '60-40']
)

plt.legend()
plt.grid(axis='y')

plt.show()
```

## OUTPUT:

<img width="741" height="570" alt="image" src="https://github.com/user-attachments/assets/92e8ce1a-c51d-4f18-a3d9-6a896b33dfea" />

### Q1:

<img width="840" height="520" alt="image" src="https://github.com/user-attachments/assets/592c57f5-e9d7-4391-823c-730ef94f1bad" />


### Q2:

<img width="897" height="539" alt="image" src="https://github.com/user-attachments/assets/0eefba2b-8112-4266-b65b-0f0046d3968a" />


### Q3:

<img width="1001" height="629" alt="image" src="https://github.com/user-attachments/assets/3e07425d-960b-458d-8cac-441c23fa6af9" />


### Q4:


<img width="341" height="110" alt="image" src="https://github.com/user-attachments/assets/de8fcb84-2524-4514-a8bd-f5a6dc938ed0" />

### Q5:


<img width="327" height="114" alt="image" src="https://github.com/user-attachments/assets/2a709426-1933-4dd5-96f6-eb38afdebeb5" />

### Q6:


<img width="556" height="158" alt="image" src="https://github.com/user-attachments/assets/6948ecbf-8923-4301-ac2a-2eadd0bfd5cb" />

### Accuracy:


<img width="924" height="677" alt="image" src="https://github.com/user-attachments/assets/80ad4de1-893c-42d9-aaa4-2f799225fc22" />



## Result:

Thus, the program to implement Linear Regression for analyzing and predicting CO2 emissions using Cylinders and Combined Fuel Consumption
was written and verified successfully using Python programming. The models were also evaluated using different train-test ratios and their
R² accuracies were compared.




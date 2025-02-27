## Scatter plot

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("y_test (Testing Values)")
plt.ylabel("y_pred (Predicted Values)")
plt.title("y_pred vs y_test")
plt.show()

## Histograms 

import pandas as pd
import matplotlib.pyplot as plt

# Filter out missing fare values
fare_data = df['Fare'].dropna()

# Create a histogram
plt.hist(fare_data, bins=40)  # 40 bins - the data range will divide into 40 parts

# Set the labels and title
plt.xlabel('Fare', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Histogram of Fare in Titanic Dataset', fontsize=18)

# Increase the size of tick labels
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Display the chart
plt.show()

## Bar charts

# Calculate the count of passengers for each gender
gender_counts = df['Sex'].value_counts()

# Create a bar chart with larger font size
plt.bar(gender_counts.index, gender_counts.values)
plt.xlabel('Gender', fontsize=18)
plt.ylabel('Count', fontsize=18)

# Set the title with larger font size
plt.title('Gender Distribution', fontsize=18)

# Increase the size of tick labels
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Display the chart
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Count the number of survivors and non-survivors by gender
survival_counts = df.groupby(['Sex', 'Survived']).size().unstack()

# Create a bar chart
ax = survival_counts.plot(kind='bar', stacked=True)

# Set the labels and title using the plot object
ax.set_xlabel('Gender', fontsize=20)
ax.set_ylabel('Count', fontsize=20)
ax.set_title('Survival Outcome by Gender', fontsize=20)

# Set the font size of the tick labels
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)


## Pie Chart 

# Calculate the number of passengers from each embarked port
port_counts = df['Embarked'].value_counts()

# Create a pie chart
plt.pie(port_counts, labels=port_counts.index, autopct='%1.1f%%', textprops={'fontsize': 20})

# Set the title
plt.title('Embarked Port Distribution', fontsize=24)

# Display the chart
plt.show()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 20:54:07 2023
@author: Sharanya
"""

#----------Importing required libraries -------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis


def read_world_bank_data(filePath):
    """
      Alter the columns labeled "country" and "date" after reading
      World Bank data from a CSV file.

       Parameters:
       - filePath: the path of the CSV file holding data from the World Bank.

       Returns:
       - df (pd.DataFrame): The original DataFrame read from the CSV file.
       - transposed (pd.DataFrame): The transposed DataFrame with 'country'
        and 'date' columns interchanged.
    """
    yearDF = pd.read_csv(filePath)
    countryDf = yearDF.copy()
    countryDf[['Country Name' , 'Time']] = countryDf[['Time' , 'Country Name']]
    countryDf = countryDf.rename(columns ={'Country Name': 'Time' ,
                                          'Time': 'Country Name'})


    return countryDf , yearDF


def histogram_plot(data , kurtosis_value):
    # Create a histogram
    sns.histplot(data , kde = True)
    plt.title(f'Distribution with Kurtosis {kurtosis_value:.2f} '
              f'(Renewable energy consumption)' , fontsize = 18)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


def BarGraph(data , columnName):
    specific_data_India = data[data['Country Name'] == 'India']
    specific_data_Srilanka = data[data['Country Name'] == 'Sri Lanka']
    specific_data_Bangladesh = data[data['Country Name'] == 'Bangladesh']
    bar_width = 0.2
    positions = np.arange(len(specific_data_India['Time']))
    position_variable1 = positions - bar_width
    position_variable2 = positions
    position_variable3 = positions + bar_width
    plt.bar(position_variable1 ,
            specific_data_India[columnName] , width = bar_width ,
            label = 'India' , color ='green')
    plt.bar(position_variable2 ,
            specific_data_Srilanka[columnName] ,
            width = bar_width ,
            label = 'Srilanka' , color = 'red')
    plt.bar(position_variable3 ,
            specific_data_Bangladesh[columnName] ,
            width = bar_width ,
            label = 'Bangladesh' , color = 'blue')
    # Adding labels and title
    plt.xlabel('Year')
    plt.xticks(positions , specific_data_India['Time'])
    plt.ylabel('Percentage')
    plt.title(columnName , fontsize =18)
    # Adding legend
    plt.legend()
    # Display a grid
    plt.grid(True)
    # Show the plot
    plt.tight_layout()
    plt.show()


def lineGraph(data):
    for country in ['Pakistan' , 'Bangladesh' , 'Nepal' , 'India' , 'Maldives']:
        country_data = data_Year[data_Year['Country Name'] == country]
        # print(country_data['Total greenhouse gas emissions '])
        plt.plot(country_data['Time'] , country_data['Total greenhouse gas emissions '] ,
                 label = country)

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Total greenhouse gas emissions')
    plt.title('Total greenhouse gas emissions Over the Years' , fontsize = 18)

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)
    plt.show()


def pieChart(data):
    # pie chart
    piedata = data[(data['Time'] == 2020)]
    countrynames = []
    sizes = []
    for country in ['Pakistan', 'Bangladesh' , 'Nepal' , 'India' , 'Maldives']:
        countrynames.append(country)
        country_data = piedata[piedata['Country Name'] == country]
        sizes.append(country_data['Agricultural land (% of land area)'].values[0])
        # print(country_data['Agricultural land (% of land area)'].values)

    # Create a pie chart
    plt.pie(sizes , labels = countrynames , autopct = '%1.1f%%' , startangle = 90)

    # Customize the plot as needed
    plt.title('Agricultural land area of different countries' , fontsize = 18)

    # Show the plot
    plt.show()


def heatmap(correlation_matrix):
    if not correlation_matrix.empty:
        # Create a heatmap for the correlation matrix
        plt.figure(figsize = (12 , 8))
        heatmap = sns.heatmap(correlation_matrix , annot = True , cmap = 'coolwarm' ,
                              fmt = ".2f" , linewidths = .5)
        plt.xlabel('Indicators')
        plt.ylabel('Indicators')
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Correlation Coefficient')
        plt.title('Correlation Matrix for Selected World Bank Indicators' ,
                  fontsize = 18)
        plt.show()
    else:
        print("Correlation matrix is empty.")


country_data , year_data = read_world_bank_data('Input.csv')
print("             *************** COUNTRY DATA ******************             ")
print(country_data.head())
print("             *************** YEAR DATA  ********************              ")
print(year_data.head())

#statistical analysis
print("            ***************** STATISTICAL ANALYSIS **************          ")

"""
METHOD 1 :
DESCRIBE:
"""
country_data['Forest_area%'] = pd.to_numeric(country_data['Forest_area%'] ,
                                             errors = 'coerce')
describes = country_data['Forest_area%'].describe()
print("******** METHOD 1 : Describes ")
print(describes)

"""
METHOD 2 :
SKEWNESS
"""
print("********** METHOD 2 : Skewness  ")
skew_column_name = 'Total greenhouse gas emissions '
# Use apply with a lambda function to convert the column to numeric and calculate skewness
country_data[skew_column_name] = country_data[skew_column_name]\
    .apply(lambda x: pd.to_numeric(x , errors = 'coerce'))
# Now, you can calculate skewness
skewness_value = country_data[skew_column_name].skew()
print("Skewness of Total greenhouse gas emissions is" , skewness_value)

"""
METHOD 3:
KURTOSIS
"""

print("*********** METHOD 3: Kurtosis ")
kurtosis_column = 'Renewable_energy_consumption'
# Replace non-numeric values with NaN
country_data[kurtosis_column] = pd.to_numeric(country_data[kurtosis_column] ,
                                              errors = 'coerce')
# Calculate kurtosis
kurtosis_value = country_data[kurtosis_column].kurtosis()
print('Kurtosis for Renewable energy consumption (% of total final energy consumption) is ' ,
      kurtosis_value)
histogram_plot(country_data[kurtosis_column] , kurtosis_value)

"""CORELATION"""
print(country_data.columns)
country_data['greenhouseGasEmissions'] = (country_data['Total greenhouse gas emissions '] /
                                          country_data['Total greenhouse gas emissions ']
                                          .sum()) * 100


#Heat map
# Select a few indicators for analysis
selected_indicators = ['greenhouseGasEmissions' , 'Forest_area%' ,
                       'Renewable_energy_consumption']
# Extract the relevant data for the selected indicators
df_selected_indicators = country_data[selected_indicators]
# Calculate the correlation matrix
correlation_matrix = df_selected_indicators.corr()
#heat map
heatmap(correlation_matrix)


#bar graph
data_Year = country_data[(country_data['Time'] >= 2015) &
                         (country_data['Time'] <= 2020)]
BarGraph(data_Year , 'Renewable_energy_consumption')

#Line Graph
data_Year = country_data[(country_data['Time'] >= 2013) &
                         (country_data['Time'] <= 2020)]
lineGraph(data_Year)

#pieGraph
pieChart(country_data)

#bargraph
data_Year = country_data[(country_data['Time'] >= 2015) & (country_data['Time'] <= 2020)]
BarGraph(data_Year , 'Oil rents (% of GDP)')


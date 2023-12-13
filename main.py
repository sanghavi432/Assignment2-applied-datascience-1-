"""
@author: Sanghavi
"""

#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis


def dataframExtraction(filename):
    """
       Used pandas to import the dataset, transpose the dataframe, and
       get the date and country columns.

       Parameters:
       - filename : The path of the data file.

       Returns:
       - dataFrameOrg: The dataframe with time as column.
       - dataFrame : The dataframe with country as column.

    """
    dataFrame = pd.read_csv(filename)
    dataFrameOrg = dataFrame.copy()
    dataFrame[['Country Name' , 'Time']] = dataFrame[['Time' , 'Country Name']]
    dataFrame = dataFrame.rename(columns = {'Country Name': 'Time' ,
                                            'Time': 'Country Name'})

    return dataFrameOrg , dataFrame


def plot_line(dataframe , countries , x_column , y_column , title):
    """
        Plot line charts for specified countries and their data over time.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the necessary columns.
        - countries (list): List of countries to plot.
        - x_column (str): The column representing the x-axis (e.g., 'Year').
        - y_column (str): The column representing the y-axis (e.g., 'GDP').
        - title (str): The title of the plot.

        Returns:
        None
    """
    fig , ax = plt.subplots()

    for country in countries:
        country_data = dataframe[dataframe['Country Name'] == country]
        ax.plot(country_data[x_column] , country_data[y_column] ,
                label = country)

    ax.set(xlabel = 'Year' , ylabel = y_column ,
           title = title)
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_correlation_matrix(correlation_matrix):
    """
        Plot a heatmap for the given correlation matrix.

        Parameters:
        - correlation_matrix (pd.DataFrame): The correlation matrix
        to be visualized.

        Returns:
        None
    """

    # Check if the correlation matrix is not empty
    if not correlation_matrix.empty:
        # Create a heatmap for the correlation matrix
        plt.figure(figsize = (12 , 8))
        heatmap = plt.imshow(correlation_matrix , cmap = 'coolwarm' ,
                             interpolation = 'nearest')

        # Add annotations
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j , i , f'{correlation_matrix.iloc[i , j]:.2f}' ,
                         ha = 'center' , va = 'center' , color = 'w')

        # Add axis labels and title
        plt.xticks(range(len(correlation_matrix.columns)) , correlation_matrix.columns)
        plt.yticks(range(len(correlation_matrix)) , correlation_matrix.index)
        plt.xlabel('Indicators')
        plt.ylabel('Indicators')
        plt.title('Correlation Matrix for Selected World Bank Indicators')

        # Add color bar
        cbar = plt.colorbar(heatmap)
        cbar.set_label('Correlation Coefficient')

        # Display the heatmap
        plt.show()
    else:
        print("Correlation matrix is empty.")


def bargraph_Access_to_electricity_urban_rural(bargraphData , indicatorName1 ,
                                               indicatorName2 , title , label1 , label2):
    """
        Plot a bar graph comparing the percentages of access to electricity for
         urban and rural areas over time.

        Parameters:
        - bargraphData (pd.DataFrame): A DataFrame containing the necessary columns,
        including 'Time', indicatorName1, and indicatorName2.
        - indicatorName1 (str): The column name representing the first
        indicator (e.g., 'Urban Access to Electricity (%)').
        - indicatorName2 (str): The column name representing the second
         indicator (e.g., 'Rural Access to Electricity (%)').
        - title (str): The title of the plot.
        - label1 (str): Label for the first set of bars.
        - label2 (str): Label for the second set of bars.

        Returns:
        None
    """

    bar_width = 0.35
    positions = np.arange(len(bargraphData['Time']))
    position_variable1 = positions - bar_width
    position_variable2 = positions
    # Plotting bar graph
    plt.figure(figsize = (12 , 8))
    plt.bar(position_variable1 , bargraphData[indicatorName1] , width = bar_width ,
            label = label1 , color = 'yellow')
    plt.bar(position_variable2 , bargraphData[indicatorName2] , width = bar_width ,
            label = label2 , color = 'pink')

    # Adding labels and title
    plt.xlabel('Year')
    plt.xticks(positions , bargraphData['Time'])
    plt.ylabel('Percentage')
    plt.title(title)

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Task 1 :
#Returning two dataframes: one transposed with years as columns and
# another with countries as columns.
YearDataFrame , countryDataFrame = dataframExtraction('Dataset.csv')
print("Year Data Frame")
print(YearDataFrame.head())
print("Country Data Frame")
print(countryDataFrame.head())

# Task 2 :
# STATISTICAL ANALYSIS
#METHOD 1
countryDataFrame['Electricity production from oil, gas and coal sources (% of total)'] = \
    pd.to_numeric(countryDataFrame['Electricity production from oil, gas and coal sources (% of total)'] ,
                                             errors = 'coerce')
describes_electricityProduction_fromOil = countryDataFrame['Electricity production from oil, gas and ' \
                                                           'coal sources (% of total)'].describe()
print("Describes Analysis for electric city production from oil")
print(describes_electricityProduction_fromOil)
#METHOD 2
YearDataFrame['Electric power consumption (kWh per capita) '] = pd.to_numeric(
    YearDataFrame['Electric power consumption (kWh per capita) '] , errors='coerce'
)
maxValue = pd.to_numeric(countryDataFrame[countryDataFrame['Time'] ==
                                          '2019']['Renewable energy consumption (%)'] , errors = 'coerce').max()
minValue = pd.to_numeric(countryDataFrame[countryDataFrame['Time'] ==
                                          '2019']['Renewable energy consumption (%)'] , errors = 'coerce').min()
print("Maximum renewable energy consumption in the year 2019: " , maxValue)
print("Minimum renewable energy consumption in the year 2019: " , minValue)
#METHOD 3
print("Skewness")
countryDataFrame['Access to electricity, urban (% of urban population) '] = \
    pd.to_numeric(countryDataFrame['Access to electricity, urban (% of urban population) '] , errors='coerce')
# Calculate skewness
skewness = skew(countryDataFrame['Access to electricity, urban (% of urban population) '].dropna())
print("Skewness for 'Access to electricity, urban (% of urban population) " , skewness)
#METHOD 4
print("Kurtosis")
data_for_kurtosis = countryDataFrame['Forest area (% of land area)'].dropna()
kurtosis_value = kurtosis(data_for_kurtosis , fisher=False)
print("Kurtosis:" , kurtosis_value)

#Line Graph
#GDP per capita of usa over years

countries_to_plot = ['India' , 'United Kingdom' , 'Thailand' , 'Pakistan' , 'Norway' , 'Netherlands']
plot_line(countryDataFrame , countries_to_plot , x_column = 'Time' ,
          y_column = 'Electricity production from oil, gas and coal sources (% of total)',
          title='Electricity production from oil, gas and coal sources')

#Heat Map
#Fertilizer production vs agriculture productivity
specificIndicators = ['Crop production' , 'Fertilizerconsumption(%)']
# Extract the relevant data for the selected indicators
df_specificIndicators = countryDataFrame[specificIndicators]
df_specificIndicators  = df_specificIndicators .copy()
df_specificIndicators .replace('..' , np.nan , inplace = True)
df_selected_indicators = df_specificIndicators .apply(pd.to_numeric , errors = 'coerce')

# Calculate the correlation matrix
correlationMatrix = df_specificIndicators.corr()
print(correlationMatrix)

plot_correlation_matrix(correlationMatrix)

bargraphData = countryDataFrame[countryDataFrame['Country Name'] == 'India']
bargraph_Access_to_electricity_urban_rural(bargraphData ,
                                           'Access to electricity, rural (% of rural population) ' ,
                                           'Access to electricity, rural (% of rural population) ' ,
                                           'Access to electricity urban vs rural','Access to electricity urban' ,
                                           'Access to electricity rural')
bargraph_Access_to_electricity_urban_rural(bargraphData , 'Total natural resources rents (% of GDP) ' ,
                                           'Forest area (% of land area)','Total natural resource rents vs forest land' ,
                                           'natural resource rents','forest area')

plot_line(countryDataFrame , countries_to_plot , x_column = 'Time' ,
          y_column = 'GDP per capita, PPP (current international $)' ,
          title = 'GDP per capita')

specificIndicators = ['Crop production' , 'GDP per capita, PPP (current international $)']
# Extract the relevant data for the selected indicators
df_specificIndicators = countryDataFrame[specificIndicators]
df_specificIndicators  = df_specificIndicators .copy()
df_specificIndicators .replace('..' ,  np.nan ,  inplace = True)
df_selected_indicators = df_specificIndicators .apply(pd.to_numeric , errors = 'coerce')

# Calculate the correlation matrix
correlationMatrix = df_specificIndicators.corr()
print(correlationMatrix)

plot_correlation_matrix(correlationMatrix)


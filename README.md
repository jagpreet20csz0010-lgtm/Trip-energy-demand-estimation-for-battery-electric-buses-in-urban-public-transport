# Trip-energy-demand-estimation-for-battery-electric-buses-in-urban-public-transport

Battery electric buses (BEBs) are increasingly used in urban public transport systems as a cleaner and more energy-efficient alternative to conventional buses. For these buses to operate reliably, it is important to estimate how much energy they will need for each trip. Trip energy demand estimation (TED) helps public transport operators to dorecast energy demandsand plan charging infrastructure.

## Features
- Dataset and python code to model TED (trip energy demand) of BEBs.
- Example scripts to reproduce results.

## Dataset Description and Data Dictionary

The dataset contains information about battery electric bus trips, including route characteristics and energy consumption. In the manuscript, the following variables are considered as independent (input) features:  

- **Average Trip Speed (Km/h) (ATS)** 
- **Average Passenger Count (APC)**  
- **Trip Duration (minutes)**  
- **Trip Length (Km)**  
- **Number of Bus Stops**  
- **External Temperature (°C)**  

The actual dataset column names and their corresponding manuscript variables are mapped below for clarity:

| Dataset Column Name         | Manuscript Variable Name              | Description |
|-----------------------------|--------------------------------------|-------------|
| `Speed`                     | Average Trip Speed                   | Average speed of the bus during the trip (km/h). |
| `Passengers`                | Average Passenger Count              | Average number of passengers on the trip. |
| `Trip_Duration_Minutes`     | Trip Duration (minutes)              | Total duration of the trip in minutes. |
| `Route_Length_km`           | Trip Length (Km)                     | Total distance of the trip in kilometers. |
| `Number_of_Bus_Stops`       | Number of Bus Stops                  | Total number of stops along the route. |
| `Trip_Temprature`           | External Temperature (°C)            | City temperature during the trip (°C). |

The output (dependent) variable representing energy consumption is:

| Dataset Column Name         | Manuscript Variable Name  Description |
|-----------------------------|-------------|
| `Trip_Energy_kw`            | TED (Trip Energy Demand)  estimation for the trip in kilowatt. |

**Note:** This mapping ensures consistency between the dataset and the variables described in the manuscript.

## Requirements
...


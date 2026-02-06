import tensorflow as tf
import numpy as np
import os as os
tfkl = tf.keras.layers

def calculate_error(forecasted_data, actual_data, changeratetolog=False):
    # Create a dictionary for fast lookup of actual rates by (geo, year, age)
    actual_dict = {tuple(row[:3]): float(row[3]) for row in actual_data}

    # Build aligned arrays by matching on common keys
    forecasted_rates = []
    actual_rates = []
    for row in forecasted_data:
        key = tuple(row[:3])
        if key in actual_dict:
            forecasted_rates.append(float(row[3]))
            actual_rates.append(actual_dict[key])

    forecasted_rates = np.array(forecasted_rates)
    actual_rates = np.array(actual_rates)

    if len(forecasted_rates) == 0:
        raise ValueError("No common geo/year/age combinations found between forecasted and actual data")

    if changeratetolog:
        # Replace zeros with small value to avoid log(0)
        forecasted_rates = np.where(forecasted_rates == 0, 1e-05, forecasted_rates)
        actual_rates = np.where(actual_rates == 0, 1e-05, actual_rates)

        forecasted_rates = np.log(forecasted_rates)
        actual_rates = np.log(actual_rates)

    # Calculate errors
    squared_errors = (forecasted_rates - actual_rates) ** 2
    mse = np.mean(squared_errors)


    return mse

def calculate_error_old(forecasted_data, actual_data, changeratetolog=False):
    # Ensure both arrays are sorted by geo, year, and age
    forecasted_data = forecasted_data[np.lexsort((forecasted_data[:, 2], forecasted_data[:, 1], forecasted_data[:, 0]))]
    actual_data = actual_data[np.lexsort((actual_data[:, 2], actual_data[:, 1], actual_data[:, 0]))]

    # Find common geo/year/age combinations between forecasted and actual rates
    common_keys = set(map(tuple, forecasted_data[:, :3])) & set(map(tuple, actual_data[:, :3]))

    # Filter both forecasted and actual rates based on common combinations
    filtered_forecasted = np.array([row for row in forecasted_data if tuple(row[:3]) in common_keys])
    filtered_actual = np.array([row for row in actual_data if tuple(row[:3]) in common_keys])
    forecasted_rates = filtered_forecasted[:, 3].astype(float)
    actual_rates = filtered_actual[:, 3].astype(float)

    if changeratetolog:
        forecasted_rates[forecasted_rates == 0] = 1e-05
        actual_rates[actual_rates == 0] = 1e-05

        forecasted_rates = np.log(forecasted_rates)
        actual_rates = np.log(actual_rates)

    mse = np.mean((forecasted_rates - actual_rates) ** 2)
    rmse = np.sqrt(np.mean(((forecasted_rates - actual_rates) ** 2)))
    rrmse = np.sqrt(np.mean((forecasted_rates - actual_rates) ** 2)) / np.mean(actual_rates)
        
    return mse, rmse, rrmse

def calculate_error_by_category(forecasted_data, actual_data, feature_index, changeratetolog=False):
    # Ensure both arrays are sorted by geo, gender, year, and age
    forecasted_data = forecasted_data[np.lexsort((forecasted_data[:, 3], forecasted_data[:, 2], forecasted_data[:, 1], forecasted_data[:, 0]))]
    actual_data = actual_data[np.lexsort((actual_data[:, 3], actual_data[:, 2], actual_data[:, 1], actual_data[:, 0]))]

    # Find common geo/gender/year/age combinations between forecasted and actual rates
    common_keys = set(map(tuple, forecasted_data[:, :4])) & set(map(tuple, actual_data[:, :4]))

    # Filter both forecasted and actual rates based on common combinations
    filtered_forecasted = np.array([row for row in forecasted_data if tuple(row[:4]) in common_keys])
    filtered_actual = np.array([row for row in actual_data if tuple(row[:4]) in common_keys])

    categories = np.unique(filtered_forecasted[:, feature_index].astype(int))
    
    mses_by_category = {}
    rmses_by_category = {}
    rrmses_by_category = {}

    for category in categories:
        forecasted = filtered_forecasted[filtered_forecasted[:, feature_index] == category]
        actual = filtered_actual[filtered_actual[:, feature_index] == category]

        forecasted_rates = forecasted[:, 4].astype(float)
        actual_rates = actual[:, 4].astype(float)

        if changeratetolog:
            forecasted_rates[forecasted_rates == 0] = 9e-06
            actual_rates[actual_rates == 0] = 9e-06

            forecasted_rates = np.log(forecasted_rates)
            actual_rates = np.log(actual_rates)

        mses_by_category[category] = np.mean((forecasted_rates - actual_rates) ** 2)
        rmses_by_category[category] = np.sqrt(np.mean(((forecasted_rates - actual_rates) ** 2)))
        rrmses_by_category[category] = np.sqrt(np.mean((forecasted_rates - actual_rates) ** 2)) / np.mean(actual_rates)
        
    return mses_by_category, rmses_by_category, rrmses_by_category

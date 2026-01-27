import csv
import numpy as np
import os as os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# loading in HMD data
data = []
ages = []
countries = []
genders = []


with open("../../data/asfr/asfrVH.txt", "r") as file:
    #reader = csv.reader(file, delimiter="\t")
    for row_index, row in enumerate(file):
        row = row.strip()
        if row_index <= 2:
            print(row)
        if row_index >= 3:
            columns = row.split()
            country, year, age, rate = columns
            year = int(year)
            try:
                age = int(age)
            except:
                age = -1
            if age not in ages and age != -1 and age <= 54:
                ages.append(age)
            if country not in countries:
                countries.append(country)
            country = countries.index(country)
            try:
                rate = float(rate)
            except:
                rate = -1
            if rate > 1:
                rate = 1
            if age != -1 and rate != -1 and age <= 54:
                data.append([country, year, age, rate])

asfr_cohort_data = np.array(data)
print("ASFR data shape:", asfr_cohort_data.shape)
# getting unique values for geographic location column 
#country_data[:,0] = country_data[:,0] + 50

# Below, I create a joint list of populations and their 
# corresponding numeric code that identifies them in the data
geos_list = countries
geos_index = np.arange(len(geos_list))
geos_key = np.column_stack((np.array(geos_list), geos_index))
np.save('../../data/geos_key.npy', geos_key)

# create combined data
#combined = np.vstack((state_data, country_data))

##### Country Splits #####
training_index = np.logical_and(asfr_cohort_data[:, 1] >= 1950, asfr_cohort_data[:, 1] <= 2010)
asfr_training = asfr_cohort_data[training_index, :]
np.savetxt('../../data/asfr_cohort_training.txt', asfr_training)
print("ASFR training data shape:", asfr_training.shape)

test_index = np.logical_and(asfr_cohort_data[:, 1] > 2010, asfr_cohort_data[:, 1] <= 2019)
asfr_test = asfr_cohort_data[test_index, :]
np.savetxt('../../data/asfr_cohort_test.txt', asfr_test)
print("ASFR test data shape:", asfr_test.shape)

final_test_index = np.logical_and(asfr_cohort_data[:, 1] > 2019, asfr_cohort_data[:, 1] <= 2025)
asfr_final_test = asfr_cohort_data[final_test_index, :]
np.savetxt('../../data/asfr_cohort_final_test.txt', asfr_final_test)
print("ASFR final test data shape:", asfr_final_test.shape)


training_index = np.logical_and(asfr_cohort_data[:, 1] >= 1950, asfr_cohort_data[:, 1] <= 2021)
asfr_training = asfr_cohort_data[training_index, :]
np.savetxt('../../data/asfr_cohort_training_llm.txt', asfr_training)
print("ASFR training data shape:", asfr_training.shape)

test_index = asfr_cohort_data[:, 1] == 2022
asfr_test = asfr_cohort_data[test_index, :]
np.savetxt('../../data/asfr_cohort_test_llm.txt', asfr_test)
print("ASFR test data shape:", asfr_test.shape)

final_test_index = np.logical_and(asfr_cohort_data[:, 1] > 2022, asfr_cohort_data[:, 1] <= 2025)
asfr_final_test = asfr_cohort_data[final_test_index, :]
np.savetxt('../../data/asfr_cohort_final_test_llm.txt', asfr_final_test)
print("ASFR final test data shape:", asfr_final_test.shape)








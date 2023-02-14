import pandas as pd
import numpy as np
import random
import copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

FILE_NAME = "WAdata.xlsx"

# Takes in the lists of folds and which fold number is the test set
# The rest of the folds become the training set
# Returns the R-squared value when using that fold as the test set
# Called once per fold (a total of 5 times here)
def train_and_test_a_single_fold(input, actual_results, fold_num):
   input_copy = copy.deepcopy(input)
   actual_results_copy = copy.deepcopy(actual_results)
   test_input = input_copy.pop(fold_num)
   training_input = pd.concat(input_copy, axis=0)
   test_results = actual_results_copy.pop(fold_num).Total.values.tolist()
   training_results = pd.concat(actual_results_copy, axis=0).Total.values.tolist()
   regr = RandomForestRegressor()
   regr.fit(training_input, training_results)
   prediction = regr.predict(test_input)
   r2 = r2_score(test_results, prediction)
   return r2

# Runs through each fold and trains/tests on that fold specifically
# Called once
def test_each_fold(input_folds, result_folds):
   r2_scores = []
   for i in range(len(input_folds)):
      r2_scores.append(train_and_test_a_single_fold(input_folds, result_folds, i))

   return np.mean(r2_scores)

# Takes in the two DataFrames (the inputs and the target values)
# Returns two lists of 5 DataFrames each. Each DataFrame is one fold,
# one is the list of inputs and the other is the list of targets
def split_df_into_folds(combined_input_df, result_df):
   fold_sizes = [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
   fold_nums = []
   for size in fold_sizes:
      for frequency in range(size[1]):
         fold_nums.append(size[0])
   random.shuffle(fold_nums)

   combined_input_df["Fold Number"] = fold_nums
   result_df["Fold Number"] = fold_nums

   input_folds = []
   result_folds = []
   clean_input_folds = []

   for i in range(len(fold_sizes)):
      input_folds.append(combined_input_df[combined_input_df["Fold Number"] == i])
      result_folds.append(result_df[result_df["Fold Number"] == i])
   for fold in input_folds:
      clean_input_folds.append(fold.drop(labels = "Fold Number", axis=1))
   return clean_input_folds, result_folds

# Opens the Excel file, makes DataFrames from the data
# Removes the indicated outliers from the DataFrames and returns them
def read_file_make_df(months):
   data_dict = pd.read_excel(FILE_NAME, sheet_name=months, index_col=0)
   combined_df = pd.concat(data_dict, axis=1)
   combined_df.drop(labels = [2015, 2020, 2021], axis = 0, inplace=True)
   acres_burned = pd.read_excel(FILE_NAME, sheet_name="Fire Acres", index_col=0)
   acres_burned.drop(labels = [2015, 2020, 2021], axis = 0, inplace=True)
   return combined_df, acres_burned

def main():
   #months = ["Last Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
   months = ["Sep"]
   combined_months_df, actual_acres_burned = read_file_make_df(months)
   input_folds, result_folds = split_df_into_folds(combined_months_df, actual_acres_burned)
   r2_score = test_each_fold(input_folds, result_folds)
   print(r2_score)

main()
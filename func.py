import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay








def best_boost_metric(cv_results, metric = "mean_test_f1_all", rows_to_show = 5, ascending = False, show_other_metrics = False):
    """
        Returns dataframe sorted by the given metric with best result on top.
        
        - cv_results:
            cv_results_ of a gridsearch
        
        - metric: "str", default "f1_macro"
            Could be one among:
            - "mean_test_accuracy"
            - "mean_test_f1_all"
            - "mean_test_scorer_class_0"
            - "mean_test_scorer_class_1"
            - "mean_test_scorer_class_2"
        
        - rows_to_show:
            int or slice, like [10:20]
            if int = n then the top n results will be displayed
        
        - ascending: Bool, default: False
        
        - show_other_metrics: Bool, default: False
            if True the dataframe will contain the scores from the other metrics
    """
    # convert to df
    res = pd.DataFrame(cv_results)
    
    # add "classifier" column
    if not "classifier" in res.columns:
        res = pd.concat([res, pd.DataFrame(data = [res.params[i]["boost__init"].__class__.__name__ for i in range(len(res.params))],
                                           columns = ["classifier"])],
                        axis = 1)
    
    # sort by metric
    res = res[["classifier", "mean_test_accuracy", "mean_test_f1_all",
                        "mean_test_scorer_class_0", "mean_test_scorer_class_1", "mean_test_scorer_class_2",
                        "mean_fit_time", "mean_score_time"]].sort_values(by = [metric], axis = 0, ascending = ascending)

    # remove duplicates
    res = res.loc[:,~res.columns.duplicated()]
    
    # show_other_metrics
    if not show_other_metrics:
        res = res[["classifier", metric, "mean_fit_time", "mean_score_time"]].copy()
    
    # rows_to_show
    if isinstance(rows_to_show, int):
        res = res.head(rows_to_show)
    else:
        res = res[rows_to_show[0]]
    
    return res



def best_metric(cv_results, metric = "f1_macro", rows_to_show = 5, ascending = False, show_other_metrics = False):
    """
        Returns dataframe sorted by the given metric with best result on top.
        
        - cv_results:
            cv_results_ of a gridsearch
        
        - metric: "str", default "f1_macro"
            Could one among:
            - mean_test_accuracy
            - mean_test_f1_all
            - mean_test_scorer_class_0
            - mean_test_scorer_class_1
            - mean_test_scorer_class_2
        
        - rows_to_show:
            int or slice, like [10:20]
            if int = n then the top n results will be displayed
        
        - ascending: Bool, default: False
        
        - show_other_metrics: Bool, default: False
            if True the dataframe will contain the scores from the other metrics
    """
    # convert to df
    res = pd.DataFrame(cv_results)
    
    # add "classifier" column
    if not "classifier" in res.columns:
        res = pd.concat([res, pd.DataFrame(data = [m["clf"].__class__.__name__ for m in res.params],
                                           columns = ["classifier"])],
                        axis = 1)
    
    # sort by metric
    res = res[["classifier", "mean_test_accuracy", "mean_test_f1_all",
                        "mean_test_scorer_class_0", "mean_test_scorer_class_1", "mean_test_scorer_class_2",
                        "mean_fit_time", "mean_score_time"]].sort_values(by = [metric], axis = 0, ascending = ascending)

    # remove duplicates
    res = res.loc[:,~res.columns.duplicated()]
    
    # show_other_metrics
    if not show_other_metrics:
        res = res[["classifier", metric, "mean_fit_time", "mean_score_time"]].copy()
    
    # rows_to_show
    if isinstance(rows_to_show, int):
        res = res.head(rows_to_show)
    else:
        res = res[rows_to_show[0]]
    
    return res


def convert_cv_results(cv_results):
    # convert the results to dataframe
    res = pd.DataFrame(cv_results)

    # remove std and split scores
    res_clean = res[res.columns.drop(list(res.filter(regex = "split|std")))].copy()

    # add "classifier" column
    res_clean = pd.concat([res_clean,
                           pd.DataFrame(data = [m["clf"].__class__.__name__ for m in res_clean.params], columns = ["classifier"])
                          ],
                          axis = 1)

    res_clean_group = res_clean.groupby(by = "classifier").describe()
    ovr_res = pd.DataFrame(data = {"mean_accuracy":res_clean_group["mean_test_accuracy"]["mean"],
                                   "mean_f1":res_clean_group["mean_test_f1_all"]["mean"],
                                   "mean_f1_class_0":res_clean_group["mean_test_scorer_class_0"]["mean"],
                                   "mean_f1_class_1":res_clean_group["mean_test_scorer_class_1"]["mean"],
                                   "mean_f1_class_2":res_clean_group["mean_test_scorer_class_2"]["mean"],
                                   "mean_fit_time":res_clean_group["mean_fit_time"]["mean"],
                                  },
                           index = res_clean_group.index,
                           columns = ["mean_accuracy", "mean_f1", "mean_f1_class_0",
                                      "mean_f1_class_1", "mean_f1_class_2", "mean_fit_time"])
    
    return ovr_res


def convert_boost_cv_results(cv_results):
    # convert the results to dataframe
    res = pd.DataFrame(cv_results)

    # remove std and split scores
    res_clean = res[res.columns.drop(list(res.filter(regex = "split|std")))].copy()

    # add "classifier" column
    res_clean = pd.concat([res_clean,
                           pd.DataFrame(
                               data = [res_clean.params[i]["boost__init"].__class__.__name__ for i in range(len(res_clean.params))],
                               columns = ["classifier"]
                           )
                          ],
                          axis = 1)

    res_clean_group = res_clean.groupby(by = "classifier").describe()
    ovr_res = pd.DataFrame(data = {"mean_accuracy":res_clean_group["mean_test_accuracy"]["mean"],
                                   "mean_f1":res_clean_group["mean_test_f1_all"]["mean"],
                                   "mean_f1_class_0":res_clean_group["mean_test_scorer_class_0"]["mean"],
                                   "mean_f1_class_1":res_clean_group["mean_test_scorer_class_1"]["mean"],
                                   "mean_f1_class_2":res_clean_group["mean_test_scorer_class_2"]["mean"],
                                   "mean_fit_time":res_clean_group["mean_fit_time"]["mean"],
                                  },
                           index = res_clean_group.index,
                           columns = ["mean_accuracy", "mean_f1", "mean_f1_class_0",
                                      "mean_f1_class_1", "mean_f1_class_2", "mean_fit_time"])
    
    return ovr_res


def confusion_matrix_scorer(clf, X, y):
    clf.fit(X, y)
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {"0_p_0_t": cm[0, 0], "1_p_0_t": cm[0, 1], "2_p_0_t": cm[0, 2],
            "0_p_1_t": cm[1, 0], "1_p_1_t": cm[1, 1], "2_p_1_t": cm[1, 2],
            "0_p_2_t": cm[2, 0], "1_p_2_t": cm[2, 1], "2_p_2_t": cm[2, 2],
           }

def fit_predict_report(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Counts of real classes:      {dict(enumerate(np.unique(y_test, return_counts = True)[1]))}")
    print(f"Counts of predicted classes: {dict(enumerate(np.unique(y_pred, return_counts = True)[1]))}")
    ConfusionMatrixDisplay(matrix).plot(cmap=plt.cm.Blues)
    plt.show()
    


def calc_prediction_success(df_odds, df_results, results_labels = None):
    """
    Calculates the success rate of  a bookmaker
    considering the lowest odd as prediction (lowest odd = highest calculated probability)
    
    Returns np.array with two elements:
    The first element is df with boolean mask where Trues are the correct predictions.
    The second element is calculated percentage of correctly predicted results(accuracy).
    
    df_odds: dataframe with odds for H/D/A columns in this order.
    
    df_results: pandas series containing the outcome.
    
    results_labels: default: ["H", "D", "A"], list of three strings
    indicating the label for home win / draw / away win in this order.
    
    """
    odds = df_odds.copy()
    predicted = pd.DataFrame()
    
    if results_labels == None:
        results_labels = ["H", "D", "A"]
    
    # change the labels to match the results df
    odds.columns = results_labels
    
    # leave only the min odd and replace the value with column label
    lowest_odds = odds.idxmin(axis = 1)
    
    # create nwe df with true/false values for the predictions
    predicted_df = (lowest_odds == df_results)
    percentage = round(len(predicted_df[predicted_df == True]) / len(predicted_df), 2)
    
    return print(f"Accuracy: {percentage}.")


# convert odds to probability
def odd_to_proba(df):
    """
    Returns probabilities based on the provided odds.
    In order to eliminate the bookmaker's margin
    the odds should represent all possible outcomes of an event.
    """
    
    df_proba = pd.DataFrame(index = df.index, columns = [col + "_proba" for col in df.columns])
    
    for col in df.columns:
        df_proba[col + "_proba"] = (1/df[col]) * 100
    
    df_proba["sum"] = df_proba.sum(axis = 1)
    
    for col in df.columns: 
        df_proba[col + "_proba"] /= df_proba["sum"]
        
    return df_proba.drop("sum", axis = 1)


# In[1]:


def get_matches_from_year(df, start, end):
    dft = df.copy()
    dft = dft[(dft.year >= start) & (dft.year <= end)]
    return dft



def tts_matches_from_year(df, start, end, test_size, random_state):
    df_tts = df.copy()
    df_tts = df_tts[(df_tts.year >= start) & (df_tts.year <= end)]
    
    X = pd.get_dummies(df_tts.drop("FTR", axis = 1))
    y = df_tts.FTR.replace(["H","D","A"],[0, 1, 2]).values.ravel()
    
    
    df_tts_X_train, df_tts_X_test, df_tts_y_train, df_tts_y_test = train_test_split(X, y,
                                                                                    test_size = test_size,
                                                                                    random_state = random_state,
                                                                                    stratify = y)
    
    return df_tts_X_train, df_tts_X_test, df_tts_y_train, df_tts_y_test



def tts_matches_from_year_with_target(df, start, end, target_year):
    df_tts = df.copy()
    df_tts = df_tts[((df_tts.year >= start) & (df_tts.year <= end)) | (df_tts.year == target_year)]
    
    X = pd.get_dummies(df_tts.drop("FTR", axis = 1))
    
    X_train = df_tts[(X.year >= start) & (X.year <= end)]
    X_test = df_tts[X.year == target_year]
    
    y_train = df_tts[(df_tts.year >= start) & (df_tts.year <= end)].FTR.replace(["H","D","A"],[0, 1, 2]).values.ravel()
    y_test = df_tts[df_tts.year == target_year].FTR.replace(["H","D","A"],[0, 1, 2]).values.ravel()
        
    return X_train, X_test, y_train, y_test


# Required Libraries
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy import stats


# The function is use to check is a column only contain numerical value
def is_number(string):
    for i in range(0, len(string)):
        if not pd.isnull(string[i]):
            try:
                float(string[i])
                return True
            except ValueError:
                return False


# The function is use to check is a column only contain numerical value
def is_number_value(value):
    if not pd.isnull(value):
        try:
            float(value)
            return True
        except ValueError:
            return False


# The function is use to performs a Chi_Squared Test or Fisher Exact Test
def chi_squared_test(label_df, feature_df):
    label_df.reset_index(drop=True, inplace=True)
    feature_df.reset_index(drop=True, inplace=True)
    data = pd.concat([pd.DataFrame(label_df.values.reshape((label_df.shape[0], 1))), feature_df], axis=1)
    data.columns = ["label", "feature"]
    contigency_table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1], margins=False)
    m = contigency_table.values.sum()
    p_value = stats.fisher_exact(contigency_table) if m <= 10000 and contigency_table.shape == (
        2, 2) else stats.chi2_contingency(contigency_table, correction=False)
    return p_value[1]


# The function is predict the testing dataset
def prediction_dt_c45(model, Xdata):
    Xdata = Xdata.reset_index(drop=True)
    ydata = pd.DataFrame(index=range(0, Xdata.shape[0]), columns=["Prediction"])
    data = pd.concat([ydata, Xdata], axis=1)
    rule = []

    dt_model = deepcopy(model)

    for i in range(0, len(dt_model)):
        dt_model[i] = dt_model[i].replace("{", "").replace("}", "").replace(";", "").replace("=", "")
        dt_model[i] = dt_model[i].replace("IF ", "").replace("AND", "").replace("THEN", "").replace("<", "<=")

    for i in range(0, len(dt_model) - 2):
        splited_rule = [x for x in dt_model[i].split(" ") if x]
        rule.append(splited_rule)

    for i in range(0, Xdata.shape[0]):
        for j in range(0, len(rule)):
            rule_confirmation = len(rule[j]) / 2 - 1
            rule_count = 0
            for k in range(0, len(rule[j]) - 2, 2):
                if not is_number_value(data[rule[j][k]][i]):
                    if data[rule[j][k]][i] in rule[j][k + 1]:
                        rule_count += 1
                        if rule_count == rule_confirmation:
                            data.iloc[i, 0] = rule[j][len(rule[j]) - 1]
                    else:
                        k = len(rule[j])
                else:
                    if rule[j][k + 1].find("<=") == 0:
                        if data[rule[j][k]][i] <= float(rule[j][k + 1].replace("<=", "")):
                            rule_count += 1
                            if rule_count == rule_confirmation:
                                data.iloc[i, 0] = rule[j][len(rule[j]) - 1]
                        else:
                            k = len(rule[j])
                    elif rule[j][k + 1].find(">") == 0:
                        if data[rule[j][k]][i] > float(rule[j][k + 1].replace(">", "")):
                            rule_count += 1
                            if rule_count == rule_confirmation:
                                data.iloc[i, 0] = rule[j][len(rule[j]) - 1]
                        else:
                            k = len(rule[j])

    for i in range(0, Xdata.shape[0]):
        if pd.isnull(data.iloc[i, 0]):
            data.iloc[i, 0] = dt_model[len(dt_model) - 1]

    return data


# Function: Calculates the Information Gain Ratio
def info_gain_ratio(target, feature=None, split=None):
    if split is None:
        split = []
    if feature is None:
        feature = []
    info_gain_r, intrinsic_v, entropy = 0, 0, 0
    denominator_1 = pd.DataFrame(feature).count()
    uniques = np.unique(target)
    data = pd.concat([pd.DataFrame(target.values.reshape((target.shape[0], 1))), feature], axis=1)

    for i in range(0, len(uniques)):
        numerator_1 = data.iloc[:, 0][(data.iloc[:, 0] == uniques[i])].count()
        if numerator_1 > 0:
            entropy = entropy - (numerator_1 / denominator_1) * np.log2((numerator_1 / denominator_1))
    info_gain = float(entropy)

    for i in range(0, len(split)):
        denominator_2 = pd.DataFrame(feature[(feature == split[i])]).count()
        if denominator_2[0] > 0:
            intrinsic_v = intrinsic_v - (denominator_2 / denominator_1) * np.log2((denominator_2 / denominator_1))
        for j in range(0, len(uniques)):
            numerator_2 = data.iloc[:, 0][
                (data.iloc[:, 0] == uniques[j]) & (data.iloc[:, 1] == split[i])].count()
            if numerator_2 > 0:
                info_gain = info_gain + (denominator_2 / denominator_1) * (numerator_2 / denominator_2) * np.log2(
                    (numerator_2 / denominator_2))
    if intrinsic_v[0] > 0:
        info_gain_r = info_gain / intrinsic_v

    return float(info_gain_r)


# The method is use to find the suitable spliting feature
def split_me(feature, split):
    result = pd.DataFrame(feature.values.reshape((feature.shape[0], 1)))
    lower, upper = "<=" + str(split), ">" + str(split)
    for i in range(0, len(feature)):
        result.iloc[i, 0] = feature.iloc[i]
        result.iloc[i, 0] = lower if float(feature.iloc[i]) <= float(split) else upper
    return result, [lower, upper]


# Function: C4.5 Algorithm
# Reference: https://github.com/Valdecy/C4.5
def dt_c45(Xdata, ydata, pre_pruning="none", chi_lim=0.1, min_lim=5):
    # Preprocessing - Creating Dataframe
    name = ydata.name
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))

    dataset = pd.concat([ydata, Xdata], axis=1)

    # Preprocessing - Unique Words List
    unique = []
    uniqueWords = []
    for j in range(0, dataset.shape[1]):
        for i in range(0, dataset.shape[0]):
            token = dataset.iloc[i, j]
            if not token in unique:
                unique.append(token)
        uniqueWords.append(unique)
        unique = []

    # Preprocessing - Label Matrix
    label = np.array(uniqueWords[0])
    label = label.reshape(1, len(uniqueWords[0]))

    # C4.5 - Initializing Variables
    i = 0
    branch = [[]]
    branch[0] = dataset
    gain_ratio = np.empty([1, branch[i].shape[1]])
    impurity = 0
    rule = [" "] * 1
    rule[0] = "IF "
    skip_update = False
    stop = 2
    upper = "1"

    # C4.5 - Algorithm
    while i < stop:
        impurity = np.amax(gain_ratio)
        gain_ratio.fill(0)
        for element in range(1, branch[i].shape[1]):
            if len(branch[i]) == 0:
                skip_update = True
                break
            if len(np.unique(branch[i][0])) == 1 or len(branch[i]) == 1:
                if ";" not in rule[i]:
                    rule[i] = rule[i] + " THEN " + name + " = " + branch[i].iloc[0, 0] + ";"
                    rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                skip_update = True
                break
            if (i > 0) and (not is_number(dataset.iloc[:, element])) and (pre_pruning == "chi_2") and (chi_squared_test(
                    branch[i].iloc[:, 0], branch[i].iloc[:, element]) > chi_lim):
                if ";" not in rule[i]:
                    rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x: x.value_counts().index[0])[
                        0] + ";"
                    rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                skip_update = True
                continue
            if is_number(dataset.iloc[:, element]):
                gain_ratio[0, element] = 0.0
                value = np.sort(branch[i].iloc[:, element].unique())
                skip_update = False
                if branch[i][(branch[i].iloc[:, element] == value[0])].count()[0] > 1:
                    start = 0
                    finish = len(branch[i].iloc[:, element].unique()) - 2
                else:
                    start = 1
                    finish = len(branch[i].iloc[:, element].unique()) - 2
                if len(branch[i]) == 2 or len(value) == 1 or len(value) == 2:
                    start = 0
                    finish = 1
                if len(value) == 3:
                    start = 0
                    finish = 2
                for bin_split in range(start, finish):
                    bin_sample = split_me(feature=branch[i].iloc[:, element], split=value[bin_split])
                    if i > 0 and pre_pruning == "chi_2" and chi_squared_test(branch[i].iloc[:, 0],
                                                                             bin_sample[0]) > chi_lim:
                        if ";" not in rule[i]:
                            rule[i] = rule[i] + " THEN " + name + " = " + \
                                      branch[i].agg(lambda x: x.value_counts().index[0])[0] + ";"
                            rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                        skip_update = True
                        continue
                    igr = info_gain_ratio(target=branch[i].iloc[:, 0], feature=bin_sample[0], split=bin_sample[1])
                    if igr > float(gain_ratio[0, element]):
                        gain_ratio[0, element] = igr
                        uniqueWords[element] = bin_sample[1]
            if not is_number(dataset.iloc[:, element]):
                gain_ratio[0, element] = 0.0
                skip_update = False
                igr = info_gain_ratio(target=branch[i].iloc[:, 0], feature=pd.DataFrame(
                    branch[i].iloc[:, element].values.reshape((branch[i].iloc[:, element].shape[0], 1))),
                                      split=uniqueWords[element])
                gain_ratio[0, element] = igr
            if i > 0 and pre_pruning == "min" and len(branch[i]) <= min_lim:
                if ";" not in rule[i]:
                    rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x: x.value_counts().index[0])[
                        0] + ";"
                    rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                skip_update = True
                continue
        if (i > 0) and (pre_pruning == "impur") and (impurity > np.amax(gain_ratio) > 0):
            if ";" not in rule[i]:
                rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x: x.value_counts().index[0])[
                    0] + ";"
                rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
            skip_update = True
            continue

        if not skip_update:
            root_index = np.argmax(gain_ratio)
            rule[i] = rule[i] + str(list(branch[i])[root_index])

            for word in range(0, len(uniqueWords[root_index])):
                uw = uniqueWords[root_index][word].replace("<=", "")
                uw = uw.replace(">", "")
                lower = "<=" + uw
                upper = ">" + uw
                if uniqueWords[root_index][word] == lower:
                    branch.append(branch[i][branch[i].iloc[:, root_index] <= float(uw)])
                elif uniqueWords[root_index][word] == upper:
                    branch.append(branch[i][branch[i].iloc[:, root_index] > float(uw)])
                else:
                    branch.append(branch[i][branch[i].iloc[:, root_index] == uniqueWords[root_index][word]])

                rule.append(rule[i] + " = " + "{" + uniqueWords[root_index][word] + "}")

            for logic_connection in range(1, len(rule)):
                if (len(np.unique(branch[i][0])) != 1) and (not rule[logic_connection].endswith(" AND ")) and (rule[
                    logic_connection].endswith("}")):
                    rule[logic_connection] = rule[logic_connection] + " AND "
        skip_update = False
        i = i + 1
        print("iteration: ", i)
        stop = len(rule)

    for i in range(len(rule) - 1, -1, -1):
        if not rule[i].endswith(";"):
            del rule[i]

    rule.append("Total Number of Rules: " + str(len(rule)))
    rule.append(dataset.agg(lambda x: x.value_counts().index[0])[0])
    print("End of Iterations")

    return rule


if __name__ == "__main__":
    # read in the data
    database = pd.read_csv("cardio_train.csv", sep=';')
    print(database.head())
    # check number of Nan value in the data base
    print("Number of Nan value in the dataset")
    print(database.isna().sum())
    database.dropna()
    ######## Important Note ####################
    # for testing efficency, we only include the first 1000 of data
    X = database.iloc[:1000, 1:11]
    y = database.iloc[:1000, 12].apply(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = database.iloc[:100, 1:11]
    y_train = database.iloc[:100, 12].apply(str)
    # Run C4.5
    dt_model = dt_c45(Xdata=X_train, ydata=y_train)
    print(dt_model)
    # Prediction
    y_pred = prediction_dt_c45(dt_model, X_test)
    # Confusion Matrix Report
    print("Confusion Matrix without pre_pruning")
    print(classification_report(y_test, y_pred.iloc[:, 0]))

    # Model with pre_pruning
    dt_model_chi_2 = dt_c45(Xdata=X_train, ydata=y_train, pre_pruning="chi_2")
    y_pred_chi_2 = prediction_dt_c45(dt_model_chi_2, X_test)
    print("Confusion Matrix with chi_2 pre_pruning")
    print(classification_report(y_test, y_pred_chi_2.iloc[:, 0]))

    # Model with impurity pre_pruning
    dt_model_impurity = dt_c45(Xdata=X_train, ydata=y_train, pre_pruning="impur")
    y_pred_impurity = prediction_dt_c45(dt_model_impurity, X_test)
    print("Confusion Matrix with impurity value pre_pruning")
    print(classification_report(y_test, y_pred_impurity.iloc[:, 0]))

    # Model with impurity pre_pruning
    dt_model_min = dt_c45(Xdata=X_train, ydata=y_train, pre_pruning="min")
    y_pred_min = prediction_dt_c45(dt_model_min, X_test)
    print("Confusion Matrix with minimum branches pre_pruning")
    print(classification_report(y_test, y_pred_min.iloc[:, 0]))

    # Since we found using chi^2 as pre-pruning method will increase the running time and have no effect on accuracy,
    # we will use chi^2 pre-pruning on larger dataset training.
    # Model with impurity pre_pruning
    print("Testing value for larger training dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_train = database.iloc[:200, 1:11]
    y_train = database.iloc[:200, 12].apply(str)

    dt_model_chi_2_full = dt_c45(Xdata=X_train, ydata=y_train, pre_pruning="chi_2")
    y_pred_chi_2_full = prediction_dt_c45(dt_model_chi_2_full, X_test)
    print("Confusion Matrix with chi_2 pre_pruning")
    print(classification_report(y_test, y_pred_chi_2_full.iloc[:, 0]))

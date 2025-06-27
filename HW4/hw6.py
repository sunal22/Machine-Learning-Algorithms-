import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",", dtype = "int")
true_labels = (true_labels == 1).astype(int)
predicted_probabilities1 = np.genfromtxt("hw06_predicted_probabilities1.csv", delimiter = ",")
predicted_probabilities2 = np.genfromtxt("hw06_predicted_probabilities2.csv", delimiter = ",")



# STEP 3
# given the predicted probabilities of size (N,),
# it should return the calculated threshold_values of size (N + 1,)
def calculate_threholds(predicted_probabilities):
    
    sorted_probablities = np.sort(np.unique(predicted_probabilities))
    middle_point = (sorted_probablities[:-1] + sorted_probablities[1:]) / 2
    beginning = (0.0 + sorted_probablities[0]) / 2
    end_point = (1.0 + sorted_probablities[-1]) / 2
    threshold_values = np.concatenate(([beginning], middle_point, [end_point]))
    
    return threshold_values

threshold_values1 = calculate_threholds(predicted_probabilities1)
print(threshold_values1)

threshold_values2 = calculate_threholds(predicted_probabilities2)
print(threshold_values2)



# STEP 4
# given the true labels of size (N,), the predicted probabilities of size (N,) and
# the threshold_values of size (N + 1,), it should return the FP and TP rates of size (N + 1,)
def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, threshold_values):
    

    fp_rates = []
    tp_rates = []

    for threshold_value in threshold_values:
        predicted_labels = (predicted_probabilities >= threshold_value).astype(int)

        true_positive = np.sum((predicted_labels == 1) & (true_labels == 1))
        false_positive = np.sum((predicted_labels == 1) & (true_labels == 0))
        false_negative = np.sum((predicted_labels == 0) & (true_labels == 1))
        true_negative = np.sum((predicted_labels == 0) & (true_labels == 0))

        TPRate = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        FPRate = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0.0

        tp_rates.append(TPRate)
        fp_rates.append(FPRate)
    
    return np.array(fp_rates), np.array(tp_rates)

fp_rates1, tp_rates1 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities1, threshold_values1)
print(fp_rates1[495:505])
print(tp_rates1[495:505])

fp_rates2, tp_rates2 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities2, threshold_values2)
print(fp_rates2[495:505])
print(tp_rates2[495:505])

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates1, tp_rates1, label = "Classifier 1")
plt.plot(fp_rates2, tp_rates2, label = "Classifier 2")
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend()
plt.show()
fig.savefig("hw06_roc_curves.pdf", bbox_inches = "tight")


# STEP 5
# given the FP and TP rates of size (N + 1,),
# it should return the area under the ROC curve
def calculate_auroc(fp_rates, tp_rates):
    
    auroc = -np.trapezoid(tp_rates, fp_rates)
    
    return auroc
auroc1 = calculate_auroc(fp_rates1, tp_rates1)
print("The area under the ROC curve for Algorithm 1 is {}.".format(auroc1))
auroc2 = calculate_auroc(fp_rates2, tp_rates2)
print("The area under the ROC curve for Algorithm 2 is {}.".format(auroc2))
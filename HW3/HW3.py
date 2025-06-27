import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",", skip_header = 1)

# get X and y values
X_train = data_set_train[:, 0:2]
y_train = data_set_train[:, 2]
X_test = data_set_test[:, 0:2]
y_test = data_set_test[:, 2]



# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_features = {}
    node_splits = {}
    node_means = {}
    is_terminal = {}
    need_split = {}

    N_train, D = X_train.shape  
    #i put the all training nodes into the root node (node 1) , so initialized the root node
    node_indices = {1: np.arange(N_train)}
    is_terminal[1] = False
    need_split[1] = True

    # your implementation starts below

    while True:
        split_nodes = [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break

        for split_node in split_nodes:
            index_of_data = node_indices[split_node]
            need_split[split_node] = False


            if len(index_of_data) <= P:
                is_terminal[split_node] = True
                node_means[split_node] = np.mean(y_train[index_of_data])
                continue

            best_split_score_MSE = float("inf") 
            best_split_feature = None
            best_split_value = None

            
            for index_feature in range(X_train.shape[1]):
                unique_values = np.sort(np.unique(X_train[index_of_data, index_feature]))
                split_positions = (unique_values[:-1] + unique_values[1:]) / 2

                for split_value in split_positions:
                    left_indices = index_of_data[X_train[index_of_data, index_feature] > split_value]
                    right_indices = index_of_data[X_train[index_of_data, index_feature] <= split_value]

                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue

                    left_mean = np.mean(y_train[left_indices])
                    right_mean = np.mean(y_train[right_indices])
                    MSE_left = np.mean((y_train[left_indices] - left_mean) ** 2)
                    MSE_right = np.mean((y_train[right_indices] - right_mean) ** 2)
                    weighted_MSE = len(left_indices) * MSE_left + len(right_indices) * MSE_right

                    if weighted_MSE < best_split_score_MSE:
                        best_split_score_MSE = weighted_MSE
                        best_split_feature = index_feature
                        best_split_value = split_value

            if best_split_feature is None:
                is_terminal[split_node] = True
                node_means[split_node] = np.mean(y_train[index_of_data])
                continue

           
            node_features[split_node] = best_split_feature
            node_splits[split_node] = best_split_value
            is_terminal[split_node] = False

            
            left_indices = index_of_data[X_train[index_of_data, best_split_feature] > best_split_value]
            right_indices = index_of_data[X_train[index_of_data, best_split_feature] <= best_split_value]

            node_indices[2 * split_node] = left_indices
            is_terminal[2 * split_node] = False
            need_split[2 * split_node] = True

            node_indices[2 * split_node + 1] = right_indices
            is_terminal[2 * split_node + 1] = False
            need_split[2 * split_node + 1] = True

   
    for node in node_indices:
        if is_terminal[node]:
            node_means[node] = np.mean(y_train[node_indices[node]])

    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)


def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    y_hat = []

    for row in X_query:
        node = 1 
       
        while not is_terminal[node]:
            feature_index = node_features[node]
            threshold_value = node_splits[node]

            if row[feature_index] > threshold_value:
                node = 2 * node
            else:
                node = 2 * node + 1

        y_hat.append(node_means[node])

    return np.array(y_hat)



# STEP 4
# assuming that there are T terminal nodes
# should print T rule_set sets as described
def extract_rule_set_sets(is_terminal, node_features, node_splits, node_means):
    for node in is_terminal:
        if not is_terminal[node]:
            continue

        decision_tree_path = []
        current_node = node

        while current_node > 1:
            parent_node = current_node // 2
            is_left_child = (current_node % 2 == 0)
            feature = node_features[parent_node] + 1  
            split_value = node_splits[parent_node]

            rule_set = f"x{feature} > {split_value:.2f}" if is_left_child else f"x{feature} <= {split_value:.2f}"
            decision_tree_path.append(rule_set)
            current_node = parent_node

        decision_tree_path.reverse()
        print(f"Node {node:02}: {decision_tree_path} => {node_means[node]}")
    

P = 256
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_set_sets(is_terminal, node_features, node_splits, node_means)

P_set = [2, 4, 8, 16, 32, 64, 128, 256]
rmse_train = []
rmse_test = []
for P in P_set:
    is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)

    y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
    rmse_train.append(np.sqrt(np.mean((y_train - y_train_hat)**2)))

    y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
    rmse_test.append(np.sqrt(np.mean((y_test - y_test_hat)**2)))

fig = plt.figure(figsize = (8, 4))
plt.semilogx(P_set, rmse_train, "ro-", label = "train", base = 2)
plt.semilogx(P_set, rmse_test, "bo-", label = "test", base = 2)
plt.legend()
plt.xlabel("$P$")
plt.ylabel("RMSE")
plt.show()
fig.savefig("decision_tree_P_comparison.pdf", bbox_inches = "tight")

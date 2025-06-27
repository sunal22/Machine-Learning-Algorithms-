import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw03_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw03_data_set_test.csv", delimiter = ",", skip_header = 1)

# get X and y values
X_train = data_set_train[:, 0:2]
y_train = data_set_train[:, 2]
X_test = data_set_test[:, 0:2]
y_test = data_set_test[:, 2]

minimum_value = -2.0
maximum_value = +2.0

def plot_figure(y, y_hat):
    fig = plt.figure(figsize = (4, 4))
    plt.axline([-12, -12], [52, 52], color = "r", linestyle = "--")
    plt.plot(y, y_hat, "k.")
    plt.xlabel("True value ($y$)")
    plt.ylabel("Predicted value ($\widehat{y}$)")
    plt.xlim([-12, 52])
    plt.ylim([-12, 52])
    plt.show()
    return(fig)


# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(X_query, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders):
    # your implementation starts below

    y_hat = np.zeros(X_query.shape[0])

    for i in range(X_query.shape[0]):
        x1_coor_query, x2_coor_query = X_query[i]

        for j in range(len(x1_right_borders)):
            if (x1_left_borders[j] <= x1_coor_query < x1_right_borders[j]) and (x2_left_borders[j] <= x2_coor_query < x2_right_borders[j]):

                points_in_bin = np.where(
                    (x1_left_borders[j] <= X_train[:, 0]) & (X_train[:, 0] < x1_right_borders[j]) &
                    (x2_left_borders[j] <= X_train[:, 1]) & (X_train[:, 1] < x2_right_borders[j]) )[0]

                if len(points_in_bin) > 0:
                    y_hat[i] = np.mean(y_train[points_in_bin])
                else:
                    y_hat[i] = 0  

                break  

    # your implementation ends above
    return(y_hat)
    
bin_width = 0.50

left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

x1_left_borders = np.meshgrid(left_borders, left_borders)[0].flatten()
x1_right_borders = np.meshgrid(right_borders, right_borders)[0].flatten()
x2_left_borders = np.meshgrid(left_borders, left_borders)[1].flatten()
x2_right_borders = np.meshgrid(right_borders, right_borders)[1].flatten()

y_train_hat = regressogram(X_train, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Regressogram => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("regressogram_training.pdf", bbox_inches = "tight")

y_test_hat = regressogram(X_test, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("regressogram_test.pdf", bbox_inches = "tight")



# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(X_query, X_train, y_train, bin_width):
    # your implementation starts below

    N_query_points = X_query.shape[0]
    y_hat = np.zeros(N_query_points)

    for i in range(N_query_points):
        x1_coor_query, x2_coor_query = X_query[i]
        sum_bins = 0
        num_bins = 0

        for j in range(X_train.shape[0]):
            x1_vertical, x2_horizontal = X_train[j]

            if abs(x1_coor_query - x1_vertical) < (bin_width / 2) and abs(x2_coor_query - x2_horizontal) < (bin_width / 2):
                sum_bins += y_train[j]
                num_bins += 1

        if num_bins > 0:
            y_hat[i] = sum_bins / num_bins
        else:
            y_hat[i] = 0

    # your implementation ends above
    return (y_hat)

    
bin_width = 0.50


y_train_hat = running_mean_smoother(X_train, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Running Mean Smoother => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("running_mean_smoother_training.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(X_test, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("running_mean_smoother_test.pdf", bbox_inches = "tight")



# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(X_query, X_train, y_train, bin_width):
    # your implementation starts below

    N_query_points = X_query.shape[0]
    N_train_points = X_train.shape[0]
    D = X_train.shape[1]  

    y_hat = np.zeros(N_query_points)

    for i in range(N_query_points):
        n = 0
        d = 0

        for j in range(N_train_points):
            
            u_1 = (X_query[i][0] - X_train[j][0]) / bin_width
            u_2 = (X_query[i][1] - X_train[j][1]) / bin_width

           
            squared_of_u = u_1 ** 2 + u_2 ** 2

            
            kernel_value = (1 / (2 * np.pi)) * np.exp(-0.5 * squared_of_u)

            n += kernel_value * y_train[j]
            d += kernel_value

        
        if d > 0:
            y_hat[i] = n / d
        else:
            y_hat[i] = 0

    # your implementation ends above
    return y_hat
bin_width = 0.08


y_train_hat = kernel_smoother(X_train, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Kernel Smoother => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("kernel_smoother_training.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(X_test, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("kernel_smoother_test.pdf", bbox_inches = "tight")

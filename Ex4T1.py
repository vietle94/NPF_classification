import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as pl


# _____________________________________________________________________________
def list_diff(a, b):
    s = set(b)
    return np.array([x for x in a if x not in s])


def choose_n_random_variables(n, data_x, data_y, replace=True):
    """
    Splits the data into "training set" and "testing set". The training set will
    contain n values. If replace is True, there can be same values multiple times
    from the data that is to be sampled. If it is False, the random.choice
    function does not replace values already taken from the sublist, resulting
    in the training set containing unique values, and
    len(trainin set) = len(testing set)
    :param n:
    :param data_x:
    :param data_y:
    :param replace:
    :return:
    """
    # Training set index
    i_tr = np.random.choice(len(data_x), n, replace=replace)
    # test set index
    i_te = list_diff(range(len(data_x)), i_tr)
    # Training set
    set_tr = [data_x.iloc[i_tr], data_y.iloc[i_tr]]
    # Testing set
    set_te = [data_x.iloc[i_te], data_y.iloc[i_te]]

    return set_tr, set_te


def set_integer_tick_frequency(ax, stepsize, x_or_y='x'):
    """
    Control the tick frequency of input axes ax by setting the step
    size between each tick with input stepsize. The
    ticks should be integer ticks!
    :param ax:          axes, axes object whose tick frequency is
                        modified
    :param stepsize:    integer, the step size between each ticks
                        (e.g. if stepsize=2 -> ticks 0, 2, 4, 6, ...)
    :param x_or_y:      string, determines which axis tick should be
                        modified. Default 'x' -> x-ticks are modified
                        if 'y' -> y-ticks are modified. In all other
                        cases x-ticks are modified
    :return:
    """
    start, end = ax.get_xlim()

    oaxis = ax.xaxis
    if x_or_y == 'y':
        oaxis = ax.yaxis
        start, end = ax.get_ylim()

    oaxis.set_ticks(np.arange(np.floor(start), np.ceil(end), stepsize))
    return


def do_fit(model, data_training, data_test):
    """
    Does the fit and calculates relevant values. Assumes x data
    is in loc 0 and y data in loc 1 (X_training = data_training[0])
    :param model:
    :param data_training:
    :param data_test:
    :return:            list, values of the coefficients from the model
                        list, values of estimated y
                        float, mean squared error
                        float, R^2 value of the fit
    """
    lm = model()
    fit_lm = lm.fit(X=data_training[0], y=data_training[1])

    # Predict y values
    y_predicted = fit_lm.predict(data_test[0])
    coeffs = fit_lm.coef_
    MSE = mean_squared_error(y_pred=y_predicted,
                             y_true=data_test[1])
    R2 = r2_score(y_true=data_test[1], y_pred=y_predicted)

    return coeffs, y_predicted, MSE, R2, fit_lm


def plot_fit_results(fig, ax, y_pred, y_true, R2, MSE, title, marker='.r'):
    """

    :param fig:
    :param ax:
    :param y_pred:
    :param y_true:
    :param R2:
    :param MSE:
    :return:
    """

    ax.plot(y_pred, y_true, marker,
            label='$R^2$=%.2f, loss=%.2f' % (R2, MSE))
    ax.set_title(title)

    set_integer_tick_frequency(ax, 1, 'y')
    ax.set_ylim([-0.1, 1.1])

    fig.tight_layout()
    return


def estimate_accuracy(y_predicted, y_true):
    """

    :param y_predicted:
    :param y_true:
    :return:
    """
    return np.sum(y_predicted == y_true)/len(y_true)
# _____________________________________________________________________________
data = pd.read_csv("../data/npf_train_full.csv", index_col=0)
# get rid of HYY_META
data.columns = list(map(lambda x: x.replace("HYY_META.", ""), data.columns))

# Choose height to look at
height = 84

# Drop "partlybad" column, as it contains no info based on Ex1T3
data.drop('partlybad', axis=1, inplace=True)

# dummy variable for event=1 or nonevent=0
event = data.event.copy().values
event[event != 'nonevent'] = 1
event[event == 'nonevent'] = 0
data.loc[:, 'isevent'] = event

# Set the dtype of isevent to int64
data['isevent'] = data['isevent'].astype('int64')

# Choosing explanatory variables (choose correct height, drop std)
explanatory_variables = ['RH', 'CS']
keys_X = [k for k in data.keys() if ((explanatory_variables[0] in k
                                    and str(height) in k)
                                   or explanatory_variables[1] in k)
        and 'std' not in k]
# add dummy variable
keys = keys_X.copy()
keys.append('isevent')

# Choose only relevant data
data_relevant = data[keys].copy()
# Reset index
data_relevant.reset_index(inplace=True, drop=True)

n_data_points = len(data_relevant.index.values)

# Split to training and testing data 1/3
set_tr, set_test = choose_n_random_variables(int(2*n_data_points/3),
                                             data_relevant[keys_X],
                                             data_relevant['isevent'],
                                             replace=False)

# Do linear regression
coeffs, y_predicted, MSE, R2, fit = do_fit(LinearRegression, set_tr, set_test)

# Do logistic regression
coeffs_log, y_predicted_log, MSE_log, R2_log, fit_log = do_fit(LogisticRegression,
                                                      set_tr, set_test)

# Predict probability distribution, output [samples, classes]
prob_predicted_log = fit_log.predict_proba(X=set_test[0])

# Calculate accuracy
accuracy = estimate_accuracy(y_predicted_log, set_test[1])


# ___PLOTTING___
fig, ax = pl.subplots(num='lin reg', figsize=[6, 6])
fig2, ax2 = pl.subplots(ncols=2, num='log prob', figsize=[8, 6],
                         sharey=True)

plot_fit_results(fig, ax, y_predicted, set_test[1], R2, MSE,
                 title='Linear regression: true event as a '
                       'function of predicted event')

for i, ax_i in enumerate(ax2):
    ax_i.plot(np.sort(prob_predicted_log[:, i]), '.')
    ax_i.set_title('Sorted predicted probability\nconditioned on y=%i' %i)
    ax_i.set_xlabel('Sample')
ax2[0].legend(['Accuracy %.3f' % accuracy])
ax2[0].set_ylabel('Probability')
fig2.tight_layout()

fig.savefig('../figures/Ex4T1_LinearRegression_predVsTrue.png')
fig2.savefig('../figures/Ex4T1_LogisticRegression.png')
pl.show()

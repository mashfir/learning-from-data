import numpy as np
from scipy import stats

class target_func:
    """
    Generate target function
    """
    def __init__(self):
        self._p1 = np.random.uniform(-1.0, 1.0, 2)
        self._p2 = np.random.uniform(-1.0, 1.0, 2)

    def test_pt(self, x):
        v1 = self._p2 - self._p1
        v2 = x - self._p1
        return np.sign(np.cross(v1, v2, axisa=0))

def linear_regressor(X, y):
    """
    For a given input array X and output array y, return weight vector w
    such that w minimizes least squared error
    """
    X_dag = np.linalg.pinv(X)
    return np.matmul(X_dag, y)

def add_x0(X):
    X_new = np.ones((X.shape[0], X.shape[1]+1))
    X_new[:, 1:] = X
    return X_new

def err_simulator(runs, num_train_pts, num_test_pts):
    """
    Defines average E_in for num_train_pts simulated data points over given number of runs
    """
    E_in = np.zeros(runs)
    E_out = np.zeros(runs)
    for i in range(runs):
        test_func = target_func()
        X_train = np.random.uniform(-1, 1, (num_train_pts, 2))
        y_train = test_func.test_pt(X_train)
        X_train = add_x0(X_train)
        X_test = np.random.uniform(-1, 1, (num_test_pts, 2))
        y_test = test_func.test_pt(X_test)
        X_test = add_x0(X_test)
        w = linear_regressor(X_train, y_train)
        g_train = np.matmul(X_train, w)
        g_test = np.matmul(X_test, w)
        E_in[i] = np.mean(np.not_equal(np.sign(g_train), y_train))
        E_out[i] = np.mean(np.not_equal(np.sign(g_test), y_test))

    return np.mean(E_in), np.mean(E_out)

def main():
    np.random.seed(1)
    E_in, E_out = err_simulator(1000, 100, 1000)
    print("E_in: {}\nE_out: {}".format(E_in, E_out))

if __name__ == '__main__':
    main()

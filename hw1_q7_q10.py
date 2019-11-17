import numpy as np

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
        return np.sign(np.cross(v1, v2))

def pla(X, y, iter_limit=1000):
    w = np.zeros(3)
    learned = False
    iters = 0
    X_y = np.c_[X, y]

    def update_weights():
        nonlocal w, X_y
        np.random.shuffle(X_y)
        X_shuffled = X_y[:, :-1]
        y_shuffled = X_y[:, -1]
        for i, x_i in enumerate(X_shuffled):
            h = np.sign(np.matmul(w.T, x_i.T))
            if h != y_shuffled[i]:
                w += y_shuffled[i] * x_i.T
                return False
        return True

    while not learned:
        if iters < iter_limit:
            iters += 1
            learned = update_weights()
        else:
            break

    return w, iters

def add_x0(X):
    X_new = np.ones((X.shape[0], X.shape[1]+1))
    X_new[:, 1:] = X
    return X_new

def err_simulator(runs, num_train_pts, num_test_pts):
    """
    Defines average E_in for num_train_pts simulated data points over given
    number of runs
    """
    total_iters = np.zeros(runs)
    E_out = np.zeros(runs)
    for i in range(runs):
        test_func = target_func()
        X_train = np.random.uniform(-1, 1, (num_train_pts, 2))
        y_train = test_func.test_pt(X_train)
        X_train = add_x0(X_train)
        X_test = np.random.uniform(-1, 1, (num_test_pts, 2))
        y_test = test_func.test_pt(X_test)
        X_test = add_x0(X_test)
        w, iters = pla(X_train, y_train)
        g_train = np.sign(np.matmul(X_train, w))
        g_test = np.sign(np.matmul(X_test, w))
        total_iters[i] = iters
        E_out[i] = np.mean(np.not_equal(g_test, y_test))

    return np.mean(total_iters), np.mean(E_out)

def main():
    np.random.seed(1)
    avg_iters, avg_E_out = err_simulator(1000, 10, 1000)
    print("N: 10, Avg iterations: {}, Avg error: {}".format(avg_iters, avg_E_out))
    avg_iters, avg_E_out = err_simulator(1000, 100, 1000)
    print("N: 100, Avg iterations: {}, Avg error: {}".format(avg_iters, avg_E_out))

if __name__ == '__main__':
    main()

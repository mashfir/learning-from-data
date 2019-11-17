import numpy as np

def flip_coin(flips=1):
    return np.random.randint(2, size=flips)

def sim_coin_flips(coins, flips):
    flip_log = np.empty([coins, flips])
    for coin in range(coins):
        flip_log[coin] = flip_coin(flips)
    return flip_log

def sim_fractions(coins, flips):
    sim_result = sim_coin_flips(coins, flips)
    sim_means = sim_result.mean(axis=1)
    v_1 = sim_means[1]
    v_rand = sim_means[np.random.randint(sim_means.size)]
    v_min = sim_means.min()
    return np.array([v_1, v_rand, v_min])

def main():
    np.random.seed(1)
    simulations = 100000
    sim_dist = np.empty([simulations, 3])
    for sim in range(simulations):
        sim_dist[sim] = sim_fractions(1000, 10)
    print(sim_dist.mean(axis=0))

if __name__ == '__main__':
    main()

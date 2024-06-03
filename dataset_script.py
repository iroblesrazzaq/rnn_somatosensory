
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch import optim as optim
import matplotlib.pyplot as plt


def apply_random_unitary_trans(X: np.array, random_rotation: np.array):
    res = np.einsum('ijk,kl->ijl', X, random_rotation)
    return res


def gen_random_unitary_trans() -> np.array:
    input_dim = 3 # hard coded so doesn't need input array for dimensions
    random_rotation = np.random.randn(input_dim, input_dim)
    q, r = np.linalg.qr(random_rotation)
    random_rotation = q  # q is a unitary matrix
    return random_rotation


# todo: add transformations -- after datasets created so rotation is uniform


def make_training_set() -> tuple[np.array]:
    # Training Set Parameters
    conditions = [[1, 1, 1], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]]
    num_trials_per_cond = 400
    num_trials = num_trials_per_cond * 8
    time_steps = 20
    input_size = 3

    # making training dataset
    X = np.zeros((num_trials, time_steps, input_size), dtype=np.float32)

    y_easy_c1 = np.zeros(num_trials, dtype=np.int64)
    y_easy_c2 = np.zeros(num_trials, dtype=np.int64)
    y_easy_c3 = np.zeros(num_trials, dtype=np.int64)

    y_med_c12 = np.zeros(num_trials, dtype=np.int64)
    y_med_c13 = np.zeros(num_trials, dtype=np.int64)
    y_med_c23 = np.zeros(num_trials, dtype=np.int64)

    y_hard = np.zeros(num_trials, dtype=np.int64)

    trial_idx = 0
    for condition in conditions:
        c1, c2, c3 = condition
        for _ in range(num_trials_per_cond):
            delay1 = random.randint(0, 9)
            delay2 = random.randint(0, 9)
            delay3 = random.randint(0, 9)
            for step in range(time_steps):
                X[trial_idx][step][0] = 0 if delay1 >= step else bernoulli.rvs(c1)
                X[trial_idx][step][1] = 0 if delay2 >= step else bernoulli.rvs(c2)
                X[trial_idx][step][2] = 0 if delay3 >= step else bernoulli.rvs(c3)
            y_easy_c1[trial_idx] = c1
            y_easy_c2[trial_idx] = c2
            y_easy_c3[trial_idx] = c3
            
            y_med_c12[trial_idx] = c1 ^ c2 
            y_med_c13[trial_idx] = c1 ^ c3
            y_med_c23[trial_idx] = c2 ^ c3

            y_hard[trial_idx] = 0 if (c1 + c2 + c3) % 2 == 0 else 1

            trial_idx += 1

    return X, y_easy_c1, y_easy_c2, y_easy_c3, y_med_c12, y_med_c13, y_med_c23, y_hard



def make_easy_test_set() -> tuple[np.array]:
    """
    Generates test sets for easy tasks

    Input: None
    Output: tuple of:
        X_test_easy: test dataset np array
        y_easy_c1_test: target feature array for c1
        y_easy_c2_test: target feature array for c2
        y_easy_c3_test: target feature array for c3
    """
    # testing set params
    # all lambda values are high lambda values, as input to the bernoulli function is high
    easy_lambda = 0.65
    medium_lambda = 0.7
    hard_lambda = 0.77

    conditions = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    num_trials_per_cond = 40
    num_trials = num_trials_per_cond * 8
    time_steps = 20
    input_size = 3

    # making test set for easy task

    # Initialize arrays for test inputs and targets
    X_test_easy = np.zeros((num_trials, time_steps, input_size), dtype=np.float32)
    y_easy_c1_test = np.zeros(num_trials, dtype=np.int64)
    y_easy_c2_test = np.zeros(num_trials, dtype=np.int64)
    y_easy_c3_test = np.zeros(num_trials, dtype=np.int64)

    trial_idx = 0
    for condition in conditions:
        cond1, cond2, cond3 = condition
        c1 = easy_lambda if cond1 == 1 else 1- easy_lambda
        c2 = easy_lambda if cond2 == 1 else 1- easy_lambda
        c3 = easy_lambda if cond3 == 1 else 1- easy_lambda
        for _ in range(num_trials_per_cond):
            delay1 = random.randint(0, 9)
            delay2 = random.randint(0, 9)
            delay3 = random.randint(0, 9)
            for step in range(time_steps):
                X_test_easy[trial_idx][step][0] = 0 if delay1 >= step else bernoulli.rvs(c1)
                X_test_easy[trial_idx][step][1] = 0 if delay2 >= step else bernoulli.rvs(c2)
                X_test_easy[trial_idx][step][2] = 0 if delay3 >= step else bernoulli.rvs(c3)

            y_easy_c1_test[trial_idx] = cond1
            y_easy_c2_test[trial_idx] = cond2   
            y_easy_c3_test[trial_idx] = cond3
            
            trial_idx += 1
    return X_test_easy, y_easy_c1_test, y_easy_c2_test, y_easy_c3_test


def make_med_test_set() -> tuple[np.array]:
    """
    Generates test sets for medium tasks

    Input: None
    Output: tuple of:
        X_test_med: test dataset np array
        y_med_c12_test: target feature array for c1 - c2 XOR
        y_med_c13_test: target feature array for c1 - c3 XOR
        y_med_c23_test: target feature array for c2 - c3 XOR
    """
    # testing set params
    # all lambda values are high lambda values, as input to the bernoulli function is high
    easy_lambda = 0.65
    medium_lambda = 0.7
    hard_lambda = 0.77

    conditions = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    num_trials_per_cond = 40
    num_trials = num_trials_per_cond * 8
    time_steps = 20
    input_size = 3
    # making test set for medium task

    # Initialize arrays for test inputs and targets
    X_test_med = np.zeros((num_trials, time_steps, input_size), dtype=np.float32)
    y_med_c12_test = np.zeros(num_trials, dtype=np.int64)
    y_med_c13_test = np.zeros(num_trials, dtype=np.int64)
    y_med_c23_test = np.zeros(num_trials, dtype=np.int64)

    trial_idx = 0
    for condition in conditions:
        cond1, cond2, cond3 = condition
        c1 = medium_lambda if cond1 == 1 else 1 - medium_lambda
        c2 = medium_lambda if cond2 == 1 else 1 - medium_lambda
        c3 = medium_lambda if cond3 == 1 else 1 - medium_lambda

        for _ in range(num_trials_per_cond):
            delay1 = random.randint(0, 9)
            delay2 = random.randint(0, 9)
            delay3 = random.randint(0, 9)
            for step in range(time_steps):
                X_test_med[trial_idx][step][0] = 0 if delay1 >= step else bernoulli.rvs(c1)
                X_test_med[trial_idx][step][1] = 0 if delay2 >= step else bernoulli.rvs(c2)
                X_test_med[trial_idx][step][2] = 0 if delay3 >= step else bernoulli.rvs(c3)

            y_med_c12_test[trial_idx] = cond1 ^ cond2
            y_med_c13_test[trial_idx] = cond1 ^ cond3        
            y_med_c23_test[trial_idx] = cond2 ^ cond3

            trial_idx += 1

    return X_test_med, y_med_c12_test, y_med_c13_test, y_med_c23_test


def make_hard_test_set() -> tuple[np.array]:
    """
    Generates test sets for complex task

    Input: None
    Output: tuple of:
        X_test_hard: test dataset np array
        y_test_hard: target feature array for 3D parity task
        
    """
    # testing set params
    # all lambda values are high lambda values, as input to the bernoulli function is high
    easy_lambda = 0.65
    medium_lambda = 0.7
    hard_lambda = 0.77

    conditions = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    num_trials_per_cond = 40
    num_trials = num_trials_per_cond * 8
    time_steps = 20
    input_size = 3
    # making test set for hard task

    # Initialize arrays for test inputs and targets
    X_test_hard = np.zeros((num_trials, time_steps, input_size), dtype=np.float32)
    y_test_hard = np.zeros(num_trials, dtype=np.int64)

    trial_idx = 0
    for condition in conditions:
        cond1, cond2, cond3 = condition
        c1 = hard_lambda if cond1 == 1 else 1 - hard_lambda
        c2 = hard_lambda if cond2 == 1 else 1 - hard_lambda
        c3 = hard_lambda if cond3 == 1 else 1 - hard_lambda

        for _ in range(num_trials_per_cond):
            delay1 = random.randint(0, 9)
            delay2 = random.randint(0, 9)
            delay3 = random.randint(0, 9)
            for step in range(time_steps):
                X_test_hard[trial_idx][step][0] = 0 if delay1 >= step else bernoulli.rvs(c1)
                X_test_hard[trial_idx][step][1] = 0 if delay2 >= step else bernoulli.rvs(c2)
                X_test_hard[trial_idx][step][2] = 0 if delay3 >= step else bernoulli.rvs(c3)

            y_test_hard[trial_idx] = 0 if (c1 + c2 + c3) % 2 == 0 else 1


            trial_idx += 1

    return X_test_hard, y_test_hard
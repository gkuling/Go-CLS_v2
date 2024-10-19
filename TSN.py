import matplotlib.pyplot as plt
import numpy as np
import project_team as proteam
from project_team.dt_project.DataProcessors._Processor import _Processor
import pandas as pd
import torch
from tqdm import tqdm

class TSNModel_config(proteam.models.project_config):
    def __init__(self,
                 S_x=100,
                 S_y=1,
                 P=100,
                 N_n_units=2000,
                 N_a=0.05,
                 N_inhibition=0.6,
                 N_U=-0.15,
                 N_ncycle=9,
                 **kwargs):
        super(TSNModel_config, self).__init__('TSNModel', **kwargs)
        self.S_x = S_x
        self.S_y = S_y

        # Notebook parameters
        # see Buhmann, Divko, and Schulten, 1989 for details regarding gamma and U terms
        self.P = P
        self.N_n_units = N_n_units
        self.N_a = N_a
        self.N_inhibition = N_inhibition
        self.N_U = N_U
        self.N_ncycle = N_ncycle

class TSNModel(torch.nn.Module):
    def __init__(self, config):
        super(TSNModel, self).__init__()
        self.config = config
        ## Notebook Network
        # Generate P random binary indices with sparseness a
        self.N_patterns = np.zeros((self.config.P, self.config.N_n_units))
        for i in range(self.config.P):
            self.N_patterns[i , :] = np.random.choice(
                [0, 1],
                size=(1, self.config.N_n_units),
                p=[1 - self.config.N_a, self.config.N_a]
            )
        # Hebbian learning for notebook recurrent weights
        self.W_N = np.matmul(
            (self.N_patterns - self.config.N_a).T,
            (self.N_patterns - self.config.N_a)
        ) / (self.config.N_n_units * self.config.N_a * (1 - self.config.N_a))
        # add global inhibition term, see Buhmann, Divko, and Schulten, 1989
        self.W_N = self.W_N - self.config.N_inhibition / (
                    self.config.N_a * self.config.N_n_units)
        # no self connection
        self.W_N = self.W_N * (~np.eye(self.W_N.shape[0], dtype=bool))

        # Student Network
        self.W_s = np.zeros([self.config.S_x, self.config.S_y])
        # set student's weights, with zero weight initialization

    def forward(self, x):
        return {
            'student': np.matmul(x, self.W_s),
            'notebook': np.matmul(
                np.matmul(
                    np.matmul(x, self.W_S_N_Lin),
                    self.W_N
                ),
                self.W_N_S_Lout
            )
        }

    def W_S_N_Learning(self, x, y):
        # Hebbian learning for Notebook-Student weights (bidirectional)

        norm = (
                self.config.N_n_units * self.config.N_a * (1 - self.config.N_a)
        )
        # Notebook to student weights, for reactivating student
        self.W_N_S_Lin = np.matmul((self.N_patterns - self.config.N_a).T, x)
        self.W_N_S_Lin /= norm

        self.W_N_S_Lout = np.matmul((self.N_patterns - self.config.N_a).T, y)
        self.W_N_S_Lout /= norm

        # student to notebook weights, for providing partial cues
        self.W_S_N_Lin = np.matmul(x.T, (self.N_patterns - self.config.N_a))
        self.W_S_N_Lout = np.matmul(y.T, (self.N_patterns - self.config.N_a))

    def pattern_completion_through_recurrent_activation(self):
        # Dynamic threshold
        Activity_dyn_t = np.zeros((self.config.P, self.config.N_n_units))

        # First round of pattern completion through recurrent activtion
        # cycles given random initial input.
        for cycle in range(self.config.N_ncycle):
            if cycle <= 1:
                clamp = 1
            else:
                clamp = 0
            rand_patt = (np.random.rand(self.config.P,
                                        self.config.N_n_units) <=
                         self.config.N_a).astype(Activity_dyn_t.dtype)  #
            # random seeding activity
            # Seeding notebook with random patterns
            M_input = Activity_dyn_t + (rand_patt * clamp)
            ### GCK: In the original codes this line was commented out(???):
            ## Seeding notebook with origginal patterns
            ## # M_input = Activity_dyn_t + (N_patterns * clamp)

            # GCK: Are these matrices?
            M_current = np.linalg.matmul(M_input, self.W_N)

            # scale currents between 0 and 1
            scale = 1.0 / (
                    np.max(M_current, axis=1) - np.min(M_current, axis=1)
            )

            M_current = (
                                M_current - np.min(M_current, axis=1)[:,
                                            np.newaxis]
                        ) * scale[:, np.newaxis]

            # find threshold based on desired sparseness
            # Sort each row of M_current in descending order
            sorted_M_current = np.sort(M_current, axis=1)[:,
                               ::-1]  # Sort in descending order along axis 1 (rows)

            # Calculate the index for the threshold based on 'a'
            t_ind = np.floor(
                Activity_dyn_t.shape[1] * self.config.N_a
            ).astype(int)  # Convert to integer

            # Ensure t_ind is at least 1
            t_ind = np.maximum(t_ind, 1)

            # Use t_ind to select the threshold value from each row of sorted_M_current
            t = sorted_M_current[:,
                t_ind - 1]  # -1 for zero-indexing in Python

            # Set Activity_dyn_t based on the threshold comparison
            Activity_dyn_t = (M_current >= t[:, np.newaxis])
        return Activity_dyn_t

    def pattern_completion_with_fix_threshold(self, Actvty_dyn_t):
        # Second round of pattern completion, with fix threshold
        Activity_fix_t = np.zeros((self.config.P, self.config.N_n_units))
        for cycle in range(self.config.N_ncycle):
            if cycle <= 1:
                clamp = 1
            else:
                clamp = 0

            M_input = Activity_fix_t + (Actvty_dyn_t * clamp)

            M_current = np.linalg.matmul(M_input, self.W_N)

            # Set Activity_fix_t based on the threshold comparison
            Activity_fix_t = (M_current >= self.config.N_U)
        return Activity_fix_t

    def seed_the_notebook(self, x_t_input):
        Activity_notebook_train = np.zeros((x_t_input.shape[0], self.config.N_n_units))
        for cycle in range(self.config.N_ncycle):
            if cycle <= 1:
                clamp = 1
            else:
                clamp = 0
            seed_patt = np.matmul(x_t_input, self.W_S_N_Lin)
            M_input = Activity_notebook_train + (seed_patt * clamp)

            M_current = np.matmul(M_input, self.W_N)

            # scale currents between 0 and 1
            scale = 1.0 / (
                    np.max(M_current, axis=1) - np.min(M_current, axis=1)
            )

            M_current = (
                                M_current - np.min(M_current, axis=1)[:,
                                            np.newaxis]
                        ) * scale[:, np.newaxis]

            # find threshold based on desired sparseness
            # Sort each row of M_current in descending order
            sorted_M_current = np.sort(M_current, axis=1)[:,
                               ::-1]  # Sort in descending order along axis 1 (rows)

            # Calculate the index for the threshold based on 'a'
            t_ind = np.floor(self.config.N_n_units * self.config.N_a).astype(
                int)  # Convert to integer

            # Ensure t_ind is at least 1
            t_ind = np.maximum(t_ind, 1)

            # Use t_ind to select the threshold value from each row of sorted_M_current
            t = sorted_M_current[:, t_ind - 1]  # -1 for zero-indexing in Python

            # Set Activity_notebook_train based on the threshold comparison
            Activity_notebook_train = (M_current >= t[:, np.newaxis])
        return np.matmul(Activity_notebook_train, self.W_N_S_Lout)

    def reactivate_pattern(self, pattern):
        return (np.matmul(pattern, self.W_N_S_Lin),
                np.matmul(pattern, self.W_N_S_Lout))

class TSN_DTProcessor_config(proteam.dt_project.DT_config):
    def __init__(self,
                 n_training=100,
                 n_test=1000,
                 SNR = np.inf,
                 input_dim=100,
                 output_dim=1,

                 **kwargs):
        super(TSN_DTProcessor_config, self).__init__('TSN_DTProcessor',
                                                     **kwargs)
        self.n_training = n_training
        self.n_test = n_test
        self.input_dim = input_dim
        self.output_dim = output_dim

        # According to SNR, set variances for teacher's weights (variance_w) and
        # output noise (variance_e) that sum to 1
        self.SNR = SNR
        if SNR == np.inf:
            self.variance_w = 1
            self.variance_e = 0
        else:
            self.variance_w = SNR / (SNR + 1)
            self.variance_e = 1 / (SNR + 1)

class TSN_DTProcessor(_Processor):
    def generate_data(self):
        ## Teacher Network
        W_t = np.random.normal(0,
                               self.config.variance_w ** 0.5,
                               [self.config.input_dim, self.config.output_dim])
        noise_train = np.random.normal(0,
                                       self.config.variance_e ** 0.5,
                                       [self.config.n_training, self.config.output_dim])
        # Training data
        self.x_t_input = np.random.normal(0,
                                          (1 / self.config.input_dim) ** 0.5,
                                          [self.config.n_training, self.config.input_dim])
        self.y_t_output = np.linalg.matmul(self.x_t_input, W_t) + noise_train

        # Testing data
        noise_test = np.random.normal(0,
                                      self.config.variance_e ** 0.5,
                                      [self.config.n_test, self.config.output_dim])
        self.x_t_input_test = np.random.normal(0,
                                               (1 / self.config.input_dim) ** 0.5,
                                               [self.config.n_test, self.config.input_dim])
        self.y_t_output_test = np.linalg.matmul(self.x_t_input_test,
                                                W_t) + noise_test
        self.W_t = W_t

class TSNPractitioner_config(proteam.ml_project.PTPractitioner_config):
    def __init__(self,
                 **kwargs):
        super(TSNPractitioner_config, self ).__init__('TSNPractitioner',
                                                     **kwargs)

class TSNPractitioner(object):
    def __init__(self, model, dt_processor, config=TSNPractitioner_config()):
        self.config = config
        self.model = model
        self.dt_processor = dt_processor

    def get_N_error(self, x, y):
        N_S_output_train = self.model.seed_the_notebook(
            x
        )

        # Notebook training error
        delta_N_train = y - N_S_output_train
        error_N_train = (delta_N_train ** 2).mean()

        # Since notebook errors stay constant throughout training,
        # populating each epoch with the same error value
        error_N_train_vector = np.matmul(
            np.ones((self.config.n_epochs, 1)),
            error_N_train.reshape((1, 1))
        )
        return error_N_train_vector[:, 0]
    def train(self):
        print('')
        # train the input out transforms of the notebook
        self.model.W_S_N_Learning(self.dt_processor.x_t_input,
                                  self.dt_processor.y_t_output)
        # array for storing retrieved notebook patterns, pre-calculating all epochs for speed considerations
        S_error_train = []
        S_error_test = []
        for epoch in tqdm(range(self.config.n_epochs), desc='Epoch'):
            # Simulates hippocampal reactivations with random binary seed patterns.
            # Retrieval process:
            # (1) Dynamic threshold ensures sparse pattern retrieval,
            # preventing silent attractor dominance.
            # (2) Fixed-threshold completes the pattern
            # with global inhibition, allowing varied sparseness.
            # Silent state occurs if seed is far from encoded patterns.
            Activity_dyn_t = \
                self.model.pattern_completion_through_recurrent_activation()

            # Second round of pattern completion, with fix threshold
            N_pattern_reactivated = (
                self.model.pattern_completion_with_fix_threshold(Activity_dyn_t)
            )
            ## Generate offline training data from notebook reactivations
            N_S_input, N_S_output = self.model.reactivate_pattern(N_pattern_reactivated)

            S_prediction = self.model.forward(
                self.dt_processor.x_t_input
            )['student']
            S_prediction_test = self.model.forward(
                self.dt_processor.x_t_input_test
            )['student']

            # Train error
            delta_train = self.dt_processor.y_t_output - S_prediction
            S_error_train.append((delta_train ** 2).mean())

            # Generalization error
            delta_test = self.dt_processor.y_t_output_test - S_prediction_test
            S_error_test.append((delta_test ** 2).mean())

            # Grad descent
            w_delta = np.matmul(N_S_input.T, N_S_output)
            w_delta -= self.model.forward(
                np.matmul(N_S_input.T, N_S_input)
            )['student']
            self.model.W_s = self.model.W_s + self.config.lr * w_delta

        # Seed the notebook with original patterns to calculate training error.
        # Seed with student input via Notebook weights, complete the pattern,
        # and use the retrieved pattern to activate the student's output through
        # Notebook-to-Student weights.
        N_training_error = self.get_N_error(self.dt_processor.x_t_input,
                                            self.dt_processor.y_t_output)
        N_test_error = self.get_N_error(self.dt_processor.x_t_input_test,
                                        self.dt_processor.y_t_output_test)
        return {'student_train_error': np.array(S_error_train),
                'student_test_error': np.array(S_error_test),
                'notebook_train_error': N_training_error,
                'notebook_test_error': N_test_error}



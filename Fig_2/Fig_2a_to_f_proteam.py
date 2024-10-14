import matplotlib.pyplot as plt
import numpy as np
import project_team as proteam
from project_team.dt_project.DataProcessors._Processor import _Processor
import pandas as pd
import torch
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
                size=(1, self.config.M),
                p=[1 - self.config.N_a, self.config.N_a]
            )
        # Hebbian learning for notebook recurrent weights
        self.W_N = np.dot(
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
            'studnet': np.dot(x, self.W_s),
            'notebook': np.dot(
                np.dot(
                    np.dot(self.W_S_N_Lin, x),
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
        self.W_N_S_Lin = np.dot((self.N_patterns - self.config.N_a).T, x)
        self.W_N_S_Lin /= norm

        self.W_N_S_Lout = np.dot((self.N_patterns - self.config.N_a).T, y)
        self.W_N_S_Lout /= norm

        # student to notebook weights, for providing partial cues
        self.W_S_N_Lin = np.dot(x.T, (self.N_patterns - self.config.N_a))
        self.W_S_N_Lout = np.dot(y.T, (self.N_patterns - self.config.N_a))

    def pattern_completion_through_recurrent_activation(self):
        # Dynamic threshold
        Activity_dyn_t = np.zeros((self.config.P, self.config.N_n_units))

        print(
            'First round of pattern completion through recurrent activtion '
            'cycles given random initial input.')
        # First round of pattern completion through recurrent activtion
        # cycles given random initial input.
        for cycle in range(self.config.N_ncycle):
            print('Cycle: ' + str(cycle))
            if cycle <= 1:
                clamp = 1
            else:
                clamp = 0
            # GCK: Is the clamp correct???
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

    def pattern_completion_with_fix_threshold(self):
        # Second round of pattern completion, with fix threshold
        Activity_fix_t = np.zeros((self.config.P, self.config.N_n_units))
        for cycle in range(self.config.N_ncycle):
            print('Cycle: ' + str(cycle))
            if cycle <= 1:
                clamp = 1
            else:
                clamp = 0

            M_input = Activity_fix_t + (self.N_patterns * clamp)

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
            t = self.config.N_U

            # Set Activity_fix_t based on the threshold comparison
            Activity_fix_t = (M_current >= t)
        return Activity_fix_t

    def seed_the_notebook(self, x_t_input):
        Activity_notebook_train = np.zeros((self.config.P, self.config.N_n_units))
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
                               self.variance_w ** 0.5,
                               [self.input_dim, self.output_dim])
        noise_train = np.random.normal(0,
                                       self.variance_e ** 0.5,
                                       [self.n_training, self.output_dim])
        # Training data
        self.x_t_input = np.random.normal(0,
                                          (1 / self.input_dim) ** 0.5,
                                          [self.n_training, self.input_dim])
        self.y_t_output = np.linalg.matmul(self.x_t_input, W_t) + noise_train

        # Testing data
        noise_test = np.random.normal(0,
                                      self.variance_e ** 0.5,
                                      [self.n_test, self.output_dim])
        self.x_t_input_test = np.random.normal(0,
                                               (1 / self.input_dim) ** 0.5,
                                               [self.n_test, self.input_dim])
        self.y_t_output_test = np.linalg.matmul(self.x_t_input_test,
                                                W_t) + noise_test


class TSNPractitioner_config(proteam.ml_project.PTPractitioner_config):
    def __init__(self,
                 **kwargs):
        super(TSNPractitioner_config, self ).__init__('TSNPractitioner',
                                                     **kwargs)

class TSNPractitioner(object):
    def __init__(self, config, model, dt_processor):
        self.config = config
        self.model = model
        self.dt_processor = dt_processor

    def train(self):
        print('')
        # train the input out transforms of the notebook
        self.model.W_S_N_Learning(self.dt_processor.x_t_input,
                                  self.dt_processor.y_t_output)
        ## Generate offline training data from notebook reactivations
        N_patterns_reactivated = np.zeros((P, M, nepoch))
        # array for storing retrieved notebook patterns, pre-calculating all epochs for speed considerations

        for epoch in range(self.config.n_epochs):
            print('Epoch: ' + str(epoch))
            # Simulates hippocampal reactivations with random binary seed patterns.
            # Retrieval process:
            # (1) Dynamic threshold ensures sparse pattern retrieval,
            # preventing silent attractor dominance.
            # (2) Fixed-threshold completes the pattern
            # with global inhibition, allowing varied sparseness.
            # Silent state occurs if seed is far from encoded patterns.
            self.model.pattern_completion_through_recurrent_activation()
            # Second round of pattern completion, with fix threshold
            N_patterns_reactivated[:, :, epoch] = (
                self.model.pattern_completion_with_fix_threshold())

        # Seed the notebook with original patterns to calculate training error.
        # Seed with student input via Notebook weights, complete the pattern,
        # and use the retrieved pattern to activate the student's output through
        # Notebook-to-Student weights.
        N_S_output_train= self.model.seed_the_notebook(
            self.dt_processor.x_t_input
        )

        # Notebook training error
        delta_N_train = self.dt_processor.y_t_output - N_S_output_train
        error_N_train = (delta_N_train ** 2).mean()

        # Since notebook errors stay constant throughout training,
        # populating each epoch with the same error value
        error_N_train_vector = np.matmul(
            np.ones((self.config.n_epochs, 1)),
            error_N_train.reshape((1, 1))
        )
        N_train_error_all[r, :] = error_N_train_vector[:, 0]

        # Notebook generalization error
        N_S_output_test = self.model.seed_the_notebook(
            self.dt_processor.x_t_input_test
        )
        # Notebook test error
        delta_N_test = self.dt_processor.y_t_output_test - N_S_output_test
        error_N_test = (delta_N_test ** 2).mean()
        # populating each epoch with the same error value
        error_N_test_vector = (np.ones((self.config.n_epochs, 1)) @
                               error_N_test.reshape(
            (1, 1)))
        N_test_error_all[r, :] = error_N_test_vector[:, 0]

        # Student training through offline notebook reactivations at each epoch
        print(
            'Student training through offline notebook reactivations at each epoch.')
        for m in range(self.config.n_epochs):
            print('Epoch: ' + str(m))
            N_S_input = N_patterns_reactivated[:, :, m - 1] @ W_N_S_Lin
            N_S_output = N_patterns_reactivated[:, :, m - 1] @ W_N_S_Lout
            N_S_prediction = N_S_input @
            S_prediction = x_t_input @ W_s
            S_prediction_test = x_t_input_test @ W_s

            # Train error
            delta_train = y_t_output - S_prediction
            error_train = sum(delta_train ** 2) / P
            error_train_vector[m - 1] = error_train

            # Generalization error
            delta_test = y_t_output_test - S_prediction_test
            error_test = sum(delta_test ** 2) / P_test
            error_test_vector[m - 1] = error_test

            # Grad descent
            w_delta = N_S_input.T @ N_S_output - N_S_input.T @ N_S_input @ W_s
            W_s = W_s + learnrate * w_delta



mdl_config = TSNModel_config()
dt_config = TSN_DTProcessor_config(
    SNR = np.inf
)
pt_config = TSNPractitioner_config(
    n_epochs=2000,
    lr=0.0015
)

r_n = 20        # number of repeats

# Storing train error, test error, reactivation error (driven by
# notebook)
# Without early stopping (r_n, nepoch)
train_error_all = []     # student train error
test_error_all = []     # student test error
N_train_error_all = []   # notebook train error
N_test_error_all = []    # notebook test error

# With early stopping (r_n, nepoch)
train_error_all_es = []     # student train error
test_error_all_es = []      # student test error

# Run simulation for r_n times
for r in range(r_n):
    print('Running Simulation: ' + str(r))
    np.random.seed(r) #set random seed for reproducibility

    # Errors
    error_train_vector = []
    error_test_vector = []
    error_react_vector = []

    dt_processor = TSN_DTProcessor(dt_config)
    dt_processor.generate_data()

    mdl = TSNModel(mdl_config)

    pt = TSNPractitioner(pt_config, mdl, dt_processor)

    pt.train()


    # Storing train and test errors for the current iteration
    train_error_all[r-1, :] = error_train_vector[:,0]
    test_error_all[r-1, :] = error_test_vector[:, 0]

    # Early stopping
    min_p = np.argmin(error_test_vector)  # Find index of minimum test error
    train_error_early_stop = error_train_vector.copy()
    train_error_early_stop[min_p + 1:] = error_train_vector[
        min_p]  # Set subsequent values to the minimum point

    test_error_early_stop = error_test_vector.copy()
    test_error_early_stop[min_p + 1:] = error_test_vector[
        min_p]  # Set subsequent values to the minimum point

    # Store early stopped errors
    train_error_all_es[r-1, :] = train_error_early_stop[:,0]
    test_error_all_es[r-1, :] = test_error_early_stop[:,0]
    print('Simulation ' + str(r) + ' complete.')
    print('-----------------------------------')

# Define color scheme and plot settings
color_scheme = np.array([[137, 152, 193], [245, 143, 136]]) / 255
line_w = 2
font_s = 12

# Plot without early stopping
plt.figure(figsize=(4.5, 4))  # Similar to setting figure position and size in MATLAB
plt.plot(range(1, nepoch + 1), np.mean(train_error_all, axis=0), color=color_scheme[0], linewidth=line_w, label='Train Error')
plt.plot(range(1, nepoch + 1), np.mean(test_error_all, axis=0), color=color_scheme[1], linewidth=line_w, label='Test Error')
plt.plot(range(1, nepoch + 1), np.mean(N_train_error_all, axis=0), 'b--', linewidth=line_w, label='N Train Error')
plt.plot(range(1, nepoch + 1), np.mean(N_test_error_all, axis=0), 'r--', linewidth=line_w, label='N Test Error')

plt.xlabel('Epoch', fontsize=font_s, color='k')
plt.ylabel('Error', fontsize=font_s, color='k')
plt.xlim([0, nepoch])
plt.ylim([0, 2])
plt.gca().spines['top'].set_linewidth(1)
plt.gca().spines['right'].set_linewidth(1)
plt.gca().spines['bottom'].set_linewidth(1)
plt.gca().spines['left'].set_linewidth(1)
plt.xticks(fontsize=font_s)
plt.yticks(fontsize=font_s)
plt.legend()
plt.tight_layout()

# Optional: Save the plot
# plt.savefig(f'Errors_No_ES_SNR_{SNR}_{date.today()}.pdf')

# Plot with early stopping
plt.figure(figsize=(4.5, 4))  # Similar to setting figure position and size in MATLAB
plt.plot(range(1, nepoch + 1), np.mean(train_error_all_es, axis=0), color=color_scheme[0], linewidth=line_w, label='Train Error (ES)')
plt.plot(range(1, nepoch + 1), np.mean(test_error_all_es, axis=0), color=color_scheme[1], linewidth=line_w, label='Test Error (ES)')
plt.plot(range(1, nepoch + 1), np.mean(N_train_error_all, axis=0), 'b--', linewidth=line_w, label='N Train Error')
plt.plot(range(1, nepoch + 1), np.mean(N_test_error_all, axis=0), 'r--', linewidth=line_w, label='N Test Error')

plt.xlabel('Epoch', fontsize=font_s, color='k')
plt.ylabel('Error', fontsize=font_s, color='k')
plt.xlim([0, nepoch])
plt.ylim([0, 2])
plt.gca().spines['top'].set_linewidth(1)
plt.gca().spines['right'].set_linewidth(1)
plt.gca().spines['bottom'].set_linewidth(1)
plt.gca().spines['left'].set_linewidth(1)
plt.xticks(fontsize=font_s)
plt.yticks(fontsize=font_s)
plt.legend()
plt.tight_layout()

# Optional: Save the plot
# plt.savefig(f'Errors_ES_SNR_{SNR}_{date.today()}.pdf')

plt.show()

print('debug')

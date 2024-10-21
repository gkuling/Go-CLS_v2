import matplotlib.pyplot as plt
import numpy as np
import project_team as proteam
from project_team.dt_project.DataProcessors._Processor import _Processor
import pandas as pd
import torch
from project_team.ml_project import PT_Practitioner
from tqdm import tqdm
from torch.nn.functional import linear
from torch.nn import functional as F
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

class TSNModel(torch.nn.Linear):
    def __init__(self, config):
        super(TSNModel, self).__init__(config.S_x,
                                       config.S_y,
                                       bias=False,
                                       )
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

        # Notebook-Student weights
        self.W_NS = None


    def forward(self, x):
        return linear(x, self.weight, self.bias)

    #     return {
    #         'student': np.matmul(x, self.W_s),
    #         'notebook': np.matmul(
    #             np.matmul(
    #                 np.matmul(x, self.W_NS['S->N in']),
    #                 self.W_N
    #             ),
    #             self.W_NS['N->S out']
    #         )
    #     }

    def W_S_N_Learning(self, x, y):
        # Hebbian learning for Notebook-Student weights (bidirectional)
        weights = {}
        norm = (
                self.config.N_n_units * self.config.N_a * (1 - self.config.N_a)
        )
        # Notebook to student weights, for reactivating student
        weights['N->S in'] = np.matmul((self.N_patterns - self.config.N_a).T, x)
        weights['N->S in'] /= norm

        weights['N->S out'] = np.matmul((self.N_patterns -
                                         self.config.N_a).T, y)
        weights['N->S out'] /= norm

        # student to notebook weights, for providing partial cues
        weights['S->N in'] = np.matmul(x.T, (self.N_patterns - self.config.N_a))
        weights['S->N out'] = np.matmul(y.T, (self.N_patterns -
                                              self.config.N_a))
        if self.W_NS is None:
            self.W_NS = weights
        else:
            self.W_NS = {
                k: (self.W_NS[k] + weights[k])/2 for k in weights.keys()
            }

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
        return (np.matmul(pattern, self.W_NS['N->S in']),
                np.matmul(pattern, self.W_NS['N->S out'] ))

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
                 batch_epochs=100,
                 **kwargs):
        super(TSNPractitioner_config, self ).__init__(**kwargs)
        # batch scheme parameters
        self.batch_epochs = batch_epochs
class TSNPractitioner(PT_Practitioner):
    def __init__(self, model, dt_processor, manager,
                 config=TSNPractitioner_config()):
        super().__init__(model, manager, dt_processor, config)
        pass
        # self.config = config
        # self.model = model
        # self.dt_processor = dt_processor

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
    def train_model(self):

        # Set up data
        tr_dtldr = self.setup_dataloader('training')

        if hasattr(self.data_processor, 'vl_dset'):
            vl_dtldr = self.setup_dataloader('validation')
        else:
            vl_dtldr = None
        self.setup_steps(len(tr_dtldr))
        self.setup_loss_functions()
        self.setup_training_accessories()

        print('ML Message: ')
        print('-' * 5 + ' ' + self.practitioner_name + ' Practitioner Message:  \
                    The Beginning of Training ')

        # beginning epoch loop
        for epoch in range(1, self.config.n_epochs + 1):

            epoch_iterator = tqdm(tr_dtldr, desc="Epoch " + str(epoch)
                                                 + " Iteration: ",
                                  position=0, leave=True,
                                  ncols=80
                                  )
            epoch_iterator.set_postfix({'loss': 'Initialized'})
            tr_loss = []
            # begin batch loop
            for batch_idx, data in enumerate(epoch_iterator):
                print('')
                # Model will not be trained more steps than asked for
                if self.config.trained_steps >= self.config.n_steps:
                    break
                self.config.trained_steps += 1

                btch_x, btch_y = self.organize_input_and_output(data)

                btch_x = torch.nn.Flatten()(btch_x)
                btch_y = btch_y[:,np.newaxis]
                # btch_y = torch.nn.Flatten()(btch_y[0])
                # btch_y = btch_y.numpy()[:,np.newaxis]

                # train the input out transforms of the notebook
                self.model.W_S_N_Learning(
                    btch_x,
                    btch_y
                )

                # array for storing retrieved notebook patterns, pre-calculating all epochs for speed considerations
                S_error_train = []
                batch_epoch_iterator = tqdm(range(self.config.batch_epochs), desc="Batch_epoch ",
                                      position=0, leave=True,
                                      ncols=70
                                      )
                batch_epoch_iterator.set_postfix({'loss': 'Initialized'})
                for batch_epoch in batch_epoch_iterator:
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
                        self.model.pattern_completion_with_fix_threshold(
                            Activity_dyn_t)
                    )
                    ## Generate offline training data from notebook reactivations
                    N_S_input, N_S_output = self.model.reactivate_pattern(
                        N_pattern_reactivated)

                    self.model.train()
                    self.optmzr.zero_grad()

                    pred = self.model(N_S_input.to(torch.float32))

                    loss = self.calculate_loss(
                        F.log_softmax(pred/5, dim=-1),
                        F.softmax(N_S_output.to(torch.float32)/5, dim=-1))
                    loss.backward()
                    self.optmzr.step()
                    # # Grad descent
                    # w_delta = np.matmul(N_S_input.T, N_S_output)
                    # w_delta -= self.model.forward(
                    #     np.matmul(N_S_input.T, N_S_input)
                    # )['student']
                    # self.model.W_s = self.model.W_s + self.config.lr * w_delta

                    # Student prediction
                    S_prediction = self.model.forward(
                        btch_x
                    )

                    # Train error
                    delta_train = btch_y - S_prediction
                    delta_train = (delta_train ** 2).mean()
                    S_error_train.append(delta_train.item())

                    batch_epoch_iterator.set_postfix({'loss': delta_train.item()})




                loss = np.mean(S_error_train)
                dir = 'C:\Project_Data\saves'
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(btch_x[0].reshape(28, 28), cmap='gray')
                axs[0].axis('off')
                axs[0].set_title(str(btch_y[0].argmax()))
                axs[1].imshow(N_S_input[0].reshape(28, 28), cmap='gray')
                axs[1].axis('off')
                axs[1].set_title(str(btch_y[0].argmax()))
                plt.savefig(dir + f'/{self.config.trained_steps}.png')
                plt.close(fig)

                # report and record the loss
                epoch_iterator.set_postfix({'loss': loss.item()})
                tr_loss.append(loss.item())

            # Validation
            vl_loss = str(self.validate_model(vl_dtldr))
            print('Validation Loss: ' + vl_loss)
        print('ML Message: Finished Training ' + self.practitioner_name)

    def validate_model(self, val_dataloader):
        print('Running Validation ')
        val_loss = []
        with torch.no_grad():
            for data in val_dataloader:
                btch_x = torch.nn.Flatten()(data['X'])
                btch_y = torch.nn.Flatten()(data['y'][0])
                S_prediction = self.model.forward(
                    btch_x
                )
                delta_val = btch_y - S_prediction
                val_loss.append((delta_val ** 2).mean().item())
        return np.mean(val_loss)

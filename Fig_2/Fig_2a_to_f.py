import matplotlib.pyplot as plt
import numpy as np

r_n = 1        # number of repeats
nepoch = 200
learnrate = 0.0015
N_x_t = 10     # teacher input dimension
N_y_t = 1       # teacher output dimension
P = 100           # number of training examples
P_test = 1000   # number of testing examples

# According to SNR, set variances for teacher's weights (variance_w) and
# output noise (variance_e) that sum to 1
SNR = np.inf

if SNR == np.inf:
    variance_w = 1
    variance_e = 0
else:
    variance_w = SNR/(SNR + 1)
    variance_e = 1/(SNR + 1)

# Student and teacher share the same dimensions
N_x_s = N_x_t
N_y_s = N_y_t

# Notebook parameters
# see Buhmann, Divko, and Schulten, 1989 for details regarding gamma and U terms

M = 2000    # num of units in notebook
a = 0.05    # notebook sparseness
gamma = 0.6 # inhibtion parameter
U = -0.15   # threshold for unit activation
ncycle = 9  # number of recurrent cycles

# Matrices for storing train error, test error, reactivation error (driven by
# notebook)
# Without early stopping
train_error_all = np.zeros((r_n,nepoch))      # student train error
test_error_all = np.zeros((r_n,nepoch))       # student test error
N_train_error_all = np.zeros((r_n,nepoch))    # notebook train error
N_test_error_all = np.zeros((r_n,nepoch))     # notebook test error

# With early stopping
train_error_all_es = np.zeros((r_n,nepoch))      # student train error
test_error_all_es = np.zeros((r_n,nepoch))       # student test error

# Run simulation for r_n times
for r in range(1,r_n+1,1):
    print('Running Simulation: ' + str(r))
    np.random.seed(r) #set random seed for reproducibility

    # Errors
    error_train_vector = np.zeros((nepoch, 1))
    error_test_vector = np.zeros((nepoch, 1))
    error_react_vector = np.zeros((nepoch, 1))

    ## Teacher Network
    W_t = np.random.normal(0,variance_w**0.5,[N_x_t,N_y_t]) # set teacher's 
    # weights with variance_w
    noise_train = np.random.normal(0,variance_e**0.5,[P,N_y_t]) # set the 
    # variance for label noise
    # Training data
    x_t_input = np.random.normal(0,(1/N_x_t)**0.5,[P,N_x_t]) # inputs
    y_t_output = x_t_input @ W_t + noise_train # outputs

    # Testing data
    noise_test = np.random.normal(0,variance_e**0.5,[P_test,N_y_t])
    x_t_input_test = np.random.normal(0, (1 / N_x_t) ** 0.5, [P_test, N_x_t])  # inputs
    y_t_output_test = x_t_input_test @ W_t + noise_test  # outputs

    ## Notebook Network
    # Generate P random binary indices with sparseness a
    N_patterns = np.zeros((P, M))
    for i in range(1,P+1,1):
        N_patterns[i-1,:] = np.random.choice([0, 1], size=(1, M), p=[1-a, a])

    # Hebbian learning for notebook recurrent weights
    W_N = np.dot((N_patterns - a).T, (N_patterns - a)) / (M * a * (1 - a))
    W_N = W_N - gamma / (a * M) # add global inhibition term, see Buhmann, Divko, and Schulten, 1989
    W_N = W_N * (~np.eye(W_N.shape[0], dtype=bool)) # no self connection

    # Hebbian learning for Notebook-Student weights (bidirectional)

    # Notebook to student weights, for reactivating student
    W_N_S_Lin = np.dot((N_patterns - a).T, x_t_input) / (M * a * (1 - a))
    W_N_S_Lout = np.dot((N_patterns - a).T, y_t_output) / (M * a * (1 - a))
    # student to notebook weights, for providing partial cues
    W_S_N_Lin = np.dot(x_t_input.T, (N_patterns - a))
    W_S_N_Lout = np.dot(y_t_output.T, (N_patterns - a))

    # Student Network
    W_s = np.random.normal(0, 0**0.5,[N_x_t,N_y_t]) # set student's weights, with zero weight initialization

    ## Generate offline training data from notebook reactivations
    N_patterns_reactivated = np.zeros((P,M,nepoch))
    # array for storing retrieved notebook patterns, pre-calculating all epochs for speed considerations

    # beginning the simulation
    for epoch in range(1,nepoch+1,1):
        print('Epoch: ' + str(epoch))
        # Simulates hippocampal reactivations with random binary seed patterns.
        # Retrieval process:
        # (1) Dynamic threshold ensures sparse pattern retrieval,
        # preventing silent attractor dominance.
        # (2) Fixed-threshold completes the pattern
        # with global inhibition, allowing varied sparseness. Silent state occurs if seed
        # is far from encoded patterns.

        # Dynamic threshold
        Activity_dyn_t = np.zeros((P, M))

        print('First round of pattern completion through recurrent activtion '
              'cycles given random initial input.')
        # First round of pattern completion through recurrent activtion
        # cycles given random initial input.
        for cycle in range(1,ncycle+1,1):
            print('Cycle: ' + str(cycle))
            if cycle <= 1:
                clamp = 1
            else:
                clamp = 0

            rand_patt = (np.random.rand(P, M) <= a) # random seeding activity
            # Seeding notebook with random patterns
            M_input = Activity_dyn_t + (rand_patt * clamp)
            ### GCK: In the original codes this line was commented out(???):
            ## Seeding notebook with origginal patterns
            ## # M_input = Activity_dyn_t + (N_patterns * clamp)

            M_current = M_input @ W_N

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
            t_ind = np.floor(Activity_dyn_t.shape[1] * a).astype(
                int)  # Convert to integer

            # Ensure t_ind is at least 1
            t_ind = np.maximum(t_ind, 1)

            # Use t_ind to select the threshold value from each row of sorted_M_current
            t = sorted_M_current[:, t_ind - 1]  # -1 for zero-indexing in Python

            # Set Activity_dyn_t based on the threshold comparison
            Activity_dyn_t = (M_current >= t[:, np.newaxis])

        print('Second round of pattern completion, with fix threshold')
        # Second round of pattern completion, with fix threshold
        Activity_fix_t = np.zeros((P, M))
        for cycle in range(1,ncycle+1,1):
            print('Cycle: ' + str(cycle))
            if cycle <= 1:
                clamp = 1
            else:
                clamp = 0

            M_input = Activity_fix_t + (Activity_dyn_t * clamp)

            M_current = M_input @ W_N

            # scale currents between 0 and 1
            scale = 1.0 / (
                    np.max(M_current, axis=1) - np.min(M_current, axis=1)
            )

            M_current = (
                                M_current - np.min(M_current, axis=1)[:,
                                            np.newaxis]
                        ) * scale[:, np.newaxis]

            # find threshold based on desired sparseness
            t = U

            # Set Activity_fix_t based on the threshold comparison
            Activity_fix_t = (M_current >= t)
        N_patterns_reactivated[:, :, epoch-1] = Activity_fix_t

    # Seed the notebook with original patterns to calculate training error.
    # Seed with student input via Notebook weights, complete the pattern,
    # and use the retrieved pattern to activate the student's output through
    # Notebook-to-Student weights.

    print('Seed the notebook with original patterns to calculate training error.')
    Activity_notebook_train = np.zeros((P, M))
    for cycle in range(1,ncycle+1,1):
        print('Cycle: ' + str(cycle))
        if cycle <= 1:
            clamp = 1
        else:
            clamp = 0
        seed_patt = x_t_input @ W_S_N_Lin
        M_input = Activity_notebook_train + (seed_patt * clamp)

        M_current = M_input @ W_N

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
        t_ind = np.floor(Activity_dyn_t.shape[1] * a).astype(
            int)  # Convert to integer

        # Ensure t_ind is at least 1
        t_ind = np.maximum(t_ind, 1)

        # Use t_ind to select the threshold value from each row of sorted_M_current
        t = sorted_M_current[:, t_ind - 1]  # -1 for zero-indexing in Python

        # Set Activity_notebook_train based on the threshold comparison
        Activity_notebook_train = (M_current >= t[:, np.newaxis])
    N_S_output_train = Activity_notebook_train @ W_N_S_Lout
    # Notebook training error
    delta_N_train = y_t_output - N_S_output_train
    error_N_train = np.sum(delta_N_train ** 2) / P
    # Since notebook errors stay constant throughout training,
    # populating each epoch with the same error value
    error_N_train_vector = np.ones((nepoch, 1)) @ error_N_train.reshape((1,1))
    N_train_error_all[r-1,:] = error_N_train_vector[:,0]

    print('Notebook training error.')
    # Notebook generalization error
    Activity_notebook_test = np.zeros((P_test, M))
    for cycle in range(1,ncycle+1,1):
        print('Cycle: ' + str(cycle))
        if cycle <= 1:
            clamp = 1
        else:
            clamp = 0
        seed_patt = x_t_input_test @ W_S_N_Lin
        M_input = Activity_notebook_test + (seed_patt * clamp)

        M_current = M_input @ W_N

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
                           ::-1]

        # Calculate the index for the threshold based on 'a'
        t_ind = np.floor(Activity_dyn_t.shape[1] * a).astype(
            int)

        # Ensure t_ind is at least 1
        t_ind = np.maximum(t_ind, 1)

        # Use t_ind to select the threshold value from each row of sorted_M_current
        t = sorted_M_current[:, t_ind - 1]

        # Set Activity_notebook_test based on the threshold comparison
        Activity_notebook_test = (M_current >= t[:, np.newaxis])
    N_S_output_test = Activity_notebook_test @ W_N_S_Lout
    # Notebook test error
    delta_N_test = y_t_output_test - N_S_output_test
    error_N_test = np.sum(delta_N_test ** 2) / P_test
    # populating each epoch with the same error value
    error_N_test_vector = np.ones((nepoch, 1)) @ error_N_test.reshape((1,1))
    N_test_error_all[r-1,:] = error_N_test_vector[:,0]

    # Student training through offline notebook reactivations at each epoch
    print('Student training through offline notebook reactivations at each epoch.')
    for m in range(1,nepoch + 1,1):
        print('Epoch: ' + str(m))
        N_S_input = N_patterns_reactivated[:,:,m-1] @ W_N_S_Lin
        N_S_output = N_patterns_reactivated[:,:,m-1] @ W_N_S_Lout
        N_S_prediction = N_S_input @ W_s
        S_prediction = x_t_input @ W_s
        S_prediction_test = x_t_input_test @ W_s

        # Train error
        delta_train = y_t_output - S_prediction
        error_train = sum(delta_train**2)/P
        error_train_vector[m-1] = error_train

        # Generalization error
        delta_test = y_t_output_test - S_prediction_test
        error_test = sum(delta_test**2)/P_test
        error_test_vector[m-1] = error_test

        # Grad descent
        w_delta = N_S_input.T @ N_S_output - N_S_input.T @ N_S_input @ W_s
        W_s = W_s + learnrate * w_delta

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

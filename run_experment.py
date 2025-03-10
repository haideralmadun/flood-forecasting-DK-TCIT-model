import pandas as pd
import numpy as np
from DK_TCIT_Model import DK_TCIT_Model
from summarize_average_performance import summarize_average_performance

# Load the dataset
dataset = pd.read_csv(r'tunxi 1981-2016_interpolated.csv')

# Remove date column and handle missing data
dataset_new = dataset.iloc[:, 1:]
dataset_new = dataset_new.dropna()

# Apply scaling to the entire dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset_new_no = scaler.fit_transform(dataset_new)

# Create a separate scaler for the 'streamflow' column (for predictions)
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(dataset_new['streamflow'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

# Convert to DataFrame for easier splitting
df = pd.DataFrame(dataset_new_no)

# Split the data into train, validation, and test sets
dataset_train = df.iloc[:37622, :].values
dataset_val = df.iloc[37622:42997, :].values
dataset_test = df.iloc[42997:, :].values

# Function to split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Set the number of time steps for input and output
n_steps_in, n_steps_out = 12, 6

# Convert into input/output sequences
X_train, y_train = split_sequences(dataset_train, n_steps_in, n_steps_out)
print(X_train.shape, y_train.shape)
X_val, y_val = split_sequences(dataset_val, n_steps_in, n_steps_out)
print(X_val.shape, y_val.shape)
X_test, y_test = split_sequences(dataset_test, n_steps_in, n_steps_out)
print(X_test.shape, y_test.shape)

# Reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = X_train.shape[2]
n_seq = 2  # You might want to adjust this if needed
n_steps = 6  # Ensure this matches your model input configuration
X_train = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))
X_val = X_val.reshape((X_val.shape[0], n_seq, n_steps, n_features))
X_test = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))

n_features = X_train.shape[3]




# Initialize the DK-TCIT model
model = DK_TCIT_Model(input_shape=(None, n_steps, n_features), n_steps_out=n_steps_out, n_steps_in=n_steps_in)
model.summary()

# Define early stopping callback
from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

# Fit the model
history = model.fit(X_train,
                    y_train,
                    batch_size=10,
                    epochs=100,
                    callbacks=[earlystopping],
                    validation_data=(X_val, y_val))

# Predict with the model
y_pred = model.predict(X_test, batch_size=10, verbose=1)

# Unscale the predicted values
y_pred_unscaled = scaler_pred.inverse_transform(y_pred)
# Unscale the actual values
y_test_unscaled = scaler_pred.inverse_transform(y_test)




# Summarize average performance
summarize_average_performance('DK-TCIT', y_test_unscaled, y_pred_unscaled)


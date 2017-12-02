import tensorflow as tf
import numpy as np
import sys
import os

print "\nCompleted imports"

####---------------------- DATA PROCESSING ------------------------####

def loadData():
    npzfile = np.load('data.npz')
    X = npzfile['arr_0']
    Y = npzfile['arr_1']
    n = X.shape[0]
    n_trn = int(n * 0.8)
    X_trn = X[0:n_trn, :]
    X_test = X[n_trn:n, :]
    Y_trn = Y[0:n_trn]
    Y_test = Y[n_trn:n]
    return X_trn, X_test, Y_trn, Y_test

X_trn, X_test, Y_trn_1, Y_test_1 = loadData()


# Preprocess the data to make it ammenable to a classification problem

# Format training data
Y_trn = [0 for i in range(len(Y_trn_1))]
for x in range(len(Y_trn)): Y_trn[x] = np.zeros(X_trn.shape[1])
Y_trn = np.asarray(Y_trn)

for i in range(len(Y_trn_1)): Y_trn[i][int(Y_trn_1[i])] = 1
Y_trn = Y_trn.reshape([len(Y_trn), X_trn.shape[1]])


# Format testing data
Y_test = [0 for i in range(len(Y_test_1))]
for x in range(len(Y_test)): Y_test[x] = np.zeros(X_test.shape[1])
Y_test = np.asarray(Y_test)

for i in range(len(Y_test_1)): Y_test[i][int(Y_test_1[i])] = 1
Y_test = Y_test.reshape([len(Y_test), X_test.shape[1]])


# Function returns a slice of 100 inputs/outputs from the given sets
# of inputs and outputs
def get_batch(X, Y, batch_size):
    slice = np.random.randint(0, len(X), batch_size)
    return X[slice], Y[slice]



print "Processed training and testing data\n"



####--------------------- TENSORFLOW SCRIPT -----------------------####


# Parameters                                                                       
learning_rate = 0.3
num_steps = 3000
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 64 # 1st layer number of neurons                                      
n_hidden_2 = 20 # 2nd layer number of neurons                                      
num_input = X_trn.shape[1] # Size of input
num_classes = Y_trn.shape[1] # Size of output

# An accuracy threshold used to prevent overfitting
early_stopping_threshold = 0.95 


# tf Graph input                                                                   
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


# Store layers weight & bias                                                       
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model. This is a simple feedforward network with 2 hidden layers
def neural_net(x):

    # Hidden fully connected layer with 256 neurons                                
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))

    # Hidden fully connected layer with 256 neurons                                
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

    # Output fully connected layer with a neuron for each class                    
    out_layer = tf.nn.relu((tf.matmul(layer_2, weights['out']) + biases['out']))

    return out_layer


# Construct model                                                                  
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer                                                        
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model                                    
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Calculate the variances needed to compute R^2
error_pred_var = tf.reduce_mean(tf.nn.moments(prediction - Y, axes=[1])[1])
train_var = tf.reduce_mean(tf.nn.moments(Y, axes=[1])[1])


# Initialize the variables (i.e. assign their default value)                       
init = tf.global_variables_initializer()


# Start training                                                                   
with tf.Session() as sess:

    # Run the initializer                                                          
    sess.run(init)

    print "\n\nRunning optimization\n\n"

    for step in range(1, num_steps+1):
        batch_x,batch_y = get_batch(X_trn, Y_trn, batch_size)

        # Run optimization op (backprop)                                           
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Calculate batch loss and accuracy                                    
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
        if step % display_step == 0 or step == 1:
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

        if acc > early_stopping_threshold:
            print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            break

    print("Optimization Finished!\n")

    # Calculate accuracy for MNIST test images                                      
    print("Testing Accuracy: ", sess.run(accuracy, feed_dict={X: X_test,
                                      Y: Y_test}))

    # Compute (a - a_hat).var() and a.var()
    error_pred = sess.run(error_pred_var, feed_dict = {X: X_test, Y: Y_test})
    y_var = sess.run(train_var, feed_dict = {X: X_test, Y: Y_test})

    # Print the R^2 value
    name = 'nonl nn'
    R2 = 1 - (error_pred / y_var)
    print ( "R^2: " + str(R2) )
    if os.path.exists('results_nnn.npz'):
        os.remove('results_nnn.npz')
    np.savez('results_nnn.npz', [name], [R2])







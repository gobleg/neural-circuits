import tensorflow as tf
import numpy as np

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

Y_trn = [0 for i in range(len(Y_trn_1))]
for x in range(len(Y_trn)): Y_trn[x] = np.zeros(X_trn.shape[1])
Y_trn = np.asarray(Y_trn)

for i in range(len(Y_trn_1)): Y_trn[i][int(Y_trn_1[i])] = 1
Y_trn = Y_trn.reshape([len(Y_trn), X_trn.shape[1]])


Y_test_var = Y_test_1.var()
print Y_test_1

Y_test = [0 for i in range(len(Y_test_1))]
for x in range(len(Y_test)): Y_test[x] = np.zeros(X_test.shape[1])
Y_test = np.asarray(Y_test)

for i in range(len(Y_test_1)): Y_test[i][int(Y_test_1[i])] = 1
Y_test = Y_test.reshape([len(Y_test), X_test.shape[1]])



def get_batch(X, Y):
    slice = np.random.randint(0, len(X), 100)
    return X[slice], Y[slice]


print "\nCompleted imports\n"

# Parameters                                                                       
learning_rate = 0.3
num_steps = 10000
batch_size = 70
display_step = 1

# Network Parameters                                                               
n_hidden_1 = 64 # 1st layer number of neurons                                      
n_hidden_2 = 20 # 2nd layer number of neurons                                      
num_input = X_trn.shape[1] # MNIST data input (img shape: 28*28)
num_classes = Y_trn.shape[1] # MNIST total classes (0-9 digits)                                

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


# Create model                                                                     
def neural_net(x):

    # Hidden fully connected layer with 256 neurons                                
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    # Hidden fully connected layer with 256 neurons                                
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    # Output fully connected layer with a neuron for each class                    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer


Rsq = lambda a, a_hat: 1 - tf.nn.moments(a - a_hat, axes=[1])[1] / tf.nn.moments(a, axes=[1])[1]

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

# Initialize the variables (i.e. assign their default value)                       
init = tf.initialize_all_variables()



# Start training                                                                   
with tf.Session() as sess:

    # Run the initializer                                                          
    sess.run(init)

    print "\n\nRunning optimization\n\n"

    for step in range(1, num_steps+1):
        batch_x,batch_y = get_batch(X_trn, Y_trn)

        # Run optimization op (backprop)                                           
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % (100 * display_step) == 0 or step == 1:
            # Calculate batch loss and accuracy                                    
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!\n")

    # Calculate accuracy for MNIST test images                                      
    print("Testing Accuracy: ", sess.run(accuracy, feed_dict={X: X_test,
                                      Y: Y_test}))

    accu = sess.run(accuracy, feed_dict = {X: X_test, Y: Y_test})

    #print(sess.run(prediction))

    print str(accu * (1 - accu))
    print Y_test_var
    
    print(  "R^2: " +  str( 1 - (accu * (1 - accu)) / Y_test_var )  )





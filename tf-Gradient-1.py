#       Tensorflow #1 Example
#   Tensorflow example of Gradient Descent
#   on a linear equation (y = mx + b)


import tensorflow as tf

m = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = m * x + b # y = mx + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) # Also known as r^2
loss = tf.reduce_sum(squared_deltas)

# If you decrease the learning rate, you have to increase the loop range value
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

x_set = [1, 2]
y_set = [5,6]
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(train, {x: x_set, y: y_set})
        print("M: ",session.run(m),"B: ",session.run(b))
    
    m_value, b_value, loss = session.run([m, b, loss], {x: x_set, y: y_set})
    print ("y = {}x + {}".format(repr(m_value[0]), repr(b_value[0]))) 
    print ("Loss: ", loss)
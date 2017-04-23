import tensorflow as tf
import matplotlib.pyplot as plt
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(-3.0)
W_val=[]
costW_val=[]

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W), sess.run(cost))
    W_val.append(sess.run(W))
    costW_val.append(sess.run(cost))
    sess.run(train)

plt.plot(W_val, costW_val)
plt.show()

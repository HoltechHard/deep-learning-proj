#####################################
# INTRODUCTION TO TENSORFLOW - HSE  #
#####################################

#import package
import tensorflow as tf

#PlaceHolder: which will be fed during graph execution
x = tf.placeholder(tf.float32, (None, 10))

#Variable: tensor with some value is updated during execution
w = tf.get_variable('w', shape = (10, 20), dtype = tf.float32)
w = tf.random_uniform((10, 20), name = 'w')

#Constant: tensor that cannot be changes
c = tf.constant(np.ones((4, 4)))

#Operation: matrix product
x = tf.placeholder(tf.float32, (None, 10))
w = tf.Variable(tf.random_uniform((10, 20)), name = 'w')
z = x @ w
z = tf.matmul(x, w)
print(z)

#Running a graph: step by step
#We need a session because operations are not executed in Python (slow)
#operations are written in c++ and executed in CPU, GPU or TPU

# Definition of a session: 
#object encapsulates the environment in tf.Operation objects => are executed in c++
#and tf.Tensor objects => are evaluated

# (1) - create a session 
sesion = tf.InteractiveSession()

# (2) - define a graph
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b 

# (3) - running the graph
print(c)
print(session.run(c))  # sessions are created to execute the graph

#Initialize a variable in other environment like a GPU
session.run(tf.global_variables_initializer())

# -- Running approprially variables -- 
#Running with automatically initialization 

# (1) - definition
tf.reset_default_graph()
a = tf.constant(np.ones((2, 2)), dtype = np.float32)
b = tf.Variable(tf.ones((2, 2)))
c = a @ b

# (2) - attempting to use uninitializer value
session.run(tf.global_variables_initializer())
session.run(c)

# -- Running approprially placeholders -- 

# (1) - definition => every node in our graph is an operation
tf.reset_default_graph()
a = tf.placeholder(np.float32, (2, 2))
b = tf.Variable(tf.ones((2, 2)))
c = a @ b

# (2) - feed a value for placeholder tensor
session.run(tf.global_variables_initializer())
session.run(c, feed_dict = {a: np.ones((2, 2))})

#to reset a graph => clear your graphs
tf.reset_default_graph()



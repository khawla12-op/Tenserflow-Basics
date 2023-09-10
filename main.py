import tensorflow as tf
#initialization of tensors:
x=tf.constant(4,shape=(1,1),dtype=tf.float32)# A scalar
x=tf.constant([[1,2,3],[4,5,6]])
x=tf.ones((3,3))
x=tf.zeros((2,3))
x=tf.eye(3) #Identity matrix
x=tf.random.normal((3,3),0,stddev=1)#  mean (average) of 0 and a standard deviation of 1
x=tf.random.uniform((1,3),minval=0,maxval=1)
x=tf.range(start=1,limit=10,delta=2)
x=tf.cast(x,dtype=tf.float64)#to convert between different types
#Mathematical Operations
#Adding
x=tf.constant([1,2,3])
y=tf.constant([9,8,4])
z=tf.add(x,y)  #or z=x+y
#Substraction
z=tf.subtract(x,y) #or z=x-y
#Division
z=tf.divide(x,y) # or z=x/y
#Multiplication
z=tf.multiply(x,y) #or z=x*y
#Produit cartesien
z=tf.tensordot(x,y,axes=1)
#Matrice Multiplication
z=tf.matmul(x,y) # or x@y

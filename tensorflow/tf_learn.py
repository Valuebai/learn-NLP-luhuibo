#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/9/16 15:46
@Desc   ：learn tensorflow record, all is well
=================================================='''
import tensorflow as tf

# # a simple matrix multiply
#
# create a constant v1, it is a 1x2 matrix
v1 = tf.constant([[2, 3]])

# create a constant v2, it is a 2x1 matrix
v2 = tf.constant([[2], [3]])

# create a matrix multiply, notices: after create it, it would not run until it run in session
product = tf.matmul(v1, v2)

# we can print product, as we can see, it's the result of multiply but the multiply itself
print('print product:', product)

# define a session, and we can enter to the docstring of tf.Session()
sess = tf.compat.v1.Session()

# run the matmul and get the result
result = sess.run(product)

# print the result
print('print the result:', result)

# close the session
sess.close()

# # a simple matrix multiply
#
# create a variable num
num = tf.Variable(0, name="count")

# create a add operation , current num add 1
new_value = tf.add(num, 1)

# create a assign operation, num = new_value
op = tf.compat.v1.assign(num, new_value)

# use this write method with xx as xx: , after running the code, session will be auto closed.
with tf.compat.v1.Session() as sess:
    # initialize global variables.
    sess.run(tf.compat.v1.global_variables_initializer())
    # print the original value of num
    print(sess.run(num))
    # create a for loop, add 1 each time and print it
    for i in range(5):
        sess.run(op)
        print(sess.run(num))

# 在声明变量的时候不赋值，计算的时候才进行赋值，这个时候feed就派上用场了
#
# create a variable placeholder input1
input1 = tf.placeholder(tf.float32)
# create a variable placeholder input2
input2 = tf.placeholder(tf.float32)

# create a add operation, let input1 and input2 multiply
new_value = tf.multiply(input1, input2)

# use this write method with xx as xx: , after running the code, session will be auto closed.
with tf.compat.v1.Session() as sess:
    # print the value of new_value, when multiply the num, feed set the input's value
    print(sess.run(new_value, feed_dict={input1: 23.0, input2: 11.0}))

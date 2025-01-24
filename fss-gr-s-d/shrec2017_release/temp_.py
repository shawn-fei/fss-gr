import tensorflow as tf
batch_size=1
npoint=9
nsample=4

a=4
b=5
out_channel=a
new_points=tf.zeros([batch_size,npoint,nsample,out_channel])
new_points_re=tf.reshape(new_points,[batch_size, -1, a])
grouped_time=tf.zeros([batch_size,npoint,nsample,b])

ccat1=tf.concat([new_points, grouped_time], -1)


with tf.Session() as sess:
    print("new_points.shape:",sess.run(new_points).shape)
    print("new_points_re.shape:",sess.run(new_points_re).shape)
    print("grouped_time.shape:",sess.run(grouped_time).shape)
    print("tf.concat结果的shape:",sess.run(ccat1).shape)
    print("new_points.shape:", sess.run(new_points).shape)
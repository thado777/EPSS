import tensorflow as tf

# 데이터
모델번호 = tf.constant([ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32)
등락률 = tf.constant([0.222
, 0.264, 0.382, 0.773, 0.080, 0.278, 0.156, 0.310, 0.093, 0.487], dtype=tf.float32)

# 변수
a = tf.Variable(0.765, dtype=tf.float32)
b = tf.Variable(0.12156, dtype=tf.float32)

# 예측값
def 예측값():
    return 모델번호 * a + b

# 손실 함수
def 손실함수():
    return tf.reduce_mean(tf.square(등락률 - 예측값()))

# 옵티마이저
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# 학습
for i in range(1000):
    opt.minimize(손실함수, var_list=[a, b])
    print(f"Iteration {i+1}: a = {a.numpy()}, b = {b.numpy()}")


print ((a*16+b)*100)
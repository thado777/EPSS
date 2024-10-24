import tensorflow as tf

# 데이터
모델번호 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32)
등락률 = tf.constant([10, 27, 24, 8, -9, 18, 0, 0, 13, 18, 0, -8, 5, 3, 8], dtype=tf.float32)

# 변수
a = tf.Variable(0.1, dtype=tf.float32)
b = tf.Variable(0.4, dtype=tf.float32)

# 예측값
def 예측값():
    return 모델번호 * a + b

# 손실 함수
def 손실함수():
    return tf.reduce_mean(tf.square(등락률 - 예측값()))

# 옵티마이저
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# 학습
for i in range(300):
    opt.minimize(손실함수, var_list=[a, b])
    print(f"Iteration {i+1}: a = {a.numpy()}, b = {b.numpy()}")

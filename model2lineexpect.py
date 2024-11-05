import tensorflow as tf

모델번호 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32)
등락률 = tf.constant([10, 27, 24, 8, -9, 18, 0, 0, 13, 18, 0, -8, 5, 3, 8], dtype=tf.float32)

a = tf.Variable(0.365, dtype=tf.float32)
b = tf.Variable(0.723, dtype=tf.float32)


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


print ((a*16+b))




모델번호cpu = tf.constant([ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32)
등락률cpu = tf.constant([0.222
, 0.264, 0.382, 0.773, 0.080, 0.278, 0.156, 0.310, 0.093, 0.475191816 ], dtype=tf.float32)


c = tf.Variable(0.865, dtype=tf.float32)
d = tf.Variable(0.12156, dtype=tf.float32)


def 예측값():
    return 모델번호cpu * c + d

# 손실 함수
def 손실함수():
    return tf.reduce_mean(tf.square(등락률cpu - 예측값()))

# 경사하강
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# 학습시키기
for i in range(1000):
    opt.minimize(손실함수, var_list=[c, d])
    print(f"Iteration {i+1}: c = {c.numpy()}, d = {d.numpy()}")

print ((a*16+b))
print ((c*16+d)*100)


#종합 4종목 합산을 100이라고 할때 내가 4개를 입력
print((c*16+d)*1955+1955)

e, f, g = map(int, input('반영비율(CPU,battery,memory)').split())
 
print(e/100*((c*16+d)*1955+1955) + f/100*4000 + g/100*16)
 
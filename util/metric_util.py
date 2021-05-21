import tensorflow as tf

def MAE(x_hat,
        x):
    mae = tf.reduce_mean(tf.log(1+tf.exp(tf.abs(x_hat-x))))
    return mae

def F_measure(gt,map):
    mask = tf.greater(map,0.5)
    mask = tf.cast(mask,tf.float32) # Hard Thresholding

    gtCnt = tf.reduce_sum(gt)

    hitMap = tf.where(gt>0,mask,tf.zeros(tf.shape(mask)))

    hitCnt = tf.reduce_sum(hitMap)
    algCnt = tf.reduce_sum(mask)

    prec = hitCnt / (algCnt + 1e-12)
    recall = hitCnt / (gtCnt + 1e-12)

    beta_square = 0.3
    F_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall + 1e-12)

    return  prec,recall,F_score

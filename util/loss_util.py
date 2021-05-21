import tensorflow as tf

# GDL with GDL_s weighting
def fuse_loss_for_saliency_detection(yp,
                                     gt):
    card_gt_p = tf.reduce_sum(gt)
    card_yp_p = tf.reduce_sum(yp)
    w_l_p = card_gt_p * card_gt_p
    card_gt_n = tf.reduce_sum(1-gt)
    card_yp_n = tf.reduce_sum(1-yp)
    w_l_n = card_gt_n * card_gt_n
    s_p = tf.multiply(gt,yp)
    s_p = tf.reduce_sum(s_p)
    s_n = tf.multiply(1-gt,1-yp)
    s_n = tf.reduce_sum(s_n)
    return 1-2*(w_l_p*s_p+w_l_n*s_n)/(w_l_p*(card_gt_p+card_yp_p)+w_l_n*(card_gt_n+card_yp_n))

def 

                                    

import tensorflow as tf

def _variable_scheduler(var_list, pivot_list, name=None):
    """Schedule variable according to the global step.
       e.g. var_list = [0.1, 0.01, 0.001], pivot_list = [0, 1000, 2000] then
         0    <= gstep < 1000 --> return 0.1
         1000 <= gstep < 2000 --> return 0.01
         2000 <= gstep        --> return 0.001
    Args:
      var_list: List of variables to return.
      pivot_list: List of pivots when to change the variable.
      name(Optional): Name of the operation.
    """
    assert(len(var_list) == len(pivot_list))
    if len(var_list) == 1:
        return tf.constant(var_list[0])

    def between(x, a, b):
        return tf.logical_and(tf.greater_equal(x, a), tf.less(x, b))

    # This class is necessary to declare constant lambda expressions
    class temp(object):
        def __init__(self, var):
            self.func = lambda: tf.constant(var)

    gstep = tf.to_int32(tf.train.get_global_step())
    conds = {}
    for idx in range(len(pivot_list)-1):
        min_val = tf.constant(pivot_list[idx], tf.int32)
        max_val = tf.constant(pivot_list[idx+1], tf.int32)
        conds[between(gstep, min_val, max_val)] = temp(var_list[idx]).func
    return tf.case(conds, default=temp(var_list[-1]).func, exclusive=True, name=name)

def learning_rate_schedule(var_list, pivot_list):
    """Learning rate scheduling. Wrapper of variable_scheduler.
    Args:
      var_list:
      pivot_list:
    """
    with tf.name_scope('learning_rate'):
        lr = _variable_scheduler(var_list, pivot_list)
        tf.add_to_collection('learning_rate', lr)
        return lr

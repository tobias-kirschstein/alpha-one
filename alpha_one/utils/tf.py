import tensorflow.compat.v1 as tf


def initialize_session():
    g = tf.Graph()
    with g.as_default():
        init = tf.variables_initializer(tf.global_variables(),
                                        name="init_all_vars_op")

    session = tf.Session(graph=g)
    session.__enter__()
    session.run(init)
    return session

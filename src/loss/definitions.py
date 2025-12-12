import tensorflow as tf

def get_weighted_mse(weights):
    """
    Returns a loss function with specific feature weights baked in.
    Args:
        weights: List or numpy array of shape (n_features,)
    """
    # Convert to TF Constant once
    # Reshape to (1, 1, n_features) for broadcasting against (Batch, Time, Feat)
    weight_tensor = tf.constant(weights, dtype=tf.float32)
    weight_tensor = tf.reshape(weight_tensor, [1, 1, -1]) 

    def weighted_loss(y_true, y_pred):
        # 1. Standard Squared Error
        error = tf.square(y_true - y_pred)
        
        # 2. Apply Weights
        weighted_error = error * weight_tensor
        
        # 3. Reduce (Sum features, then Mean batch/time)
        return tf.reduce_mean(tf.reduce_sum(weighted_error, axis=-1))
    
    return weighted_loss
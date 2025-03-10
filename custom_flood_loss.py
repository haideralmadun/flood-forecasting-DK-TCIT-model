
import tensorflow as tf

def huber_loss(delta=1.0):
    """
    Returns a Huber loss function for the given delta.

    Parameters:
        delta: Float, the threshold where squared loss transitions to linear loss.

    Returns:
        A loss function that can be used in model compilation.
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.reduce_mean(tf.where(is_small_error, squared_loss, linear_loss))

    return loss


def custom_flood_loss(y_true, y_pred):
    # Custom loss function for flood forecasting

    # Define domain-specific parameters
    #peck_point_5 = 0.0831724  # NO. 330-> 540
    peck_point_4 = 0.0555906  # NO. 329-> 361
    peck_point_3 = 0.0454785  # NO. 328-> 295
    peck_point_2 = 0.0353664  # NO. 327-> 229
    peck_point_1 = 0.0252544  # NO. 326-> 164
    flood_threshold = 0.0153683  # Threshold above which flow rates are considered flood events

    # Weight parameters for penalizing errors
    alpha  = 0.2  
    alpha1 = 0.6
    alpha2 = 0.7
    alpha3 = 0.8
    alpha4 = 0.9
    #alpha5 = 1.0    
    beta = 0.5   # Regularization weight
    delta = 1.0  # Delta for Huber loss

    # Calculate Huber loss for each prediction
    huber_loss_fn = huber_loss(delta)
    
    # Separate losses during different peck points and flood events
    peck_point_4_loss = tf.where(y_true > peck_point_4, huber_loss_fn(y_true, y_pred), tf.zeros_like(y_true))
    #peck_point_4_loss = tf.where((y_true > peck_point_4) & (y_true <= peck_point_5), huber_loss_fn(y_true, y_pred), tf.zeros_like(y_true))
    peck_point_3_loss = tf.where((y_true > peck_point_3) & (y_true <= peck_point_4), huber_loss_fn(y_true, y_pred), tf.zeros_like(y_true))
    peck_point_2_loss = tf.where((y_true > peck_point_2) & (y_true <= peck_point_3), huber_loss_fn(y_true, y_pred), tf.zeros_like(y_true))
    peck_point_1_loss = tf.where((y_true > peck_point_1) & (y_true <= peck_point_2), huber_loss_fn(y_true, y_pred), tf.zeros_like(y_true))
    flood_loss = tf.where((y_true > flood_threshold) & (y_true <= peck_point_1), huber_loss_fn(y_true, y_pred), tf.zeros_like(y_true))
    non_flood_loss = tf.where(y_true <= flood_threshold, huber_loss_fn(y_true, y_pred), tf.zeros_like(y_true))

    # Penalize errors during flood peck events more heavily
    #peck_point_5_loss = tf.reduce_mean(peck_point_5_loss) * alpha5
    peck_point_4_loss = tf.reduce_mean(peck_point_4_loss) * alpha4
    peck_point_3_loss = tf.reduce_mean(peck_point_3_loss) * alpha3
    peck_point_2_loss = tf.reduce_mean(peck_point_2_loss) * alpha2
    peck_point_1_loss = tf.reduce_mean(peck_point_1_loss) * alpha1

    # Penalize errors during flood events more heavily
    flood_loss = tf.reduce_mean(flood_loss) * alpha

    # Include non-flood loss in the total loss
    non_flood_loss = tf.reduce_mean(non_flood_loss)

    # Regularization term to encourage smoothness in predicted flow rates (L2 regularization)
    regularization_term = beta * tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]))

    # Total loss
    loss = peck_point_1_loss + peck_point_2_loss + peck_point_3_loss + peck_point_4_loss + flood_loss + non_flood_loss + regularization_term

    return loss


import tensorflow as tf

# sparse_features = ['C1', 'banner_pos', 'site_category', 'app_category',
#                    'device_type', 'device_conn_type', 'C18', 'hour', 'is_device', 'C_pix']

# dense_features = ['C_site_id', 'C_site_domain', 'C_app_id', 'C_app_domain', 'C_device_ip',
#                   'C_device_model', 'C_C14', 'C_C17', 'C_C19', 'C_C20', 'C_C21']

# sparse_features = ['C1', 'banner_pos', 'site_category', 'app_category',
#                         'device_type', 'device_conn_type', 'C18', 'hour', 'is_device', 'C_pix', 'C_site_id',
#                         'C_site_domain', 'C_app_id', 'C_app_domain', 'C_device_ip',
#                         'C_device_model', 'C_C14', 'C_C17', 'C_C19', 'C_C20', 'C_C21']
# dense_features_test = ['site_id', 'site_domain', 'app_id', 'app_domain', 'device_id',
#                   'device_model', 'C14', 'C17', 'C19', 'C20', 'C21']
# dense_features_test = ['site_id', 'site_domain', 'app_id', 'device_id', 'device_ip', 'device_model', 'C14', 'C17',
#                        'C20']
dense_features = ['site_id', 'site_domain', 'app_id', 'app_domain', 'device_id',
                   #'device_ip',
                  'device_model', 'C14', 'C17', 'C19', 'C20', 'C21']

categorical_features = ['hour', 'C1', 'banner_pos',
                        'site_category', 'app_category',
                        'device_type', 'device_conn_type', 'C15', 'C16', 'C18']

sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                   'site_category', 'app_id', 'app_domain', 'app_category', #'device_ip',
                   'device_id',
                   'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18',
                   'C19', 'C20', 'C21']
target = ['click']

METRICS = [
    # tf.keras.metrics.TruePositives(name='tp'),
    # tf.keras.metrics.FalsePositives(name='fp'),
    # tf.keras.metrics.TrueNegatives(name='tn'),
    # tf.keras.metrics.FalseNegatives(name='fn'),
    # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    # tf.keras.metrics.Precision(name='precision'),
    # tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
    tf.keras.metrics.AUC(name='auc'),
    # tf.keras.metrics.Recall(name='recall'),
]

prefix_dir = './result/'

data_type = "raw"

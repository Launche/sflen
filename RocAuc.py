from sklearn.metrics import roc_auc_score
import tensorflow as tf


# class RocAuc(tf.keras.callbacks.Callback):
#     def __init__(self, validation_generate, interval=1):
#         self.interval = interval
#         self.validation_generate = validation_generate
#
#     def on_epoch_end(self, epoch, logs={}):
#         # 每次epoch,读取一批生成的数据
#         x_val, y_val = next(self.validation_generate)
#         # print(y_val)
#         if epoch % self.interval == 0:
#             try:
#                 y_pred = self.model.predict(x_val, verbose=0)
#                 score = roc_auc_score(y_val, y_pred)
#                 print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch + 1, score * 100))
#             except:
#                 print('\n  epoch:%d  only one class!!\n' % (epoch + 1))

class RocAuc(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

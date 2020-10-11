import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback


class Mylosscallback(Callback):
    def __init__(self, log_dir):
        super(Callback, self).__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.num = 0

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.num = self.num + 1
        # print(logs.keys())
        self.losses = logs.get('loss')
        self.accuracy = logs.get('accuracy')
        # self.val_loss = logs.get('val_loss')
        # self.val_acc = logs.get('val_acc')
        tf.summary.scalar("losses", self.losses,self.num)
        tf.summary.scalar("accuracy", self.accuracy,self.num)
        # print('debug success!!!')
        # summary = tf.Summary()
        # summary.value.add(tag='losses', simple_value=self.losses)
        # summary.value.add(tag='accuracy', simple_value=self.accuracy)
        # summary.value.add(tag='val_loss', simple_value=self.val_loss)
        # summary.value.add(tag='val_acc', simple_value=self.val_acc)

        self.writer.flush()



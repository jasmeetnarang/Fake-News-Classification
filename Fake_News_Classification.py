"""
Project : Fake News Classification
Author: Jasmeet Narang

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import matplotlib.pyplot as plt
from sklearn import metrics
import csv

max_length = 128  # Maximum length of input sentence to the model.
def_batch_size = 256
epochs = 5


def readfile(filename):
    data = pd.read_csv(filename)
    return data


def splitXY(data):
    # all columns except the last are features
    features = data.iloc[:, 3:5]
    label = data.iloc[:, -1]

    return features, label


train = readfile('train.csv')

features, label = splitXY(train)

labels = np.array(pd.get_dummies(label))

training_samples = 2 * int(features.shape[0] / 3)
validation_samples = features.shape[0] - training_samples


x_train = features[:training_samples]
y_train = labels[:training_samples]

x_val = features[training_samples:]
y_val = labels[training_samples:]


# Create a data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            sps,
            outputs,
            shuffle=True,
            batch=def_batch_size,
            include_targets=True,
    ):
        self.sps = sps
        self.labels = outputs
        self.shuffle = shuffle
        self.batch_size = batch
        self.include_targets = include_targets

        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sps))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.sps) // self.batch_size) + 1

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        curr_sentence_pairs = self.sps[indexes]

        encoded = self.tokenizer.batch_encode_plus(
            curr_sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors="tf",
            truncation=True
        )

        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


input_ids = tf.keras.layers.Input(
    shape=(max_length,), dtype=tf.int32, name="input_ids"
)

attention_masks = tf.keras.layers.Input(
    shape=(max_length,), dtype=tf.int32, name="attention_masks"
)

token_type_ids = tf.keras.layers.Input(
    shape=(max_length,), dtype=tf.int32, name="token_type_ids"
)

bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
# Freeze the BERT model to reuse the pretrained features without modifying them.
bert_model.trainable = False

sequence_output, pooled_output = bert_model(
    input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
)

bi_lstm = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, return_sequences=True)
)(sequence_output)

avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
concat = tf.keras.layers.concatenate([avg_pool, max_pool])
dropout = tf.keras.layers.Dropout(0.3)(concat)
output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
model = tf.keras.models.Model(
    inputs=[input_ids, attention_masks, token_type_ids], outputs=output
)

#compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["acc"],
)

model.summary()

train_data = DataGenerator(
    x_train[['title1_en', 'title2_en']].values.astype("str"),
    y_train,
    batch=def_batch_size,
    shuffle=True,
)
valid_data = DataGenerator(
    x_val[['title1_en', 'title2_en']].values.astype("str"),
    y_val,
    batch=def_batch_size,
    shuffle=False
)

# to retrain
# model.load_weights("models/BERT_weights_bidirectional.h5")

# perform fitting
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)

model.save_weights("BERT_weights_bidirectional.h5")

model.load_weights("BERT_weights_bidirectional.h5")


test = readfile('test.csv')

test_features = test.iloc[:, 3:5]

test_data = DataGenerator(
    test_features[['title1_en', 'title2_en']].values.astype("str"),
    [],
    batch=def_batch_size,
    shuffle=False,
    include_targets=False
)

#test the model on test dataset
prediction = model.predict(test_data, verbose=1)

y_pred = np.argmax(prediction, axis=1)


test_labels = []

for y in y_pred:
    if y == 0:
        test_labels.append('agreed')
    elif y == 1:
        test_labels.append('disagreed')
    elif y == 2:
        test_labels.append('unrelated')

test_labels = np.array(test_labels)
ids = test['id'].to_numpy()
titles = ['id', 'label']

with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([t for t in titles])
    writer.writerows(zip(ids, test_labels))

prediction = model.predict(valid_data, verbose=1)

y_pred = np.argmax(prediction, axis=1)


val_labels = []
for y in y_pred:
    if y == 0:
        val_labels.append('agreed')
    elif y == 1:
        val_labels.append('disagreed')
    elif y == 2:
        val_labels.append('unrelated')
val_pred = np.array(val_labels)
ids = train['id'].to_numpy()
titles = ['id', 'label']
with open('val_labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([t for t in titles])
    writer.writerows(zip(ids, test_labels))

y_cal_cm = np.argmax(np.array(y_val[:len(y_pred)]), axis=1)


# print the metrics
print(metrics.confusion_matrix(y_cal_cm, y_pred))
print(metrics.classification_report(y_cal_cm, y_pred, digits=3))


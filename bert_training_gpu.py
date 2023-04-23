import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from tensorflow.keras.layers import Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import nltk
nltk.download('punkt')

print("defining stratergy variable")
strategy = tf.distribute.MirroredStrategy()

print("importing Completed")

with strategy.scope():
    bert_model = TFBertModel.from_pretrained('bert-base-uncased' )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',)
    print("model and tokenizer downloading completed")

with open('preprocessed_text_file.txt','r',encoding='utf8') as file:
    text = file.readlines()
text_list = text
for item_index in range(len(text_list)):
    text_list[item_index]=text_list[item_index].replace('\n',' ') 
print("Reading files completed")


nltk.word_tokenize(text_list[0]),len(nltk.word_tokenize(text_list[0]))
text = []
sentence_length = 124
for item in text_list:
    if len(nltk.word_tokenize(item)) < sentence_length :
        text.append(item)
    elif len(nltk.word_tokenize(item)) >= sentence_length:

        while(len(nltk.word_tokenize(item)) >= sentence_length):
#             print(item,end='\n')
            text.append(item[:sentence_length])
            item = item[sentence_length:]
#             print(item,end='\n')

print("truncating sentences to 124 length completed")

text_list[728],text[728]

encoded_data = tokenizer.batch_encode_plus(text,add_special_tokens=True,
                                           return_attention_mask=True,return_token_type_ids=False,
                                           pad_to_max_length=True,max_length=500,truncation=True)
print("data encodedation completed")
dataset = tf.data.Dataset.from_tensor_slices(({'input_ids':encoded_data['input_ids'],
                                              'attention_mask':encoded_data['attention_mask']},
                                             np.zeros(len(encoded_data['input_ids']))))
print("Dataset Created")


with strategy.scope():
    input_ids = Input(shape=(None,),dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    bert_output = bert_model({'input_ids':input_ids,
                            'attention_mask':attention_mask})[0]
    embedding_layer = GlobalAveragePooling1D()(bert_output)
    embedding_model = Model(inputs=[input_ids,attention_mask], outputs=embedding_layer)
    print("Model Initialised")

#compile model
with strategy.scope():
    embedding_model.compile(optimizer=tf.optimizers.Adam(),
                        loss=tf.losses.CosineSimilarity(),
                        metrics=[tf.metrics.CosineSimilarity()])
    print("Model Compilation Completed")
# train model

with strategy.scope():
    early_stopping = EarlyStopping(monitor='loss',patience=3,restore_best_weights=True)
    print("Early Stopping Defined")
    print("Starting model Fitting ....................")
    embedding_model.fit(dataset.batch(32),epochs=10,callbacks=[early_stopping])


print("saving model for further checking...")
embedding_model.save("./outputs/embedding_model.h5")
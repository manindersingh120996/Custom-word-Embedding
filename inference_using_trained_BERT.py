import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from sklearn.neighbors import NearestNeighbors

bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Defining Custom_object as this trained model is having TFBertModel Layer which tensorflow does not understands by default.
# Load trained embedding model
embedding_model = tf.keras.models.load_model('embedding_model.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

# Get word embedding for all words in the vocabulary
vocabulary = ['word1', 'word2', 'word3', 'word4', 'word5']
vectors = []
for word in vocabulary:
    input_ids = tokenizer.encode(word, add_special_tokens=True, return_tensors='tf')
    vector = embedding_model.predict({'input_ids': input_ids, 'attention_mask': tf.ones_like(input_ids)})[0]
    vectors.append(vector)

# than as shown in the code below we can use "nn.kneighbor" 
# to find the nearest word

    # Build nearest neighbors model
k = 1
nn = NearestNeighbors(n_neighbors=k)
nn.fit(vectors)


# Find nearest word to input word
input_word = 'word'
input_vector = embedding_model.predict({'input_ids': tokenizer.encode(input_word, add_special_tokens=True, return_tensors='tf'), 'attention_mask': tf.ones((1, len(tokenizer.encode(input_word))))})[0]
distances, indices = nn.kneighbors([input_vector])
nearest_word = vocabulary[indices[0][0]]

print(f'Nearest word to "{input_word}" is "{nearest_word}"')

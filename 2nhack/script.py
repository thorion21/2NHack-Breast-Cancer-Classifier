from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from keras.preprocessing.image import img_to_array

# load model
loaded_model = load_model('demo_fep.h5')
# summarize model.
#loaded_model.summary()
# load dataset

X = load_img("mlpass_test.png", target_size=(48,48))
X = img_to_array(X)
X = np.expand_dims(X, axis=0)
print(X.shape)

Y = loaded_model.predict(X, batch_size=1)
print(Y)

# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(np.array(X), np.array(Y), verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
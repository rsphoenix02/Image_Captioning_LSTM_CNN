import pickle as pickle
from keras.utils import load_img, img_to_array
from vgg16 import VGG16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input	

counter = 0

def load_image(path):
    img = load_img(path, target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.asarray(x)

def load_encoding_model():
	model = VGG16(include_top=True, weights='imagenet', input_shape = (224, 224, 3))
	return model

def get_encoding(model, img):
	global counter
	counter += 1
	p = "G:\\rsphoenix02\\caption_generator\\Flicker8k_Dataset\\"+str(img)
	image = load_image(p)
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	print("Encoding image: "+str(counter))
	print(pred.shape)
	return pred

def prepare_dataset(no_imgs = -1):
	f_train_images = open("G:\\rsphoenix02\\caption_generator\\Flickr8k_text\\Flickr_8k.trainImages.txt",'r')
	
	# train_imgs = []

	# for line in f_train_images.readlines():
	# 	line = line.rstrip('\n')
	# 	if not line:
	# 		continue
	# 	train_imgs.append(line)

	train_imgs = f_train_images.read().split('\n') if no_imgs == -1 else f_train_images.read().strip().split('\n')[:no_imgs]
	# print(train_imgs)
	f_train_images.close()

	f_test_images = open("G:\\rsphoenix02\\caption_generator\\Flickr8k_text\\Flickr_8k.testImages.txt",'r')
	test_imgs = f_test_images.read().split('\n') if no_imgs == -1 else f_test_images.read().strip().split('\n')[:no_imgs]
	f_test_images.close()

	f_train_dataset = open('G:\\rsphoenix02\\caption_generator\\Flickr8k_text\\flickr_8k_train_dataset','w')
	f_train_dataset.write("image_id\tcaptions\n")

	f_test_dataset = open('G:\\rsphoenix02\\caption_generator\\Flickr8k_text\\flickr_8k_test_dataset','w')
	f_test_dataset.write("image_id\tcaptions\n")

	f_captions = open("G:\\rsphoenix02\\caption_generator\\Flickr8k_text\\Flickr8k.token.txt", 'r')
	captions = f_captions.read().split('\n')
	# print(captions)

	descriptions = {}

	for caption in captions:
		first, second = caption.split('\t')
		img_name = first.split(".")[0]
		img_name += ".jpg"

		if descriptions.get(img_name) is None:
			descriptions[img_name] = []

		if second != '':
			descriptions[img_name].append(second)
	f_captions.close()

	#print(descriptions)

	encoded_images = {}
	encoding_model = load_encoding_model()

	c_train = 0
	for img in train_imgs:
		encoded_images[img] = get_encoding(encoding_model, img)
		for capt in descriptions[img]:
			caption = "<start> "+capt+" <end>"
			f_train_dataset.write(img+"\t"+caption+"\n")
			f_train_dataset.flush()
			c_train += 1
	f_train_dataset.close()

	c_test = 0
	for img in test_imgs:
		encoded_images[img] = get_encoding(encoding_model, img)
		for capt in descriptions[img]:
			caption = "<start> "+capt+" <end>"
			f_test_dataset.write(img+"\t"+caption+"\n")
			f_test_dataset.flush()
			c_test += 1
	f_test_dataset.close()
	with open( "encoded_images.p", "wb" ) as pickle_f:
		pickle.dump( encoded_images, pickle_f )  
	return [c_train, c_test]

if __name__ == '__main__':
	c_train, c_test = prepare_dataset()
	print("Training samples = "+str(c_train))
	print("Test samples = "+str(c_test))

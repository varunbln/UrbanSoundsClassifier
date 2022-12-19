# Urban Sounds Classifier
A Convolutional Neural Network(CNN) model to perform multi-label classification on audio data by extracting spectrograms. 
## Dataset
This project will be using the UrbanSounds8K dataset. This dataset contains 8732 labelled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn,children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. All excerpts are taken from field recordings uploaded to http://www.freesound.org/  
## Tech Stack
The classifier model is built using Keras interface for Tensorflow. The model was trained in a free-tier colab and the weights were saved to reconstruct the model on the server. The frontend of the website is made using vanilla HTML, CSS and JavaScript. The backend has been made using FastAPI in Python.  
## Final Metrics
Accuracy, precision, and recall are used to measure and grade the performance of the model. The loss metric used is categorical cross-entropy.
![image](https://user-images.githubusercontent.com/25721272/208441317-830f7ad3-a545-4a9e-9b7a-a344d947f652.png)  
![image](https://user-images.githubusercontent.com/25721272/208441745-58d3c99b-0475-48c0-af77-af826bc4d8a7.png)


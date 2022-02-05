# Age-and-Gender-Recognition-System Using Convolutional Neural Networks

Objective:

We attempt to close the gap between automatic face recognition capabilities and those of age and gender estimation methods. To this end, we follow the successful example laid down by recent face recognition systems: Face recognition techniques described in the last few years have shown that tremendous progress can be made by the use of deep convolutional neural networks. We demonstrate similar gains with a simple network architecture, designed by considering the rather limited availability of accurate age and gender labels in existing face data sets. 

In this proposed system, we developed a deep learning model using Convolutional neural networks and tkinter framework. The model receives the image, passes it through different layers by reducing the size in each layer. We then train our model using the which contains over 5 lakh images of human beings which include different ethnicity, color, and many more factors. The dataset also provides a label for each image. Once the model is trained, we can use the model for testing. We first detect the presence of a human in each image. Then, we process the image using the Open CV’s BlobFromImage function to obtain all the human faces present in the image. Each face is then processed by a developed deep learning model to get the output label which essentially gives us the age and gender of each person in the image.

Technologies Used:
Python, CNN, Open CV

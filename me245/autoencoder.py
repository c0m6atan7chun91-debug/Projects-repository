import torch.nn as nn
'''
The autoencoder class was developed using the help of a tutorial: https://www.youtube.com/watch?v=zp8clK9yCro&t=214s and https://www.youtube.com/watch?v=VVDHU_TWwUg
It only was used to establish syntax and theory in regarding the coding side of pytorch. I still had to change this section with my own theory and found syntax to the project.
'''
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__() # required to initialize the parent class that is nn to use its functions
        
        #Define the NN layers for the autoencoder
        '''
        we have 15 features and have 60 data points being entered into the dataset
        So the linear layer also known as the input layer will be condensing our input features into the NN
        60 days to cover a quater of the year and each day has 15 features that has been extracted from it so 60*15 = 900 features to enter the neural network
        
        As there is a tanh function it will only output values between 1 and -1, which is suitable for normalized datasets
        '''
        #Define the encoder:
        self.encoder = nn.Sequential(
            nn.Linear(900,90),#input layer so 900 represents the total features and 90 represents the vector it gets compressed into for the next input space
            nn.Tanh(),#activation function
            nn.Linear(90,45),#hidden layer
            nn.Tanh()#activation function at this point there are 60 samples having its features compressed into 45 values in a input space
            #Don't use the ReLU to allow negative values from the normalization. If I used ReLU function there will be no negative values to return so it will destroy performance.
            #You should use Tanh as it maps the values between -1 and 1 which is slightly smaller scale than a normalized dataset but still would get the model to learn the fundermental patterns. Using a adjusted scale.
        )
        
        #For the decoder it needs to do the reverse operations of the encoder to recreate the dataset
        self.decoder = nn.Sequential(
            nn.Linear(45,90),#hidden layer
            nn.Tanh(),
            nn.Linear(90,900)#output layer and no activation function required as the raw output should be the normalized values limiting it between 1 and -1 would skew the MSEs evaluation of the output
        )  
        
    def forward(self,input_for_neuron):
        encoded = self.encoder(input_for_neuron)
        decoded = self.decoder(encoded)
        return decoded
    
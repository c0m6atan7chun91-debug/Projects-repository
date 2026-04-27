from autoencoder import Autoencoder
import torch
import torch.nn as nn


class AutoencoderModelManager():
    def __init__(self):
        self.autoencoder_model = Autoencoder() #Establish the model we use to execute said calculations
        self.training_criterion = nn.MSELoss() #You do this to establish the function that will be used to compare the input and output values of the autoencoder, so it is the criterion
        self.prediction_criterion = nn.MSELoss(reduction='none') #reduction='none' instead of allowing it to return one average loss value for the entire batch entered (like the training one). It returns a tensor of the same shape as the input (meaning each input sequence has an assigned loss value).
        self.optimization = torch.optim.Adam(self.autoencoder_model.parameters(), lr = 0.0001) #this updates the weights based on the learning rate which controls how much these weights are updated by. Also, self.autoencoder_model.parameters() directs which weights to update for the ML algorithm and lr is in reference to lr
        #Adam seems to be the best optimizer based from this source: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
        
    def training_loop(self, sequences):
        #A sequence is how many 60 day sliding windows we can recreate from a given training dataset
        #An epoch is how many times the model goes over a training dataset during training (forward and backwards (backwards is to check the weights and see if they need to be changed))
        for epoch in range(20):
            for batch_increment in range(0, len(sequences), 32):
                #do the forward then backward pass of the loop and update the weights
                batch = sequences[batch_increment : batch_increment+32]
                input_vector_sequence = batch.reshape(-1,900) #doing -1 will infer the 2nd dimension and the 900 will say how many data points are expected per column. as 60*15 is = 900 it will create a vector with 900 elements. So it takes in batches of 32 sequences at a time
                reconstructed_sequence_data = self.autoencoder_model(input_vector_sequence) #enter the required sequence for training into the model to create a model output of it
                loss = self.training_criterion(reconstructed_sequence_data, input_vector_sequence) #you check the difference between the reconstructed and input sequence by using the MSE loss function to find how different it is
                
                self.optimization.zero_grad() #this clears the gradients history that was previously calculated, otherwise you will have incremented the sum of the previous one
                loss.backward()#Go back across the dataset using chain rule to check the derivatives and see how the loss function to see the d loss/ d Weight for the gradient of the loss against the weight function.
                self.optimization.step()#this updates the weights of the autoencoder to try and minimize the difference between the output from the autoencoder and the input to the autoencoder. It then uses the gradient as such: weight - lr * gradient.
            print(f"Epoch: {epoch + 1 }, Loss: {loss.item():.4f}") #loss is the difference between the constructed and reconstructed output
    
    #This loop was made entirely without tutorials
    def prediction(self,sequences):
        '''
        To understand what makes an anomaly in this case we will have to look at how the model is trained. It needs to be trained on a large pool of data to understand what is considered a normal dataset.
        So assuming it has learned what is a normal behaviour from that; datasets that are entered should have "normal" periods to the anomaly detection system. So if it cannot be recreated then it must be an extreme outlier compared to what is normal market data.
        So if a sequence is above the mean of the loss and 2 times more than standard deviations summed up in a one tailed test then it is an anomaly.
        It is using a one tailed test as you can't get an error below zero has MSE uses the square difference.
        '''
        list_of_loss_values = [] #append mean value of a sequence and the starting index (which is the ordinal index that is the same to the dfs so i can find the starting date)
        self.autoencoder_model.eval() #this sets the model into evaluation mode. It disables dropout and batch normalization layers
        #Ensures that it stops making a computation graph during a forward pass. So it doesn't take up more memory.
        with torch.no_grad():
            for sequence in sequences:
                input_vector_sequence = sequence.reshape(-1,900) #doing -1 will infer the 2nd dimension and the 900 will say how many data points are expected per column. as 60*15 is = 900 it will create a vector with 900 elements
                reconstructed_sequence_data = self.autoencoder_model(input_vector_sequence) #enter the required sequence for training into the model to create a model output of it
                losses_per_sequence = self.prediction_criterion(reconstructed_sequence_data, input_vector_sequence).mean(dim=1) #you check the difference between the reconstructed and input sequence by using the MSE loss function to find how different it is and .mean() returns each sequences loss values from a batch in a vector of 1 dimension but as an average for that sequence
                for loss in losses_per_sequence:
                    list_of_loss_values.append(loss.item())
                
            #Now we have the list of loss values and each index is corresponding to the date it starts at from the df and dataset it is from
            #convert it to a torch temporarily to a tensor. This is becuase Tensors build off of the numpy extension meaning their sizes are fixed
            tensor_of_loss_values = torch.tensor(list_of_loss_values)
            mean_loss = tensor_of_loss_values.mean() #returns a tensor and in this case is the mean of the mean of sequences
            std_loss = tensor_of_loss_values.std() #return a tensor  so need to use .item() to return the actual std.std reconstruction errors are used to create the threshold to find how many lie in a 99% confidence based on the relative distance from the mean of sequences more or less a span of how much the population should lie in
            #This is where I found the measuremetn of a 99% confidence interval for a one tailed test: https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Statistics_with_Technology_2e_(Kozak)/12%3A_Appendix-_Critical_Value_Tables/12.02%3A_Normal_Critical_Values_for_Confidence_Levels
            #so you use the 98% confidence interval (which is for two tail tests) as we are using one tail it will cover 99% of the population so it is 2.33 std away from the mean
            list_of_loss_values_and_dates = [i for i, value in enumerate(list_of_loss_values) if value > 2.33* std_loss.item() + mean_loss.item()]
            return list_of_loss_values_and_dates
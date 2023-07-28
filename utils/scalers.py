from sklearn import preprocessing


def standard_scaler(scaler=None, Xtrain=None):
    
    # Initialize a scaler, if none exists
    if scaler == None:
        scaler = preprocessing.StandardScaler()
    else:
        scaler = scaler.fit(Xtrain)  # Only fit on training data
    # Xtrain = scaler.transform(Xtrain)
    # Xtest = scaler.transform(Xtest)
                            
    return scaler
                            
def penultimate_processing(type='standard_scaling'):
    """
    Performs data processing (e.g. standard scaling) on values in the 
    penultimate layer of a neural network. e.g. just before the final
    fully connected layer, we may choose to perform standard scaling, fitted
    on all the results of the training set. Then transform all results again
    of the penultimate hidden values for the training, validation, and test set.
    """
    processed_values = None
    
    return processed_values

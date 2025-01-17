Dataset = imageDatastore('Dataset1', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[Training_Data, Validation_Data] = splitEachLabel(Dataset, 0.7,'randomized');

net = squeezenet;
analyzeNetwork(net)



Input_Layer_Size = net.Layers(1).InputSize(1:2);
Resized_Training_Data = augmentedImageDatastore(Input_Layer_Size, Training_Data);
Resized_Validation_Data = augmentedImageDatastore(Input_Layer_Size, Validation_Data);

Network_Architecture = layerGraph(net);

Number_of_Classes = numel(categories(Training_Data.Labels));

New_Convolutional_Layer = convolution2dLayer([1, 1], Number_of_Classes, ...
    'WeightLearnRateFactor', 10, ...2
    'BiasLearnRateFactor', 10, ... 
    'Name', 'Facial Feature Learner'); 

New_Classification_Layer = classificationLayer('Name', 'Face Classifier');

New_Network = replaceLayer(Network_Architecture, 'conv10', New_Convolutional_Layer);
New_Network = replaceLayer(New_Network, 'ClassificationLayer_predictions', New_Classification_Layer);
    

Training_Options = trainingOptions('sgdm', ...
    'MiniBatchSize', 4,  ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 4e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Training_Data, ... 
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

Trained_Network = trainNetwork(Resized_Training_Data, New_Network, Training_Options);

save('Face_Recognizer.mat', 'Trained_Network');
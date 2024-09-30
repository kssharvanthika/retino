% Load your image dataset and corresponding labels
% Assuming you have two variables: images and labels

% Preprocess the images and labels accordingly

% Load the SqueezeNet model
% Assuming you have already loaded and compiled the model
loaded_Network = load('Face_Recognizer.mat');
net = loaded_Network.Trained_Network;
% Predict the labels for the images
predictedLabels = classify(net, Resized_Validation_Data);

% Convert the categorical predicted labels to numeric labels
predictedLabels = grp2idx(predictedLabels);

% Create the confusion matrix
cm = confusionmat(labels, predictedLabels);

% Define class labels for visualization
classLabels = {'Advanced PDR', 'Mild(or early) NPDR', 'Moderate NPDR','PDR','Severe NPDR','No DR signs','Vert Severe NPDR'};  % Replace with your actual class labels

% Plot the confusion matrix
figure;
heatmap(classLabels, classLabels, cm);
title('Confusion Matrix');
xlabel('Predicted Label');
ylabel('True Label');
colorbar;

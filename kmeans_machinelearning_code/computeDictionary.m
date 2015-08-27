clear all;
load('data.mat'); % load image paths and filter bank  


%--------------------------------------
%   YOUR CODE STARTS HERE
%--------------------------------------

Sample = [];
%For each image in the dataset
for i = 1:length(imagePaths)
    
    %get the image path
    img_path = imagePaths{1};
    
    %read the image
    I = imread(img_path);
    
    %extract feature points
    featurePoints = extractFilterResponses(I, filterBank);
    
    
    %TODO: store just a random subsample of the feature points
    N   = size(featurePoints,1);
    idx = randperm( N); 
    sample = featurePoints( idx(1: floor(.01*N)),:);
    Sample = [ Sample; sample];
end


    
%TODO: apply K-Means to the set of all the stored feature vectors
k = 100; 
p = 99; 
n = size( Sample, 1);
idx = randperm(n); 
center0 = Sample ( idx(1:k),:);
[center, label] = TAsolution(Sample, 100, center0);
%TODO: Save dictionary
dictionary = center; 
save('dictionary.mat','dictionary');



%--------------------------------------
%   YOUR CODE ENDS HERE
%--------------------------------------



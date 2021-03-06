function act = nnComputeActivations(theta, X, output_size, opt)
%NNCOMPUTEACTIVATIONS  Compute the activations from the last layer of the
%                     neural network.
%
% function act = nnComputeActivations(theta, X, output_size, opt)
%
% theta       - learned parameter vector of all weights in the NN.
% X           - m x n design matrix. Note that in this case we will not
%               add a bias vector to the X's directly, since we are not
%               going to regularize the biases.
% layer_sizes - size of hidden layers and output layer.
% opt         - NN options
%
% act - activations of the output layer.
%
% Example usage:
%  acts = nnComputeActivations(theta, X, layers, opt);
%  acts = nnComputeActivations(theta, X, [hidsizes; outsize], opt);
%
    [m, visible_size] = size(X);
    layer_sizes = [visible_size; opt.hidden_sizes; output_size];
    
    [Ws, bs] = unflattenParameters(theta, layer_sizes);
        
    %% Compute the activations of the output layer. Our solution is approx 
    %  10 lines.
    
    % NOT YET IMPLEMENTED %

    act = zeros(output_size, m);

    %% BEGIN SOLUTION
    act = X';
    for i = 1:length(layer_sizes)-1
        act = sigmoid(bsxfun(@plus, Ws{i}*act, bs{i}));
    end
    %% END SOLUTION
end

function y = sigmoid(x)
    y = 1./(1.+exp(-x));
end
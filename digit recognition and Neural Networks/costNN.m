function [cost, grad] = costNN(X, y, theta, opt)
% costNN  Neural network cost function.
%
% function [cost, grad] = costNN(X, y, theta, opt)
%
%   X     - m x n design matrix of m data points.
%   y     - k x m labels.
%   theta - flattened parameters for NN.
%   opt   - Struct must contain:
%          lambda        - regularization strength
%          hidden_sizes  - vector of number of units in each hidden layer. 
%                          In the case of a single hidden layer NN this 
%                          will be a scalar.
%
%   cost - cost at theta
%   grad - gradient at theta. i.e. [dJ/dx1, ..., dJ/dxp]'
%
    visible_size = size(X, 2);
    hidden_size = opt.hidden_sizes;
    output_size = size(y, 1);
    
    n_layers = length(opt.hidden_sizes) + 1;

    all_layer_sizes = [visible_size; opt.hidden_sizes; output_size];
    
    [Ws, bs] = unflattenParameters(theta, all_layer_sizes);
    
    % You may find the following variables helpful.
    
    Wgrads = cell(n_layers, 1); % Wgrads{i} = Wigrad
    bgrads = cell(n_layers, 1); % bgrads{i} = bigrad
    
    % in the case of single hidden layer NN.
    %      layer 1 = input layer
    %      layer 2 = hidden layer
    %      layer 3 = output layer
    
    %% Write your code below to compute the cost and gradients.
    % our solution is ~14 lines. You may assume that the NN has a single
    % hidden layer.

    cost = 0;

    % NOT YET IMPLEMENTED %
    
    %% BEGIN SOLUTION
    
    % forward propagation
    acts = cell(n_layers+1,1);
    acts{1} = X';
    for i = 1:n_layers
        acts{i+1} = sigmoid(bsxfun(@plus, Ws{i}*acts{i}, bs{i}));
    end
    
    % cost computation
    diff = acts{end} - y;
    cost = 0.5*sum(diff(:).^2) + 0.5*opt.lambda*sum(cellfun(@(w) sum(w(:).^2), Ws));

    if nargout < 2, return; end % only need cost.
    
    delta = diff.*acts{n_layers+1}.*(1-acts{n_layers+1});
    Wgrads{n_layers} = delta*acts{n_layers}' + opt.lambda*Ws{n_layers};
    bgrads{n_layers} = sum(delta, 2);
    
    % back propagation and grad computation
    for i = n_layers-1:-1:1
        delta = Ws{i+1}'*delta.*acts{i+1}.*(1-acts{i+1});
        Wgrads{i} = delta*acts{i}' + opt.lambda*Ws{i};
        bgrads{i} = sum(delta, 2);
    end
    
    %% END SOLUTION
    
    grad = flattenParameters(Wgrads, bgrads);
    
end

function y = sigmoid(x)
    y = 1 ./ (1. + exp(-x));
end

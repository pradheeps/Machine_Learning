function [cost, grad, hess] = costLR(X, y, theta, lambda)
% costLR  Logistic Regression cost function.
%
%   X      - m x (n+1) design matrix of m examples
%   y      - m x 1 labels
%   theta  - current logistic regression parameters.
%   lambda - regularization term.
%
%   cost - cost at theta
%   grad - gradient at theta. i.e. [dJ/dx1, ..., dJ/dxn]' (n x 1 vector)
%   hess - hessian (symmetric matrix of second partial derivatives) at
%          theta, i.e.  (n x n matrix)
%             [d2J/dx1*dx1, d2J/dx1*dx2, ..., d2J/dx1*dxn;
%              d2J/dx2*dx1, d2J/dx2*dx2, ..., d2J/dx2*dxn;
%                 .               .                  .
%                 .               .                  .
%              d2J/dxn*dx1, d2J/dxn*dx2, ..., d2J/dxn*dxn]
%
%  When theta is a scalar, cost is just J(X,y,theta), grad is just
%   dJ(X,y,theta)/dtheta and hess is just d2J(X,y,theta)/dtheta^2.
%
%  Example usage:
%    cost = costLR(randn(100, 2), rand(100, 1) < 0.5, randn(2, 1));
%    [cost, grad] = costLR([1, 2; 1, 4], [0; 1], [0.1; 0.5])
%    [cost, grad, hess] = costLR(A, B, C); % hessian is optional
    
    % Write your solution below. If you choose to implement Newton's
    % method, please also compute hess. If you choose to just use gradient
    % descent, then the hessian is not required (you can just leave it
    % blank).
    % Our solution is approx 6 lines.
    
    cost = 0;
    grad = zeros(size(theta));
    
    %% BEGIN SOLUTION
    m = size(X, 1);
    
    % compute the cost
    
    z = X*theta;
    cost = sum(log1p(exp(z))) - y'*z + 0.5*lambda*sum(theta.^2);       
    if nargout < 2, return; end % only need cost.
    
    % compute the gradient
    
    act = sigmoid(z);
    grad = X'*(act - y) + lambda*theta;   
    if nargout < 3, return; end % only need cost and grad.
    
    % compute the hessian (only required for Newton step).
   
    hess = X'*bsxfun(@times, act.*(1-act), X) + lambda*eye(length(theta));
    size(hess)
    %% END SOLUTION
end

function y = logsumexp(x)
% LOGSUMEXP More numerically stable version of y = log(sum(exp(x), 2));
    b = max(x,[],2);
    y = log(sum(exp(x-repmat(b,[1, size(x,2)])),2))+b;
end

function y = sigmoid(x)
% SIGMOID  Computes sigmoid function, y = e^x/(1+e^x) = 1/(1+e^-x).
    y = 1./(1. + exp(-x));
end
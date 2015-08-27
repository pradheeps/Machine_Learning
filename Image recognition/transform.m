function [xZCAwhite]=transform(x,k,epsilon)
%u,xRot,xTilde,xPCAwhite,
%zero center the data
if(nargin<2)
    k=size(x,1);
end
if(nargin<3)
    epsilon=1e-5;
end
avg=mean(x,1);
%x = x - repmat(avg, size(x, 1), 1);

%calculate sigma/covariance
sigma=(x*x')/size(x,2);

% svd/ find eig values
%u will have eig vectors
%s will have eig values
% v is same as u' and can be ignored
[u,s,v]=svd(sigma);

%find rotated x
%xRot = u' * x;          % rotated version of the data. 
%xTilde = u(:,1:k)' * x; % reduced dimension representation of the data, 

%xPCAwhite = diag(1./sqrt(diag(s) + epsilon)) * u' * x;

xZCAwhite = u * diag(1./sqrt(diag(s) + epsilon)) * u' * x;


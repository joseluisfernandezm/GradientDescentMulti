function [J] = costFunction(Thj, X, y)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y) computes the cost of using
%   theta as the parameter for logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = size(X, 1); % number of training examples

%{ 
====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% =============================================================
%}

m=size(X,1);       
h = sigmoid((Thj*X')');%lo que va dentro de sigmod es z
J = (1/m)*(sum( -y.*log(h) - (1-y).*log(1-h) ));

end

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon J = SIGMOID(z) computes the sigmoid of z.
% You need to return the following variables correctly 
    g = zeros(size(z));%inicializamos g(z)

%{ 
====================== YOUR CODE HERE ======================
Instructions: Compute the sigmoid of each value of z (z can be a matrix,
              vector or scalar). Con esta funcion vamos a calcular g(z)
 =============================================================
 %}
    
    g=1./(1+exp(-z));%formula teórica de g(z)

end

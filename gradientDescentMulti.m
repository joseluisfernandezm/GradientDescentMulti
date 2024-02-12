function [J,Thetas_finales, T_mat] = gradientDescentMulti(X,y,alpha,Thj,n_iters)
%{
La salida del algoritmo deberá ser:

- Un vector de la funcion de coste para cada una de las iteraciones del
  algoritmo.
- Una matriz con los parametros thetaj para cada una de las iteraciones

- Matriz X con los datos de entrenamiento.
- Vector y con las etiquetas de los datos de entrenamiento.
- Coeficiente de aprendizaje α.
- Valores de inicialización de los parámetros θj.
- Número máximo de iteraciones a realizar del algoritmo.

%}

    m=size(X,1);%cantidad de datos  
    n = size(Thj, 2);%numero de variables
    T_mat = zeros(n_iters, n);%inicializamos T_mat a cero
    T=Thj;%inicializo T auxiliar
    
    %Inicializar h
    z = (Thj*X')'; 
    h = sigmoid(z);
    
    
    % for para el numero de iteraciones indicado
    
    for i = 1:n_iters 
        
        % Ec DxG, y vamos guardando las nuevas thetas en cada vector, uno
        % para theta0 y el otro para theta1
        
        %actualizamos el valor de cada theta en cada iteracion
        for j = 1:n
            T_mat(i, j) = T(j) - alpha*(1/m)*sum((h-y).*X(:, j));
            T(j) =  T_mat(i, j); 
        end        
        h = sigmoid((T*X')');
        J(i) = costFunction(T, X, y);
        
    end
    Thetas_finales = T;%al final del bucle seran las theas optimas

end

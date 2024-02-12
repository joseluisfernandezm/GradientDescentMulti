clear all
close all
clc

%{
CONTENIDOS:

Nos piden volver a ejecutar el algoritmo de descenso por gradiente para
ajustar el nuevo clasificador cuto vector theta tendrá una dimension 9 nos
piden que encontremos el alpha adecuado para que el algoritmo converja. 

%}

%% Carga de datos
tic
data=load('DATA_exams.txt'); %cargamos los datos del fichero con load

% extraemos los datos en 2 vectores x1 y  para hacerlo generico

x1=data(:,1);%nota examen 1 (Datos de entrenamiento)
x2=data(:,2);%nota examen 2 (Datos de entrenamiento)
y=data(:,3);%Admitido o No admitido (Etiquetas o Datos de salida)

x3 = x1.*x2;
x4 = x1.^2;
x5 = x2.^2;
x6 = x1.^2 .* x2;
x7 = x1.* x2.^2;
x8 = x1.^3;
x9 = x2.^3;


%% Descenso por gradiente

alpha=10^(-11);%me lo da el ejercicio
n_iters=5*10^5;%me lo da el ejercicio

X=horzcat(ones(size(x1,1),1), x1, x2, x3, x4, x5, x6, x7, x8, x9);% solo añado x1 porque mis datos de entrenamiento son x, los datos de entrada
Thj=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; %inicializamos los theta a 0
[J,Thetas_finales, T_mat] = gradientDescentMulti(X,y,alpha,Thj,n_iters);

figure()
plot(0:1:n_iters-1,J);
title(['alpha =', num2str(alpha), ' Evolucion de J(Theta)'])
xlabel('N iters')
ylabel('Coste J(Theta)')

%% Pintar frontera

%{

Para pintar la frontera entre los datos vamos a usar el ultimo valor de
thetas hallado por el algoritmo de descenso por gradiente. el umbral de
decision se tomara en z=0, es cedir que paralo calores de z=thj*x mayores
que cero se decidira que y=1 y para valores menores de 0 se decidira y=0

%}

Thj=Thetas_finales; % para este caso en el que usamos las finales del algoritmo

figure()

hold on
index=find(y == 1);%con find encuentro los valores que son 1
plot(data(index,1), data(index,2), 'bx');%filtro los datos que son solo 1 y los represento
title('Datosexams.txt');
xlabel('Resultados examen 1');
ylabel('Resultados examen 2');

index=find(y ==0);
plot(data(index,1), data(index,2), 'ro');%hago lo mismo pero ahora con los daros en 0


u = linspace(min(X(:,2)), max(X(:,2)), 50);
v = linspace(min(X(:,3)), max(X(:,3)), 50);
z = zeros(length(u), length(v));
%z estará definido para todas las notas definidas por u y v
for i = 1:length(u)
 for j = 1:length(v)
        z(i,j) = Thetas_finales(1) + Thetas_finales(2)*u(i) + Thetas_finales(3)*v(j)+ Thetas_finales(4)*u(i).*v(j) + Thetas_finales(5)*u(i).*u(i) + Thetas_finales(6)*v(j).*v(j) + Thetas_finales(7)*u(i).*u(i).*v(j) + Thetas_finales(8)*v(j).*v(j).*u(i)+ Thetas_finales(9)*u(i).*u(i).*u(i)+ Thetas_finales(10)*v(j).*v(j).*v(j); 
 end
end
z = z';
contour(u, v, sigmoid(z), [0.5, 0.5], 'LineWidth', 2);% [0.5, 0.5] es el umbral de decisión. Modifiquenlo entre 0 y 1 para ver los resultados
hold on;
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;



%% Errores de clasificacion 

%{
Ahora sobre el clasificador ajustado del apartado anterior vamos a calcular
los errores de clasificacion que se cometerian sobre el conjunto de datos
de entrenamiento.

Para ello nos piden calcular el valor de z para cada uno de los datos, y
dependiento de si cada dato de z es mayor o menor que 0 asignaremos cada
dato a la clase 1 o 2
%}


z_aux = zeros(length(x1),1);%inicializamos el vector z con el tamaño de los datos m

for k=1:length(x1)
        z_aux(k) = Thetas_finales(1)+Thetas_finales(2)*x1(k,1)+Thetas_finales(3)*x2(k,1);%ecuacion de teoria
        if (z_aux(k)>=0.5)
           y_prima(k,1)=1;
        else 
           y_prima(k,1)=0;
        end
end

pos = 0; %positivos
neg = 0; %negativos
fal_neg = 0; %falso negativo
fal_pos = 0; %falso positivo
error = y_prima - y;%calculo del error

%Me recorro los vectores de salida y cuento los errores
for j=1:length(y)
    if y(j)==1
        pos = pos+1; % Pos totales reales
    else 
        neg = neg+1;% Neg totales reales
    end
    if error(j)== 1
        fal_pos = fal_pos+1;
    elseif error(j)== -1
        fal_neg = fal_neg+1;
    else
    end
end
Error_negativo = fal_pos/neg;
Error_positivo = fal_neg/pos;
Error_total=(fal_pos+fal_neg)/length(y);%error medio

toc

%{
PREGUNTAS: 
¿Qué forma tiene ahora la frontera resultante?

¿Es la esperada?

¿Qué diferencia hay con respecto a las fronteras dibujadas en los apartados anteriores?


¿Cuál es el valor de cada una de las tasas de error (clase positiva, negativa y error total)
para esta frontera de decisión?

¿Están los errores equilibrados?

¿Cómo influye la forma de la frontera de decisión?

¿Cree que puede haber un problema de sobreentrenamiento?

%}


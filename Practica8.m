clear all
close all
clc

%{
CONTENIDOS:

El objetivo de esta practica es el de implementar el alforitmo de regresion
logistica que nos ayude a predecir si un alumno entrará o no en la
universidad. 

Para esta decision nos centraremos en las notas de dos examenes. Para cada
uno de los cansidaros camos a tener los resultados de dos exames y la
decision final adoptada entre admitido o no admitido.

Por tanto vamos a tener 3 columnas de datos, la primera con la nota del
primer examen, la segunda con la nota del segundo examen y la tercera con
la decision de si entra o no en la universidad.

Por tanto nuestros datos de entrenamiento van a ser las columnas 1 y 2 y
por otra parte nuestras etiquetas estan en la columna 3

%}

%% Carga de datos

data=load('DATA_exams.txt'); %cargamos los datos del fichero con load

% extraemos los datos en 2 vectores x1 y  para hacerlo generico

x1=data(:,1);%nota examen 1 (Datos de entrenamiento)
x2=data(:,2);%nota examen 2 (Datos de entrenamiento)
y=data(:,3);%Admitido o No admitido (Etiquetas o Datos de salida)

%% Visualizacion de los datos 

%plot de los datos

%Creo los ejes de la grafica


figure()

hold on
index=find(y == 1);%con find encuentro los valores que son 1
plot(data(index,1), data(index,2), 'bx');%filtro los datos que son solo 1 y los represento

title('Datosexams.txt');
xlabel('Resultados examen 1');
ylabel('Resultados examen 2');

index=find(y ==0);
plot(data(index,1), data(index,2), 'ro');%hago lo mismo pero ahora con los daros en 0
hold off


%% Funcion logistica

% Ejemplo para pobar la funcion sigmoid

z=[-5:0.05:5];

gz=sigmoid(z);%sigmoid nos calcula la funcion g(z) que es la funcion logistica

%plot del ejemplo

figure()
plot(z,gz);
title('Funcion sigmod')
xlabel('z')
ylabel('g(z)')



%{
PREGUNTAS

¿La funcion logistica tiene el aspecto esperado?

¿Cuando tiende a 0?

¿Y a 1?

¿Cuanto vale para z=0?


%}

%% Descenso por gradiente

alpha=0.001;%me lo da el ejercicio
n_iters=5*10^5;%me lo da el ejercicio

X=horzcat(ones(size(x1,1),1),x1,x2);% solo añado x1 porque mis datos de entrenamiento son x, los datos de entrada
Thj=[0,0,0]; %inicializamos los theta a 0
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
 z(i,j) = [1, u(i) v(j)]*Thj'; %theta contiene las 3 thetas calculadas con descenso del gradiente, para que salga bien tienen que poner muchas iteraciones y jugar con alpha
 end
end
z = z';
contour(u, v, sigmoid(z), [0.5, 0.5], 'LineWidth', 2);% [0.5, 0.5] es el umbral de decisión. Modifiquenlo entre 0 y 1 para ver los resultados
hold on;
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

%{
PREGUNTAS: 
¿Qué forma tiene la frontera resultante?

¿Es la esperada? 

Sobre la gráfica anterior¿cuántos errores se cometerían sobre los datos de entrenamiento para la clase positiva?

¿y para la negativa?
%}

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





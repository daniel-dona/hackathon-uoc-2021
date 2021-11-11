import numpy as np
from sklearn import linear_model

from sklearn.metrics import mean_squared_error

modelo = linear_model.LinearRegression()

train_in = [ -40, -10, 0, 8, 15, 22, 38 ]
train_out = [ -40, 14, 32, 46, 59, 72, 100 ]

x = np.array(train_in)
y = np.array(train_out)

print("Entrada de entrenamiento:", train_in)
print("Salida de entrenamiento:", train_out)

print("-----------------------------------------------------------------------------------------")

mse = 100
i = 1

eq = ""

while mse > 0.000001:

	eq = np.polynomial.polynomial.polyfit(x,y,i)

	test = []

	for v in train_in:

		test.append(np.polynomial.polynomial.polyval(v,eq))

	mse = mean_squared_error(train_out, test)

	print("Iteración:",i)
	print("Polinomio:\n", np.poly1d(eq))
	print("Predicciones del entrenamiento:",test)
	print("Error Cuadrático Medio:", mse)
	print("-----------------------------------------------------------------------------------------")

	i += 1


print("Calculando conjunto de prueba: ")

test_in = [5, 10, -7, 24, 89]

x_test = np.array(test_in)

print("Resultados:")

print(np.polynomial.polynomial.polyval(x_test,eq))

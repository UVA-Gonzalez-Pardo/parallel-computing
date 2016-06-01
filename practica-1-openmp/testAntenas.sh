#!/bin/bash

# El script ejecuta el programa a.out el numero de iteraciones que le pases
# como parametro. Los resultados se añadiran al final del fichero
# output.txt creado en el mismo directorio.

i=0
nIteraciones=$1

echo "Datos:" >> output.txt
while [ $i -lt $nIteraciones ]
do
	./a.out 500 500 5 1 1 1 | grep 'Time:' | grep -o '[0-9]*\.[0-9]*' >> output.txt
	i=$((i+1))
	echo "iteracion: $i"
done
echo "" >> output.txt

echo "FIN: datos al final de output.txt"

/**
 * Computación Paralela (curso 1516)
 *
 * Colocación de antenas
 * Versión paralela
 *
 * @author Daniel González Alonso
 * @author Santos Pardo Ramos
 */

// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

// Include para las utilidades de computación paralela
#include "cputils.h"
#include <omp.h>


/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
} Antena;

typedef struct {
	int valor;
	int x;
	int y;
} Registro;

int num_hilos;		// Numero de hilos


/**
 * Función de ayuda para imprimir el mapa
 */
void print_mapa(int ** mapa, int rows, int cols, Antena * a)
{
	if(rows > 50 || cols > 30){
		printf("Mapa muy grande para imprimir\n");
		return;
	}

	#define ANSI_COLOR_RED     "\x1b[31m"
	#define ANSI_COLOR_GREEN   "\x1b[32m"
	#define ANSI_COLOR_RESET   "\x1b[0m"

	printf("Mapa [%d,%d]\n",rows,cols);
	for(int i=0; i < rows; i++){
		for(int j=0; j < cols; j++)
		{
			int val = mapa[i][j];

			if(val == 0){
				if(a != NULL && a->x == j && a->y == i){
					printf( ANSI_COLOR_RED "   A"  ANSI_COLOR_RESET);
				} else { 
					printf( ANSI_COLOR_GREEN "   A"  ANSI_COLOR_RESET);
				}
			} else {
				printf("%4d", val);
			}
		}
		printf("\n");
	}
	printf("\n");
}


/**
 * Distancia de una antena a un punto (y,x)
 * @note Es el cuadrado de la distancia para tener más carga
 */
int manhattan(Antena a, int y, int x)
{
	int dist = abs(a.x - x) + abs(a.y - y);
	return dist * dist;
}


/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar(int ** mapa, int rows, int cols, Antena antena)
{
	//printf("INICIO ACTUALIZAR\n");

	mapa[antena.y][antena.x] = 0;
	int r;
	//int distancia, flag = 1, delta, r=1;
	#pragma omp for nowait
	for(r = 1; r < (rows + cols); r++)
	{
		Antena arriba = { antena.y - r, antena.x };
		Antena abajo = { antena.y + r, antena.x };
		Antena izquierda = { antena.y, antena.x - r };
		Antena derecha = { antena.y, antena.x + r };

		int distancia = manhattan(antena, arriba.y, arriba.x);
		int flag = 0;

		//printf("radio: %d, dist_manhattan: %d\n", r, distancia);
		//print_mapa(mapa,rows,cols,NULL);
		//printf("Hilo:%d r=%d\n",omp_get_thread_num(),r);
		//fflush(stdout);
		for(int delta = 0; delta < r; delta++)
		{
			//printf("delta: %d\n", delta);
			// Arriba
			if( (arriba.y + delta >= 0) && (arriba.y + delta < rows) &&
				(arriba.x + delta >= 0) && (arriba.x + delta < cols) &&
				(mapa[arriba.y + delta][arriba.x + delta] > distancia) )
			{
				mapa[arriba.y + delta][arriba.x + delta] = distancia;
				flag = 1;
				//printf("Arriba: (%d, %d)\n", arriba.y + delta, arriba.x + delta);
			}
			// Abajo
			if( (abajo.y - delta >= 0) && (abajo.y - delta < rows) &&
				(abajo.x - delta >= 0) && (abajo.x - delta < cols) &&
				(mapa[abajo.y - delta][abajo.x - delta] > distancia) )
			{
				mapa[abajo.y - delta][abajo.x - delta] = distancia;
				flag = 1;
				//printf("Abajo: (%d, %d)\n", abajo.y + delta, abajo.x + delta);
			}
			// Izquierda
			if( (izquierda.y - delta >= 0) && (izquierda.y - delta < rows) &&
				(izquierda.x + delta >= 0) && (izquierda.x + delta < cols) &&
				(mapa[izquierda.y - delta][izquierda.x + delta] > distancia) )
			{
				mapa[izquierda.y - delta][izquierda.x + delta] = distancia;
				flag = 1;
				//printf("Izquierda: (%d, %d)\n", izquierda.y + delta, izquierda.x + delta);
			}
			// Derecha
			if( (derecha.y + delta >= 0) && (derecha.y + delta < rows) &&
				(derecha.x - delta >= 0) && (derecha.x - delta < cols) &&
				(mapa[derecha.y + delta][derecha.x - delta] > distancia) )
			{
				mapa[derecha.y + delta][derecha.x - delta] = distancia;
				flag = 1;
				//printf("Derecha: (%d, %d)\n", derecha.y + delta, derecha.x + delta);
			}
		}

		// No es necesario mas iteraciones puesto que no se van a actualizar
		// mas posiciones en el mapa.
		if(flag == 0) {
			r = (rows + cols);
			//#pragma omp flush(r)
		}
	}

	//printf("Despues actualizar\n");
	//print_mapa(mapa,rows,cols,NULL);

	//printf("FIN ACTUALIZAR\n");
}


/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar_primera_antena(int ** mapa, int rows, int cols, Antena antena)			// TODO: Mejora de Actualizar
{
	mapa[antena.y][antena.x] = 0;
	int r;
	//int distancia, flag = 1, delta, r=1;
	#pragma omp parallel for shared(mapa,rows,cols,miAntena,miAntena,r) schedule(static)
	for(r = 1; r < (rows + cols); r++)
	{
		Antena arriba = { antena.y - r, antena.x };
		Antena abajo = { antena.y + r, antena.x };
		Antena izquierda = { antena.y, antena.x - r };
		Antena derecha = { antena.y, antena.x + r };

		int distancia = manhattan(antena, arriba.y, arriba.x);
		int flag = 0;

		//printf("radio: %d, dist_manhattan: %d\n", r, distancia);
		//print_mapa(mapa,rows,cols,NULL);
		//printf("Hilo:%d r=%d\n",omp_get_thread_num(),r);
		//fflush(stdout);
		for(int delta = 0; delta < r; delta++)
		{
			//printf("delta: %d\n", delta);
			// Arriba
			if( (arriba.y + delta >= 0) && (arriba.y + delta < rows) &&
				(arriba.x + delta >= 0) && (arriba.x + delta < cols) )
			{
				mapa[arriba.y + delta][arriba.x + delta] = distancia;
				flag = 1;
				//printf("Arriba: (%d, %d)\n", arriba.y + delta, arriba.x + delta);
			}
			// Abajo
			if( (abajo.y - delta >= 0) && (abajo.y - delta < rows) &&
				(abajo.x - delta >= 0) && (abajo.x - delta < cols) )
			{
				mapa[abajo.y - delta][abajo.x - delta] = distancia;
				flag = 1;
				//printf("Abajo: (%d, %d)\n", abajo.y + delta, abajo.x + delta);
			}
			// Izquierda
			if( (izquierda.y - delta >= 0) && (izquierda.y - delta < rows) &&
				(izquierda.x + delta >= 0) && (izquierda.x + delta < cols) )
			{
				mapa[izquierda.y - delta][izquierda.x + delta] = distancia;
				flag = 1;
				//printf("Izquierda: (%d, %d)\n", izquierda.y + delta, izquierda.x + delta);
			}
			// Derecha
			if( (derecha.y + delta >= 0) && (derecha.y + delta < rows) &&
				(derecha.x - delta >= 0) && (derecha.x - delta < cols) )
			{
				mapa[derecha.y + delta][derecha.x - delta] = distancia;
				flag = 1;
				//printf("Derecha: (%d, %d)\n", derecha.y + delta, derecha.x + delta);
			}
		}

		// No es necesario mas iteraciones puesto que no se van a actualizar
		// mas posiciones en el mapa.
		if(flag == 0) {
			r = (rows + cols);
			//#pragma omp flush(r)
		}
	}

	//printf("Despues actualizar\n");
	//print_mapa(mapa,rows,cols,NULL);

	//printf("FIN ACTUALIZAR\n");
}


/**
 * Calcular la distancia máxima en el mapa
 */
Antena calcular_max(int ** mapa, int rows, int cols, Registro * registros)
{
	int maximo=0, numHilo;

	#pragma omp parallel for shared(mapa,registros) firstprivate(maximo) private(numHilo)
	for(int i=0; i < rows; i++)
		for(int j=0; j < cols; j++)
		{
			if(mapa[i][j] > maximo)
			{
				maximo = mapa[i][j];
				numHilo = omp_get_thread_num();
				registros[numHilo].x = i;
				registros[numHilo].y = j;
				registros[numHilo].valor = mapa[i][j];
			}//if
		}//j

	maximo=0;
	for(int i=0; i < num_hilos; i++)
		if(registros[i].valor > registros[maximo].valor)
			maximo = i;

	Antena antena = { registros[maximo].x, registros[maximo].y };
	return antena;
}


/**
 * Función principal
 */
int main(int nargs, char ** vargs)
{
	//
	// 1. LEER DATOS DE ENTRADA
	//
	
	// Comprobar número de argumentos
	if(nargs < 7)
	{
		fprintf(stderr,"Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",vargs[0]);
		return -1;
	}

	// Leer los argumentos de entrada
	int rows = atoi(vargs[1]);
	int cols = atoi(vargs[2]);
	int distMax = atoi(vargs[3]);
	int nAntenas = atoi(vargs[4]);

	if( nAntenas<1 || nargs != (nAntenas*2+5) )
	{
		fprintf(stderr, "Error en la lista de antenas\n");
		return -1;
	}

	// Mensaje
	printf("Calculando el número de antenas necesarias para cubrir un mapa de"
		   " (%d x %d)\ncon una distancia máxima no superior a %d "
		   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);

	// Reservar memoria para las antenas
	Antena * antenas = malloc(sizeof(Antena) * (size_t) nAntenas);
	if( !antenas )
	{
		fprintf(stderr, "Error al reservar memoria para las antenas inicales\n");
		return -1;
	}
	
	// Leer antenas
	for(int i=0; i < nAntenas; i++)
	{
		antenas[i].x = atoi(vargs[5+i*2]);
		antenas[i].y = atoi(vargs[6+i*2]);

		if( antenas[i].y<0 || antenas[i].y>=rows || antenas[i].x<0 || antenas[i].x>=cols )
		{
			fprintf(stderr, "Antena #%d está fuera del mapa\n", i);
			return -1;
		}
	}

	//
	// 2. INICIACIÓN
	//

	// Medir el tiempo
	double tiempo = cp_Wtime();

	// Obtenemos el numero de hilos
	num_hilos = 0;
	#pragma omp parallel shared(num_hilos)
	{
		#pragma omp single nowait
		{
			num_hilos = omp_get_num_threads();
		}
	}

	// Reservar memoria para los registros de los maximos
	Registro * registros = malloc(sizeof(Registro) * (size_t) (num_hilos));	
	if( !registros )
	{
		fprintf(stderr, "Error al reservar memoria para los registros iniciales\n");
		return -1;
	}

	// Crear el mapa
	int i;
	int ** mapa = (int**)malloc(cols*sizeof(int*));
    for(i=0; i < rows; i++){
        mapa[i] = (int*)malloc(rows*sizeof(int*));
    }
	
	// Colocar las antenas iniciales
	actualizar_primera_antena(mapa, rows, cols, antenas[0]);

	for(i=1; i < nAntenas; i++)
	{
		actualizar(mapa, rows, cols, antenas[i]);
	}
	
	// Debug
#ifdef DEBUG
	print_mapa(mapa, rows, cols, NULL);
#endif


	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	int nuevas = 0;		// Contador de antenas
	while(1)
	{
		// Calcular la posicion de la antena a distancia maxima.
		Antena antena = calcular_max(mapa, rows, cols, registros);
		
		// Salimos si ya hemos cumplido el maximo
		if (mapa[antena.y][antena.x] <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Actualización del mapa con la nueva antena y nuvas distancias.
		actualizar(mapa, rows, cols, antena);
	}

	// Debug
#ifdef DEBUG
	print_mapa(mapa, rows, cols, NULL);
#endif

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;	

	// Salida
	printf("Result: %d\n", nuevas);
	printf("Time: %f\n", tiempo);
	print_mapa(mapa, rows, cols, NULL);

	return 0;
}

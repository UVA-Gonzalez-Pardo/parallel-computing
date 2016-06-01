/**
 * Computación Paralela (curso 1516)
 *
 * Colocación de antenas
 * Versión secuencial
 *
 * @author Javier
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


/**
 * Macro para acceder a las posiciones del mapa
 */
#define m(y,x) mapa[ (y * cols) + x ]

/**
 * Función de ayuda para imprimir el mapa
 */
void print_mapa(int * mapa, int rows, int cols, Antena * a)
{
	if(rows > 50 || cols > 30){
		printf("Mapa muy grande para imprimir\n");
		return;
	}

	#define ANSI_COLOR_RED     "\x1b[31m"
	#define ANSI_COLOR_GREEN   "\x1b[32m"
	#define ANSI_COLOR_RESET   "\x1b[0m"

	printf("Mapa [%d,%d]\n",rows,cols);
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++)
		{
			int val = m(i,j);

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
	int dist = abs(a.x -x) + abs(a.y - y);
	return dist * dist;
}


/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar(int * mapa, int rows, int cols, Antena antena)
{
	#pragma omp parallel
	{
		#pragma omp single
		m(antena.y,antena.x) = 0;
	
		#pragma omp for
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){

				int nuevadist = manhattan(antena,i,j);

				if(nuevadist < m(i,j)){
					m(i,j) = nuevadist;
				}

			} // j
		} // i
	}
}


/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar_primera_antena(int * mapa, int rows, int cols, Antena antena){
	#pragma omp parallel
	{	
		#pragma omp single
		m(antena.y,antena.x) = 0;

		#pragma omp for
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				m(i,j) = manhattan(antena,i,j);
			}//j
		}//i
	}
}


/**
 * Calcular la distancia máxima en el mapa
 */
int calcular_max(int * mapa, int rows, int cols, Registro * registros, int num_hilos)
{
	int maximo=0, ii=0, jj=0, i, j;
	#pragma omp parallel for shared(mapa,registros) firstprivate(ii,jj,maximo) private(i,j)
	for(i=0; i < rows; i++)
	{//i

		for(j=0; j < cols; j++)
		{//j
			if(m(i,j) > maximo)
			{//if
				maximo = m(i,j);
				ii = i;
				jj = j;
			}//if
		}//j
	
		registros[omp_get_thread_num()].x = ii;
		registros[omp_get_thread_num()].y = jj;
		registros[omp_get_thread_num()].valor = maximo;

	}//i

	maximo=0;

	for(i=0; i < num_hilos; i++)
		if(registros[i].valor > maximo)
		{//if
			maximo = registros[i].valor;
			registros[num_hilos].valor = maximo;
			registros[num_hilos].x = registros[i].x;
			registros[num_hilos].y = registros[i].y;

		}//if

	//printf("Maximo devuelto es: %d y esta en %d , %d \n", registros[num_hilos].valor, registros[num_hilos].x, registros[num_hilos].y);
	return registros[num_hilos].valor;
}


/**
 * Calcular la posición de la nueva antena
 */
Antena nueva_antena(Registro * registros, int rows, int cols, int min, int num_hilos)
{	
	int i=registros[num_hilos].x;
	int j=registros[num_hilos].y;

	Antena antena = {i,j};
	return antena;
}


/**
 * Función principal
 */
int main(int nargs, char ** vargs)

{
	char *num_hilos2=getenv("OMP_NUM_THREADS");
	int num_hilos=atoi(num_hilos2);
	
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

	if( nAntenas<1 || nargs != (nAntenas*2+5) ){
		fprintf(stderr, "Error en la lista de antenas\n");
		return -1;
	}

	// Mensaje
	printf("Calculando el número de antenas necesarias para cubrir un mapa de"
		   " (%d x %d)\ncon una distancia máxima no superior a %d "
		   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);

	// Reservar memoria para las antenas
	Antena * antenas = malloc(sizeof(Antena) * (size_t) nAntenas);
	if(!antenas){
		fprintf(stderr, "Error al reservar memoria para las antenas inicales\n");
		return -1;
	}
	
	// Leer antenas
	for(int i=0; i < nAntenas; i++){
		antenas[i].x = atoi(vargs[5+i*2]);
		antenas[i].y = atoi(vargs[6+i*2]);

		if( antenas[i].y<0 || antenas[i].y>=rows || antenas[i].x<0 || antenas[i].x>=cols )
		{
			fprintf(stderr, "Antena #%d está fuera del mapa\n", i);
			return -1;
		}
	}
	
	//Reservar memoria para los registros de los maximos

	Registro * registros = malloc(sizeof(Registro) * (size_t) (num_hilos+1));	
	if(!registros){
		fprintf(stderr, "Error al reservar memoria para los registros iniciales\n");
		return -1;
	}


	//
	// 2. INICIACIÓN
	//

	// Medir el tiempo
	double tiempo = cp_Wtime();

	// Crear el mapa
	int * mapa = malloc((size_t) (rows*cols) * sizeof(int) );

	// Iniciar el mapa con el valor MAX INT
	//for(int i=0; i<(rows*cols); i++){
	//	mapa[i] = INT_MAX;
	//}
	
	// Colocar las antenas iniciales
	
	actualizar_primera_antena(mapa, rows, cols, antenas[0]);

	for(int i=1; i<nAntenas; i++)
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

	// Contador de antenas
	int nuevas = 0;
	
	while(1)
	{
		// Calcular el máximo
		int max = calcular_max(mapa, rows, cols, registros,num_hilos);

		// Salimos si ya hemos cumplido el maximo
		if (max <= distMax) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Calculo de la nueva antena y actualización del mapa
		Antena antena = nueva_antena(registros, rows, cols, max,num_hilos);
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
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);
	print_mapa(mapa,rows,cols,NULL);

	return 0;
}

/**
 * Computación Paralela (curso 1516)
 *
 * Colocación de antenas
 * Versión paralela con MPI
 *
 * Grupo 32
 * @author Daniel González Alonso
 * @author Santos Pardo Ramos
 */

// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

// Include para las utilidades de computación paralela
#include "cputils.h"

// MPI
#include <mpi.h>

/**
 * Estructura antena
 */
typedef struct {
	int y;
	int x;
	int valor;
} Antena;



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
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++)
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
		}// j
		printf("\n");
	}// i
	printf("\n");
}


/**
 * Distancia de una antena a un punto (y,x)
 * @note Es el cuadrado de la distancia para tener más carga
 */
int manhattan(Antena *a, int y, int x)
{
	int dist = abs(a->x - x) + abs(a->y - y);
	return dist * dist;
}


/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar(int **mapa, int rows, int cols,	int p_glob, Antena *antena)
{
	int encima = 0,
		debajo = 0,
		r = 1,
		flag = -1;

	// Comprobamos si la entena esta dentro del sub-mapa, por encima o por
	// debajo.
	if(antena->y >= p_glob)
	{
		if(antena->y < p_glob + rows) {
			mapa[antena->y - p_glob][antena->x] = 0;
		} else {
			debajo = 1;
			r = antena->y - (p_glob + rows - 1);
		}
	} else {
		encima = 1;
		r = p_glob - antena->y;
	}

	while( flag != 0 )
	{
		flag = 0;		// Reiniciamos la bandera

		int distancia = manhattan(antena, antena->y - r, antena->x);

		// Si antena->y esta por encima de nuestro mapa, no hace falta
		// recorrer la parte superior del rombo.
		if( !encima )
		{
			Antena arriba = { antena->y - p_glob - r, antena->x, -1 };

			// Si arriba.y esta fuera del mapa, iniciamos delta en la primera
			// interseccion del rombo con el mapa por arriba.
			int delta = (arriba.y < 0)? -(arriba.y) : 0;

			// Descendemos por los lateriales superiores del rombo desde
			// arriba->y hasta antena->y
			// Descendiendo por la Izquierda
			int deltaI = delta;
			while( (deltaI <= r)
					&& (arriba.y + deltaI < rows)
					&& (arriba.x - deltaI >= 0)
			) {
				if(mapa[arriba.y + deltaI][arriba.x - deltaI] > distancia)
				{
					mapa[arriba.y + deltaI][arriba.x - deltaI] = distancia;
					flag = 1;
				}
				deltaI++;
			}
			// Descendiendo por la Derecha
			int deltaD = delta;
			while( (deltaD <= r)
					&& (arriba.y + deltaD < rows)
					&& (arriba.x + deltaD < cols)
			) {				
				if(mapa[arriba.y + deltaD][arriba.x + deltaD] > distancia)
				{
					mapa[arriba.y + deltaD][arriba.x + deltaD] = distancia;
					flag = 1;
				}
				deltaD++;
			}
		}

		// Si antena->y esta por debajo de nuestro mapa, no hace falta
		// recorrer la parte inferior del rombo.
		if( !debajo )
		{
			Antena abajo = { antena->y - p_glob + r, antena->x, -1 };

			// Si abajo.y esta fuera del mapa, iniciamos delta en la primera
			// interseccion del rombo con el mapa por abajo.
			int delta = (abajo.y >= rows)? abajo.y + 1 - rows : 0;

			// Ascendemos por los lateriales inferiores del rombo desde
			// abajo->y hasta antena->y
			// Ascendiendo por la Izquierda
			int deltaI = delta;
			while( (deltaI < r)
					&& (abajo.y - deltaI >= 0)
					&& (abajo.x - deltaI >= 0)
			) {
				if(mapa[abajo.y - deltaI][abajo.x - deltaI] > distancia)
				{
					mapa[abajo.y - deltaI][abajo.x - deltaI] = distancia;
					flag = 1;
				}
				deltaI++;
			}
			// Ascendiendo por la Derecha
			int deltaD = delta;
			while( (deltaD < r)
					&& (abajo.y - deltaD >= 0)
					&& (abajo.x + deltaD < cols)
			) {
				if(mapa[abajo.y - deltaD][abajo.x + deltaD] > distancia)
				{
					mapa[abajo.y - deltaD][abajo.x + deltaD] = distancia;
					flag = 1;
				}
				deltaD++;
			}
		}

		r++;
	}
}


/**
 * Actualizar el mapa con la nueva antena
 */
void actualizar_primera_antena(int **mapa, int rows, int cols,
	int p_glob, Antena *antena)
{
	if( (p_glob <= antena->y) && (antena->y < p_glob + rows) )
		mapa[antena->y - p_glob][antena->x] = 0;

	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++)
		{
			mapa[i][j] = manhattan(antena, i+p_glob, j);
		}// j
	}// i
}


/**
 * Operacion Reduce de MPI para obtener el maximo global a partir de los
 * maximos locales
 */
void calcula_maximo_global(void *invec, void *inoutvec, int *len,
	MPI_Datatype *datatype)
{
	Antena* in = (Antena*) invec;
	Antena* inout = (Antena*) inoutvec;

	for(int i = 0; i < *len; i++)
	{
		if( (in[i].valor > inout[i].valor)
			|| ( (in[i].valor == inout[i].valor) && (in[i].y < inout[i].y) )
		) {
			inout[i].y = in[i].y;
			inout[i].x = in[i].x;
			inout[i].valor = in[i].valor;
		}
	}
}


/**
 * Calcular la distancia máxima en el mapa
 */
void calcular_max(int **mapa, int rows, int cols, int p_glob,
	MPI_Datatype* MPI_antena, MPI_Op* MPI_Maximo_Antena, Antena* maximo_global)
{
	Antena maximo_local = { -1, -1, -1 };

	for(int i=0; i < rows; i++) {
		for(int j=0; j < cols; j++)
		{
			if(mapa[i][j] > maximo_local.valor)
			{
				maximo_local.y = i + p_glob;
				maximo_local.x = j;
				maximo_local.valor = mapa[i][j];
			}// if
		}// j
	}// i

	// Reduccion para obtener el maximo global
	MPI_Allreduce( &maximo_local, maximo_global, 1, *MPI_antena,
		*MPI_Maximo_Antena, MPI_COMM_WORLD );
}


/**
 * Calcular el numero de filas de los sub-mapas de cada proceso y la
 * posicion relativa de su primera fila en el mapa global
 */
void calcula_Submatrices(int rank, int n_procs, int rows, int* p_rows,
	int* p_glob)
{
	*p_rows = rows / n_procs;

	int	p_rows2 = *p_rows + 1,
		p_resto = rows % n_procs;	// Se reparten a los primeros procesos

	if( rank < p_resto )
	{
		*p_rows = p_rows2;
		*p_glob = rank * p_rows2;
	} else {
		*p_glob = p_resto*p_rows2 + (*p_rows)*(rank - p_resto);
	}
}


/**
 * Función principal
 */
int main(int nargs, char **vargs)
{
	//
	// 1. LEER DATOS DE ENTRADA
	//
	
	int n_procs, rank;

	// Inicialización del entorno MPI.
	MPI_Init(&nargs, &vargs);

	// Obtenemos el identificador y el número de procesos.
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

	
	// Comprobar número de argumentos
	if( nargs < 7 )
	{
		fprintf(stderr,
				"Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",
				vargs[0]
		);
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
	if(rank == 0)
		printf("Calculando el número de antenas necesarias para cubrir un"
			" mapa de (%d x %d)\ncon una distancia máxima no superior a %d "
			"y con %d antenas iniciales\n\n", rows, cols, distMax, nAntenas);

	// Reservar memoria para las antenas
	Antena* antenas = malloc( sizeof(Antena) * (size_t)nAntenas );
	if( !antenas )
	{
		fprintf(stderr,
				"Error al reservar memoria para las antenas inicales\n"
		);
		return -1;
	}
	
	// Leer antenas
	for(int i = 0; i < nAntenas; i++)
	{
		antenas[i].x = atoi(vargs[5+i*2]);
		antenas[i].y = atoi(vargs[6+i*2]);

		if( antenas[i].y < 0 || antenas[i].y >= rows ||
			antenas[i].x < 0 || antenas[i].x >= cols )
		{
			fprintf(stderr, "Antena #%d está fuera del mapa\n", i);
			return -1;
		}
	}

	//
	// 2. INICIACIÓN
	//

	// Medir el tiempo
	MPI_Barrier(MPI_COMM_WORLD);
	double tiempo = cp_Wtime();


	/*
	 * MPI
	 */
	// Definicion del tipo antena de MPI
	MPI_Datatype MPI_antena;
	Antena miAntena;

	// Direcciones de los campos
	MPI_Aint direccion_Antena, direccion_y, direccion_x, direccion_valor;
	MPI_Get_address(&miAntena, &direccion_Antena);
	MPI_Get_address(&miAntena.y, &direccion_y);
	MPI_Get_address(&miAntena.x, &direccion_x);
	MPI_Get_address(&miAntena.valor, &direccion_valor);

	// Calculo de los desplazamientos
	MPI_Aint despl_y = direccion_y - direccion_Antena;
	MPI_Aint despl_x = direccion_x - direccion_Antena;
	MPI_Aint despl_valor = direccion_valor - direccion_Antena;

	// Creacion del tipo
	int longitudes[3] = { 1, 1, 1 };
	MPI_Aint desplazamientos[3] = {	despl_y, despl_x, despl_valor };
	MPI_Datatype tipos[3] = { MPI_INT, MPI_INT, MPI_INT };
	MPI_Type_create_struct(3, longitudes, desplazamientos, tipos, &MPI_antena);
	MPI_Type_commit(&MPI_antena);

	// Operacion reduce para obtener el maximo global en MPI
	MPI_Op MPI_Maximo_Antena;
	MPI_Op_create(calcula_maximo_global, 1, &MPI_Maximo_Antena);


	int p_rows,	// Numero de filas del sub-mapa de cada proceso
		p_glob;	// Posicion de la primera fila del sub-mapa respecto al resto
	// Calculamos cuantas filas tiene el sub-mapa de cada proceso.
	calcula_Submatrices(rank, n_procs, rows, &p_rows, &p_glob);


	// Reservamos memoria para los sub-mapas
	int** mapa = (int**)malloc( p_rows * sizeof(int*) );
	for(int i=0; i < p_rows; i++)
		mapa[i] = (int*)malloc( cols * sizeof(int*) );


	// Colocamos las antenas iniciales
	actualizar_primera_antena(mapa, p_rows, cols, p_glob, &antenas[0]);
	for(int i = 1; i < nAntenas; i++)
	{
		actualizar(mapa, p_rows, cols, p_glob, &antenas[i]);
	}

	// Debug
#ifdef DEBUG
	print_mapa(mapa, p_rows, cols, NULL);
#endif

	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	int nuevas = 0;		// Contador de antenas

	while(1)
	{
		Antena maximo_global = { -1, -1, -1 };

		// Calcular la posicion de la antena a distancia maxima.
		calcular_max( mapa, p_rows, cols, p_glob, &MPI_antena,
			&MPI_Maximo_Antena, &maximo_global );

		// Salimos si ya hemos cumplido el maximo
		if( maximo_global.valor <= distMax ) break;	
		
		// Incrementamos el contador
		nuevas++;
		
		// Actualización del mapa con la nueva antena y nuvas distancias.
		actualizar(mapa, p_rows, cols, p_glob, &maximo_global);
	}

	// Debug
#ifdef DEBUG
	print_mapa(mapa, p_rows, cols, NULL);
#endif

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	MPI_Barrier(MPI_COMM_WORLD);
	tiempo = cp_Wtime() - tiempo;

	// Salida
	if(rank == 0){
		printf("Result: %d\n", nuevas);
		printf("Time: %f\n", tiempo);
	}

	MPI_Finalize();

	return 0;
}

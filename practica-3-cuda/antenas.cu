/**
 * Computación Paralela (curso 1516)
 *
 * Colocación de antenas
 * Versión Cuda
 *
 * @author Daniel González Alonso
 * @author Santos Pardo Ramos
 */

// Includes generales
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// CUDA
#include <cuda_runtime.h>

// Include para las utilidades de computación paralela
#include <cutil.h>


/**
 * Estructura antena
 */
typedef struct {
    int y;
    int x;
    int valor;
} Antena;

/**
 * Macro para acceder a las posiciones del mapa
 */
#define m(y,x) mapa[ ((y) * cols) + x ]


#define NUM_THREADS_PER_BLOCK       512
#define NUM_BLOCKS                  4
#define NUM_THREADS                 (NUM_THREADS_PER_BLOCK * NUM_BLOCKS)

//Declaración del vector resultados de la reducción
static int* d_Result;




/**
 * Función de ayuda para imprimir el mapa
 */
__global__ void print_mapa(int* mapa, int rows, int cols, Antena* a)
{
    if(rows > 50 || cols > 30){
        printf("Mapa muy grande para imprimir\n");
        return;
    }

    #define ANSI_COLOR_RED     "\x1b[31m"
    #define ANSI_COLOR_GREEN   "\x1b[32m"
    #define ANSI_COLOR_RESET   "\x1b[0m"

    printf("Mapa [%d,%d]\n", rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++)
        {
            int val = m(i, j);

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
__device__ void manhattan(Antena* a, int y, int x, int* resultado)
{
    int dist = abs(a->x - x) + abs(a->y - y);
    (*resultado) = dist * dist;
}


/**
 * Actualizar el mapa con la nueva antena
 */
__global__ void actualizar(int* mapa, int rows, int cols, Antena* antena)
{
    // El bloque es un cuadrado 2x2, si estamos en el hilo (0,0) actualizamos
    // la antena.
    if (threadIdx.x == 0 && threadIdx.y == 0)
        m(antena->y, antena->x) = 0;

    int r       = 1,
        flag    = -1,
        distancia[1];

    while (flag != 0)
    {
        flag = 0;       // Reiniciamos la bandera

        manhattan(antena, antena->y - r, antena->x, &distancia[0]);

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            Antena izquierda = (Antena){ antena->y, (antena->x - r), -1 };

            // Si izquierda.x esta fuera del mapa, iniciamos delta en la
            // primera interseccion del rombo con el mapa por la izquierda.
            int delta = (izquierda.x < 0)? -(izquierda.x) : 0;

            // Ascendemos por el lateral superior izquierdo del rombo
            for (; (delta < r) && (izquierda.y - delta >= 0); delta++)
            {
                if (m(izquierda.y - delta, izquierda.x + delta) > distancia[0])
                {
                    //printf("IDBX:%d, IDBY:%d dist=%d, aby=%d, abx=%d delta:%d, r:%d\n", threadIdx.x, threadIdx.y, distancia[0], izquierda.y - delta, izquierda.x + delta, delta, r);
                    m(izquierda.y - delta, izquierda.x + delta) = distancia[0];
                    flag = 1;
                }
            }
        }

        if (threadIdx.x == 1 && threadIdx.y == 0)
        {
            Antena arriba = (Antena){ (antena->y - r), antena->x, -1 };

            // Si arriba.y esta fuera del mapa, iniciamos delta en la primera
            // interseccion del rombo con el mapa por arriba.
            int delta = (arriba.y < 0)? -(arriba.y) : 0;
            
            // Descendemos por el lateral superior derecho del rombo
            for (; (delta < r) && (arriba.x + delta < cols); delta++)
            {
                if (m(arriba.y + delta, arriba.x + delta) > distancia[0])
                {
                    //printf("IDBX:%d, IDBY:%d dist=%d, aby=%d, abx=%d delta:%d, r:%d\n", threadIdx.x, threadIdx.y, distancia[0], arriba.y + delta, arriba.x + delta, delta, r);
                    m(arriba.y + delta, arriba.x + delta) = distancia[0];
                    flag = 1;
                }
            }
        }

        if (threadIdx.x == 1 && threadIdx.y == 1)
        {
            Antena derecha = (Antena){ antena->y, (antena->x + r), -1 };

            // Si derecha.x esta fuera del mapa, iniciamos delta en la primera
            // interseccion del rombo con el mapa por la derecha.
            int delta = (derecha.x >= cols)? derecha.x + 1 - cols : 0;
            
            // Descendemos por el lateral inferior derecho del rombo
            for (; (delta < r) && (derecha.y + delta < rows); delta++)
            {
                if (m(derecha.y + delta, derecha.x - delta) > distancia[0])
                {
                    //printf("IDBX:%d, IDBY:%d dist=%d, aby=%d, abx=%d delta:%d, r:%d\n", threadIdx.x, threadIdx.y, distancia[0], derecha.y + delta, derecha.x - delta, delta, r);
                    m(derecha.y + delta, derecha.x - delta) = distancia[0];
                    flag = 1;
                }
            }
        }

        if (threadIdx.x == 0 && threadIdx.y == 1)
        {
            Antena abajo = (Antena){ (antena->y + r), antena->x, -1 };

            // Si abajo.y esta fuera del mapa, iniciamos delta en la primera
            // interseccion del rombo con el mapa por abajo.
            int delta = (abajo.y >= rows)? abajo.y + 1 - rows : 0;

            // Ascendemos por el lateral inferior izquierdo del rombo
            for (; (delta < r) && (abajo.x - delta >= 0); delta++)
            {
                if (m(abajo.y - delta, abajo.x - delta) > distancia[0])
                {
                    //printf("IDBX:%d, IDBY:%d dist=%d, aby=%d, abx=%d delta:%d, r:%d\n", threadIdx.x, threadIdx.y, distancia[0], abajo.y - delta, abajo.x - delta, delta, r);
                    m(abajo.y - delta, abajo.x - delta) = distancia[0];
                    flag = 1;
                }
            }
        }

        r++;
    }

    __syncthreads();
}


/**
 * Actualizar el mapa con la primera antena
 */
__global__ void actualizar_primera_antena(int* mapa, int rows, int cols,
    Antena* antena)
{
    // El bloque es un cuadrado 2x2, si estamos en el hilo (0,0) actualizamos
    // la antena.
    if (threadIdx.x == 0 && threadIdx.y == 0)
        m(antena->y, antena->x) = 0;

    int r       = 1,
        flag    = -1,
        distancia[1];

    while (flag != 0)
    {
        flag = 0;       // Reiniciamos la bandera

        manhattan(antena, antena->y - r, antena->x, &distancia[0]);

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            Antena izquierda = (Antena){ antena->y, (antena->x - r), -1 };

            // Si izquierda.x esta fuera del mapa, iniciamos delta en la
            // primera interseccion del rombo con el mapa por la izquierda.
            int delta = (izquierda.x < 0)? -(izquierda.x) : 0;

            // Ascendemos por el lateral superior izquierdo del rombo
            for (; (delta < r) && (izquierda.y - delta >= 0); delta++)
            {
                m(izquierda.y - delta, izquierda.x + delta) = distancia[0];
                flag = 1;
            }
        }

        if (threadIdx.x == 1 && threadIdx.y == 0)
        {
            Antena arriba = (Antena){ (antena->y - r), antena->x, -1 };

            // Si arriba.y esta fuera del mapa, iniciamos delta en la primera
            // interseccion del rombo con el mapa por arriba.
            int delta = (arriba.y < 0)? -(arriba.y) : 0;
            
            // Descendemos por el lateral superior derecho del rombo
            for (; (delta < r) && (arriba.x + delta < cols); delta++)
            {
                m(arriba.y + delta, arriba.x + delta) = distancia[0];
                flag = 1;
            }
        }

        if (threadIdx.x == 1 && threadIdx.y == 1)
        {
            Antena derecha = (Antena){ antena->y, (antena->x + r), -1 };

            // Si derecha.x esta fuera del mapa, iniciamos delta en la primera
            // interseccion del rombo con el mapa por la derecha.
            int delta = (derecha.x >= cols)? derecha.x + 1 - cols : 0;
            
            // Descendemos por el lateral inferior derecho del rombo
            for (; (delta < r) && (derecha.y + delta < rows); delta++)
            {
                m(derecha.y + delta, derecha.x - delta) = distancia[0];
                flag = 1;
            }
        }

        if (threadIdx.x == 0 && threadIdx.y == 1)
        {
            Antena abajo = (Antena){ (antena->y + r), antena->x, -1 };

            // Si abajo.y esta fuera del mapa, iniciamos delta en la primera
            // interseccion del rombo con el mapa por abajo.
            int delta = (abajo.y >= rows)? abajo.y + 1 - rows : 0;

            // Ascendemos por el lateral inferior izquierdo del rombo
            for (; (delta < r) && (abajo.x - delta >= 0); delta++)
            {
                m(abajo.y - delta, abajo.x - delta) = distancia[0];
                flag = 1;
            }
        }

        r++;
    }

    __syncthreads();
}


/**
 * Kernel que realiza la reducción de un array de entrada y lo deja en un array de salida
 */
__global__ void reduce_kernel(const int* g_idata, int numValues, int* g_odata){
    extern __shared__ int sdata[];

    // cada hilo carga un elemento desde memoria global hacia memoria shared
    unsigned int tid = threadIdx.x;
    unsigned int igl = blockIdx.x;
    sdata[tid] = g_idata[blockDim.x * igl + tid];
    __syncthreads();
    
    if ((blockDim.x * igl + tid) <= numValues)
    {
        // Hacemos la reducción en memoria shared
        for(unsigned int s = 1; s < blockDim.x; s *= 2) {
            // Comprobamos si el hilo actual es activo para esta iteración
            if (tid % (2*s) == 0){
                // Hacemos la reducción sumando los dos elementos que le tocan a este hilo
                if (sdata[tid] < sdata[tid + s])
                    sdata[tid] = sdata[tid + s];
            }
            __syncthreads();
        }
    }

    // El hilo 0 de cada bloque escribe el resultado final de la reducción
    // en la memoria global del dispositivo pasada por parámetro (g_odata[])
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];

}


/**
 * Función que se encarga de lanzar los kernels para realizar la reducción
 * del maximo
 */
extern "C" int* reduce(const int* values, unsigned int numValues){

    int numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    int numBlocks = NUM_BLOCKS;

    //La primera pasada reduce el array de entrada: VALUES
    //a un array de igual tamaño que el número total de bloques del grid: D_RESULT
    int sharedMemorySize = numThreadsPerBlock * sizeof(int);

    reduce_kernel<<<numBlocks, numThreadsPerBlock, sharedMemorySize>>>(values, numValues, d_Result);

    //La segunda pasada lanza sólo un único bloque para realizar la reducción final
    numThreadsPerBlock = numBlocks;
    numBlocks = 1;
    sharedMemorySize = numThreadsPerBlock * sizeof(int);
    reduce_kernel<<<numBlocks, numThreadsPerBlock, sharedMemorySize>>>(d_Result, numValues, d_Result);

    return d_Result;
}


/**
 * Calcular la posición de la nueva antena
 */
__global__ void nueva_antena(int* mapa, int rows, int cols, int min, Antena* maximo)
{
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){

            if(m(i,j)==min)
            {
                (*maximo) = (Antena){ i, j, min };
            }

        } // j
    } // i

}



/**
 * Función principal
 */
int main(int nargs, char** vargs)
{
    //
    // 1. LEER DATOS DE ENTRADA
    //
    
    // Comprobar número de argumentos
    if (nargs < 7)
    {
        fprintf(stderr,
                "Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",
                vargs[0]);
        return -1;
    }

    // Leer los argumentos de entrada
    int rows = atoi(vargs[1]);
    int cols = atoi(vargs[2]);
    int distMax = atoi(vargs[3]);
    int nAntenas = atoi(vargs[4]);

    if (nAntenas<1 || nargs != (nAntenas*2+5))
    {
        fprintf(stderr, "Error en la lista de antenas\n");
        return -1;
    }

    // Mensaje
    printf("Calculando el número de antenas necesarias para cubrir un"
        " mapa de (%d x %d)\ncon una distancia máxima no superior a %d "
        "y con %d antenas iniciales\n\n", rows, cols, distMax, nAntenas);

    // Reservar memoria para las antenas
    Antena* antenas = (Antena*)malloc(sizeof(Antena) * nAntenas);
    if (!antenas)
    {
        fprintf( stderr,
                "Error al reservar memoria para las antenas inicales\n" );
        return -1;
    }
    
    // Leer antenas
    for (int i = 0; i < nAntenas; i++)
    {
        antenas[i].x = atoi(vargs[5+i*2]);
        antenas[i].y = atoi(vargs[6+i*2]);

        if( antenas[i].y < 0 || antenas[i].y >= rows
            || antenas[i].x < 0 || antenas[i].x >= cols )
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


    unsigned int numValues = rows * cols;

    // Antena sobre la que trabajaremos en el device
    Antena* antenaDevice;
    cudaMalloc((void**)&antenaDevice, sizeof(Antena));

    // Array sobre el que haremos la reduccion en reduce_kernel
    cudaMalloc((void**)&d_Result, NUM_BLOCKS * sizeof(int));

    // Crear el mapa
    int* mapa;
    cudaMalloc((void**)&mapa, rows * cols * sizeof(int)) ;

    // Declaración del shape de los bloques y del grid
    dim3 gridShape1(1);
    dim3 bloqShape1(2, 2);


    // Colocar las antenas iniciales
    cudaMemcpy(antenaDevice, &antenas[0], sizeof(Antena), cudaMemcpyHostToDevice);
    actualizar_primera_antena<<<gridShape1, bloqShape1>>>(mapa, rows, cols, antenaDevice);
    for (int i = 1; i < nAntenas; i++)
    {
        cudaMemcpy(antenaDevice, &antenas[i], sizeof(Antena), cudaMemcpyHostToDevice) ;
        actualizar<<<gridShape1, bloqShape1>>>(mapa, rows, cols, antenaDevice);
    }

    // Debug
#ifdef DEBUG
    print_mapa(mapa, rows, cols, NULL);
#endif


    //
    // 3. CALCULO DE LAS NUEVAS ANTENAS
    //
    
    int nuevas = 0;     // Contador de antenas
    int* valor = (int*)malloc( sizeof(int)) ;

    while (1)
    {
        // Calcular la posicion de la antena a distancia maxima.
        reduce(mapa, numValues);

        cudaMemcpy(valor, d_Result, sizeof(int), cudaMemcpyDeviceToHost) ;

        // Salimos si ya hemos cumplido el maximo
        if ((*valor) <= distMax) break;

        // Incrementamos el contador
        nuevas++;

        nueva_antena<<<1,1>>>(mapa, rows, cols, (*valor), antenaDevice);

        // Actualización del mapa con la nueva antena y nuevas distancias.
        actualizar<<<gridShape1, bloqShape1>>>(mapa, rows, cols, antenaDevice);
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
    //print_mapa<<<1, 1>>>(mapa, rows, cols, NULL);

    // Liberamos memoria del device
    cudaFree(mapa);
    cudaFree(antenaDevice);
    cudaFree(d_Result);
    
    // Liberamos memoria del HOST
    free(antenas);
    free(valor);

    // Liberamos los hilos del DEVICE
    cudaDeviceReset();

    return 0;
}

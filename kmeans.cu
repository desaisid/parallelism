/* 
  k means in n dimensions
*/

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#define DIMENSIONALITY 2
#define MAX_ALLOWED_K  100

struct point
{
  int location[DIMENSIONALITY];
  int closest_mean_index;

};

typedef struct point point_t;

struct mean_point
{
  int location[DIMENSIONALITY];
  int num_associated_points;
};

typedef mean_point mean_point_t;
__device__ int get_distance(point_t p1, mean_point_t p2)
{
  int result;
  //Ideally, euclidean distance would have been
  // dist = square_root(sum( square(p1[i]-p2[i]) )for all i=0..DIMENSIONALITY )
  //we use a fast approximation instead 
  
  for(int i =0; i < DIMENSIONALITY;i++)
    result = abs(p1.location[i]-p2.location[i]);
    
  return result;
}

__global__ void calculate_centroid_part1(point_t * point_array, int num_points, mean_point_t* new_means_array)
{
  
  int threadid = blockDim.x*blockIdx.x + threadIdx.x; // this better be a grid of (X, 1, 1) dimension
  
  if(threadid > num_points -1)
    return;
    
  int index = point_array[threadid].closest_mean_index;
  new_means_array[index].num_associated_points = 0; // Yes, we are overdoing it right now. Multiple times in every thread. Is it better serially? at scale?
  
  __syncthreads();
  for(int i =0; i<DIMENSIONALITY; i++)
  {
    atomicAdd(&(new_means_array[index].location[i]), point_array[threadid].location[i]);
  }   
  atomicInc((unsigned int *)&(new_means_array[index].num_associated_points),1);
}

void calculate_centroid_part2(mean_point_t* new_means_array, int k)
{
//normalize the centroid values
  for(int i =0; i < k; i++)
    for(int j =0; j < DIMENSIONALITY; j++)
      new_means_array[i].location[j] /= new_means_array[i].num_associated_points;

}

/* I dont like this, but until I find a way out, point_array is input and output */
__global__ void find_clusters(point_t* point_array, int point_array_len, mean_point_t* old_means_array,
                               int k)
{
  
  //First lets copy the mean_list into shared mem, so that we are faster  
  __shared__ mean_point_t mean_list[MAX_ALLOWED_K];
  
  if(threadIdx.x <k)
  {
    for(int i=0;i < DIMENSIONALITY; i++)
      mean_list[threadIdx.x].location[i] = old_means_array[threadIdx.x].location[i];
  }
  
  __syncthreads();
  
  int threadid = blockDim.x*blockIdx.x + threadIdx.x; // this better be a grid of (X, 1, 1) dimension
  
  if(threadid > (point_array_len -1) )
    return;
  
  int min_distance = 0;
  for(int i =0; i < k; i++)
  {
    int distance_to_mean = get_distance(point_array[threadid], mean_list[i]);
    if(distance_to_mean < min_distance)
    {
      min_distance = distance_to_mean;
      point_array[threadid].closest_mean_index = i;

    }
  
  }
    
} 

bool are_means_equal(mean_point_t *old_means,mean_point_t * new_means, int K)
{
  int ret = memcmp(old_means, new_means, sizeof(mean_point_t)*K); // memcmp returns 0 if no difference
  return (ret==0);
}

void print_means(mean_point_t* mean_list, int K)
{
  for(int i=0;i < K;i++)
  {
    for(int j =0; j< DIMENSIONALITY;j++)
      printf("mean[%d].location[%d] = %d\n",i,j, mean_list[i].location[j]);
    printf("mean[%d].num_associated_points = %d\n", i,mean_list[i].num_associated_points);
  }
}

void print_points(point_t *point_list, int num_points)
{
  for(int i =0;i < num_points;i++)
  {
    for(int j=0;j < DIMENSIONALITY;j++)
      printf("point[%d].location[%d] = %d\n",i,j,point_list[i].location[j]);
    printf("point[%d].closest_mean_index = %d\n",i,point_list[i].closest_mean_index);
  
  }

}

int main()
{
  int K = 2;
  printf("Starting ... \n");
  point_t the_points[] = { {{1,1}, 0}, {{1,2}, 0},{{2,1}, 0}, {{8,7}, 0}, {{8,8}, 0},{{7,8}, 0} };
  int num_points = sizeof(the_points)/sizeof(point_t);
  point_t *d_points_list = NULL;
  mean_point_t init_means[] = { {{4,3}, 0}, {{10,12}, 1} };
  
  mean_point_t *d_means = NULL;
  mean_point_t* h_new_means = (mean_point_t *)malloc(sizeof(init_means));
  mean_point_t* d_new_means = NULL;
  mean_point_t* h_old_means = (mean_point_t *)malloc(sizeof(init_means));
  
  //init device memories
  checkCudaErrors(cudaMalloc((void **)&d_points_list, sizeof(the_points)));
  checkCudaErrors(cudaMalloc((void **)&d_means,  sizeof(init_means)));
  checkCudaErrors(cudaMalloc((void **)&d_new_means,   sizeof(init_means)));
  checkCudaErrors(cudaMemcpy(d_points_list, the_points, sizeof(the_points), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_means,  init_means, sizeof(init_means), cudaMemcpyHostToDevice)); //not really needed
  memcpy(h_new_means, init_means,sizeof(init_means));
  memset(h_old_means, 0 ,sizeof(init_means)); //not really needed, i dont like uninitialized fellows
  
  // thread dimensioning
  const int blockSize = 256; //one dimensional
  const int gridSize  = 2;  // number of blocks


  while(!are_means_equal(h_old_means, h_new_means,K))  
  {
    memcpy(h_old_means, h_new_means,sizeof(init_means));
    
    checkCudaErrors(cudaMemcpy(d_means,  h_old_means, sizeof(init_means), cudaMemcpyHostToDevice));
    find_clusters<<<gridSize, blockSize, MAX_ALLOWED_K*sizeof(mean_point_t)>>>(d_points_list, num_points, d_means, K);
    calculate_centroid_part1<<<gridSize, blockSize, MAX_ALLOWED_K*sizeof(mean_point_t)>>>(d_points_list, num_points, d_new_means);
    checkCudaErrors(cudaMemcpy( h_new_means, d_new_means, sizeof(init_means), cudaMemcpyDeviceToHost));  
    calculate_centroid_part2(h_new_means, K); 
    
    //print whats going on
    print_means(h_new_means,K);
    print_points(the_points,num_points); 

  }

  printf("Done.\n");
  print_points(the_points,num_points);
  return 0;
}

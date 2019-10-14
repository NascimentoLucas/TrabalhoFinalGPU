#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 100
#define MIN 50
#define TIME 10

typedef struct { 
  int life, actualLife, 
  strength, 
  speed, actualSpeed, 
  cDamage;   
} Fighter;

int get_random(){
  return (int)(((float)rand()/RAND_MAX) * (MAX - MIN) + MIN);
}

void randomizeFighters(Fighter *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].life = get_random();
    data[i].actualLife = data[i].life;
    data[i].strength = get_random();

    data[i].speed = get_random();
    data[i].actualSpeed = data[i].speed;
    data[i].cDamage = get_random();
  }
}

void showFighters(Fighter *data, int n) {
  for (int i = 0; i < n; i++) {
    printf("\n__%d__\nlife: %d/%d\nstrength: %d\nspeed: %d/%d\ncDamage: %d", i, 
    data[i].actualLife, data[i].life,
    data[i].strength,
    data[i].actualSpeed, data[i].speed, 
    data[i].cDamage);
  }
}

__device__ 
int get_damage(Fighter *f, int atk, int target){
  int str = f[atk].strength;
  int atkSpeed = max(f[atk].actualSpeed, 1);
  int targetSpeed = max(f[target].actualSpeed, 1);

  int damage = __float2int_rd(str * ((float)atkSpeed / targetSpeed));
  return damage;
}

__device__ 
int get_corruption(Fighter *f, int atk, int target){
  int cDam = f[atk].cDamage;
  int atkLife = max(f[atk].actualLife, 1);
  int targetLife = max(f[target].actualLife, 1);

  int damage = __float2int_rd(cDam * ((float)atkLife / targetLife));
  return damage;
}
__global__
void fight(Fighter *f, int n) {
  int index = 2 * threadIdx.x + blockIdx.x * blockDim.x;
  int stride = 2 *  (blockDim.x * gridDim.x);
  int j;
  int firstDamage;
  int secondDamage;
  for(int i = index; i < n; i += stride)
  {     
    j = i + 1;

    for(int k = 0; k < TIME; k += 1){

      firstDamage = get_corruption(f, j, i);
      secondDamage = get_corruption(f, i, j);

      f[i].actualLife -=  get_damage(f, j, i);
      f[j].actualLife -= get_damage(f, i, j);
      
      f[i].actualSpeed -= firstDamage;
      f[j].actualSpeed -= secondDamage;
    }
  }
}

int main(const int argc, const char** argv) {

  int nBodies = 2;//2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);
    
  int deviceId;
  int numberOfSMs;
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  size_t threadsPerBlock = 1;//256;
  size_t numberOfBlocks = 1;//32 * numberOfSMs;

  int bytes = nBodies * sizeof(Fighter);
  Fighter *buf;
  cudaMallocManaged(&buf, bytes);
  //cudaMemPrefetchAsync(buf, bytes, deviceId);

  randomizeFighters(buf,nBodies);
        
  fight<<<numberOfBlocks, threadsPerBlock>>>(buf, nBodies); 

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

  cudaError_t asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));  

  //cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);
  
  //showFighters(buf, nBodies);
  cudaFree(buf);
}

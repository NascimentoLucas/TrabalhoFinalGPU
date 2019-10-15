#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 100
#define MIN 50
#define ADJUSTLIFE 100

#define MAXADJUST 15
#define MINADJUST 0

#define TIME 100
#define TEST 1

typedef struct { 
  int life, actualLife, 
  strength, 
  speed, actualSpeed, 
  cDamage;   
} Fighter;

void printFighter(Fighter data){
  //printf("\n__%d__", i);
  printf("\nlife %d/%d", data.actualLife, data.life);
  //printf("\nstrength %d", data.strength);
  //printf("\nspeed %d/%d", data.actualSpeed, data.speed);
  //printf("\ncDamage %d", data.cDamage);
}
//todo garantir o random
int get_random(int min, int max){
  return (int)(((float)rand()/RAND_MAX) * (max - min) + min);
}

int get_random_neg(){

  int multi = 1;

  if(get_random(0, 10) > 5){
    multi = -1;
  }

  return (int)((((float)rand()/RAND_MAX) * (MAXADJUST - MINADJUST) + MINADJUST) * multi);
}

void randomizeFighters(Fighter *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].life = get_random(MIN + ADJUSTLIFE, MAX + ADJUSTLIFE);
    data[i].actualLife = data[i].life;
    data[i].strength = get_random(MIN, MAX);

    data[i].speed = get_random(MIN, MAX);
    data[i].actualSpeed = data[i].speed;
    data[i].cDamage = get_random(MIN, MAX);
  }
}

void showFighters(Fighter *data, int n) {
  printf("\nshowing fighters");
  for (int i = 0; i < n; i++) {
    printf("\n__%d__", i);
    printf("\nlife %d/%d", data[i].actualLife,data[i].life);
    //printf("\nstrength %d", data[i].strength);
    //printf("\nspeed %d/%d", data[i].actualSpeed,data[i].speed);
    //printf("\ncDamage %d", data[i].cDamage);
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
  int cDam = f[atk].cDamage * 0.01;
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

void selectFighters(Fighter *data, int n) {
  //printf("\nselecting fighters");
  int aux = 0;
  for (int i = 0; i < n / 2; i+=2) {
    if(data[i].actualLife > data[i + 1].actualLife){
      data[aux] = data[i];
    }
    else if(data[i].actualLife < data[i + 1].actualLife){
      data[aux] = data[i + 1];
    }
    else{
      if(data[i].actualSpeed > data[i + 1].actualSpeed){
        data[aux] = data[i];
      }
      else if(data[i].actualSpeed < data[i + 1].actualSpeed){
        data[aux] = data[i + 1];
      }
      else{
        data[aux] = data[i];
      }
    }
    data[aux].actualLife =  data[aux].life;
    data[aux].actualSpeed =  data[aux].speed;
    aux++;
  }
}

void multiplyFighters(Fighter *data, int n) {
  //printf("\nmultipling fighters");
  Fighter aux;
  for (int i = n / 2; i < n; i+=2) {
    aux = data[i - n / 2];
    data[i].life = aux.life +  get_random_neg();
    data[i].actualLife = data[i].life;
    data[i].strength = aux.strength +  get_random_neg();

    data[i].speed = aux.speed +  get_random_neg();
    data[i].actualSpeed = data[i].speed;
    data[i].cDamage = aux.cDamage +  get_random_neg();
  }
}

int main(const int argc, const char** argv) {

  int nBodies = 10;//2<<11;
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

  randomizeFighters(buf, nBodies);
  showFighters(buf, nBodies);
  for (int t = 0; t < TEST; t++){
    /*printf("\n");
    printf("\n");*/
    //printf("\nTTTTT%dTTTTT", t);
    /*printf("\n");
    printf("\n");*/
            
    fight<<<numberOfBlocks, threadsPerBlock>>>(buf, nBodies); 

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    cudaError_t asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));  

    //showFighters(buf, nBodies);
    if ( t < TEST - 1 ){
      selectFighters(buf, nBodies);
      multiplyFighters(buf, nBodies);
    }
    else{
      showFighters(buf, nBodies);
    }
  }
  //cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);



  cudaFree(buf);
}

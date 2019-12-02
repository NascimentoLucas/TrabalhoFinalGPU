#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 1
#define EXPORT true

#define MAX 1000
#define MIN 1

#define MAXADJUST 10
#define MINADJUST 0

#define AMOUNTINTERACTION 100
#define AMOUNTTESTS 1000

typedef struct { 
  int id, generation,
  life, actualLife, 
  strength, 
  speed, actualSpeed, 
  cDamage;   
} Fighter;

void printFighter(Fighter data){
  //printf("\n__%d__", i);
  printf("\nID %d", data.id);
  printf("\ngeneration %d", data.generation);
  printf("\nlife %d/%d", data.actualLife, data.life);
  printf("\nstrength %d", data.strength);
  printf("\nspeed %d/%d", data.actualSpeed, data.speed);
  printf("\ncDamage %d", data.cDamage);
  printf("\n_____");
}

void printFighterExport(Fighter data){
  //printf("\n__%d__", i);
  printf("\n%d", data.id);
  printf(";%d", data.generation);
  printf(";%d", data.life);
  printf(";%d", data.strength);
  printf(";%d", data.speed);
  printf(";%d", data.cDamage);
}

int GetRandom(int min, int max){
  return (int)(((float)rand()/RAND_MAX) * (max - min) + min);
}

int GetRandomNeg(){

  int multi = 1;

  if(GetRandom(0, 10) > 5){
    multi = -1;
  }

  return (int)((((float)rand()/RAND_MAX) * (MAXADJUST - MINADJUST) + MINADJUST) * multi);
}

int MaxMin(int value, int adjust){

  if(MAX + adjust< value){
    return MAX + adjust;
  }
  else if (MIN + adjust> value){
    return MIN + adjust;
  }
  return value;
}

int GetSpeed(int life){
  return MaxMin(MAX - life, 0);
}

void CreateFighters(Fighter *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].id = i;
    data[i].generation = 0;
    data[i].life = GetRandom(MIN, 2);
    data[i].actualLife = data[i].life;
    data[i].strength = GetRandom(MIN, 2);

    #if MODESPEED > 0
      data[i].speed = GetSpeed(data[i].life);
    #else
      data[i].speed = GetRandom(MIN, 2);
    #endif
    data[i].actualSpeed = data[i].speed;
    data[i].cDamage = GetRandom(MIN, 2);
  }
}

void showFighters(Fighter *data, int n) {
  printf("\nshowing fighters");
  for (int i = 0; i < n; i++) {
    printFighter(data[i]);    
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

    for(int k = 0; k < AMOUNTINTERACTION; k += 1){

      firstDamage = get_corruption(f, j, i);
      secondDamage = get_corruption(f, i, j);

      f[i].actualLife -=  get_damage(f, j, i);
      f[j].actualLife -= get_damage(f, i, j);
      
      f[i].actualSpeed -= firstDamage;
      f[j].actualSpeed -= secondDamage;
    }
  }
}

int chooseWinner(Fighter *data, int index){
  if(data[index].actualLife > data[index + 1].actualLife){
    return index;
  }
  else if(data[index].actualLife < data[index + 1].actualLife){
    return index + 1;
  }
  else{
    if(data[index].actualSpeed > data[index + 1].actualSpeed){
      return index;
    }
    else if(data[index].actualSpeed < data[index + 1].actualSpeed){
      return index + 1;
    }
    else{
      int aux = 0;
      if(GetRandom(0,2) > 0){
        aux = 1;
      }
      return index + aux;
    }
  }
}

void selectFighters(Fighter *data, int n) {
  n /= 2;
  #if DEBUG > 1
    printf("\nSelecting#");
  #endif
  int aux = 0;
  int index;
  int start = 0;
  if(n % 2 == 1){
    start = 1;    
    data[aux] = data[0];
    data[aux].actualLife =  data[aux].life;
    data[aux].actualSpeed =  data[aux].speed;
    aux++;
    data[aux] = data[1];
    data[aux].actualLife =  data[aux].life;
    data[aux].actualSpeed =  data[aux].speed;
    aux++;
  }
  
  #if DEBUG > 1
    printf(" #fighters");)
  #endif
  for (int i = start; i < n; i++) {
    index = i * 2;
    data[aux] = data[chooseWinner(data, index)];
    data[aux].actualLife =  data[aux].life;
    data[aux].actualSpeed =  data[aux].speed;
    #if DEBUG > 1
      printFighter(data[aux]);
    #endif
    aux++;
  }
 
}

void Reproduce(Fighter *data, Fighter father, int n) {
  //printf("\nMultipling fighters %d", n);
  for (int i = 0; i < n; i++) {
    data[i].id = i;
    data[i].generation = father.generation + 1;
    data[i].life = MaxMin(father.life +  GetRandomNeg(), 0);
    data[i].actualLife = data[i].life;
    data[i].strength = MaxMin(father.strength +  GetRandomNeg(), 0);

    #if MODESPEED > 0
      data[i].speed = GetSpeed(data[i].life);
    #else
      data[i].speed = MaxMin(father.speed +  GetRandomNeg(), 0);
    #endif
    data[i].actualSpeed = data[i].speed;
    data[i].cDamage = MaxMin(father.cDamage +  GetRandomNeg(), 0);
    //printFighter(data[i]);
  }
}

Fighter copyFighter(Fighter father){
  Fighter son;

  son.id = father.id;
  son.generation = father.generation;
  son.life = father.life * 1;
  son.actualLife = father.life * 1;
  son.strength = father.strength * 1;
  son.speed = father.speed * 1;
  son.actualSpeed = father.speed * 1;
  son.cDamage = father.cDamage * 1;

  return son;
}

int main() {

  int nMaxBodies = 2<<11;
  int nBodies = nMaxBodies;
    
  int deviceId;
  int numberOfSMs;
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = 32 * numberOfSMs;

  int bytes = nBodies * sizeof(Fighter);
  Fighter *buf;
  cudaMallocManaged(&buf, bytes);
  //cudaMemPrefetchAsync(buf, bytes, deviceId);

  CreateFighters(buf, nBodies);
  Fighter champ;
  #if EXPORT
    printf("id;generation;life;strength;speed;cDamage");
  #endif
  for (int t = 0; t < AMOUNTTESTS; t++){
    #if DEBUG > 2
      printf("\n");
      printf("\n");
      printf("\nTTTTT%dTTTTT", t);
      printf("\n");
      printf("\n");
        
      printf("\n\nFirst fithters\n\n");
      showFighters(buf, nBodies);
    #endif
   
    while(nBodies > 2){

      fight<<<numberOfBlocks, threadsPerBlock>>>(buf, nBodies); 

      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

      cudaError_t asyncErr = cudaDeviceSynchronize();
      if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));  

      selectFighters(buf, nBodies);
      nBodies = (int)(nBodies / 2) ;
      nBodies += (nBodies % 2);
    }
    
    champ = buf[chooseWinner(buf, 0)];   
    
    //printf("\n***Round Champion***");
    //printFighter(champ);
    #if EXPORT
      printFighterExport(champ);
    #endif
    //printf("\n*****");

    nBodies = nMaxBodies;
    Reproduce(buf, champ, nBodies);
  }
  //cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);

  cudaFree(buf);
}

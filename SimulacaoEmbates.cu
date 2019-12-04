#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 1
#define EXPORT false

#define MAX 1000
#define MIN 1

#define MAXADJUST 12
#define MINADJUST 0

#define AMOUNTINTERACTION 100
#define AMOUNTTESTS 1000
#define SIZE 10

typedef struct { 
  int id, generation,
  life, actualLife, 
  strength, 
  speed, actualSpeed, 
  cDamage, 
  rate;   
} Fighter;

static Fighter mainFighter;

void printFighter(Fighter data){
  //printf("\n__%d__", i);
  printf("\nID %d", data.id);
  printf("\ngeneration %d", data.generation);
  printf("\nlife %d/%d", data.actualLife, data.life);
  printf("\nstrength %d", data.strength);
  printf("\nspeed %d/%d", data.actualSpeed, data.speed);
  printf("\ncDamage %d", data.cDamage);
  printf("\nrate %d", data.rate);
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
  printf(";%d", data.rate);
}

int GetRandom(int min, int max){
  return (int)(((float)rand()/RAND_MAX) * (max - min) + min);
}

int GetRandomNeg(){

  int multi = 1;

  if(GetRandom(0, 10) > 5){
    multi = -1;
  }

  return GetRandom(MINADJUST, MAXADJUST) * multi;
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

void CreateMainFighter(){  
  mainFighter.id = -1;
  mainFighter.generation = 0;
  mainFighter.life = GetRandom(MAX / 2, MAX);
  mainFighter.strength = GetRandom(MIN, MAX);  
  mainFighter.speed = GetRandom(MIN, MAX);
  mainFighter.cDamage = GetRandom(MIN, MAX);
  
  mainFighter.actualLife = mainFighter.life;  
  mainFighter.actualSpeed = mainFighter.speed;
}

void CreateFighters(Fighter *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].id = i;
    data[i].generation = 0;

    data[i].life = GetRandom(MIN, MAX);
    data[i].actualLife = data[i].life;

    data[i].strength =  GetRandom(MIN, MAX);

    data[i].speed =  GetRandom(MIN, MAX);
    data[i].actualSpeed = data[i].speed;

    data[i].cDamage =  GetRandom(MIN, MAX);
  }
}

void showFighters(Fighter *data, int n) {
  printf("\nshowing fighters");
  for (int i = 0; i < n; i++) {
    printFighter(data[i]);    
  }
}

__device__ 
__host__
int get_damage(Fighter atk, Fighter target){
  int str = atk.strength;
  int atkSpeed = max(atk.actualSpeed, 1);
  int targetSpeed = max(target.actualSpeed, 1);

  int damage = (int)(str * ((float)atkSpeed / targetSpeed));
  return damage;
}

__device__ 
__host__
int get_corruption(Fighter atk, Fighter target){
  int cDam = atk.cDamage * 0.01f;
  int atkLife = max(atk.actualLife, 1);
  int targetLife = max(target.actualLife, 1);

  int damage = (int)(cDam * ((float)atkLife / targetLife));
  return damage;
}

__global__
void fight(Fighter *f, int n, Fighter mainFighter) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int firstDamage;
  int secondDamage;
  int k = 0;
  for(int i = index; i < n; i += stride)
  {   
    while(k < AMOUNTINTERACTION & mainFighter.actualLife > 0){
      k++;
      firstDamage = get_corruption(mainFighter, f[i]);
      secondDamage = get_corruption(f[i], mainFighter);

      f[i].actualLife -=  get_damage(mainFighter, f[i]);
      mainFighter.actualLife -= get_damage(f[i], mainFighter);
      
      f[i].actualSpeed -= firstDamage;
      mainFighter.actualSpeed -= secondDamage;
    }
    k = 0;
    f[i].rate = mainFighter.actualLife - f[i].actualLife;
  }

}

int chooseWinner(Fighter *data, int index){
  float adjust = 0.5f;
  float firstDif = MAX - abs(data[index].life - mainFighter.life) * adjust;
  float secondDif = MAX - abs(data[index + 1].life - mainFighter.life) * adjust;
  float first = abs(data[index].rate) ;
  float second = abs(data[index + 1].rate) ;
  
  if(first < second){
    return index;
  }
  else if(first > second){
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

    data[i].speed = MaxMin(father.speed +  GetRandomNeg(), 0);
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
  int nMaxBodies = 2<<SIZE;
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

  CreateMainFighter();

  #if EXPORT
    printf("\nid;generation;life;strength;speed;cDamage;rate");
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

      fight<<<numberOfBlocks, threadsPerBlock>>>(buf, nBodies, mainFighter); 

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
      //printFighter(champ);
    #endif
    //printf("\n*****");

    nBodies = nMaxBodies;
    Reproduce(buf, champ, nBodies);
  }
  cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);

  champ.actualLife = champ.life;
  champ.actualSpeed = champ.speed;

  printFighter(champ);
  printFighter(mainFighter);
  #if !EXPORT
    int firstDamage;
    int secondDamage;
    int k = 0;
    while(k < AMOUNTINTERACTION & mainFighter.actualLife > 0){
      k++;
      firstDamage = get_corruption(mainFighter, champ);
      secondDamage = get_corruption(champ, mainFighter);

      champ.actualLife -=  get_damage(mainFighter, champ);
      mainFighter.actualLife -= get_damage(champ, mainFighter);
      
      champ.actualSpeed -= firstDamage;
      mainFighter.actualSpeed -= secondDamage;
      printf("\nMain.life: %d <> Champ.life: %d", mainFighter.actualLife, champ.actualLife);
    }
  #endif
  cudaFree(buf);
}

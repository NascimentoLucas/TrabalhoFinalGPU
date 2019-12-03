#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 1
#define EXPORT true
#define MODE 1

#define MAX 1000
#define MIN 1

#define MAXADJUST 2
#define MINADJUST 0

#define AMOUNTINTERACTION 10
#define AMOUNTTESTS 1000
#define SIZE 11

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

void CreateMainFighter(){  
  mainFighter.id = -1,
  mainFighter.generation = 0,
  mainFighter.life = GetRandom(MIN, MAX),
  mainFighter.strength = GetRandom(MIN, MAX),  
  mainFighter.speed = GetRandom(MIN, MAX),
  mainFighter.cDamage = GetRandom(MIN, MAX),
  
  mainFighter.actualLife = mainFighter.life;  
  mainFighter.actualSpeed = mainFighter.speed;
}

void CreateFighters(Fighter *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].id = i;
    data[i].generation = 0;

    #if MODE == 1
      data[i].life = MaxMin(father.cDamage +  GetRandomNeg(), 0);
    #else
      data[i].life = GetRandom(MIN, MAX);
    #endif
    data[i].actualLife = data[i].life;

    #if MODE == 2
      data[i].strength = MaxMin(father.cDamage +  GetRandomNeg(), 0);
    #else
      data[i].strength =  GetRandom(MIN, MAX);
    #endif

    #if MODE == 3
      data[i].speed = MaxMin(father.cDamage +  GetRandomNeg(), 0);
    #else
      data[i].speed =  GetRandom(MIN, MAX);
    #endif  
    data[i].actualSpeed = data[i].speed;

    #if MODE == 4
      data[i].cDamage = MaxMin(father.cDamage +  GetRandomNeg(), 0);
    #else
      data[i].cDamage =  GetRandom(MIN, MAX);
    #endif 
  }
}

void showFighters(Fighter *data, int n) {
  printf("\nshowing fighters");
  for (int i = 0; i < n; i++) {
    printFighter(data[i]);    
  }
}

__device__ 
int get_damage(Fighter atk, Fighter target){
  int str = atk.strength;
  int atkSpeed = max(atk.actualSpeed, 1);
  int targetSpeed = max(target.actualSpeed, 1);

  int damage = __float2int_rd(str * ((float)atkSpeed / targetSpeed));
  return damage;
}

__device__ 
int get_corruption(Fighter atk, Fighter target){
  int cDam = atk.cDamage * 0.01f;
  int atkLife = max(atk.actualLife, 1);
  int targetLife = max(target.actualLife, 1);

  int damage = __float2int_rd(cDam * ((float)atkLife / targetLife));
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
    f[i].rate = mainFighter.life - f[i].life;
  }

}

int chooseWinner(Fighter *data, int index){
  int first = abs(data[index].rate);
  int second = abs(data[index + 1].rate);
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

    #if MODE == 1
      data[i].life = father.life;
    #else
      data[i].life = MaxMin(father.life +  GetRandomNeg(), 0);
    #endif
    data[i].actualLife = data[i].life;

    #if MODE == 2
      data[i].strength = father.strength;
    #else
      data[i].strength = MaxMin(father.strength +  GetRandomNeg(), 0);
    #endif

    #if MODE == 3
      data[i].speed = father.speed;
    #else
      data[i].speed = MaxMin(father.speed +  GetRandomNeg(), 0);
    #endif  
    data[i].actualSpeed = data[i].speed;

    #if MODE == 4
      data[i].cDamage = father.cDamage;
    #else
      data[i].cDamage = MaxMin(father.cDamage +  GetRandomNeg(), 0);
    #endif 
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
  printFighter(mainFighter);

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
    #endif
    //printf("\n*****");

    nBodies = nMaxBodies;
    Reproduce(buf, champ, nBodies);
  }
  //cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);

  cudaFree(buf);
}

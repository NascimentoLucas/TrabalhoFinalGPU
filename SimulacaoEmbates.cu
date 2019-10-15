#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 1000
#define MIN 500
#define ADJUSTLIFE 500

#define MAXADJUST 10
#define MINADJUST 0

#define TIME 100
#define TEST 100

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

void printFighterExel(Fighter data){
  //printf("\n__%d__", i);
  printf("\n%d", data.id);
  printf(";%d", data.generation);
  printf(";%d", data.life);
  printf(";%d", data.strength);
  printf(";%d", data.speed);
  printf(";%d", data.cDamage);
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
    data[i].id = i;
    data[i].generation = 0;
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
      return index;
    }
  }
}

void selectFighters(Fighter *data, int n) {
  n /= 2;
  //printf("\nSelecting#");
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
    //printf(" %d", n + 1);
  }
  else{
    
    //printf(" %d", n);
  }
  //printf(" #fighters");)
  for (int i = start; i < n; i++) {
    index = i * 2;
    data[aux] = data[chooseWinner(data, index)];
    data[aux].actualLife =  data[aux].life;
    data[aux].actualSpeed =  data[aux].speed;
    //printFighter(data[aux]);
    aux++;
  }
 
}

int maxMin(int value, int adjust){

  if(MAX + adjust< value){
    return MAX + adjust;
  }
  else if (MIN + adjust> value){
    return MIN + adjust;
  }
  return value;
}

void multiplyFighters(Fighter *data, Fighter father, int n) {
  //printf("\nMultipling fighters %d", n);
  for (int i = 0; i < n; i++) {
    data[i].id = i;
    data[i].generation = father.generation + 1;
    data[i].life = maxMin(father.life +  get_random_neg(), ADJUSTLIFE);
    data[i].actualLife = data[i].life;
    data[i].strength = maxMin(father.strength +  get_random_neg(), 0);

    data[i].speed = maxMin(father.speed +  get_random_neg(), 0);
    data[i].actualSpeed = data[i].speed;
    data[i].cDamage = maxMin(father.cDamage +  get_random_neg(), 0);
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

int main(const int argc, const char** argv) {

  int nMaxBodies = 2<<11;
  int nBodies = nMaxBodies;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);
    
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

  randomizeFighters(buf, nBodies);
  Fighter champ;
  for (int t = 0; t < TEST; t++){
    /*printf("\n");
    printf("\n");*/
    //printf("\nTTTTT%dTTTTT", t);
    /*printf("\n");
    printf("\n");*/
       
    //printf("\n\nFirst fithters\n\n");
    //showFighters(buf, nBodies);
   
    while(nBodies > 2){

      fight<<<numberOfBlocks, threadsPerBlock>>>(buf, nBodies); 

      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

      cudaError_t asyncErr = cudaDeviceSynchronize();
      if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));  

      //showFighters(buf, nBodies);
      selectFighters(buf, nBodies);
      nBodies = (int)(nBodies / 2) ;
      nBodies += (nBodies % 2);
      if ( t < TEST - 1 ){
        //multiplyFighters(buf, nBodies);
      }
      else{
        //showFighters(buf, nBodies);
      }
    }
    
    champ = buf[chooseWinner(buf, 0)];
    if(t == 0 || t == TEST - 1 || true){
      //printf("\n***Champion round***");
      //printFighter(champ);
      printFighterExel(champ);
      //printf("\n*****");
    }

    nBodies = nMaxBodies;
    multiplyFighters(buf, champ, nBodies);
  }
  //cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);

  cudaFree(buf);
}
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUGTITLE false
#define DEBUGVALUE false

#define MAX 1000
#define MIN 1

#define MAXADJUST 12
#define MINADJUST 0

#define AMOUNTINTERACTION 100
#define AMOUNTTESTS 100
#define POW 10

#define MAXBODIES 2<<POW

typedef struct { 
  int id, generation,
  life, actualLife, 
  strength, 
  speed, actualSpeed, 
  cDamage, 
  rate;   
} Fighter;

static Fighter mainFighter;
static Fighter fighters[MAXBODIES];

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

void CreateFighters(int n) {
  for (int i = 0; i < n; i++) {
    fighters[i].id = i;
    fighters[i].generation = 0;

    fighters[i].life = GetRandom(MIN, MAX);
    fighters[i].actualLife = fighters[i].life;

    fighters[i].strength =  GetRandom(MIN, MAX);

    fighters[i].speed =  GetRandom(MIN, MAX);
    fighters[i].actualSpeed = fighters[i].speed;

    fighters[i].cDamage =  GetRandom(MIN, MAX);
  }
}

void SetupMainFighter(){  
  mainFighter.id = -1;
  mainFighter.generation = 0;
  mainFighter.life = GetRandom(MIN, MAX);
  mainFighter.strength = GetRandom(MIN, MAX); 
  mainFighter.speed = GetRandom(MIN, MAX);
  mainFighter.cDamage = GetRandom(MIN, MAX);

  mainFighter.actualLife = mainFighter.life;  
  mainFighter.actualSpeed = mainFighter.speed;
}

int get_damage(Fighter atk, Fighter target){
  int str = atk.strength;
  int atkSpeed = max(atk.actualSpeed, 1);
  int targetSpeed = max(target.actualSpeed, 1);

  int damage = (int)(str * ((float)atkSpeed / targetSpeed));
  return damage;
}

int get_corruption(Fighter atk, Fighter target){
  int cDam = atk.cDamage * 0.01f;
  int atkLife = max(atk.actualLife, 1);
  int targetLife = max(target.actualLife, 1);

  int damage = (int)(cDam * ((float)atkLife / targetLife));
  return damage;
}

void Fight(int index){
  int firstDamage = get_corruption(mainFighter, fighters[index]);
  int secondDamage = get_corruption(fighters[index], mainFighter);

  fighters[index].actualLife -=  get_damage(mainFighter, fighters[index]);
  mainFighter.actualLife -= get_damage(fighters[index], mainFighter);

  fighters[index].actualSpeed -= firstDamage;
  mainFighter.actualSpeed -= secondDamage;
}

void Simulation(int n) {
    
    int firstDamage;
    int secondDamage;

    #pragma acc kernels
    for(int i = 0; i < n; i ++)
    {   
      int k = 0;
     while(k < AMOUNTINTERACTION & mainFighter.actualLife > 0  & fighters[i].actualLife > 0){
        k++;
        Fight(i);
      }
      k = 0;
      fighters[i].rate = abs(mainFighter.actualLife - fighters[i].actualLife); 
      mainFighter.actualLife = mainFighter.life;   
      mainFighter.actualSpeed = mainFighter.speed;   
    }
}

int chooseWinner(Fighter *f, int index){
  int first = abs(f[index].rate) ;
  int second = abs(f[index + 1].rate) ;
  #if DEBUGVALUE
    printf("\nfirst: %d <> second %d: ", first, second);
  #endif
  if(first < second){    
    #if DEBUGVALUE
      printf("\nchosen: %d id = %d", first, f[index].id);
    #endif
    return index;
  }
  else if(first > second){
    #if DEBUGVALUE
      printf("\nchosen: %d id = %d", second, f[index + 1].id);
    #endif
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

void selectFighters(int n) {
  n /= 2;
  #if DEBUGTITLE
    printf("\nSelecting");
  #endif
  int aux = 0;
  int index;
  int start = 0;
  if(n % 2 == 1){
    start = 2;    
    fighters[aux] = fighters[0];
    fighters[aux].actualLife =  fighters[aux].life;
    fighters[aux].actualSpeed =  fighters[aux].speed;
    aux++;
    fighters[aux] = fighters[1];
    fighters[aux].actualLife =  fighters[aux].life;
    fighters[aux].actualSpeed =  fighters[aux].speed;
    aux++;
  }
  
 
  for (int i = start; i < n; i++) {
    index = i * 2;
    fighters[aux] = fighters[chooseWinner(fighters, index)];
    #if DEBUGVALUE
      printf("\nindex = %d id = %d", aux, fighters[aux].id);
    #endif
    
    fighters[aux].actualLife =  fighters[aux].life;
    fighters[aux].actualSpeed =  fighters[aux].speed;
    #if DEBUGVALUE > 1
      printFighter(fighters[aux]);
    #endif
    aux++;
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

void Reproduce(Fighter father, int n) {
  fighters[0] = copyFighter(father);
  for (int i = 1; i < n; i++) {
    fighters[i].generation = father.generation + 1;

    fighters[i].life = MaxMin(father.life +  GetRandomNeg(), 0);
    fighters[i].actualLife = fighters[i].life;

    fighters[i].strength = MaxMin(father.strength +  GetRandomNeg(), 0);

    fighters[i].speed = MaxMin(father.speed +  GetRandomNeg(), 0);
    fighters[i].actualSpeed = fighters[i].speed;

    fighters[i].cDamage = MaxMin(father.cDamage +  GetRandomNeg(), 0);
  }
}

int main() {
  int nBodies = MAXBODIES;

  CreateFighters(nBodies);
  Fighter champ;

  SetupMainFighter();

  
  printf("\nMIN;MAX;MIN;MINADJUST;MAXADJUST;AMOUNTINTERACTION;AMOUNTTESTS;MaxBodies");
  printf("\n%d;%d;%d;%d;%d;%d;%d;%d", 
  MIN,MAX,MIN,MINADJUST,MAXADJUST,AMOUNTINTERACTION,AMOUNTTESTS,MAXBODIES);
  printf("\nid;generation;life;strength;speed;cDamage;rate");
  
  for (int t = 0; t < AMOUNTTESTS; t++){
   
    while(nBodies > 2){
      #if DEBUGTITLE
        printf("\n###nBodies: %d###", nBodies);
      #endif
      Simulation(nBodies); 

      selectFighters(nBodies);
      nBodies = (int)(nBodies / 2) ;
      nBodies += (nBodies % 2);
      #if DEBUGTITLE
        printf("\n#####");
      #endif
    }
    
    champ = fighters[chooseWinner(fighters, 0)];   
    

    
    printFighterExport(champ);
    //printFighter(champ);
    

    if(champ.rate == 0){
      printf("\nEarly quit at %d/%d", t, AMOUNTTESTS);
      break;
    }

    nBodies = MAXBODIES;
    Reproduce(champ, nBodies);
  }

  printFighter(champ);
  printFighter(mainFighter);
  
  int k = 0;
  printf("\nMain.life: %d <> Champ.life: %d", mainFighter.actualLife, champ.actualLife);
  while(k < AMOUNTINTERACTION & mainFighter.actualLife > 0  & champ.actualLife > 0){
    k++;
    int firstDamage = get_corruption(mainFighter,  champ);
    int secondDamage = get_corruption( champ, mainFighter);

    champ.actualLife -=  get_damage(mainFighter,  champ);
    mainFighter.actualLife -= get_damage( champ, mainFighter);

    champ.actualSpeed -= firstDamage;
    mainFighter.actualSpeed -= secondDamage;
    printf("\nMain.life: %d <> Champ.life: %d", mainFighter.actualLife, champ.actualLife);
  }
}
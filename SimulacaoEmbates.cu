#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 100
#define MIN 50

typedef struct { 
  float life, actualLife, 
  strength, 
  speed, actualSpeed, 
  cDamage;   
} Fighter;

float get_random(){
  return ((float)rand()/RAND_MAX) * (MAX - MIN) + MIN;
}

void randomizeFighters(Fighter *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].life = get_random();
    data[i].actualLife = data[i].life;
    data[i].strength = get_random();

    data[i].speed = get_random();
    data[i].actualSpeed = data[i].speed;
    data[i].cDamage = get_random();

    //printf("life: %f\nstrength: %f\nspeed: %f\ncDamage: %f", data[i].life, data[i].strength,
    //data[i].speed, data[i].cDamage);
  }
}

/*
__global__
void bodyForce(Body *p, float dt, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride)
    {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
        
        //printf("%f",p[i].x);
        //printf("%f",p[i].y);
        //printf("%f",p[i].z);
        
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx; 
        p[i].vy += dt*Fy; 
        p[i].vz += dt*Fz; 
        
        //printf("%f", Fx);
        //printf("%f", Fy);
        //printf("%f", Fz);
    }
}
*/
int main(const int argc, const char** argv) {

  int nBodies = 1;//2<<11;
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

  randomizeFighters(buf,nBodies);
        
  /*bodyForce<<<numberOfBlocks, threadsPerBlock>>>((Body*)buf, dt, nBodies); 

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

  cudaError_t asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));   
  */
  
  cudaFree(buf);
}

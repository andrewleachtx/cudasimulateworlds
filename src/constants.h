#ifndef CONSTANTS_H
#define CONSTANTS_H

#define NUM_PARTICLES 64

// Hyperparameters //
#define DT_SIMULATION (1.0f / 60.0f)
#define FLOAT_EPS 1e-8f
#define GRAVITY -9.8f
#define AIR_FRICTION 0.0f
#define FRICTION 0.1f
#define RESTITUTION 0.85f
#define BENCHMARK true
#define STOP_VELOCITY 2.0f
#define MAX_SIMULATE_TIME_SECONDS 2000
#define STOP_PLANE_DIST 4.0f
#define WORLD_LOG_STEPSIZE 250
#define MAX_STEPS 10000

#endif // CONSTANTS_H
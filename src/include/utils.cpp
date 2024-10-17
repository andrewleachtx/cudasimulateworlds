#include "utils.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include "../include.h"

#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <random>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using std::cout, std::endl, std::cerr;
using std::shared_ptr, std::make_shared;
using std::vector, std::string;

float randFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
};

glm::vec3 randXYZ() {
    return glm::vec3(randFloat(), randFloat(), randFloat());
}

const float FLOAT_EPS = 1e-8f;
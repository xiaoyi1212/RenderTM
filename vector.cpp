#include "vector.h"
#include <cmath>

vector* vector::add(const vector* other)
{
    this->x += other->x;
    this->y += other->y;
    this->z += other->z;
    return this;
}

vector* vector::subtract(const vector* other)
{
    this->x -= other->x;
    this->y -= other->y;
    this->z -= other->z;
    return this;
}

vector* vector::multiply(const double m)
{
    this->x *= m;
    this->y *= m;
    this->z *= m;
    return this;
}

vector* vector::divide(const double m)
{
    if (m == 0) return nullptr;
    this->x /= m;
    this->y /= m;
    this->z /= m;
    return this;
}

double vector::length() const
{
    return sqrt(x * x + y * y + z * z);
}

vector* vector::normalize()
{
    if (const double len = length(); len != 0) divide(len);
    return this;
}

double vector::dot(const vector* other) const
{
    return this->x * other->x + this->y * other->y + this->z * other->z;
}

vector* vector::cross(const vector* other) const
{
    const double newX = this->y * other->z - this->z * other->y;
    const double newY = this->z * other->x - this->x * other->z;
    const double newZ = this->x * other->y - this->y * other->x;
    return new vector(newX, newY, newZ);
}

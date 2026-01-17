#pragma once

class vector
{
    double x, y, z;

public:
    [[nodiscard]] double length() const;
    vector* normalize();
    double dot(const vector* other) const;
    vector* cross(const vector* other) const;
    vector* multiply(double m);
    vector* divide(double m);
    vector* subtract(const vector* other);
    vector* add(const vector* other);

    vector(const double new_x, const double new_y, const double new_z)
    {
        this->x = new_x;
        this->y = new_y;
        this->z = new_z;
    }
};

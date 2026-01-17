#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>

#include "vector.h"

static double x_of(const vector& v)
{
    vector basis(1.0, 0.0, 0.0);
    return v.dot(&basis);
}

static double y_of(const vector& v)
{
    vector basis(0.0, 1.0, 0.0);
    return v.dot(&basis);
}

static double z_of(const vector& v)
{
    vector basis(0.0, 0.0, 1.0);
    return v.dot(&basis);
}

TEST_CASE("vector length and normalization")
{
    vector v(3.0, 0.0, 4.0);
    REQUIRE(v.length() == Catch::Approx(5.0));

    v.normalize();
    REQUIRE(v.length() == Catch::Approx(1.0));

    vector zero(0.0, 0.0, 0.0);
    zero.normalize();
    REQUIRE(zero.length() == Catch::Approx(0.0));
}

TEST_CASE("vector add/subtract/multiply/divide")
{
    vector v(1.0, 2.0, 3.0);
    vector u(1.0, 1.0, 1.0);

    v.add(&u);
    REQUIRE(x_of(v) == Catch::Approx(2.0));
    REQUIRE(y_of(v) == Catch::Approx(3.0));
    REQUIRE(z_of(v) == Catch::Approx(4.0));

    v.subtract(&u);
    REQUIRE(x_of(v) == Catch::Approx(1.0));
    REQUIRE(y_of(v) == Catch::Approx(2.0));
    REQUIRE(z_of(v) == Catch::Approx(3.0));

    v.multiply(2.0);
    REQUIRE(x_of(v) == Catch::Approx(2.0));
    REQUIRE(y_of(v) == Catch::Approx(4.0));
    REQUIRE(z_of(v) == Catch::Approx(6.0));

    vector* divided = v.divide(2.0);
    REQUIRE(divided != nullptr);
    REQUIRE(x_of(v) == Catch::Approx(1.0));
    REQUIRE(y_of(v) == Catch::Approx(2.0));
    REQUIRE(z_of(v) == Catch::Approx(3.0));

    vector v2(1.0, 2.0, 3.0);
    vector* bad = v2.divide(0.0);
    REQUIRE(bad == nullptr);
    REQUIRE(x_of(v2) == Catch::Approx(1.0));
    REQUIRE(y_of(v2) == Catch::Approx(2.0));
    REQUIRE(z_of(v2) == Catch::Approx(3.0));
}

TEST_CASE("vector dot and cross")
{
    vector a(1.0, 0.0, 0.0);
    vector b(0.0, 1.0, 0.0);

    REQUIRE(a.dot(&b) == Catch::Approx(0.0));

    std::unique_ptr<vector> c(a.cross(&b));
    REQUIRE(c);
    REQUIRE(x_of(*c) == Catch::Approx(0.0));
    REQUIRE(y_of(*c) == Catch::Approx(0.0));
    REQUIRE(z_of(*c) == Catch::Approx(1.0));

    REQUIRE(a.dot(c.get()) == Catch::Approx(0.0));
    REQUIRE(b.dot(c.get()) == Catch::Approx(0.0));
}

TEST_CASE("vector operations return self for chaining")
{
    vector v(1.0, 2.0, 3.0);
    vector u(1.0, 1.0, 1.0);

    REQUIRE(v.add(&u) == &v);
    REQUIRE(v.subtract(&u) == &v);
    REQUIRE(v.multiply(2.0) == &v);
    REQUIRE(v.divide(2.0) == &v);

    vector zero(0.0, 0.0, 0.0);
    REQUIRE(zero.normalize() == &zero);
}

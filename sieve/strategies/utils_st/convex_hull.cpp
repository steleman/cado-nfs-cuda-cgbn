#include "cado.h" // IWYU pragma: keep

#include <algorithm>
#include <tuple>

#include "convex_hull.hpp"

/* Which way the path o -> a -> b turns: positive counter-clockwise,
 * negative clockwise, zero when the three points are collinear.
 *
 * This is the only geometric decision the hull makes, and it is one
 * subtraction and one multiplication per coordinate. In particular it
 * needs no scaling of x against y: multiplying every y by some s > 0
 * multiplies the result by s and cannot change its sign, so the hull of
 * a set of (probability, time) points does not depend on how the two
 * are made commensurate. */
static double turn(point const & o, point const & a, point const & b)
{
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

tabular_point convex_hull(tabular_point const & t)
{
    /* the convex hull of the empty set is the empty set */
    if (t.empty())
        return {};

    /* Andrew's monotone chain, lower half only. */
    tabular_point sorted = t;
    std::ranges::sort(sorted, [](point const & a, point const & b) {
        /* by x, then by y, then by index, so that the result does not
         * depend on the order the points were collected in */
        return std::forward_as_tuple(a.x, a.y, a.number) <
               std::forward_as_tuple(b.x, b.y, b.number);
    });

    /* Two points whose x agree to within this are the same trade-off.
     * The tolerance is not cosmetic: the same strategies reach this
     * function both in memory and read back from a file that stores six
     * decimals, and mirror-image strategies cost the same to the last
     * bit or two, so without it the two routes disagree on which points
     * exist at all. It is inherited, along with its value. */
    constexpr double precision = 1e-10;

    tabular_point hull;
    for (auto const & p: sorted) {
        /* Of several points sharing an x, only the lowest can be on the
         * lower hull, and it is the one we have just seen. */
        if (!hull.empty() && p.x - hull.back().x < precision)
            continue;
        /* Drop the last point of the chain while it does not turn
         * strictly upwards, which leaves the vertices of the hull and
         * nothing else. A point sitting exactly on an edge is dropped
         * rather than kept: whether it is on the edge or a hair above it
         * is decided by the last bits of a cross product, so keeping it
         * would make the answer depend on rounding that the callers
         * cannot control -- the same strategies reach this code by two
         * routes, one of which has been through a file that stores six
         * decimals. */
        while (hull.size() >= 2 && turn(hull[hull.size() - 2], hull.back(), p) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }

    return hull;
}

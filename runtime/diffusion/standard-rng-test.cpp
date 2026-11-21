#include "core/rng.hpp"

#include <cmath>
#include <cstdio>

int main() {
    STDDefaultRNG first, same, different;
    first.manual_seed(43);
    same.manual_seed(43);
    different.manual_seed(44);
    const auto noise = first.randn(4096);
    if (noise.size() != 4096 || noise != same.randn(4096) || noise == different.randn(4096)) return 1;
    for (auto x : noise) if (!std::isfinite(x)) return 1;
    const auto next = first.randn(4096);
    if (next == noise || next != same.randn(4096)) return 1;
    first.manual_seed(43);
    if (first.randn(4096) != noise) return 1;
    std::puts("Standard RNG seed, continuation and reseed checks passed");
}

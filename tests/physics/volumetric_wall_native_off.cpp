// Task 19 P1: the ordinary path is unchanged by the VOLUMETRIC_WALL define when no wall is attached. The same source is
// built twice (EXTENSIONS none, and VOLUMETRIC_WALL): a force-free periodic box with a Taylor-Green initial velocity
// runs VW_N steps and dumps rho and u (the fields every build exposes) to VW_OUT; the research side compares the two
// dumps bitwise. No wall, no attachment, no physics claim.
#include "lbm.hpp"
#include "physics_check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string env(const char* name) {
#ifdef _MSC_VER
    char* value = nullptr;
    size_t length = 0u;
    std::string result{};
    if(_dupenv_s(&value, &length, name)==0 && value!=nullptr) { result = value; free(value); }
    return result;
#else
    const char* value = std::getenv(name);
    return value!=nullptr ? std::string(value) : std::string();
#endif
}

template<class T> void dump(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    if(!out) throw std::runtime_error("cannot write "+path);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()*sizeof(T)));
}

}  // namespace

void main_setup() { // VW_OUT folder, VW_N steps (default 100); 32x32x32 periodic box, nu = 0.02, Taylor-Green u0 = 0.02
    const std::string out{ env("VW_OUT") };
    const std::uint32_t steps{ env("VW_N").empty() ? 100u : static_cast<std::uint32_t>(std::stoul(env("VW_N"))) };
    if(out.empty()) { std::printf("usage: VW_OUT=DIR [VW_N=steps] physics_volumetric_wall_native_off\n"); std::exit(2); }
    const uint L{ 32u };
#ifdef VOLUME_FORCE
    LBM lbm(L, L, L, 0.02f, 1.0e-6f, -2.0e-6f, 3.0e-6f);   // task 43: the same test with a uniform force (the clamp flag of the wall define reads the forced velocity)
#else
    LBM lbm(L, L, L, 0.02f);
#endif
    const float u0{ 0.02f }, k{ 2.0f*3.14159265f/static_cast<float>(L) };
    for(ulong n=0ull; n<lbm.get_N(); n++) {
        uint x=0u, y=0u, z=0u;
        lbm.coordinates(n, x, y, z);
        const float fx{ static_cast<float>(x)+0.5f }, fy{ static_cast<float>(y)+0.5f }, fz{ static_cast<float>(z)+0.5f };
        lbm.rho[n] = 1.0f;
        lbm.u.x[n] = u0*std::sin(k*fx)*std::cos(k*fy)*std::cos(k*fz);
        lbm.u.y[n] = -u0*std::cos(k*fx)*std::sin(k*fy)*std::cos(k*fz);
        lbm.u.z[n] = 0.0f;
        lbm.flags[n] = 0u;
    }
    lbm.run(0u);
    lbm.run(static_cast<ulong>(steps));
    lbm.rho.read_from_device();
    lbm.u.read_from_device();
    std::vector<float> rho(static_cast<size_t>(lbm.get_N())), u(3u*static_cast<size_t>(lbm.get_N()));
    for(ulong n=0ull; n<lbm.get_N(); n++) { rho[static_cast<size_t>(n)] = lbm.rho[n]; u[static_cast<size_t>(n)] = lbm.u.x[n]; u[static_cast<size_t>(lbm.get_N()+n)] = lbm.u.y[n]; u[static_cast<size_t>(2ull*lbm.get_N()+n)] = lbm.u.z[n]; }
    dump(out+"/rho.f32", rho);
    dump(out+"/u.f32", u);
    std::vector<fpxx> fi{};
    lbm.read_fi_raw(fi);   // #286 item 6: the raw populations, not only the moments
    dump(out+"/fi.f32", fi);
    std::ofstream meta(out+"/meta.txt");
#ifdef VOLUMETRIC_WALL
    meta << "define VOLUMETRIC_WALL on (no attachment)\n";
#else
    meta << "define off\n";
#endif
#ifdef VOLUME_FORCE
    meta << "VOLUME_FORCE 1e-6 -2e-6 3e-6\n";
#endif
    meta << "steps " << steps << "\nt " << lbm.get_t() << "\n";
    PhysicsCheck::report("volumetric_wall_native_off_run", 0.0, 0.5);
    std::fflush(stdout);
    std::exit(0);
}

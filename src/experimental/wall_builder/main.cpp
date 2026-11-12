/// @file main.cpp
/// @brief volumetric_wall_builder: the terrain container of the volumetric slip wall (scripts/slip_wall/terrain_container.py),
/// compiled and on all cores. Every numeric section is the same, byte for byte, as the Python builder's; the provenance
/// text names this tool where the Python one names its exporter and library versions.
///
/// Usage: volumetric_wall_builder --heights CSV --z-top Z --nu NU --friction K --out FILE
///            [--velocities D3Q27|D3Q19] [--name NAME] [--rough-z0 Z0] [--rough-K0 K0] [--allow-steep]
///            [--initial-velocity ux,uy,uz] [--threads N]
/// Exit codes as terrain_container.py: 0 written, 2 refused input, 3 map identities violated (FILE.rejected.json), 1 error.

#include "container_writer.hpp"
#include "dense_algebra.hpp"
#include "maps.hpp"
#include "pass_runner.hpp"
#include "python_text.hpp"
#include "sampler.hpp"
#include "sha256.hpp"
#include "terrain.hpp"
#include "terrain_input.hpp"
#include "wall_tables.hpp"

#define WIN32_LEAN_AND_MEAN // NOLINT(cppcoreguidelines-macro-usage): the documented switch that trims windows.h
#include <windows.h>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace fluidx3d::experimental::wall_builder;

/// @brief A refused input: exit code 2 with "REFUSED: ...", as terrain_container.py.
class Refused : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// @brief The command line.
struct Arguments {
    std::string heights{};
    int z_top{};
    double nu{};
    double friction{};
    std::string out{};
    std::string velocities{ "D3Q27" };
    std::string name{ "terrain" };
    std::optional<double> rough_z0{};
    double rough_k0{};
    bool allow_steep{};
    std::optional<std::array<double, 3>> initial_velocity{};
    int threads{};
};

/// @brief Parses the command line.
class CommandLine {
public:
    /// @throws std::invalid_argument for an unknown or incomplete option
    /// @throws Refused for an option value terrain_container.py refuses
    [[nodiscard]] static Arguments parse(const int argc, char** argv)
    {
        std::map<std::string, std::string> values{};
        Arguments a{};
        a.threads = static_cast<int>(std::thread::hardware_concurrency());
        for (int n{ 1 }; n < argc; ++n) {
            const std::string flag{ argv[n] };
            if (flag == "--allow-steep") {
                a.allow_steep = true;
                continue;
            }
            if (flag.rfind("--", 0) != 0 || n + 1 >= argc) {
                throw std::invalid_argument("bad argument " + flag + "; see the usage in src/experimental/wall_builder/main.cpp");
            }
            values[flag] = argv[++n];
        }
        for (const char* required : { "--heights", "--z-top", "--nu", "--friction", "--out" }) {
            if (values.count(required) == 0) {
                throw std::invalid_argument(std::string{ "missing " } + required);
            }
        }
        a.heights = values["--heights"];
        a.z_top = std::stoi(values["--z-top"]);
        a.nu = number(values["--nu"]);
        a.friction = number(values["--friction"]);
        a.out = values["--out"];
        a.velocities = values.count("--velocities") ? values["--velocities"] : a.velocities;
        a.name = values.count("--name") ? values["--name"] : a.name;
        a.rough_k0 = values.count("--rough-K0") ? number(values["--rough-K0"]) : 0.0;
        a.threads = values.count("--threads") ? std::stoi(values["--threads"]) : a.threads;
        if (values.count("--rough-z0")) {
            a.rough_z0 = number(values["--rough-z0"]);
        } else if (a.rough_k0 != 0.0) {
            throw Refused("--rough-K0 needs --rough-z0");
        }
        if (values.count("--initial-velocity")) {
            try {
                a.initial_velocity = ContainerWriter::parse_velocity(values["--initial-velocity"]);
            } catch (const std::invalid_argument& error) {
                throw Refused(error.what());
            }
        }
        if (a.threads < 1) {
            throw std::invalid_argument("--threads must be 1 or more");
        }
        return a;
    }

private:
    /// @brief Python's float(text). @throws std::invalid_argument when it is not a number
    [[nodiscard]] static double number(const std::string& text)
    {
        char* end{ nullptr };
        const double value{ std::strtod(text.c_str(), &end) };
        if (text.empty() || *end != '\0') {
            throw std::invalid_argument("'" + text + "' is not a number");
        }
        return value;
    }
};

/// @brief Keeps Windows from power-throttling the tool. A process started without a foreground window gets EcoQoS on
/// Windows 11, and on a hybrid CPU the scheduler then keeps it on the efficiency cores (measured: 8 of 24 cores).
class FullSpeed {
public:
    /// @throws std::runtime_error when Windows refuses the setting
    static void request()
    {
        PROCESS_POWER_THROTTLING_STATE state{};
        state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        state.StateMask = 0; // the control bit set and the state bit clear: never throttle the execution speed
        const BOOL done{ SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof state) };
        if (done == 0) {
            throw std::runtime_error("SetProcessInformation(ProcessPowerThrottling) failed, error " + std::to_string(GetLastError()));
        }
    }
};

/// @brief Wall-clock seconds of the stages, printed and kept for the report.
class StageClock {
public:
    /// @brief Ends a stage: prints its seconds and keeps them under the name.
    void lap(const std::string& name)
    {
        const auto now{ std::chrono::steady_clock::now() };
        const double seconds{ std::chrono::duration<double>(now - last_).count() };
        last_ = now;
        stages.push_back(Stage{ name, seconds });
        std::cout << "  " << name << " " << seconds << " s" << std::endl;
    }

    /// @brief Seconds since the start.
    [[nodiscard]] double total() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count(); }

    /// @brief One stage.
    struct Stage {
        std::string name{};
        double seconds{};
    };
    std::vector<Stage> stages{}; ///< The stages in order.

private:
    std::chrono::steady_clock::time_point start_{ std::chrono::steady_clock::now() };
    std::chrono::steady_clock::time_point last_{ start_ };
};

/// @brief Writes a whole file. @throws std::runtime_error when it cannot be written
void write_text(const std::string& path, const std::string& text)
{
    std::ofstream file{ path, std::ios::binary };
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
}

/// @brief Writes a file as the concatenation of chunks. @throws std::runtime_error when it cannot be written
void write_file(const std::string& path, const std::vector<std::vector<char>>& chunks)
{
    std::ofstream file{ path, std::ios::binary };
    for (const std::vector<char>& chunk : chunks) {
        file.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    }
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
}

/// @brief The identity values as a JSON object.
void identities_json(JsonWriter& json, const IdentityValues& v)
{
    json.begin_object();
    json.key("donor_partition");
    json.value(v.donor_partition);
    json.key("receiver_coverage");
    json.value(v.receiver_coverage);
    json.key("prism_coverage");
    json.value(v.prism_coverage);
    json.key("return_delivery");
    json.value(v.return_delivery);
    json.end_object();
}

/// @brief OUT.rejected.json of NativeIdentityGate (the failing states or rows, largest residual first).
std::string rejection_report(const GateReport& gate, const AssembledMaps& m, const TerrainGeometry& geometry, const VelocityTable& vs, const Arguments& a)
{
    const auto q{ static_cast<std::int64_t>(vs.directions.size()) };
    JsonWriter json{ true };
    json.begin_object();
    json.key("tolerance");
    json.value(GateReport::TOLERANCE);
    json.key("definitions");
    json.value("native (src/experimental/volumetric_wall_native.cpp)");
    json.key("identities");
    identities_json(json, gate.worst);
    json.key("failing");
    json.begin_object();
    for (const GateFailure& f : gate.failing) {
        json.key(f.name);
        json.begin_list();
        for (std::size_t n{ 0 }; n < f.items.size(); ++n) {
            json.begin_object();
            const std::int64_t item{ f.items[n] };
            if (f.per_state) {
                const auto cell{ static_cast<std::size_t>(item / q) };
                const Direction& c{ vs.directions[static_cast<std::size_t>(item % q)] };
                const std::array<int, 3>& k{ geometry.cells()[cell].ijk };
                json.key("state");
                json.value(item);
                json.key("cell");
                json.begin_list();
                json.value(k[0]);
                json.value(k[1]);
                json.value(k[2]);
                json.end_list();
                json.key("c");
                json.begin_list();
                json.value(c.x);
                json.value(c.y);
                json.value(c.z);
                json.end_list();
                json.key("volume");
                json.value(m.volume[cell]);
            } else {
                const WallRow& row{ m.rows[static_cast<std::size_t>(item)] };
                const Direction& c{ vs.directions[static_cast<std::size_t>(row.direction)] };
                json.key("row");
                json.value(item);
                json.key("facet");
                json.value(geometry.facets()[static_cast<std::size_t>(row.facet)].key.python_text());
                json.key("c");
                json.begin_list();
                json.value(c.x);
                json.value(c.y);
                json.value(c.z);
                json.end_list();
                json.key("vi");
                json.value(m.vi[static_cast<std::size_t>(item)]);
            }
            json.key("residual");
            json.value(f.residuals[n]);
            json.end_object();
        }
        json.end_list();
    }
    json.end_object();
    json.key("finite");
    json.value(gate.finite);
    json.key("passed");
    json.value(gate.passed);
    json.key("container");
    json.value("null: no container written");
    json.key("requested_output");
    json.value(a.out);
    json.end_object();
    return json.text() + "\n";
}

/// @brief OUT.build.json.
std::string build_report(const std::string& sha, const std::size_t bytes, const Arguments& a, const PermutedMaps& p, const TerrainGeometry& geometry,
                         const IdentityValues& identities, const GateReport& gate, const WallList& wall, const StageClock& clock)
{
    JsonWriter json{ true };
    json.begin_object();
    json.key("container");
    json.value(a.out);
    json.key("sha256");
    json.value(sha);
    json.key("bytes");
    json.value(static_cast<std::int64_t>(bytes));
    json.key("velocity_set");
    json.value(a.velocities);
    json.key("allow_steep");
    json.value(a.allow_steep);
    json.key("domain_checks");
    json.value(a.allow_steep ? "R10, R1 (R11 slope check skipped: --allow-steep)" : "R10, R11, R1");
    json.key("cells");
    json.value(static_cast<std::int64_t>(geometry.cells().size()));
    json.key("facets");
    json.value(static_cast<std::int64_t>(geometry.facets().size()));
    json.key("rows");
    json.value(static_cast<std::int64_t>(p.vi.size()));
    json.key("W_nnz");
    json.value(static_cast<std::int64_t>(p.w.values.size()));
    json.key("identities");
    identities_json(json, identities);
    json.key("native_identities");
    identities_json(json, gate.worst);
    json.key("seconds");
    json.begin_object();
    for (const StageClock::Stage& stage : clock.stages) {
        json.key(stage.name);
        json.value(stage.seconds);
    }
    json.key("total");
    json.value(clock.total());
    json.end_object();
    json.key("threads");
    json.value(a.threads);
    json.key("wall_list_cells_added_by_rule_253");
    json.value(static_cast<std::int64_t>(wall.added_253.size()));
    json.key("builder");
    json.value("volumetric_wall_builder");
    json.end_object();
    return json.text() + "\n";
}

/// @brief The whole build. @return the exit code
int build(const Arguments& a)
{
    if (std::filesystem::exists(a.out)) {
        throw Refused(a.out + " exists");
    }
    std::optional<RoughLaw> law{};
    if (a.rough_z0) {
        try {
            law.emplace(*a.rough_z0, a.rough_k0);
        } catch (const std::invalid_argument& error) {
            throw Refused(error.what());
        }
    }
    StageClock clock{};
    const VelocityTable velocities{ VelocityTable::named(a.velocities) };
    const TerrainInput input{ TerrainInput::from_csv(a.heights, a.z_top, a.name) };
    input.domain_checks(a.allow_steep);
    const TerrainGeometry geometry{ input.record(velocities, 12) };
    std::cout << "cells " << geometry.cells().size() << ", facets " << geometry.facets().size() << "; passes on " << a.threads << " threads ..." << std::endl;
    clock.lap("geometry");
    PassRunner passes{ geometry, a.threads };
    passes.run();
    clock.lap("passes");
    SamplerSearch sampler{ geometry, 12 };
    sampler.run(a.threads);
    sampler.check();
    clock.lap("sampler");
    const MapAssembly assembly{ geometry, velocities, passes, sampler };
    const AssembledMaps maps{ assembly.assemble() };
    const MapProducts products{ assembly.products(maps) };
    const IdentityValues identities{ assembly.identities(maps, products) };
    const PermutedMaps permuted{ assembly.permute(maps) };
    clock.lap("maps");
    const WallTables tables{ geometry, velocities, assembly };
    const ArmTables arms{ tables.arms(maps, permuted) };
    const WallList wall{ tables.wall_list(permuted, arms) };
    const GradientMap gradient{ tables.gradient(wall.cells, a.threads) };
    clock.lap("arms_and_gradient");
    const GateReport gate{ assembly.gate(maps, products) };
    std::filesystem::path out_path{ a.out };
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    if (!gate.passed) {
        write_text(a.out + ".rejected.json", rejection_report(gate, maps, geometry, velocities, a));
        std::cout << "REFUSED: map identities violated (native definitions, tolerance 1e-10); no container written" << std::endl;
        std::cout << "diagnostic report: " << a.out << ".rejected.json" << std::endl;
        return 3;
    }
    ContainerOptions options{};
    options.nu = a.nu;
    options.friction = a.friction;
    options.rough_z0 = a.rough_z0;
    options.rough_k0 = a.rough_k0;
    options.initial_velocity = a.initial_velocity;
    options.allow_steep = a.allow_steep;
    ContainerParts parts{};
    parts.input = &input;
    parts.velocities = &velocities;
    parts.geometry = &geometry;
    parts.passes = &passes;
    parts.sampler = &sampler;
    parts.maps = &maps;
    parts.permuted = &permuted;
    parts.identities = identities;
    parts.arms = &arms;
    parts.wall = &wall;
    parts.gradient = &gradient;
    const ContainerWriter writer{ parts, options };
    const std::vector<std::vector<char>> chunks{ writer.chunks() };
    std::size_t size{ 0 };
    for (const std::vector<char>& chunk : chunks) {
        size += chunk.size();
    }
    // the digest on a second thread while this one writes the file (both only read the chunks)
    std::string sha{};
    std::exception_ptr hash_error{};
    std::thread hasher{ [&]() {
        try {
            sha = Sha256::of(chunks);
        } catch (...) { // a thread boundary: rethrown below
            hash_error = std::current_exception();
        }
    } };
    try {
        write_file(a.out, chunks);
    } catch (...) { // join before the error leaves this scope
        hasher.join();
        throw;
    }
    hasher.join();
    if (hash_error) {
        std::rethrow_exception(hash_error);
    }
    clock.lap("write");
    write_text(a.out + ".build.json", build_report(sha, size, a, permuted, geometry, identities, gate, wall, clock));
    std::cout << "sha256 " << sha << "\nbytes " << size << "\ncells " << geometry.cells().size() << ", facets " << geometry.facets().size()
              << ", rows " << permuted.vi.size() << ", W_nnz " << permuted.w.values.size() << "\ntotal " << clock.total() << " s" << std::endl;
    return 0;
}

} // namespace

int main(const int argc, char** argv)
{
    try {
        const Arguments arguments{ CommandLine::parse(argc, argv) };
        FullSpeed::request();
        DenseAlgebra::single_threaded(); // the tool parallelises over facets and cells; each OpenBLAS call stays on its thread
        return build(arguments);
    } catch (const Refused& error) { // the program boundary: terrain_container.py's refusal
        std::cout << "REFUSED: " << error.what() << std::endl;
        return 2;
    } catch (const std::exception& error) { // the program boundary: report and stop
        std::cerr << "volumetric_wall_builder: " << error.what() << std::endl;
        return 1;
    }
}

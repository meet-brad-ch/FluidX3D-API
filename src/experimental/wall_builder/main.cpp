/// @file main.cpp
/// @brief volumetric_wall_builder: the terrain container of the volumetric slip wall (scripts/slip_wall/terrain_container.py),
/// compiled and on all cores. Every numeric section is the same, byte for byte, as the Python builder's; the provenance
/// text names this tool where the Python one names its exporter and library versions. The sections that grow with the
/// volume are made one slab of x-columns at a time, so the memory stays within --memory-gb (GiB, default 64); D and
/// direct_arm go through temporary files in FILE.sections (next to FILE, or in --temp-dir), removed at the end.
///
/// Usage: volumetric_wall_builder --heights CSV --z-top Z --nu NU --friction K --out FILE
///            [--velocities D3Q27|D3Q19] [--name NAME] [--rough-z0 Z0] [--rough-K0 K0] [--allow-steep]
///            [--initial-velocity ux,uy,uz] [--threads N] [--memory-gb GB] [--temp-dir DIR] [--slab-columns N]
///        volumetric_wall_builder --stl STL --cell-m M [--apron N] [--cache-dir DIR] [--heights-only] (other options as above)
/// --stl samples the terrain STL into node heights (stl_heights.hpp) in place of --heights; the CSV is cached in DIR
/// (default %LOCALAPPDATA%\FluidX3D\terrain_heights) under a key of the STL bytes and the recipe, so the next build with
/// the same STL and recipe reads it. --heights-only writes (or finds) the cached CSV, prints its path and stops.
/// --skip-identity-gate writes the container also when the native identity gate fails (a test of the build and write path:
/// the report still goes to FILE.rejected.json, the build report says "failed", and the engine's check refuses such a file).
/// Exit codes as terrain_container.py: 0 written, 2 refused input, 3 map identities violated (FILE.rejected.json), 1 error.

#include "container_writer.hpp"
#include "dense_algebra.hpp"
#include "maps.hpp"
#include "pass_runner.hpp"
#include "python_text.hpp"
#include "sampler.hpp"
#include "sha256.hpp"
#include "stl_heights.hpp"
#include "terrain.hpp"
#include "terrain_input.hpp"
#include "wall_tables.hpp"

#define WIN32_LEAN_AND_MEAN // NOLINT(cppcoreguidelines-macro-usage): the documented switch that trims windows.h
#include <windows.h>

#include <psapi.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
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
    double memory_gb{ 64.0 };   ///< --memory-gb: the memory budget in GiB.
    std::string temp_dir{};     ///< --temp-dir: the parent of the temporary folder (empty: the folder of --out).
    int slab_columns{};         ///< --slab-columns: a fixed slab width in x-columns (0: from the budget).
    std::string stl{};          ///< --stl: a terrain STL in place of --heights.
    double cell_m{};            ///< --cell-m: the cell size of --stl in metres.
    int apron{ 8 };             ///< --apron: flat columns added at each x end of --stl.
    std::string cache_dir{};    ///< --cache-dir: the heights cache of --stl (empty: %LOCALAPPDATA%\FluidX3D\terrain_heights).
    bool heights_only{};        ///< --heights-only: write the cached heights CSV and stop.
    bool skip_identity_gate{};  ///< --skip-identity-gate: write the container also when the identity gate fails.
};

/// @brief Parses the command line.
class CommandLine {
public:
    /// @throws std::invalid_argument for an unknown or incomplete option
    /// @throws Refused for an option value terrain_container.py refuses
    [[nodiscard]] static Arguments parse(const int argc, char** argv)
    {
        std::map<std::string, std::string> values{ flags(argc, argv) };
        Arguments a{};
        a.allow_steep = values.count("--allow-steep") > 0;
        a.heights_only = values.count("--heights-only") > 0;
        a.skip_identity_gate = values.count("--skip-identity-gate") > 0;
        terrain(values, a);
        if (a.heights_only) {
            return a;
        }
        required(values, a);
        optional(values, a);
        physics(values, a);
        check(a);
        return a;
    }

private:
    /// @brief The options as flag -> value ("--allow-steep" has the value "1"). @throws std::invalid_argument for a bad flag
    [[nodiscard]] static std::map<std::string, std::string> flags(const int argc, char** argv)
    {
        std::map<std::string, std::string> values{};
        for (int n{ 1 }; n < argc; ++n) {
            const std::string flag{ argv[n] };
            if (flag == "--allow-steep" || flag == "--heights-only" || flag == "--skip-identity-gate") {
                values[flag] = "1";
                continue;
            }
            if (flag.rfind("--", 0) != 0 || n + 1 >= argc) {
                throw std::invalid_argument("bad argument " + flag + "; see the usage in src/experimental/wall_builder/main.cpp");
            }
            values[flag] = argv[++n];
        }
        return values;
    }

    /// @brief The terrain: --heights, or --stl with --cell-m (and --apron, --cache-dir).
    /// @throws std::invalid_argument when neither or both are given, or a value is not a number
    static void terrain(std::map<std::string, std::string>& values, Arguments& a)
    {
        const bool has_heights{ values.count("--heights") > 0 };
        const bool has_stl{ values.count("--stl") > 0 };
        if (has_heights == has_stl) {
            throw std::invalid_argument("give one of --heights CSV and --stl STL");
        }
        if (has_heights) {
            if (a.heights_only) {
                throw std::invalid_argument("--heights-only needs --stl");
            }
            a.heights = values["--heights"];
            return;
        }
        if (values.count("--cell-m") == 0) {
            throw std::invalid_argument("--stl needs --cell-m");
        }
        a.stl = values["--stl"];
        a.cell_m = number(values["--cell-m"]);
        if (values.count("--apron") > 0) {
            a.apron = std::stoi(values["--apron"]);
        }
        if (values.count("--cache-dir") > 0) {
            a.cache_dir = values["--cache-dir"];
        }
        a.threads = static_cast<int>(std::thread::hardware_concurrency());
        if (values.count("--threads") > 0) {
            a.threads = std::stoi(values["--threads"]);
        }
    }

    /// @brief The required options. @throws std::invalid_argument when one is missing or not a number
    static void required(std::map<std::string, std::string>& values, Arguments& a)
    {
        for (const char* name : { "--z-top", "--nu", "--friction", "--out" }) {
            if (values.count(name) == 0) {
                throw std::invalid_argument(std::string{ "missing " } + name);
            }
        }
        a.z_top = std::stoi(values["--z-top"]);
        a.nu = number(values["--nu"]);
        a.friction = number(values["--friction"]);
        a.out = values["--out"];
    }

    /// @brief The options of the tool (names, threads, memory, slabs); a missing one keeps its default.
    /// @throws std::invalid_argument for a value that is not a number
    static void optional(std::map<std::string, std::string>& values, Arguments& a)
    {
        a.threads = static_cast<int>(std::thread::hardware_concurrency());
        if (values.count("--velocities") > 0) {
            a.velocities = values["--velocities"];
        }
        if (values.count("--name") > 0) {
            a.name = values["--name"];
        }
        if (values.count("--threads") > 0) {
            a.threads = std::stoi(values["--threads"]);
        }
        if (values.count("--memory-gb") > 0) {
            a.memory_gb = number(values["--memory-gb"]);
        }
        if (values.count("--temp-dir") > 0) {
            a.temp_dir = values["--temp-dir"];
        }
        if (values.count("--slab-columns") > 0) {
            a.slab_columns = std::stoi(values["--slab-columns"]);
        }
    }

    /// @brief The physical options of terrain_container.py. @throws Refused for a value it refuses
    static void physics(std::map<std::string, std::string>& values, Arguments& a)
    {
        if (values.count("--rough-K0") > 0) {
            a.rough_k0 = number(values["--rough-K0"]);
        }
        if (values.count("--rough-z0") > 0) {
            a.rough_z0 = number(values["--rough-z0"]);
        } else if (a.rough_k0 != 0.0) {
            throw Refused("--rough-K0 needs --rough-z0");
        }
        if (values.count("--initial-velocity") == 0) {
            return;
        }
        try {
            a.initial_velocity = ContainerWriter::parse_velocity(values["--initial-velocity"]);
        } catch (const std::invalid_argument& error) {
            throw Refused(error.what());
        }
    }

    /// @brief The ranges of the tool's options. @throws std::invalid_argument for a value out of range
    static void check(const Arguments& a)
    {
        if (a.threads < 1) {
            throw std::invalid_argument("--threads must be 1 or more");
        }
        if (!(a.memory_gb > 0.0)) {
            throw std::invalid_argument("--memory-gb must be above 0");
        }
        if (a.slab_columns < 0) {
            throw std::invalid_argument("--slab-columns must be 0 (from the budget) or more");
        }
    }

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

/// @brief The process memory: the private bytes now and the peak working set.
class ProcessMemory {
public:
    /// @brief The private (committed) bytes of the process now. @throws std::runtime_error when Windows refuses
    [[nodiscard]] static std::size_t private_bytes() { return counters().PrivateUsage; }

    /// @brief The largest working set of the process so far. @throws std::runtime_error when Windows refuses
    [[nodiscard]] static std::size_t peak_working_set() { return counters().PeakWorkingSetSize; }

private:
    [[nodiscard]] static PROCESS_MEMORY_COUNTERS_EX counters()
    {
        PROCESS_MEMORY_COUNTERS_EX values{};
        values.cb = sizeof values;
        const BOOL done{ GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&values), sizeof values) };
        if (done == 0) {
            throw std::runtime_error("GetProcessMemoryInfo failed, error " + std::to_string(GetLastError()));
        }
        return values;
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
        const double private_gb{ static_cast<double>(ProcessMemory::private_bytes()) / 1e9 };
        const double elapsed{ std::chrono::duration<double>(now - start_).count() };
        std::cout << "  " << name << " " << seconds << " s, private " << private_gb << " GB (at " << elapsed << " s)" << std::endl;
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

/// @brief The reports of a build: OUT.rejected.json and OUT.build.json.
class Reports {
public:
    /// @brief OUT.rejected.json of NativeIdentityGate (the failing states or rows, largest residual first).
    [[nodiscard]] static std::string rejection(const GateReport& gate, const AssembledMaps& m, const TerrainGeometry& geometry, const VelocityTable& vs,
                                               const Arguments& a)
    {
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
                if (f.per_state) {
                    state_item(json, f.items[n], geometry, vs);
                } else {
                    row_item(json, f.items[n], m, geometry, vs);
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

    /// @brief What OUT.build.json reports.
    struct BuildFacts {
        std::string sha{};                    ///< The container's SHA-256.
        std::uint64_t bytes{};                ///< Its size.
        std::size_t peak{};                   ///< Peak working set of the build, bytes.
        const Arguments* arguments{};
        const PermutedMaps* permuted{};
        const TerrainGeometry* geometry{};
        IdentityValues identities{};
        IdentityValues native{};              ///< The gate's worst relative residuals.
        std::string identity_gate{};          ///< "passed", or "failed (...)" with --skip-identity-gate.
        std::int64_t added_253{};             ///< Wall-list cells added by rule #253.
        const StageClock* clock{};
    };

    /// @brief OUT.build.json.
    [[nodiscard]] static std::string build(const BuildFacts& f)
    {
        const Arguments& a{ *f.arguments };
        JsonWriter json{ true };
        json.begin_object();
        json.key("container");
        json.value(a.out);
        json.key("sha256");
        json.value(f.sha);
        json.key("bytes");
        json.value(static_cast<std::int64_t>(f.bytes));
        json.key("velocity_set");
        json.value(a.velocities);
        json.key("allow_steep");
        json.value(a.allow_steep);
        json.key("domain_checks");
        json.value(a.allow_steep ? "R10, R1 (R11 slope check skipped: --allow-steep)" : "R10, R11, R1");
        sizes_json(json, f);
        json.key("identities");
        identities_json(json, f.identities);
        json.key("native_identities");
        identities_json(json, f.native);
        json.key("identity_gate");
        json.value(f.identity_gate);
        json.key("seconds");
        json.begin_object();
        for (const StageClock::Stage& stage : f.clock->stages) {
            json.key(stage.name);
            json.value(stage.seconds);
        }
        json.key("total");
        json.value(f.clock->total());
        json.end_object();
        json.key("threads");
        json.value(a.threads);
        json.key("memory_gb");
        json.value(a.memory_gb);
        json.key("peak_working_set_bytes");
        json.value(static_cast<std::int64_t>(f.peak));
        json.key("wall_list_cells_added_by_rule_253");
        json.value(f.added_253);
        json.key("builder");
        json.value("volumetric_wall_builder");
        json.end_object();
        return json.text() + "\n";
    }

private:
    /// @brief The identity values as a JSON object.
    static void identities_json(JsonWriter& json, const IdentityValues& v)
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

    /// @brief A direction as a JSON list.
    static void direction_json(JsonWriter& json, const Direction& c)
    {
        json.begin_list();
        json.value(c.x);
        json.value(c.y);
        json.value(c.z);
        json.end_list();
    }

    /// @brief The fields of one failing state.
    static void state_item(JsonWriter& json, const std::int64_t item, const TerrainGeometry& geometry, const VelocityTable& vs)
    {
        const auto q{ static_cast<std::int64_t>(vs.directions.size()) };
        const auto cell{ static_cast<std::size_t>(item / q) };
        const Direction& c{ vs.directions[static_cast<std::size_t>(item % q)] };
        const std::array<int, 3> k{ geometry.cell_ijk(cell) };
        json.key("state");
        json.value(item);
        json.key("cell");
        json.begin_list();
        json.value(k[0]);
        json.value(k[1]);
        json.value(k[2]);
        json.end_list();
        json.key("c");
        direction_json(json, c);
        json.key("volume");
        json.value(geometry.cell_volume(cell));
    }

    /// @brief The fields of one failing row.
    static void row_item(JsonWriter& json, const std::int64_t item, const AssembledMaps& m, const TerrainGeometry& geometry, const VelocityTable& vs)
    {
        const WallRow& row{ m.rows[static_cast<std::size_t>(item)] };
        const Direction& c{ vs.directions[static_cast<std::size_t>(row.direction)] };
        json.key("row");
        json.value(item);
        json.key("facet");
        json.value(geometry.facets()[static_cast<std::size_t>(row.facet)].key.python_text());
        json.key("c");
        direction_json(json, c);
        json.key("vi");
        json.value(m.vi[static_cast<std::size_t>(item)]);
    }

    /// @brief cells, facets, rows and W_nnz.
    static void sizes_json(JsonWriter& json, const BuildFacts& f)
    {
        json.key("cells");
        json.value(static_cast<std::int64_t>(f.geometry->cell_count()));
        json.key("facets");
        json.value(static_cast<std::int64_t>(f.geometry->facets().size()));
        json.key("rows");
        json.value(static_cast<std::int64_t>(f.permuted->vi.size()));
        json.key("W_nnz");
        json.value(static_cast<std::int64_t>(f.permuted->w.values.size()));
    }
};

/// @brief The folder of the slab stream's temporary files; removed with its content when the build ends.
class TempFolder {
public:
    /// @brief Creates the folder. @throws Refused when it exists (it may be another build's)
    explicit TempFolder(std::filesystem::path path)
        : path_{ std::move(path) }
    {
        if (std::filesystem::exists(path_)) {
            throw Refused("the temporary folder " + path_.string() + " exists (another build, or one that was stopped: remove it)");
        }
        std::filesystem::create_directories(path_);
    }

    ~TempFolder()
    {
        std::error_code error{};
        std::filesystem::remove_all(path_, error);
        if (error) { // a destructor cannot throw: the folder is named so the owner can remove it
            std::cerr << "volumetric_wall_builder: cannot remove the temporary folder " << path_.string() << ": " << error.message() << std::endl;
        }
    }

    TempFolder(const TempFolder&) = delete;
    TempFolder& operator=(const TempFolder&) = delete;

    /// @brief The folder.
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_{};
};

/// @brief The disk space the build needs: the container and the temporary files of D and direct_arm.
class DiskCheck {
public:
    /// @brief Fails when a disk has less free space than the estimate plus a quarter.
    /// @param cells the fluid cells
    /// @param q the velocity set size
    /// @throws Refused naming the disk, the free space and the need
    static void require(const std::filesystem::path& out_folder, const std::filesystem::path& temp_folder, const std::size_t cells, const std::size_t q)
    {
        // per cell: cell_xyz, volume, centroid, mode_phi; per state at most: D (offsets, indices, weights), R_offsets, f0,
        // direct_arm; the surface sections add about a fifth (hill: 1814 B/cell measured against 1456 B/cell here). The
        // temporary files hold D and direct_arm once more until the container is written.
        const double container{ static_cast<double>(cells) * (52.0 + 52.0 * static_cast<double>(q)) * 1.25 };
        const double temporary{ static_cast<double>(cells) * (40.0 * static_cast<double>(q)) };
        const double margin{ 1.25 };
        const std::filesystem::space_info out_space{ std::filesystem::space(out_folder) };
        const std::filesystem::space_info temp_space{ std::filesystem::space(temp_folder) };
        const bool same_disk{ std::filesystem::absolute(out_folder).root_name() == std::filesystem::absolute(temp_folder).root_name() };
        const double out_need{ same_disk ? margin * (container + temporary) : margin * container };
        const double temp_need{ margin * temporary };
        if (static_cast<double>(out_space.available) < out_need) {
            throw Refused("the disk of " + out_folder.string() + " has " + gigabytes(static_cast<double>(out_space.available)) + " free; the build needs "
                          + gigabytes(out_need) + " (the estimate plus a quarter)");
        }
        if (static_cast<double>(temp_space.available) < temp_need) {
            throw Refused("the disk of " + temp_folder.string() + " has " + gigabytes(static_cast<double>(temp_space.available)) + " free; the temporary files need "
                          + gigabytes(temp_need) + " (the estimate plus a quarter)");
        }
    }

    /// @brief A byte count as "12.3 GB".
    [[nodiscard]] static std::string gigabytes(const double bytes)
    {
        std::ostringstream text{};
        text.precision(3);
        text << bytes / 1e9 << " GB";
        return text.str();
    }
};

/// @brief The cells of one slab for the memory budget.
class SlabBudget {
public:
    /// @brief Donor results and buffers held per cell of a slab (DonorColumn: q DonorStates, the event lists, the task
    /// list): measured 3.0 KB on the tiled hill (3.69 M cells, one slab, 2026-11-12), with a margin.
    static constexpr double BYTES_PER_CELL{ 4096.0 };

    /// @brief The cells of one slab so the process stays within the budget.
    /// @param budget_gb the memory budget in GiB
    /// @param resident the bytes the process holds before the slab loop
    /// @param column_cells the cells of the largest x-column (a slab holds at least three)
    /// @throws Refused when the budget does not hold three x-columns
    [[nodiscard]] static std::size_t cells(const double budget_gb, const std::size_t resident, const std::size_t column_cells)
    {
        const double budget{ budget_gb * 1073741824.0 };
        const double room{ budget - static_cast<double>(resident) };
        const double least{ 3.0 * static_cast<double>(column_cells) * BYTES_PER_CELL };
        if (room < least) {
            throw Refused("--memory-gb " + PythonFloat::repr(budget_gb) + ": the build holds " + DiskCheck::gigabytes(static_cast<double>(resident))
                          + " before the slabs, and one slab needs " + DiskCheck::gigabytes(least) + " more");
        }
        return static_cast<std::size_t>(room / BYTES_PER_CELL) - 2u * column_cells;
    }
};

/// @brief The steps of a build around the stages: refusals, the slab size, the gate and the output.
class BuildSteps {
public:
    /// @brief Refuses an existing output and a rough law outside its domain, as terrain_container.py.
    /// @throws Refused naming the reason
    static void refuse_bad_options(const Arguments& a)
    {
        if (std::filesystem::exists(a.out)) {
            throw Refused(a.out + " exists");
        }
        if (!a.rough_z0) {
            return;
        }
        try {
            const RoughLaw law{ *a.rough_z0, a.rough_k0 };
        } catch (const std::invalid_argument& error) {
            throw Refused(error.what());
        }
    }

    /// @brief The heights CSV: --heights, or the cached CSV of --stl (sampled when it is not in the cache).
    /// @throws std::runtime_error when the STL cannot be sampled or the cache cannot be written
    [[nodiscard]] static std::string heights_csv(const Arguments& a)
    {
        if (a.stl.empty()) {
            return a.heights;
        }
        std::filesystem::path cache{ a.cache_dir };
        if (cache.empty()) {
            const char* const local{ std::getenv("LOCALAPPDATA") }; // NOLINT(concurrency-mt-unsafe): read once at start-up
            if (local == nullptr) {
                throw std::runtime_error("LOCALAPPDATA is not set: give --cache-dir");
            }
            cache = std::filesystem::path{ local } / "FluidX3D" / "terrain_heights";
        }
        HeightsRecipe recipe{};
        recipe.cell_m = a.cell_m;
        recipe.apron = a.apron;
        return StlHeights::cached(a.stl, recipe, cache, a.threads).string();
    }

    /// @brief The cells of one slab within the memory budget (the largest x-column sets the least).
    /// @throws Refused when the budget does not hold three x-columns
    [[nodiscard]] static std::size_t slab_cells(const TerrainGeometry& geometry, const Arguments& a)
    {
        std::size_t column_cells{ 0 };
        for (int i{ 0 }; i < geometry.record().lx; ++i) {
            const std::size_t in_column{ geometry.column_first(i + 1) - geometry.column_first(i) };
            if (in_column > column_cells) { // not std::max: windows.h defines a max macro
                column_cells = in_column;
            }
        }
        const std::size_t resident{ ProcessMemory::private_bytes() };
        return SlabBudget::cells(a.memory_gb, resident, column_cells);
    }

    /// @brief Per cell, 1 for a gather donor cell (arms().wall_cells).
    [[nodiscard]] static std::vector<char> wall_flags(const ArmTables& arms, const std::size_t cells)
    {
        std::vector<char> flags(cells, 0);
        for (const std::int64_t cell : arms.wall_cells) {
            flags[static_cast<std::size_t>(cell)] = 1;
        }
        return flags;
    }

    /// @brief The S stencil cells, sorted unique.
    [[nodiscard]] static std::vector<std::int64_t> support_cells(const SparseMatrix& s)
    {
        std::vector<std::int64_t> cells{ s.indices };
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        return cells;
    }

    /// @brief The temporary folder's parent after the disk check.
    /// @throws Refused when a disk has too little free space
    [[nodiscard]] static std::filesystem::path temp_parent(const Arguments& a, const std::filesystem::path& out_path, const std::size_t cells,
                                                           const std::size_t q)
    {
        std::filesystem::create_directories(out_path.parent_path());
        std::filesystem::path parent{ out_path.parent_path() };
        if (!a.temp_dir.empty()) {
            parent = std::filesystem::absolute(a.temp_dir);
        }
        std::filesystem::create_directories(parent);
        DiskCheck::require(out_path.parent_path(), parent, cells, q);
        return parent;
    }

    /// @brief NativeIdentityGate.check from the stream's state part and the row part.
    [[nodiscard]] static GateReport gate(const SlabResults& streamed, const RowGate& rows)
    {
        GateReport gate{};
        gate.finite = streamed.finite && rows.finite;
        gate.worst = IdentityValues{ streamed.donor_worst.result(), streamed.receiver_worst.result(), rows.prism_worst, rows.return_worst };
        const std::vector<GateFailure> candidates{ streamed.donor_failures.result("donor_partition"), streamed.receiver_failures.result("receiver_coverage"),
                                                   rows.prism, rows.ret };
        for (const GateFailure& f : candidates) {
            if (!f.items.empty()) {
                gate.failing.push_back(f);
            }
        }
        gate.passed = gate.finite && gate.failing.empty();
        return gate;
    }

    /// @brief Writes the container and OUT.build.json; prints the summary.
    /// @throws std::runtime_error when a write fails or a value does not fit its section
    static void write(const Arguments& a, const ContainerParts& parts, const GateReport& gate, StageClock& clock)
    {
        ContainerOptions options{};
        options.nu = a.nu;
        options.friction = a.friction;
        options.rough_z0 = a.rough_z0;
        options.rough_k0 = a.rough_k0;
        options.initial_velocity = a.initial_velocity;
        options.allow_steep = a.allow_steep;
        const ContainerWriter writer{ parts, options };
        ContainerSink sink{ std::filesystem::absolute(a.out) };
        Reports::BuildFacts facts{};
        facts.sha = writer.write(sink);
        facts.bytes = sink.bytes();
        clock.lap("write");
        facts.peak = ProcessMemory::peak_working_set();
        facts.arguments = &a;
        facts.permuted = parts.permuted;
        facts.geometry = parts.geometry;
        facts.identities = parts.identities;
        facts.native = gate.worst;
        facts.identity_gate = "passed";
        if (!gate.passed) {
            facts.identity_gate = "failed (written with --skip-identity-gate; see the .rejected.json)";
        }
        facts.added_253 = static_cast<std::int64_t>(parts.wall->added_253.size());
        facts.clock = &clock;
        write_text(a.out + ".build.json", Reports::build(facts));
        const std::string peak_text{ DiskCheck::gigabytes(static_cast<double>(facts.peak)) };
        const double total{ clock.total() };
        std::cout << "sha256 " << facts.sha << "\nbytes " << facts.bytes << "\ncells " << parts.geometry->cell_count() << ", facets "
                  << parts.geometry->facets().size() << ", rows " << parts.permuted->vi.size() << ", W_nnz " << parts.permuted->w.values.size()
                  << "\npeak working set " << peak_text << "\ntotal " << total << " s" << std::endl;
    }
};

/// @brief The whole build. @return the exit code
int build(const Arguments& a)
{
    StageClock clock{};
    const std::string heights{ BuildSteps::heights_csv(a) };
    if (a.heights_only) {
        std::cout << heights << std::endl;
        return 0;
    }
    BuildSteps::refuse_bad_options(a);
    if (!a.stl.empty()) {
        clock.lap("heights");
    }
    const VelocityTable velocities{ VelocityTable::named(a.velocities) };
    const TerrainInput input{ TerrainInput::from_csv(heights, a.z_top, a.name) };
    input.domain_checks(a.allow_steep);
    const TerrainGeometry geometry{ input.record(velocities, 12), a.threads };
    const std::size_t cells{ geometry.cell_count() };
    std::cout << "cells " << cells << ", facets " << geometry.facets().size() << "; " << a.threads << " threads" << std::endl;
    clock.lap("geometry");
    PassRunner passes{ geometry, a.threads };
    passes.run();
    clock.lap("producer_passes");
    SamplerSearch sampler{ geometry, 12 };
    sampler.run(a.threads);
    sampler.check();
    clock.lap("sampler");
    const MapAssembly assembly{ geometry, velocities, passes, sampler };
    const AssembledMaps maps{ assembly.assemble(a.threads) };
    clock.lap("maps_assemble");
    const StateProducts state_products{ MapAssembly::state_products(maps) };
    const RowProducts row_products{ assembly.row_products(maps) };
    const RowGate row_gate{ MapAssembly::row_gate(maps, row_products) };
    clock.lap("maps_products");
    const PermutedMaps permuted{ assembly.permute(maps) };
    clock.lap("maps_permute");
    const WallTables tables{ geometry, velocities, assembly };
    const ArmTables arms{ tables.arms(maps, permuted) };
    const std::vector<char> wall_flags{ BuildSteps::wall_flags(arms, cells) };
    const std::vector<std::int64_t> support{ BuildSteps::support_cells(permuted.s) };
    clock.lap("maps_arms");
    const std::filesystem::path out_path{ std::filesystem::absolute(a.out) };
    const std::filesystem::path temp_parent{ BuildSteps::temp_parent(a, out_path, cells, velocities.directions.size()) };
    const TempFolder temp{ temp_parent / (out_path.filename().string() + ".sections") };
    const std::size_t slab_cells{ BuildSteps::slab_cells(geometry, a) };
    DiagTotals diag{ passes.diag() };
    const SlabInputs inputs{ .geometry = &geometry, .velocities = &velocities, .products = &state_products, .permuted_r = &permuted.r,
                             .wall = &wall_flags, .support = &support };
    SlabStream stream{ inputs, temp.path(), a.threads };
    stream.run(slab_cells, a.slab_columns, diag);
    const SlabResults& streamed{ stream.results() };
    clock.lap("slabs");
    const WallList wall{ tables.wall_list(wall_flags, streamed, permuted.s, support) };
    const GradientMap gradient{ tables.gradient(wall.cells, a.threads) };
    clock.lap("wall_list_and_gradient");
    IdentityValues identities{ MapAssembly::row_identities(maps, row_products) };
    identities.donor_partition = streamed.donor_partition;
    identities.receiver_coverage = streamed.receiver_coverage;
    const GateReport gate{ BuildSteps::gate(streamed, row_gate) };
    if (!gate.passed) {
        write_text(a.out + ".rejected.json", Reports::rejection(gate, maps, geometry, velocities, a));
        if (!a.skip_identity_gate) {
            std::cout << "REFUSED: map identities violated (native definitions, tolerance 1e-10); no container written" << std::endl;
            std::cout << "diagnostic report: " << a.out << ".rejected.json" << std::endl;
            return 3;
        }
        std::cout << "WARNING: map identities violated (native definitions, tolerance 1e-10); --skip-identity-gate writes the container anyway"
                  << " (diagnostic report: " << a.out << ".rejected.json)" << std::endl;
    }
    const ContainerParts parts{ .input = &input, .velocities = &velocities, .geometry = &geometry, .sampler = &sampler, .permuted = &permuted,
                                .identities = identities, .diag = diag, .stream = &stream, .arms = &arms, .wall = &wall, .gradient = &gradient };
    BuildSteps::write(a, parts, gate, clock);
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

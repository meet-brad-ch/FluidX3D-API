#pragma once
/// @file terrain.hpp
/// @brief The terrain case, its cells and facets: builder3d.py Case and Builder.__init__, terrain_facets, lid_facet and
/// candidates, ported in Python's operation order.

#include "dyadic.hpp"
#include "poly.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief A lattice velocity.
struct Direction {
    int x{}; ///< x component, -1, 0 or 1.
    int y{}; ///< y component.
    int z{}; ///< z component.
};

/// @brief The input of the builder: builder3d.py's manifest record and the velocity set.
struct TerrainRecord {
    int lx{};                            ///< Columns in x (periodic).
    int ly{};                            ///< Columns in y (periodic).
    int z_top{};                         ///< The lid height.
    std::vector<double> heights{};       ///< Node heights, index i * ly + j, 0-based; exact binary64 values.
    std::vector<bool> main_diagonal{};   ///< Per column (i * ly + j): true for the "main" diagonal.
    std::vector<Direction> directions{}; ///< The velocity set in lexicographic order.
    int neighbours{};                    ///< The sampler stencil size.
};

/// @brief A facet key: ("T", i, j, "T<tag>", k) or ("L", i, j), with i, j wrapped into the periodic box.
struct FacetKey {
    bool lid{}; ///< True for "L".
    int i{};    ///< Wrapped column x.
    int j{};    ///< Wrapped column y.
    int tag{};  ///< Triangle 0 or 1 (terrain only).
    int k{};    ///< Layer (terrain only).

    /// @brief Equal keys.
    [[nodiscard]] bool operator==(const FacetKey& other) const;
    /// @brief The text of Python's str() of the key tuple; the passes sort the facets by it.
    [[nodiscard]] std::string python_text() const;
};

/// @brief A facet in one integer frame (builder3d.py Facet).
struct Facet {
    FacetKey key{};                    ///< The key.
    std::vector<Vec3> poly{};          ///< The polygon in the frame.
    std::array<Dyadic, 3> n_raw{};     ///< The exact raw normal (-a, -b, 1) or (0, 0, -1).
    Vec3 area{};                       ///< The area vector a (binary64).
    Vec3 unit{};                       ///< The unit normal into the fluid.
    std::array<int, 3> cell{};         ///< The wrapped cell of the facet.
    Plane plane{};                     ///< The unit plane in the frame.
    Dyadic d_raw{};                    ///< n_raw . x = d_raw on the unwrapped facet (exact).
    std::array<std::int8_t, 27> signs{}; ///< Exact sign of c . n_raw per direction index.
};

/// @brief A facet with the image offset of its unwrapped column (builder3d.py candidates: (f, img)).
struct Candidate {
    Facet facet{};             ///< The facet in the frame of the pass.
    std::array<int, 2> image{}; ///< (ii - ii % Lx, jj - jj % Ly).
};

/// @brief The origin-free part of one terrain facet (the cached half of terrain_facets).
struct FacetBase {
    FacetKey key{};                 ///< The key.
    Dyadic ae{};                    ///< Exact a of the triangle plane.
    Dyadic be{};                    ///< Exact b.
    Dyadic ge{};                    ///< Exact g at the unwrapped position.
    std::vector<Vec3> part{};       ///< The clipped footprint in the facet's own frame (z unused).
    double g_own{};                 ///< g in the own frame, binary64.
    double a{};                     ///< a, binary64.
    double b{};                     ///< b, binary64.
    Vec3 area{};                    ///< The area vector.
    Vec3 unit{};                    ///< The unit normal.
    double length{};                ///< |n_raw| in binary64.
    std::array<int, 3> cell{};      ///< The wrapped cell.
    std::array<std::int8_t, 27> signs{}; ///< Exact sign of c . n_raw per direction index.
};

/// @brief One column triangle in binary64 (Case.triangles "trif" and float(exact_plane)), for surface_height.
struct ColumnTrianglePlane {
    std::array<double, 6> foot{}; ///< x0, y0, x1, y1, x2, y2 in the vertex order of the triangle.
    double a{};                   ///< float(a).
    double b{};                   ///< float(b).
    double g{};                   ///< float(g) at the unwrapped position.
};

/// @brief One fluid cell with its pieces in its own frame (made on demand for the donor pass; not stored for all cells).
struct TerrainCell {
    std::array<int, 3> ijk{};   ///< The cell.
    std::vector<Poly> pieces{}; ///< The fluid pieces, frame origin (i, j, k).
    double volume{};            ///< The fluid volume (Builder.__init__ "volume": 1.0 for a full cell).
    Vec3 centroid{};            ///< The fluid centroid in global coordinates ("centroid").
    Vec3 centroid_local{};      ///< The centroid about the cell corner ("centroid_local").
};

/// @brief The cells and facets of the terrain (builder3d.py Builder.__init__), read-only after construction.
class TerrainGeometry {
public:
    /// @brief Builds every cell, the canonical facets and the facet bases of all columns a pass can ask for.
    /// @throws std::runtime_error with the builder's message when a cell is classified fluid but has no piece
    /// @throws std::domain_error when an input value is outside what the port supports, or no cell is fluid
    /// @throws std::invalid_argument when threads is below 1
    /// @param threads the number of threads of the parallel set-up, 1 or more
    TerrainGeometry(TerrainRecord record, int threads);

    /// @brief The input record.
    [[nodiscard]] const TerrainRecord& record() const { return record_; }

    /// @brief The number of fluid cells. Cell n is the n-th in Python's sorted() of (i, j, k) tuples.
    [[nodiscard]] std::size_t cell_count() const { return volume_.size(); }

    /// @brief The (i, j, k) of cell n.
    [[nodiscard]] std::array<int, 3> cell_ijk(std::size_t n) const;

    /// @brief The fluid volume of cell n (Builder.__init__ "volume").
    [[nodiscard]] double cell_volume(const std::size_t n) const { return volume_[n]; }

    /// @brief The fluid centroid of cell n in global coordinates ("centroid").
    [[nodiscard]] const Vec3& cell_centroid(const std::size_t n) const { return centroid_[n]; }

    /// @brief The centroid of cell n about its corner ("centroid_local").
    [[nodiscard]] const Vec3& cell_local(const std::size_t n) const { return local_[n]; }

    /// @brief The index of fluid cell (i, j, k), i and j wrapped into the box, k as is; -1 when it is not a fluid cell.
    [[nodiscard]] std::int64_t cell_index(int i, int j, int k) const;

    /// @brief The index of the first cell of x-column i; column_first(lx) is the cell count. The cells of x-columns
    /// [a, b) are the indices [column_first(a), column_first(b)).
    [[nodiscard]] std::size_t column_first(const int i) const { return static_cast<std::size_t>(first_[static_cast<std::size_t>(i) * static_cast<std::size_t>(record_.ly)]); }

    /// @brief The lowest layer k of a fluid cell; the highest is z_top - 1.
    [[nodiscard]] int lowest_layer() const { return lowest_layer_; }

    /// @brief The first fluid layer of column (i, j), i and j inside the box; z_top when the column has no cell.
    [[nodiscard]] int first_layer(const int i, const int j) const { return kfirst_[static_cast<std::size_t>(i) * static_cast<std::size_t>(record_.ly) + static_cast<std::size_t>(j)]; }

    /// @brief The index of the first cell of column (i, j), i and j inside the box.
    [[nodiscard]] std::size_t column_cell(const int i, const int j) const { return static_cast<std::size_t>(first_[static_cast<std::size_t>(i) * static_cast<std::size_t>(record_.ly) + static_cast<std::size_t>(j)]); }

    /// @brief Cell n with its pieces and their cached volumes and moments, for the donor pass.
    /// @throws std::runtime_error when the cell has no piece
    [[nodiscard]] TerrainCell donor_cell(std::size_t n) const;

    /// @brief The canonical facets (frame origin 0), sorted by Python's str() of the key.
    [[nodiscard]] const std::vector<Facet>& facets() const { return facets_; }

    /// @brief The terrain facets of unwrapped column (i, j), layer k, in the frame of an integer origin.
    void terrain_facets(int i, int j, int k, const std::array<int, 3>& origin, std::vector<Facet>& out) const;

    /// @brief The lid facet of unwrapped column (i, j) in the frame of an integer origin.
    [[nodiscard]] Facet lid_facet(int i, int j, const std::array<int, 3>& origin) const;

    /// @brief The facet with the key of f, built again in the frame of an integer origin (facet_in_frame).
    /// @throws std::runtime_error when the facet is not rebuilt in that frame
    [[nodiscard]] Facet facet_in_frame(const Facet& f, const std::array<int, 3>& origin) const;

    /// @brief The facets of the 3 x 3 x 3 neighbourhood of (i, j, k) with their images, in Python's order.
    /// @param exclude a key to leave out, or nullptr
    void candidates(int i, int j, int k, const FacetKey* exclude, const std::array<int, 3>& origin, std::vector<Candidate>& out) const;

    /// @brief The two triangles of unwrapped column (i, j) in binary64, in Case.triangles order.
    [[nodiscard]] std::array<ColumnTrianglePlane, 2> column_planes(int i, int j) const;

    /// @brief Python's i % n for any sign of i.
    [[nodiscard]] static int wrap(int i, int n);

private:
    TerrainRecord record_{};
    int threads_{ 1 };                     ///< Threads of the parallel set-up.
    int zmin_{};
    int lowest_layer_{ 2147483647 };       ///< The lowest fluid layer (INT_MAX until the column index is built).                           ///< floor of the lowest node height.
    int base_i0_{};                        ///< Lowest unwrapped column x of the base table.
    int base_j0_{};                        ///< Lowest unwrapped column y.
    int base_k0_{};                        ///< Lowest layer.
    int base_ni_{};                        ///< Columns x in the table.
    int base_nj_{};                        ///< Columns y.
    int base_nk_{};                        ///< Layers.
    std::vector<std::vector<FacetBase>> bases_{}; ///< Facet bases per (i, j, k) of the table.
    std::vector<int> kfirst_{};             ///< Per column (i * ly + j): the first fluid layer.
    std::vector<std::int64_t> first_{};     ///< Per column: the index of its first cell; one more entry: the cell count.
    std::vector<double> volume_{};          ///< Per cell.
    std::vector<Vec3> centroid_{};          ///< Per cell.
    std::vector<Vec3> local_{};             ///< Per cell.
    std::vector<Facet> facets_{};

    [[nodiscard]] double height_value(int i, int j) const;
    [[nodiscard]] const std::vector<FacetBase>& bases_at(int i, int j, int k) const;
    [[nodiscard]] std::vector<FacetBase> compute_bases(int i, int j, int k) const;
    [[nodiscard]] std::vector<Poly> cell_pieces(int i, int j, int k) const;
    /// @brief Fills volume, centroid and centroid_local as Builder.__init__ does: the exact cube for a full cell, else the
    /// sums of the piece volumes and moments.
    static void cell_moments(TerrainCell& cell, bool full);
    [[nodiscard]] Facet facet_from_base(const FacetBase& base, int i, int j, int k, const std::array<int, 3>& origin) const;
    /// @brief Per column the first fluid layer (k + 1 > the lowest node, Builder.__init__) and the cell offsets.
    void build_column_index();
    /// @brief Volume, centroid and local centroid of every cell, on all threads; the pieces are not kept.
    void build_cells();
    /// @brief The canonical facets of every fluid cell and the lids, sorted by Python's str() of the key.
    void build_facets();
    void build_base_table();
};

} // namespace fluidx3d::experimental::wall_builder

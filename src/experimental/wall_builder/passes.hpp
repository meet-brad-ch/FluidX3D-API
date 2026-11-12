#pragma once
/// @file passes.hpp
/// @brief The producer and donor passes of terrain_container.py VelocityMomentTerrainBuilder (with builder3d.py
/// producer_partition, first_owner and add_w), ported in Python's operation order.

#include "poly.hpp"
#include "terrain.hpp"

#include <array>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief h[(fkey, d, cell, img)]: the immediate volume of a row in a cell image.
struct HEntry {
    FacetKey fkey{};            ///< Producer facet.
    int d{};                    ///< Direction index.
    std::array<int, 3> cell{};  ///< Wrapped cell.
    std::array<int, 2> image{}; ///< Image offset.
    double value{};             ///< Volume.
};

/// @brief W[(fkey, d, gkey, img)]: the volume a row hands to its first receiver.
struct WEntry {
    FacetKey fkey{};            ///< Producer facet.
    int d{};                    ///< Direction index.
    FacetKey gkey{};            ///< Receiver facet.
    std::array<int, 2> image{}; ///< Image offset of the receiver.
    double value{};             ///< Volume.
};

/// @brief vi[(fkey, d)]: the FP64 prism volume c . a of a row.
struct ViEntry {
    FacetKey fkey{}; ///< Producer facet.
    int d{};         ///< Direction index.
    double value{};  ///< c . a.
};

/// @brief h_moment[(fkey, d, cell)]: volume and first moment about the cell corner.
struct HMomentEntry {
    FacetKey fkey{};               ///< Producer facet.
    int d{};                       ///< Direction index.
    std::array<int, 3> cell{};     ///< Wrapped cell.
    std::array<double, 4> value{}; ///< Volume, then moment x, y, z.
};

/// @brief row_moment[(fkey, d)]: volume and global first moment of a row's free part.
struct RowMomentEntry {
    FacetKey fkey{};               ///< Producer facet.
    int d{};                       ///< Direction index.
    std::array<double, 4> value{}; ///< Volume, then moment x, y, z.
};

/// @brief direct[(cell, d)] and direct_moment[(cell, d)]: what a donor state keeps.
struct DirectEntry {
    std::array<int, 3> cell{};     ///< The cell.
    int d{};                       ///< Direction index.
    double value{};                ///< direct: the kept volume.
    std::array<double, 4> moment{}; ///< direct_moment: volume, then moment about the cell corner.
};

/// @brief Everything one producer row or one donor state writes, in Python's insertion order.
struct UnitResult {
    std::vector<DiagEvent> events{};          ///< Split defects and small volumes, in order.
    std::vector<HEntry> h{};                  ///< h, first-insertion order.
    std::vector<WEntry> w{};                  ///< W, first-insertion order.
    std::vector<ViEntry> vi{};                ///< vi (at most one).
    std::vector<HMomentEntry> h_moment{};     ///< h_moment, first-insertion order.
    std::vector<RowMomentEntry> row_moment{}; ///< row_moment (at most one).
    std::vector<DirectEntry> direct{};        ///< direct and direct_moment (at most one).
};

/// @brief The two passes for one facet or one cell; one engine per thread, the geometry is shared and read-only.
class PassEngine {
public:
    /// @brief Uses the geometry; it must outlive the engine.
    explicit PassEngine(const TerrainGeometry& geometry);

    /// @brief The producer rows of a canonical facet, one result per moving direction (in direction order).
    /// @throws std::runtime_error, std::domain_error with the builder's message (GeometryInvalid, R4)
    [[nodiscard]] std::vector<UnitResult> producer(const Facet& facet) const;

    /// @brief The donor states of a cell, one result per direction (all directions, rest included).
    /// @throws std::domain_error with the builder's message (GeometryInvalid)
    [[nodiscard]] std::vector<UnitResult> donor(const TerrainCell& cell) const;

private:
    const TerrainGeometry& geometry_;

    /// @brief One piece and the indices (into the receivers) of the eligible receivers that contain it.
    struct OwnedPiece {
        Poly piece{};             ///< The piece.
        std::vector<int> owners{}; ///< Receiver indices, in receiver order.
    };

    /// @brief A piece and its first receiver.
    struct FirstOwner {
        Poly piece{};  ///< The piece.
        int owner{};   ///< Receiver index.
    };

    void producer_row(const Facet& f, const Facet& in_frame, const std::vector<Candidate>& all, int d, UnitResult& out) const;
    void partition(const Facet& f, const Direction& c, const std::vector<const Candidate*>& receivers, UnitResult& out,
                   std::vector<Poly>& free_pieces, std::vector<FirstOwner>& owned) const;
    void first_owner(const OwnedPiece& start, const Direction& c, const std::vector<const Candidate*>& receivers, UnitResult& out,
                     std::vector<FirstOwner>& owned) const;
    void add_h(const Facet& f, int d, const Poly& piece, const std::array<int, 3>& origin, UnitResult& out) const;
    void add_h_cell(const Facet& f, int d, const Poly& clipped, const std::array<int, 3>& corner, const std::array<int, 3>& origin,
                    UnitResult& out) const;
    void add_w(const Facet& f, int d, const Candidate& receiver, const Poly& piece, UnitResult& out) const;
    void donor_state(const TerrainCell& cell, const std::vector<Candidate>& all, int d, UnitResult& out) const;
};

/// @brief The exact predicates of builder3d.py: exact_parallel and exact_sigma_difference.
class ExactOrder {
public:
    /// @brief True when the exact raw normals are parallel (their cross product is zero).
    [[nodiscard]] static bool parallel(const Facet& a, const Facet& b);

    /// @brief The sign of sigma_1 - sigma_2 for parallel planes: sign(d2 / (n2 . c) - d1 / (n1 . c)).
    /// @throws std::domain_error when n . c is zero for either facet
    [[nodiscard]] static int sigma_difference(const Facet& g1, const Facet& g2, const Direction& c);
};

} // namespace fluidx3d::experimental::wall_builder

#pragma once
/// @file poly.hpp
/// @brief The binary64 convex polyhedra of the wall builder, ported operation for operation from builder3d.py.
///
/// Every floating-point expression keeps the operation order of the Python code, and the library is built without
/// fast math (FLUIDX3D_PRECISE_FLOAT), so each result is the same binary64 value as in Python.

#include <initializer_list>
#include <memory>
#include <span>
#include <optional>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief A point or a vector in lattice units.
struct Vec3 {
    double x{}; ///< x component.
    double y{}; ///< y component.
    double z{}; ///< z component.
};

/// @brief The half-space normal . x <= offset.
struct Plane {
    Vec3 normal{};   ///< Unit normal (outward for a face of a polyhedron).
    double offset{}; ///< Offset along the normal.
};

/// @brief A volume and its first moment about the frame origin.
struct VolumeMoment {
    double volume{}; ///< The exact volume, rounded once.
    Vec3 moment{};   ///< The first moment (volume times centroid).
};

/// @brief One diagnostic event of a pass, replayed later in the serial order (terrain_container.py merge()).
struct DiagEvent {
    bool is_defect{}; ///< True: a split defect; false: a small volume.
    double value{};   ///< The defect or the volume.
};

/// @brief The vector operations of builder3d.py (dotf, crossf, subf), with the same operation order.
class VectorOps {
public:
    /// @brief a . b as ((a.x b.x + a.y b.y) + a.z b.z).
    [[nodiscard]] static double dot(const Vec3& a, const Vec3& b);
    /// @brief a x b.
    [[nodiscard]] static Vec3 cross(const Vec3& a, const Vec3& b);
    /// @brief a - b.
    [[nodiscard]] static Vec3 subtract(const Vec3& a, const Vec3& b);
};

class FanDeterminants; // poly.cpp: the exact fan determinants of a Poly

/// @brief The face cycles of a polyhedron, stored flat: every vertex index in face order and one offset per face (one
/// allocation per polyhedron instead of one per face).
class FaceList {
public:
    /// @brief The number of faces.
    [[nodiscard]] std::size_t count() const { return offsets_.size() - 1; }

    /// @brief The vertex indices of face f, 0-based.
    [[nodiscard]] std::span<const int> face(const std::size_t f) const { return { indices_.data() + offsets_[f], offsets_[f + 1] - offsets_[f] }; }

    /// @brief Every vertex index of every face, in order.
    [[nodiscard]] std::span<const int> all() const { return { indices_.data(), indices_.size() }; }

    /// @brief Appends a face.
    void add(const std::span<const int> face)
    {
        indices_.insert(indices_.end(), face.begin(), face.end());
        offsets_.push_back(indices_.size());
    }

    /// @brief Appends a face given as a list.
    void add(const std::initializer_list<int> face) { add(std::span<const int>{ face.begin(), face.size() }); }

    /// @brief Removes every face.
    void clear()
    {
        indices_.clear();
        offsets_.assign(1, 0);
    }

private:
    std::vector<int> indices_{};             ///< All face cycles, concatenated.
    std::vector<std::size_t> offsets_{ 0 };  ///< Start of each face, then the end.
};

/// @brief A binary64 convex polyhedron: vertices and face cycles (indices into the vertices).
///
/// The exact volume and moment are cached on first use: they depend only on the vertices and faces, and a split computes
/// the volume of the same polyhedron again and again (a plane that misses it returns the polyhedron itself). Code that
/// changes vertices or faces after a volume was taken must make a new Poly.
class Poly {
public:
    std::vector<Vec3> vertices{};           ///< The vertices.
    FaceList faces{};                       ///< The face cycles, 0-based vertex indices.

    /// @brief The box [lo, hi] with the face order of builder3d.py Poly.box.
    [[nodiscard]] static Poly box(const Vec3& lo, const Vec3& hi);

    /// @brief The prism of a planar polygon swept by a vector (Poly.prism).
    /// @param polygon the base polygon, 3 vertices or more
    /// @param vec the sweep vector
    [[nodiscard]] static Poly prism(const std::vector<Vec3>& polygon, const Vec3& vec);

    /// @brief The vertex mean (Poly.interior), summed left to right.
    [[nodiscard]] Vec3 interior() const;

    /// @brief The exact volume, rounded once (Poly.volume).
    /// @throws std::overflow_error when the scaled determinants need more than WideInt::BITS (1024) bits
    /// @throws std::domain_error when a coordinate is not finite or the volume is subnormal
    [[nodiscard]] double volume() const;

    /// @brief The exact volume and the first moment (Poly.moment).
    /// @throws std::overflow_error when the scaled determinants need more than WideInt::BITS (1024) bits
    /// @throws std::domain_error when a coordinate is not finite or a weight is subnormal
    [[nodiscard]] VolumeMoment moment() const;

    /// @brief The outward face planes (Poly.halfspaces); a face with a zero normal is skipped.
    [[nodiscard]] std::vector<Plane> halfspaces() const;

private:
    /// @brief The exact fan determinants, computed on first use.
    [[nodiscard]] const FanDeterminants& fan() const;

    mutable std::shared_ptr<const FanDeterminants> fan_{}; ///< Cached exact determinants, shared by the copies.
    mutable std::optional<double> volume_{};       ///< Cached exact volume.
    mutable std::optional<VolumeMoment> moment_{}; ///< Cached exact volume and moment.
};

/// @brief The clipping and splitting of builder3d.py: clip, split, subtract_convex and order_cycle.
class PolyClipper {
public:
    static constexpr double EPS_ON{ 1e-12 }; ///< builder3d.EPS_ON: the ON-plane tolerance factor.
    static constexpr double SMALL{ 1e-14 };  ///< builder3d.SMALL: volumes at or below it are counted.

    /// @brief P cut to normal . x <= offset; empty when no vertex lies strictly inside (builder3d.clip).
    [[nodiscard]] static std::optional<Poly> clip(const Poly& poly, const Plane& plane);

    /// @brief Both sides of a plane, with the split defect appended to the events (builder3d.split).
    /// @param poly the polyhedron
    /// @param plane the plane
    /// @param events receives one defect event
    /// @param inside receives the part with normal . x <= offset, if any
    /// @param outside receives the part with normal . x >= offset, if any
    static void split(const Poly& poly, const Plane& plane, std::vector<DiagEvent>& events, std::optional<Poly>& inside,
                      std::optional<Poly>& outside);

    /// @brief The pieces inside and outside a convex region, by its face planes in order (builder3d.subtract_convex).
    /// @param pieces the pieces
    /// @param planes the outward face planes of the region
    /// @param events receives the split defects
    /// @param inside receives the pieces inside the region
    /// @param outside receives the pieces outside the region
    static void subtract_convex(const std::vector<Poly>& pieces, const std::vector<Plane>& planes, std::vector<DiagEvent>& events,
                                std::vector<Poly>& inside, std::vector<Poly>& outside);

    /// @brief The volume, checked finite and not negative (builder3d.checked_volume).
    /// @throws std::domain_error when the volume is not finite or is negative
    [[nodiscard]] static double checked_volume(const Poly& poly);

    /// @brief Appends a small-volume event when volume <= SMALL (note_small).
    static void note_small(double volume, std::vector<DiagEvent>& events);

private:
    /// @brief Where a polyhedron lies with respect to a plane, by the vertex classes of clip().
    enum class Side { inside, outside, on, cut };

    /// @brief The side of a polyhedron (the same classification as clip, without building anything).
    [[nodiscard]] static Side side(const Poly& poly, const Plane& plane);

    /// @brief The vertex indices of a cap polygon sorted by angle about their mean (builder3d.order_cycle).
    [[nodiscard]] static std::vector<int> order_cycle(const std::vector<Vec3>& vertices, const std::vector<int>& indices, const Vec3& normal);
};

} // namespace fluidx3d::experimental::wall_builder

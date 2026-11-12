/// @file pass_runner.cpp
/// @brief Runs the passes on many threads; the results stay in the serial order of terrain_container.py run().

#include "pass_runner.hpp"

#include "parallel_for.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace fluidx3d::experimental::wall_builder {

// ------------------------------------------------------------------------------------------------ DiagTotals

void DiagTotals::add_unit(const std::span<const DiagEvent> events)
{
    const double before{ sum_defect };
    for (const DiagEvent& event : events) {
        if (event.is_defect) {
            max_defect = std::max(max_defect, event.value);
            sum_defect += event.value;
        } else {
            small_count += 1;
            small_total += event.value;
        }
    }
    max_unit_sum = std::max(max_unit_sum, sum_defect - before);
}

// ------------------------------------------------------------------------------------------------ PassRunner

PassRunner::PassRunner(const TerrainGeometry& geometry, const int threads)
    : geometry_{ geometry }
    , threads_{ threads }
{
    if (threads < 1) {
        throw std::invalid_argument("PassRunner: threads " + std::to_string(threads) + " (need 1 or more)");
    }
}

void PassRunner::run()
{
    const std::vector<Facet>& facets{ geometry_.facets() };
    const std::size_t total{ facets.size() };
    facets_.assign(total, {});
    std::atomic<std::size_t> done{ 0 };
    std::mutex lock{};
    const auto start{ std::chrono::steady_clock::now() };
    const std::size_t step{ total / 10 > 0 ? total / 10 : 1 };
    ParallelFor::run(total, threads_, [&](const std::size_t task) {
        const PassEngine engine{ geometry_ };
        facets_[task] = compact(engine.producer(facets[task]));
        const std::size_t finished{ done.fetch_add(1) + 1 };
        if (finished % step == 0 || finished == total) {
            const double seconds{ std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() };
            const std::lock_guard<std::mutex> guard{ lock };
            std::cout << "  producer passes " << finished << "/" << total << " " << static_cast<int>(seconds) << " s" << std::endl;
        }
    });
}

DiagTotals PassRunner::diag() const
{
    // a row without events adds nothing (its defect sum is 0.0 and max_unit_sum starts at 0.0), so the rows that compact()
    // left out do not change the totals
    DiagTotals totals{};
    for (const ProducerFacet& facet : facets_) {
        for (const ProducerRow& row : facet.rows) {
            const std::span<const DiagEvent> events{ facet.events.data() + row.e_begin, row.e_end - row.e_begin };
            totals.add_unit(events);
        }
    }
    return totals;
}

ProducerFacet PassRunner::compact(const std::vector<UnitResult>& results)
{
    ProducerFacet out{};
    for (std::size_t slot{ 0 }; slot < results.size(); ++slot) {
        const UnitResult& unit{ results[slot] };
        const bool empty{ unit.vi.empty() && unit.h.empty() && unit.w.empty() && unit.h_moment.empty() && unit.events.empty() };
        if (empty) {
            continue;
        }
        if (unit.vi.size() > 1) {
            throw std::logic_error("a producer row with more than one vi entry");
        }
        ProducerRow row{};
        row.slot = static_cast<int>(slot);
        row.has_vi = !unit.vi.empty();
        if (row.has_vi) {
            row.vi = unit.vi[0].value;
        }
        row.h_begin = static_cast<std::uint32_t>(out.h.size());
        for (const HEntry& e : unit.h) {
            out.h.push_back(HCell{ e.cell, e.value });
        }
        row.h_end = static_cast<std::uint32_t>(out.h.size());
        row.w_begin = static_cast<std::uint32_t>(out.w.size());
        for (const WEntry& e : unit.w) {
            out.w.push_back(WFacet{ e.gkey, e.value });
        }
        row.w_end = static_cast<std::uint32_t>(out.w.size());
        row.m_begin = static_cast<std::uint32_t>(out.moments.size());
        for (const HMomentEntry& e : unit.h_moment) {
            out.moments.push_back(HMoment{ e.cell, e.value });
        }
        row.m_end = static_cast<std::uint32_t>(out.moments.size());
        row.e_begin = static_cast<std::uint32_t>(out.events.size());
        out.events.insert(out.events.end(), unit.events.begin(), unit.events.end());
        row.e_end = static_cast<std::uint32_t>(out.events.size());
        out.rows.push_back(row);
    }
    out.rows.shrink_to_fit();
    out.h.shrink_to_fit();
    out.w.shrink_to_fit();
    out.moments.shrink_to_fit();
    out.events.shrink_to_fit();
    return out;
}

ProducerRowView PassRunner::row(const std::size_t facet, const int slot) const
{
    const ProducerFacet& f{ facets_[facet] };
    ProducerRowView view{};
    for (const ProducerRow& r : f.rows) {
        if (r.slot != slot) {
            continue;
        }
        view.exists = true;
        view.has_vi = r.has_vi;
        view.vi = r.vi;
        view.h = std::span<const HCell>{ f.h.data() + r.h_begin, r.h_end - r.h_begin };
        view.w = std::span<const WFacet>{ f.w.data() + r.w_begin, r.w_end - r.w_begin };
        view.moments = std::span<const HMoment>{ f.moments.data() + r.m_begin, r.m_end - r.m_begin };
        return view;
    }
    return view;
}

// ------------------------------------------------------------------------------------------------ DonorColumns

DonorColumns::DonorColumns(const TerrainGeometry& geometry, const int threads)
    : geometry_{ geometry }
    , threads_{ threads }
{
    if (threads < 1) {
        throw std::invalid_argument("DonorColumns: threads " + std::to_string(threads) + " (need 1 or more)");
    }
    columns_.resize(static_cast<std::size_t>(geometry.record().lx));
}

void DonorColumns::compute(const std::vector<int>& columns)
{
    const std::size_t q{ geometry_.record().directions.size() };
    // the cells of the new columns in one list, so the threads share the work of all of them
    std::vector<std::size_t> cells{};
    std::vector<DonorColumn*> owner{};
    for (const int i : columns) {
        std::unique_ptr<DonorColumn>& slot{ columns_[static_cast<std::size_t>(i)] };
        if (slot) {
            continue;
        }
        const std::size_t first{ geometry_.column_first(i) };
        const std::size_t last{ geometry_.column_first(i + 1) };
        slot = std::make_unique<DonorColumn>();
        slot->first = first;
        slot->states.assign((last - first) * q, DonorState{});
        slot->events.assign(last - first, {});
        slot->event_end.assign((last - first) * q, 0u);
        for (std::size_t n{ first }; n < last; ++n) {
            cells.push_back(n);
            owner.push_back(slot.get());
        }
    }
    ParallelFor::run(cells.size(), threads_, [&](const std::size_t task) {
        const std::size_t n{ cells[task] };
        DonorColumn& column{ *owner[task] };
        const std::size_t local{ n - column.first };
        const PassEngine engine{ geometry_ };
        const std::vector<UnitResult> results{ engine.donor(geometry_.donor_cell(n)) };
        std::vector<DiagEvent>& events{ column.events[local] };
        for (std::size_t d{ 0 }; d < q; ++d) {
            const UnitResult& unit{ results[d] };
            if (unit.direct.size() != 1) {
                throw std::logic_error("donor state without exactly one direct entry");
            }
            const DirectEntry& direct{ unit.direct[0] };
            column.states[local * q + d] = DonorState{ direct.value, direct.moment };
            events.insert(events.end(), unit.events.begin(), unit.events.end());
            column.event_end[local * q + d] = static_cast<std::uint32_t>(events.size());
        }
    });
}

const DonorColumn& DonorColumns::column(const int i) const
{
    const std::unique_ptr<DonorColumn>& slot{ columns_[static_cast<std::size_t>(i)] };
    if (!slot) {
        throw std::logic_error("DonorColumns: x-column " + std::to_string(i) + " is not computed");
    }
    return *slot;
}

void DonorColumns::drop(const int i)
{
    columns_[static_cast<std::size_t>(i)].reset();
}

std::size_t DonorColumns::bytes() const
{
    std::size_t total{ 0 };
    for (const std::unique_ptr<DonorColumn>& slot : columns_) {
        if (!slot) {
            continue;
        }
        total += slot->states.capacity() * sizeof(DonorState);
        total += slot->events.capacity() * sizeof(std::vector<DiagEvent>);
        total += slot->event_end.capacity() * sizeof(std::uint32_t);
        for (const std::vector<DiagEvent>& events : slot->events) {
            total += events.capacity() * sizeof(DiagEvent);
        }
    }
    return total;
}

} // namespace fluidx3d::experimental::wall_builder

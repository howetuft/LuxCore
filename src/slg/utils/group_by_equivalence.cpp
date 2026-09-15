/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#include "slg/utils/group_by_equivalence.h"

#include <ranges>
#include <unordered_map>
#include "oneapi/tbb.h"
#include "oneapi/tbb/cache_aligned_allocator.h"

namespace {

// This is the classical Union-Find algorithm,
// in a parallel implementation (tbb powered)
class alignas(std::hardware_destructive_interference_size) UnionFind {
public:
    UnionFind() {}
    explicit UnionFind(const size_t count) {
		reserve(count);
	}
	UnionFind(const UnionFind& other) :
		parent(other.parent, Allocator()),
		rank(other.rank, Allocator())
	{}
	UnionFind(UnionFind&& other) :
		parent(std::move(other.parent)),
		rank(std::move(other.rank))
	{}
	UnionFind& operator=(const UnionFind& other) {
		parent = other.parent;
		rank = other.rank;
		return (*this);
	}
	UnionFind& operator=(UnionFind&& other) {
		parent = std::move(other.parent);
		rank = std::move(other.rank);
		return (*this);
	}

    // Find the root of the set containing element i
    size_t find(const size_t i) {
        if (parent.find(i) == parent.end()) {
            parent[i] = i;
            rank[i] = 0;
        }
        if (parent[i] != i) {
            parent[i] = find(parent[i]); // Path compression
        }
        return parent[i];
    }

    // Union the sets containing elements i and j
    void unite(const size_t i, const size_t j) {
        const size_t rootI = find(i);
        const size_t rootJ = find(j);

        if (rootI != rootJ) {
            // Union by rank
            if (rank[rootI] > rank[rootJ]) {
                parent[rootJ] = rootI;
            } else if (rank[rootI] < rank[rootJ]) {
                parent[rootI] = rootJ;
            } else {
                parent[rootJ] = rootI;
                rank[rootI]++;
            }
        }
    }

	// Reserve space
	void reserve(const size_t count) {
		parent.reserve(count);
		rank.reserve(count);
	}

    // Overload the += operator to merge two UnionFind instances
    UnionFind operator+=(const UnionFind& other) {
        for (const auto& pair : other.parent) {
            unite(pair.first, pair.second);
        }
        return (*this);
    }

    size_t size() const {
		return parent.size();
	}

	// Find without compression
	size_t find_readonly(const size_t i) const {
		auto res = parent.find(i);
        if (res != parent.end()) {
			return res->second;
		} else {
			return i;
		}
	}

private:
	using Allocator = tbb::cache_aligned_allocator<std::pair<const size_t, size_t>>;
	using Hash = std::hash<size_t>;
	using Equal = std::equal_to<size_t>;
    std::unordered_map<size_t, size_t, Hash, Equal, Allocator> parent;
    std::unordered_map<size_t, size_t, Hash, Equal, Allocator> rank;

    friend std::ostream& operator<<(std::ostream& os, const UnionFind& uf);
};

std::ostream& operator<<(std::ostream& os, const UnionFind& uf) {
    os << "Parent: ";
    for (const auto& pair : uf.parent) {
        os << "(" << pair.first << ", " << pair.second << ") ";
    }
    os << "\nRank: ";
    for (const auto& pair : uf.rank) {
        os << "(" << pair.first << ", " << pair.second << ") ";
    }
    return os;
}

// Parallel GroupByEquivalence builder for use with tbb::parallel_reduce
// Processes equivalence relations in parallel using UnionFind
template<typename Range>
class ParallelGroupByEquivalence {
	UnionFind dsu;
	const size_t numPoints;
	Range relation;

public:
	// Constructor (plain)
	ParallelGroupByEquivalence(size_t p_numPoints, Range p_relation)
		: numPoints(p_numPoints),
		  relation(std::move(p_relation)),
		  dsu(p_numPoints)
	{}

	// Constructor (split)
	ParallelGroupByEquivalence(ParallelGroupByEquivalence& x, tbb::split)
		: numPoints(x.numPoints),
		  relation(x.relation),
		  dsu(x.numPoints)
	{}

	// Body: process a range of indices into the relation
	void operator()(const tbb::blocked_range<size_t>& r) {
		auto it = std::ranges::begin(relation);
		std::advance(it, r.begin());
		for (size_t i = r.begin(); i < r.end(); ++i, ++it) {
			const auto& [a, b] = *it;
			dsu.unite(a, b);
		}
	}

	// Reduction: merge two UnionFind instances
	void join(ParallelGroupByEquivalence& rhs) {
		if (dsu.size() < rhs.dsu.size()) {
			std::swap(dsu, rhs.dsu);
		}
		dsu += rhs.dsu;
	}

	// Access the resulting UnionFind
	UnionFind getResult() const {
		return dsu;
	}
};


// Parallel helper function in anonymous namespace that uses tbb::parallel_reduce
slg::Classes GroupByEquivalenceImpl(size_t numElements, auto&& relation) {
	slg::Classes clusters;
	std::unordered_map<size_t, std::vector<size_t>> clusterMap;

	// Use parallel_reduce with ParallelGroupByEquivalence
	static tbb::affinity_partitioner tbb_partitioner;

	// Determine grain size based on relation size
	const size_t relationSize = std::ranges::size(relation);
	constexpr size_t grain = 1024;

	ParallelGroupByEquivalence pqs(numElements, std::forward<decltype(relation)>(relation));

	tbb::parallel_reduce(
		tbb::blocked_range<size_t>(0, relationSize, grain),
		pqs,
		tbb_partitioner
	);

	UnionFind uf = pqs.getResult();

	// Create clusters from the UnionFind
	for (size_t i = 0; i < numElements; ++i) {
		const size_t root = uf.find(i);
		clusterMap[root].push_back(i);
	}

	clusters.reserve(clusterMap.size());
	for (auto& [root, indices] : clusterMap) {
		clusters.push_back(std::move(indices));
	}

	return clusters;
}


} // namespace slg

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

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

#ifndef SLG_UTILS_UNIONFIND_H
#define SLG_UTILS_UNIONFIND_H

#include <unordered_map>
#include "oneapi/tbb/cache_aligned_allocator.h"

namespace slg {

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
    u_int find(const u_int i) {
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
    void unite(const u_int i, const u_int j) {
        const u_int rootI = find(i);
        const u_int rootJ = find(j);

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
	u_int find_readonly(const u_int i) const {
		auto res = parent.find(i);
        if (res != parent.end()) {
			return res->second;
		} else {
			return i;
		}

	}

private:
	using Allocator = oneapi::tbb::cache_aligned_allocator<std::pair<const u_int, u_int>>;
	using Hash = std::hash<u_int>;
	using Equal = std::equal_to<u_int>;
    std::unordered_map<u_int, u_int, Hash, Equal, Allocator> parent;
    std::unordered_map<u_int, u_int, Hash, Equal, Allocator> rank;

    friend std::ostream& operator<<(std::ostream& os, const UnionFind& uf);
};

std::ostream& operator<<(std::ostream& os, const UnionFind& uf);

} // namespace slg

#endif // SLG_UTILS_UNIONFIND_H

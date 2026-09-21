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

#include <numeric>
#include <ranges>
#include <vector>
#include "oneapi/tbb.h"

namespace {

// Classical Union-Find (disjoint set union) on flat arrays: the elements
// form the dense range [0, numElements), so the parent and rank structures
// are plain vectors instead of hash maps.
class UnionFind {
	std::vector<size_t> parent;
	std::vector<size_t> rank;

public:
	UnionFind() = default;
	explicit UnionFind(const size_t count) : parent(count), rank(count, 0) {
		std::iota(parent.begin(), parent.end(), size_t(0));
	}

	// Find the root of the set containing i (path halving)
	size_t find(size_t i) {
		while (parent[i] != i) {
			parent[i] = parent[parent[i]];
			i = parent[i];
		}
		return i;
	}

	// Union the sets containing i and j (union by rank)
	void unite(const size_t i, const size_t j) {
		const size_t rootI = find(i);
		const size_t rootJ = find(j);

		if (rootI == rootJ)
			return;
		if (rank[rootI] < rank[rootJ])
			parent[rootI] = rootJ;
		else if (rank[rootI] > rank[rootJ])
			parent[rootJ] = rootI;
		else {
			parent[rootJ] = rootI;
			rank[rootI]++;
		}
	}

	// Merge another UnionFind into this one: every non-root element is
	// united with its parent, which merges all the components (the roots
	// are reached through their members)
	UnionFind& operator+=(const UnionFind& other) {
		for (size_t i = 0; i < parent.size(); ++i) {
			if (other.parent[i] != i)
				unite(i, other.parent[i]);
		}
		return *this;
	}
};

// Processes the equivalence relations in parallel: each thread unions its
// range of relations into its own UnionFind, merged by the reduce step.
// The UnionFind trees are valid at any time (only complete merges).
template<typename Range>
class ParallelGroupByEquivalence {
	UnionFind dsu;
	const size_t numPoints;
	Range relation;

public:
	ParallelGroupByEquivalence(size_t p_numPoints, Range p_relation)
		: dsu(p_numPoints), numPoints(p_numPoints), relation(std::move(p_relation)) {}

	ParallelGroupByEquivalence(ParallelGroupByEquivalence& x, tbb::split)
		: dsu(x.numPoints), numPoints(x.numPoints), relation(x.relation) {}

	// Body: process a range of indices into the relation
	void operator()(const tbb::blocked_range<size_t>& r) {
		auto it = std::ranges::begin(relation);
		std::advance(it, r.begin());
		for (size_t i = r.begin(); i < r.end(); ++i, ++it) {
			const auto& [a, b] = *it;
			dsu.unite(a, b);
		}
	}

	// Reduction: merge the sibling UnionFind into this one
	void join(ParallelGroupByEquivalence& rhs) {
		dsu += rhs.dsu;
	}

	UnionFind& getResult() {
		return dsu;
	}
};

// Helper class for building UnionFind from a callable generator
// using tbb::parallel_reduce
template<typename Generator>
class ParallelGroupByEquivalenceFromGenerator {
	const size_t numPoints;
	Generator relation;
	UnionFind dsu;

public:
	ParallelGroupByEquivalenceFromGenerator(size_t p_numPoints,
		Generator p_relation)
		: numPoints(p_numPoints),
		  relation(std::move(p_relation)),
		  dsu(p_numPoints) {}

	ParallelGroupByEquivalenceFromGenerator(
		ParallelGroupByEquivalenceFromGenerator& x, tbb::split)
		: numPoints(x.numPoints),
		  relation(x.relation),
		  dsu(x.numPoints) {}

	void operator()(const tbb::blocked_range<size_t>& r) {
		// Get pairs for this range
		auto pairsRange = relation(r.begin(), r.end());

		// Process each pair
		for (const auto& [i, j] : pairsRange) {
			dsu.unite(i, j);
		}
	}

	void join(ParallelGroupByEquivalenceFromGenerator<Generator>& rhs) {
		dsu += rhs.dsu;
	}

	UnionFind& getResult() {
		return dsu;
	}
};

// Build the classes from a final UnionFind: flatten the trees (single
// threaded: the union-find is not used concurrently anymore), then count
// the class sizes, prefix sum and fill (CSR): no hashing and no per class
// allocation until the final slicing. The elements end up in ascending
// order inside each class.
slg::Classes BuildClassesFromUnionFind(UnionFind& uf, const size_t numElements) {
	// Flatten all the trees to depth 1
	for (size_t i = 0; i < numElements; ++i)
		uf.find(i);

	// Count the class sizes
	std::vector<size_t> classStart(numElements + 1, 0);
	for (size_t i = 0; i < numElements; ++i)
		++classStart[uf.find(i) + 1];

	// Prefix sum
	for (size_t i = 0; i < numElements; ++i)
		classStart[i + 1] += classStart[i];

	// Fill the class entries
	std::vector<size_t> classEntries(numElements);
	{
		std::vector<size_t> classCursor(classStart.begin(), classStart.end() - 1);
		for (size_t i = 0; i < numElements; ++i)
			classEntries[classCursor[uf.find(i)]++] = i;
	}

	// Slice the entries into the output classes
	slg::Classes classes;
	for (size_t r = 0; r < numElements; ++r) {
		if (classStart[r + 1] > classStart[r])
			classes.emplace_back(
					std::make_move_iterator(classEntries.begin() + classStart[r]),
					std::make_move_iterator(classEntries.begin() + classStart[r + 1]));
	}

	return classes;
}

}  // namespace

namespace slg {

// Version for callable generators: relation is a functor that takes an
// interval [r1, r2) with r1 and r2 of size_t type and returns a vector of
// Relation. Functor is evaluated in multithreaded process, which may be more
// efficient than statically compute it beforehand
Classes GroupByEquivalence(size_t numElements, RelationFunction relation) {
	// Use parallel_reduce with ParallelGroupByEquivalenceFromGenerator
	static tbb::affinity_partitioner tbb_partitioner;
	constexpr size_t grain = 1024;

	ParallelGroupByEquivalenceFromGenerator<RelationFunction> solver(
		numElements, std::move(relation));

	tbb::parallel_reduce(
		tbb::blocked_range<size_t>(0, numElements, grain),
		solver,
		tbb_partitioner
	);

	return BuildClassesFromUnionFind(solver.getResult(), numElements);
}

// Version for direct ranges: relation is a span of Relation pairs
Classes GroupByEquivalence(size_t numElements, RelationSpan relation) {
	// Use parallel_reduce with ParallelGroupByEquivalence
	static tbb::affinity_partitioner tbb_partitioner;

	// Determine grain size based on relation size
	const size_t relationSize = relation.size();
	constexpr size_t grain = 1024;

	ParallelGroupByEquivalence<RelationSpan> solver(
		numElements, std::move(relation));

	tbb::parallel_reduce(
		tbb::blocked_range<size_t>(0, relationSize, grain),
		solver,
		tbb_partitioner
	);

	return BuildClassesFromUnionFind(solver.getResult(), numElements);
}


}  // namespace slg

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

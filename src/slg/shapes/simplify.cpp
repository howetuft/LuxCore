/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
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

#include <map>
#include <vector>
#include <string>
#include <queue>
#include <limits>
#include <tuple>
#include <ranges>
#include <unordered_set>
#include <algorithm>
#include <format>
#include <random>
#include <execution>

#include <tbb/mutex.h>
#include <tbb/concurrent_priority_queue.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/concurrent_unordered_set.h>

#include "luxrays/core/exttrianglemesh.h"
#include "slg/shapes/simplify.h"
#include "slg/scene/scene.h"
#include "slg/utils/harlequincolors.h"
#include "dset.h"

//using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
//
// The following code is based on Sven Forstmann's quadric mesh simplification
// code (https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification)
// and heavily modified for LuxCoreRender
//
// Papers at https://mgarland.org/research/quadrics.html

/////////////////////////////////////////////
//
// Mesh Simplification Tutorial
//
// (C) by Sven Forstmann in 2014
//
// License : MIT
// http://opensource.org/licenses/MIT
//
// https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
//
// 5/2016: Chris Rorden created minimal version for OSX/Linux/Windows compile

constexpr float FLOAT_INFINITY = std::numeric_limits<float>::infinity();

constexpr std::array<std::tuple<u_int, u_int>, 3> EDGES({ {0, 1}, {1, 2}, {2, 0}, });

// Enumerate helper (à la 'Python enumerate')
template <typename T,
          typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T && iterable) {
    struct iterator {
        size_t i;
        TIter iter;
        bool operator != (const iterator & other) const { return iter != other.iter; }
        void operator ++ () { ++i; ++iter; }
        auto operator * () const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper {
        T iterable;
        auto begin() { return iterator{ 0, std::begin(iterable) }; }
        auto end() { return iterator{ 0, std::end(iterable) }; }
    };
    return iterable_wrapper{ std::forward<T>(iterable) };
}

// TODO use SIMD
class SymetricMatrix {
public:
	// Constructor
	SymetricMatrix(const float c = 0.f) {
		for (u_int i = 0; i < 10; ++i)
			m[i] = c;
	}

	SymetricMatrix(
			const float m11, const float m12, const float m13, const float m14,
			const float m22, const float m23, const float m24,
			const float m33, const float m34,
			const float m44) {
		m[0] = m11;
		m[1] = m12;
		m[2] = m13;
		m[3] = m14;
		m[4] = m22;
		m[5] = m23;
		m[6] = m24;
		m[7] = m33;
		m[8] = m34;
		m[9] = m44;
	}

	// Make plane
	SymetricMatrix(const float a, const float b, const float c, const float d) {
		m[0] = a * a;
		m[1] = a * b;
		m[2] = a * c;
		m[3] = a * d;
		m[4] = b * b;
		m[5] = b * c;
		m[6] = b * d;
		m[7] = c * c;
		m[8] = c * d;
		m[9] = d * d;
	}

	float operator[](size_t c) const {
		return m[c];
	}

	// Determinant
	float det(
			const u_int a11, const u_int a12, const u_int a13,
			const u_int a21, const u_int a22, const u_int a23,
			const u_int a31, const u_int a32, const u_int a33) const {
		const float det = m[a11] * m[a22] * m[a33] + m[a13] * m[a21] * m[a32] + m[a12] * m[a23] * m[a31]
				- m[a13] * m[a22] * m[a31] - m[a11] * m[a23] * m[a32] - m[a12] * m[a21] * m[a33];
		return det;
	}

	const SymetricMatrix operator+(const SymetricMatrix &n) const {
		return SymetricMatrix(
				m[0] + n[0], m[1] + n[1], m[2] + n[2], m[3] + n[3],
				m[4] + n[4], m[5] + n[5], m[6] + n[6],
				m[7] + n[7], m[8] + n[8],
				m[9] + n[9]);
	}

	SymetricMatrix& operator+=(const SymetricMatrix& n) {
		m[0] += n[0];
		m[1] += n[1];
		m[2] += n[2];
		m[3] += n[3];
		m[4] += n[4];
		m[5] += n[5];
		m[6] += n[6];
		m[7] += n[7];
		m[8] += n[8];
		m[9] += n[9];

		return *this;
	}

	float m[10];
};


// Hash function for Point
inline void hash_combine(std::size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hash_combine(seed, rest...);
}

template<>
struct std::hash<luxrays::Point> {
    std::size_t operator()(const luxrays::Point& p) const noexcept
    {
		std::size_t h=0;
		hash_combine(h, p.x, p.y, p.z);
        return h;
    }
};

struct SimplifyRef {
	u_int tid = 0;
	u_int tvertex = std::numeric_limits<u_int>::infinity();

	SimplifyRef(u_int p_tid, u_int p_tvertex): tid(p_tid), tvertex(p_tvertex) {}
	SimplifyRef() {}
};

using RefVector = std::vector< SimplifyRef, tbb::cache_aligned_allocator<SimplifyRef> >;

using BatchVector = std::vector<RefVector>;

struct SimplifyVertex {
	// Core data
	luxrays::Point p;              // Position
	tbb::concurrent_vector<SimplifyRef, tbb::cache_aligned_allocator<SimplifyRef>> refs;       // Incident edges in topology
	SymetricMatrix q;     // Quadric
	bool border;          // Border status

	// Compacting data
	// TODO used only in CompactMesh -> Move there
	bool keep = false;
	size_t newIndex = 0;

	// LuxCore specific data
	Normal norm;
	UV uv;
	Spectrum col;
	float alpha;

};

using VertexVector = std::vector<
	SimplifyVertex,
	tbb::cache_aligned_allocator<SimplifyVertex>
>;

struct SimplifyTriangle {
	std::array<u_int, 3> v;  // Vertex indices
	luxrays::Normal geometryN;
	std::array<float, 3> err;
	bool deleted = false;
	bool dirty = false;

	// Cached values to limit false sharing
	std::array<luxrays::Point, 3> cachedVerts;

	// Update error of the triangle
	void UpdateTriangleError(
		const VertexVector& vertices,
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	);
};

using TriangleVector = std::vector<
	SimplifyTriangle,
	tbb::cache_aligned_allocator<SimplifyTriangle>
>;

// Error between vertex and Quadric
float VertexError(
	const SymetricMatrix &q,
	const float x,
	const float y,
	const float z
) {
	return  q[0] * x * x
			+ 2.f * q[1] * x * y
			+ 2.f * q[2] * x * z
			+ 2.f * q[3] * x
			+ q[4] * y * y
			+ 2.f * q[5] * y * z
			+ 2.f * q[6] * y
			+ q[7] * z * z
			+ 2.f * q[8] * z
			+ q[9];
}


// Error for one edge
// Returns: error, interpolated point
std::tuple<float, luxrays::Point> CalculateCollapseError(
	const SimplifyVertex& v0,
	const SimplifyVertex& v1,
	const bool preserveBorder
) {

	const SymetricMatrix q = v0.q + v1.q;

	luxrays::Point pResult;

	// Compute interpolated vertex
	const luxrays::Point &p1 = v0.p;
	const luxrays::Point &p2 = v1.p;
	const luxrays::Point p3 = (p1 + p2) / 2;

	// Error can be negative, I add 1 to have screenErrorScale can than
	// work as expected
	const float error1 = VertexError(q, p1.x, p1.y, p1.z) + 1.f;
	const float error2 = VertexError(q, p2.x, p2.y, p2.z) + 1.f;
	const float error3 = VertexError(q, p3.x, p3.y, p3.z) + 1.f;

	float error;
	if (preserveBorder && v0.border) {
		error = error1;
		pResult = p1;
	} else if (preserveBorder && v1.border) {
		error = error2;
		pResult = p2;
	} else {
		error = std::min({error1, error2, error3});

		if (error1 == error)
			pResult = p1;
		if (error2 == error)
			pResult = p2;
		if (error3 == error)
			pResult = p3;
	}

	// Adding 1.0 because error have negative values
	error = std::max(error + 1.f, 0.f);
	return std::tuple(error , std::move(pResult));
}

float CalculateCollapseScreenErrorScale(
	const SimplifyVertex& v0,
	const SimplifyVertex& v1,
	float edgeScreenSize,
	const Camera& camera
) {
	if (edgeScreenSize > 0.f) {
		const Point& p0 = v0.p;
		const Point& p1 = v1.p;
		const float notVisibleScale = .5f;

		float p0x, p0y;
		if (!camera.GetSamplePosition(p0, &p0x, &p0y) ||
				!IsValid(p0x) || !IsValid(p0y)) {
			return notVisibleScale;
		}

		// Normalize
		p0x /= camera.filmWidth;
		p0y /= camera.filmHeight;

		float p1x, p1y;
		if (!camera.GetSamplePosition(p1, &p1x, &p1y) ||
				!IsValid(p1x) || !IsValid(p1y)) {
			return notVisibleScale;
		}

		// Normalize
		p1x /= camera.filmWidth;
		p1y /= camera.filmHeight;

		const float edge = sqrtf(Sqr(p0x - p1x) + Sqr(p0y - p1y));
		if (edge == 0.f) {
			return notVisibleScale;
		}

		return Max(edge / edgeScreenSize, notVisibleScale);
	} else {
		return 1.f;
	}
}


class Simplify {
public:
	// TODO Parallelize constructor
	Simplify(const ExtTriangleMesh &srcMesh) {
		const u_int vertCount = srcMesh.GetTotalVertexCount();
		const u_int triCount = srcMesh.GetTotalTriangleCount();
		const Point *verts = srcMesh.GetVertices();
		const Triangle *tris = srcMesh.GetTriangles();

		vertices.resize(srcMesh.GetTotalVertexCount());

		for (u_int i = 0; i < vertCount; ++i) {
			vertices[i].p = verts[i];
			vertices[i].refs.reserve(9);  // Seems reasonable
		}

		if (srcMesh.HasNormals()) {
			const Normal *norms = srcMesh.GetNormals();
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].norm = norms[i];

			hasNormals = true;
		} else {
			hasNormals = false;
		}

		if (srcMesh.HasUVs(0)) {
			const UV *uvs = srcMesh.GetUVs(0);
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].uv = uvs[i];

			hasUVs = true;
		} else {
			hasUVs = false;
		}

		if (srcMesh.HasColors(0)) {
			const Spectrum *cols = srcMesh.GetColors(0);
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].col = cols[i];

			hasColors = true;
		} else
			hasColors = false;

		if (srcMesh.HasAlphas(0)) {
			const float *alphas = srcMesh.GetAlphas(0);
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].alpha = alphas[i];

			hasAlphas = true;
		} else
			hasAlphas = false;

		triangles.resize(triCount);
		for (u_int i = 0; i < triCount; ++i) {
			triangles[i].v[0] = tris[i].v[0];
			triangles[i].v[1] = tris[i].v[1];
			triangles[i].v[2] = tris[i].v[2];
		}
	}

	ExtTriangleMesh *GetExtMesh() const {
		const u_int vertCount = vertices.size();
		const u_int triCount = triangles.size();

		Point *newVertices = ExtTriangleMesh::AllocVerticesBuffer(vertCount);
		for (u_int i = 0; i < vertCount; ++i)
			newVertices[i] = vertices[i].p;

		Normal *newNorms = nullptr;
		if (hasNormals) {
			newNorms = new Normal[vertCount];
			for (u_int i = 0; i < vertCount; ++i)
				newNorms[i] = vertices[i].norm;
		}

		UV *newUVs = nullptr;
		if (hasUVs) {
			newUVs = new UV[vertCount];
			for (u_int i = 0; i < vertCount; ++i)
				newUVs[i] = vertices[i].uv;
		}

		Spectrum *newCols = nullptr;
		if (hasColors) {
			newCols = new Spectrum[vertCount];
			for (u_int i = 0; i < vertCount; ++i)
				newCols[i] = vertices[i].col;
		}

		float *newAlphas = nullptr;
		if (hasAlphas) {
			newAlphas = new float[vertCount];
			for (u_int i = 0; i < vertCount; ++i)
				newAlphas[i] = vertices[i].alpha;
		}

		Triangle *newTris = ExtTriangleMesh::AllocTrianglesBuffer(triCount);
		for (u_int i = 0; i < triCount; ++i) {
			assert (triangles[i].v[0] < vertCount);
			newTris[i].v[0] = triangles[i].v[0];

			assert (triangles[i].v[1] < vertCount);
			newTris[i].v[1] = triangles[i].v[1];

			assert (triangles[i].v[2] < vertCount);
			newTris[i].v[2] = triangles[i].v[2];
		}

		return new ExtTriangleMesh(vertCount, triCount, newVertices, newTris, newNorms,
				newUVs, newCols, newAlphas);
	}

	// Partition candidate list into components (aka "batches")
	//
	// We use parallelized connected components algorithm
	//
	//
	BatchVector
	PartitionIndependentEdgeBatches(const RefVector& candidateList) const {
		// Step 1: Build closure set (= candidate vertices + their neighborhoods)
		tbb::enumerable_thread_specific<std::unordered_set<u_int>> closures;
		tbb::blocked_range<u_int> candidate_range(0, candidateList.size());
		auto closure_task = [&](const decltype(candidate_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				auto& c = candidateList[i];
				const auto& t = triangles[c.tid];
				const u_int i0 = t.v[c.tvertex];
				const u_int i1 = t.v[(c.tvertex + 1) % 3];
				auto neighbors = edgeNeighbors(i0, i1);
				closures.local().insert(neighbors.begin(), neighbors.end());
			}
		};
		tbb::parallel_for(candidate_range, closure_task);
		std::unordered_set<u_int> closure;
		for (auto& local: closures) {
			closure.merge(local);
		}

		// Step 2: Build connected components (union-find)
		DisjointSets unionFind(vertices.size());
		// TODO
		tbb::parallel_for(
			tbb::blocked_range<u_int>(0, triangles.size()),
			[&](const tbb::blocked_range<u_int>& r) {
				for (u_int i = r.begin(); i != r.end(); ++i) {
					const auto& t = triangles[i];
					for (const auto e: EDGES) {
						u_int i0 = t.v[std::get<0>(e)];
						u_int i1 = t.v[std::get<1>(e)];
						if (closure.contains(i0) and closure.contains(i1)) {
							unionFind.unite(i0, i1);
						}
					}
				}
			}
		);

		// Build batches (sequential)
		std::unordered_map<u_int, RefVector> batches;
		for (const auto& c: candidateList) {
			u_int i = triangles[c.tid].v[c.tvertex];  // Vertex index
			u_int batchIndex = unionFind.find(i);
			batches[batchIndex].push_back(c);
		}
		BatchVector res;
		res.reserve(batches.size());
		for (const auto& [i, batch]: enumerate(batches)) {
			res.push_back(batch.second);
		}

		return res;
	}

	u_int DeleteTriangles(
		const BatchVector& batches,
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	) {
		tbb::enumerable_thread_specific<u_int> batchDeleted;
		tbb::blocked_range<u_int> batch_range(0, batches.size());
		auto delete_task = [&](const tbb::blocked_range<u_int>& r) {
			for (u_int i = r.begin(); i != r.end(); ++i) {
				for (auto& ref : batches[i]) {
					batchDeleted.local() += CollapseEdge(
						ref, edgeScreenSize, camera, preserveBorder
					);
				}
			}
		};
		tbb::parallel_for(batch_range, delete_task);

		u_int deletedTriangles = std::accumulate(
			batchDeleted.begin(),
			batchDeleted.end(),
			0,
			std::plus<u_int>()
		);
		return deletedTriangles;
	}

	void Decimate(
		const u_int targetTriangleCount,
		const Camera& camera,
		const float edgeScreenSize,
		const bool preserveBorder
	) {
		// Work on 10% of all triangles for each iteration
		u_int maxCandidateQueueSize = std::max(64u, Floor2UInt(triangles.size() * .1f));

		// Main iteration loop
		const u_int startTriangleCount = triangles.size();
		u_int deletedTriangles = 0;
		for (u_int iteration = 0; iteration < 64; ++iteration) {

			if (startTriangleCount - deletedTriangles <= targetTriangleCount) break;

			SDL_LOG("Simplify - Start iteration #" << iteration);


			// Compute iteration data (including mesh topology)
			SDL_LOG("Simplify - Initialize data #" << iteration);
			InitIteration(iteration, edgeScreenSize, camera, preserveBorder);

			// Build candidate list
			SDL_LOG("Simplify - Build candidate list #" << iteration);
			auto candidateList = BuildCandidateList(
				preserveBorder, maxCandidateQueueSize
			);

			// Partition candidates into independent batches
			SDL_LOG("Simplify - Partition candidate list #" << iteration);
			auto batches = PartitionIndependentEdgeBatches(candidateList);

			// Delete triangles (run batches)
			SDL_LOG(
				"Simplify - Delete triangles ("
				<< batches.size() << " batches)"
				<< " #" << iteration
				);
			const u_int iterationDeletedTriangles = DeleteTriangles(
				batches, edgeScreenSize, camera, preserveBorder
			);

			deletedTriangles += iterationDeletedTriangles;

			SDL_LOG(
				"Simplify - End iteration #" << iteration
				<< " (" << candidateList.size()
				<< " edge candidates, deleted "
				<< iterationDeletedTriangles
				<< " triangles in current iteration, "
				<< deletedTriangles
				<< " of " << startTriangleCount << " triangles"
				<< " in all iterations)"
			);

			// No more work?
			if (!iterationDeletedTriangles) break;

			//std::exit(0);  // DEBUG - Stop here
		}

		// Clean up mesh
		CompactMesh();
		SDL_LOG("Simplify - Mesh compacted");

	//std::exit(0);  // DEBUG - Stop here
	}

private:

	VertexVector vertices;
	TriangleVector triangles;

	bool hasNormals, hasUVs, hasColors, hasAlphas;

	// Neighbor features
	using NeighborSet = std::unordered_set<u_int>;


	// Find edge neighbors, ie vertices that could be affected
	// by collapsing the given edge
	NeighborSet edgeNeighbors(const u_int i0, const u_int i1) const {
		NeighborSet neighbors;

		neighbors.insert(i0);
		neighbors.insert(i1);
		for (const auto& ref: vertices[i0].refs) {
			for (u_int vertexIndex: triangles[ref.tid].v) {
				neighbors.insert(vertexIndex);
			}
		}
		for (const auto& ref: vertices[i1].refs) {
			for (u_int vertexIndex: triangles[ref.tid].v) {
				neighbors.insert(vertexIndex);
			}
		}
		return neighbors;
	}

	// Set intersection
	static bool disjoint(const NeighborSet& p_s0, const NeighborSet& p_s1) {
		bool order = (p_s0.size() <= p_s1.size());
		const NeighborSet& s0 = order ? p_s0 : p_s1;
		const NeighborSet& s1 = order ? p_s1 : p_s0;
		for (auto i: s0) {
			if (s1.count(i)) {
				return false;
			}
		}
		return true;
	}

	static bool connected(const NeighborSet& p_s0, const NeighborSet& p_s1) {
		return not disjoint(p_s0, p_s1);
	}

	// Collapse an edge
	// Returns: number of deleted triangles
	// Modifies: triangles, vertices
	u_int CollapseEdge(
		const SimplifyRef& vertex,  /* Candidate vertex to collapse */
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	) {
		// Check triangle
		const u_int triangleIndex = vertex.tid;
		SimplifyTriangle &t = triangles[triangleIndex];

		if (t.deleted)
			return 0;
		if (t.dirty)
			return 0;

		// Get explicit edge to collapse
		auto [e1, e2] = EDGES[vertex.tvertex];
		const u_int i0 = t.v[e1];
		const u_int i1 = t.v[e2];

		// Prepare shortcuts
		SimplifyVertex &v0 = vertices[i0];
		SimplifyVertex &v1 = vertices[i1];

		u_int deletedTriangles = 0;

		// Border check
		if (v0.border != v1.border) {
			return 0;
		}

		// Compute vertex to collapse to
		// Doesn't modify state (neither vertices nor triangles)
		const auto [error, p] = CalculateCollapseError(v0, v1, preserveBorder);

		// Do not collapse edge if it makes a face flip
		// deleted0, deleted1: true/false if the triangles referencing the
		// vertex are deleted
		auto [will_flip0, deleted0] = Flipped<true>(p, i0, i1);
		auto [will_flip1, deleted1] = Flipped<true>(p, i1, i0);
		if (will_flip0 || will_flip1) {
			return 0;
		}


		// At this stage, no triangle flip is to fear anymore,
		// so we can collapse edge


		// Compute new position
		v0.p = p;
		v0.q = v1.q + v0.q;

		// Interpolate other vertex attributes
		const auto& tv0 = vertices[t.v[0]];
		const auto& tv1 = vertices[t.v[1]];
		const auto& tv2 = vertices[t.v[2]];
		const luxrays::Point& triPoint0 = tv0.p;
		const luxrays::Point& triPoint1 = tv1.p;
		const luxrays::Point& triPoint2 = tv2.p;
		float b1, b2;
		if (Triangle::GetBaryCoords(
				triPoint0,
				triPoint1,
				triPoint2,
				p, &b1, &b2)) {
			const float b0 = 1.f - b1 - b2;

			if (hasNormals) {
				const Normal triNorm0 = tv0.norm;
				const Normal triNorm1 = tv1.norm;
				const Normal triNorm2 = tv2.norm;
				v0.norm = Normalize(b0 * triNorm0 + b1 * triNorm1 + b2 * triNorm2);
			}
			if (hasUVs) {
				const UV triUV0 = tv0.uv;
				const UV triUV1 = tv1.uv;
				const UV triUV2 = tv2.uv;
				v0.uv = b0 * triUV0 + b1 * triUV1 + b2 * triUV2;
			}
			if (hasColors) {
				const Spectrum triCol0 = tv0.col;
				const Spectrum triCol1 = tv1.col;
				const Spectrum triCol2 = tv2.col;
				v0.col = b0 * triCol0 + b1 * triCol1 + b2 * triCol2;
			}
			if (hasAlphas) {
				const float triAlpha0 = tv0.alpha;
				const float triAlpha1 = tv1.alpha;
				const float triAlpha2 = tv2.alpha;
				v0.alpha = b0 * triAlpha0 + b1 * triAlpha1 + b2 * triAlpha2;
			}
		} else {
			// Must be a malformed triangle
			if (hasNormals) {
				const Normal triNorm0 = tv0.norm;
				v0.norm = triNorm0;
			}
			if (hasUVs) {
				const UV triUV0 = tv0.uv;
				v0.uv = triUV0;
			}
			if (hasColors) {
				const Spectrum triCol0 = tv0.col;
				v0.col = triCol0;
			}
			if (hasAlphas) {
				const float triAlpha0 = tv0.alpha;
				v0.alpha = triAlpha0;
			}
		}

		auto [newRefs0, deletedTriangles0] =
			UpdateTriangles(i0, v0, deleted0, edgeScreenSize, camera, preserveBorder);
		auto [newRefs1, deletedTriangles1] =
			UpdateTriangles(i0, v1, deleted1, edgeScreenSize, camera, preserveBorder);

		// Update incident edges of vertex
		auto& refs = v0.refs;
		refs.clear();
		refs.grow_by(newRefs0.begin(), newRefs0.end());
		refs.grow_by(newRefs1.begin(), newRefs1.end());
		deletedTriangles = deletedTriangles0 + deletedTriangles1;

		return deletedTriangles;
	}

	// Check if a triangle flips when this edge is removed
	// Returns: check status, deleted status of each ref (vector)
	using FlippedFullReturn = std::tuple<bool, std::vector<bool>>;

	template<bool F=true>
	constexpr std::conditional<F, FlippedFullReturn, bool>::type
	Flipped(const luxrays::Point p, const u_int i0, const u_int i1) const {

		const SimplifyVertex &v0 = vertices[i0];
		std::vector<bool> deleted;

		if constexpr(F) {
			deleted.resize(v0.refs.size());
		}

		for (size_t k = 0; k < v0.refs.size(); ++k) {
			auto& ref = v0.refs[k];
			const SimplifyTriangle &t = triangles[ref.tid];

			if (t.deleted) continue;

			const u_int s = ref.tvertex;
			const u_int id1 = t.v[(s + 1) % 3];
			const u_int id2 = t.v[(s + 2) % 3];

			// Delete ?
			if (id1 == i1 || id2 == i1) {
				if constexpr(F) {
					deleted[k] = true;
				}
				continue;
			}

			// Check if the triangle is too narrow
			// (avoiding sqrt function)
			const luxrays::Vector d1 = vertices[id1].p - p;
			const luxrays::Vector d2 = vertices[id2].p - p;
			const float sqrlen1 = d1.LengthSquared();
			const float sqrlen2 = d2.LengthSquared();

			const float dot = Dot(d1, d2);
			const float sqrdot = dot * dot;
			constexpr float sqrthreshold = .999f * .999f;
			if (sqrdot > sqrthreshold * sqrlen1 * sqrlen2) {
				if constexpr (F) {
					return FlippedFullReturn(true, deleted);
				} else {
					return true;
				}
			}

			// Check if the Normal is changing side
			// (avoiding sqrt function)
			luxrays::Vector rawnormal = Cross(d1, d2);
			const float sqrlen_rawnormal = rawnormal.LengthSquared();
			const float normdot = Dot(rawnormal, t.geometryN);
			const float sqrnormdot = normdot * normdot;
			constexpr float sqrthreshold2 = .2f * .2f;
			if  (std::signbit(normdot)  or sqrnormdot < sqrthreshold2 * sqrlen_rawnormal) {
				if constexpr(F) {
					return FlippedFullReturn(true, deleted);
				} else {
					return true;
				}
			}

			if constexpr(F) {
				deleted[k] = false;
			}
		}

		if constexpr(F) {
			return FlippedFullReturn(false, deleted);
		} else {
			return false;
		}
	}

	// Update triangle connections and edge error after a edge is collapsed
	// Returns: new incident edges list, number of deleted triangles
	std::tuple<RefVector, u_int> UpdateTriangles(
		const u_int i0,
		const SimplifyVertex &v,  // Collapsed vertex
		const std::vector<bool> &deleted,
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	) {
		u_int deletedTriangles = 0;
		RefVector refs;
		refs.reserve(v.refs.size());
		for (const auto& [k, r]: enumerate(v.refs)) {
			SimplifyTriangle &t = triangles[r.tid];

			if (t.deleted)
				continue;

			if (deleted[k]) {
				t.deleted = true;
				deletedTriangles++;
				continue;
			}

			t.v[r.tvertex] = i0;
			t.dirty = true;
			t.UpdateTriangleError(vertices, edgeScreenSize, camera, preserveBorder);

			refs.push_back(r);
		}
		return std::tuple(refs, deletedTriangles);
	}

	// Initialize quadrics on vertices
	//
	// Modify triangles and vertices (q values)
	void InitQuadrics(
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	) {
		// Starting values
		for (u_int i = 0; i < vertices.size(); ++i)
			vertices[i].q = SymetricMatrix(0.0);

		// Parallel computation of per-triangle quadric contributions
		std::vector<std::array<SymetricMatrix, 3>> triangleQuadrics(triangles.size());

		// TODO
		tbb::parallel_for(
			tbb::blocked_range<u_int>(0, triangles.size()),
			[&](const tbb::blocked_range<u_int>& r) {
				for (u_int i = r.begin(); i != r.end(); ++i) {
					SimplifyTriangle &t = triangles[i];

					SimplifyVertex &v0 = vertices[t.v[0]];
					SimplifyVertex &v1 = vertices[t.v[1]];
					SimplifyVertex &v2 = vertices[t.v[2]];

					const Normal geometryN(Normalize(Cross(v1.p - v0.p, v2.p - v0.p)));
					t.geometryN = geometryN;

					const SymetricMatrix sm(geometryN.x, geometryN.y, geometryN.z,
										   -Dot(Vector(geometryN), Vector(v0.p)));

					triangleQuadrics[i][0] = sm;
					triangleQuadrics[i][1] = sm;
					triangleQuadrics[i][2] = sm;
				}
			}
		);

		// Accumulation into vertex quadrics
		std::vector<tbb::mutex> v_mtx(vertices.size());
		tbb::blocked_range<u_int> tri_range(0, triangles.size());
		auto acc_task = [&](decltype(tri_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				const auto &t = triangles[i];
				for (u_int j = 0; j < 3; ++j) {
					auto vertex_index = t.v[j];
					tbb::mutex::scoped_lock lock(v_mtx[vertex_index]);
					vertices[vertex_index].q += triangleQuadrics[i][j];
				}
			}
		};
		tbb::parallel_for(tri_range, acc_task);

		// Triangle error update
		auto update_task = [&](decltype(tri_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				auto &t = triangles[i];
				t.UpdateTriangleError(vertices, edgeScreenSize, camera, preserveBorder);
			}
		};
		tbb::parallel_for(tri_range, update_task);
	}

	// Build incident edge tables on vertices
	//
	// Modify: vertices
	void InitIncidentEdges() {

		// Clear previous data
		for (auto& v: vertices) {
			v.refs.clear();
		}

		// Build incident map
		struct IncidentTask {
			TriangleVector& triangles;
			VertexVector& vertices;

			IncidentTask(
				TriangleVector& p_triangles,
				VertexVector& p_vertices
			) :
				triangles(p_triangles),
				vertices(p_vertices)
			{}

			IncidentTask(const IncidentTask&) = default;

			void operator()(const tbb::blocked_range<size_t>& r) const {
				for (auto i = r.begin(); i != r.end(); ++i) {
					triangles[i].dirty = false;  // Clear triangle dirty flags, by the way
					const auto& v = triangles[i].v;
					vertices[v[0]].refs.push_back(SimplifyRef(i, 0));
					vertices[v[1]].refs.push_back(SimplifyRef(i, 1));
					vertices[v[2]].refs.push_back(SimplifyRef(i, 2));
				}
			}
		};

		IncidentTask task(triangles, vertices);
		auto range = tbb::blocked_range<size_t>(0, triangles.size());

		tbb::parallel_for(range, task);
	}


	// Init border indicators on vertices
	//
	// Modify: vertices
	// TODO Parallelize
	void InitBorders() {
		// Set borders to false
		for (auto& v: vertices)
			v.border = false;

		std::vector<u_int> vcount, vids;
		for (const auto& v: vertices) {
			vcount.clear();
			vids.clear();

			for (const auto& ref: v.refs) {
				auto k = ref.tid;
				SimplifyTriangle &t = triangles[k];

				for (u_int k = 0; k < 3; ++k) {
					u_int ofs = 0;
					u_int id = t.v[k];

					while (ofs < vcount.size()) {
						if (vids[ofs] == id)
							break;

						ofs++;
					}

					if (ofs == vcount.size()) {
						vcount.push_back(1);
						vids.push_back(id);
					} else
						vcount[ofs]++;
				}
			}

			for (u_int j = 0; j < vcount.size(); ++j) {
				if (vcount[j] == 1)
					vertices[vids[j]].border = true;
			}
		}
	}

	// Build candidate list
	//
	RefVector BuildCandidateList(
		bool preserveBorder,
		u_int maxCandidateQueueSize
	) const {

		// Ref comparison predicate
		auto refErrorCompare = [&](const SimplifyRef& left, const SimplifyRef& right) {
			const auto left_error = triangles[left.tid].err[left.tvertex];
			const auto right_error = triangles[right.tid].err[right.tvertex];
			return  left_error > right_error;
		};

		tbb::concurrent_priority_queue<SimplifyRef, decltype(refErrorCompare) >
			candidateQueue(vertices.size(), refErrorCompare);
		tbb::blocked_range<u_int> tri_range(0, triangles.size());
		tbb::auto_partitioner partitioner;

		auto build_task = [&](const decltype(tri_range)& r) {
			// Main loop
			for (u_int i = r.begin(); i != r.end(); ++i) {
				const SimplifyTriangle &t = triangles[i];
				auto tv0 = t.v[0];
				auto tv1 = t.v[1];
				auto tv2 = t.v[2];
				const std::array<std::tuple<u_int, u_int>, 3> edges(
					{
						{tv0, tv1},
						{tv1, tv2},
						{tv2, tv0},
					}
				);

				u_int minErrorIndex = NULL_INDEX;
				float minError = FLOAT_INFINITY;
				for (u_int j = 0; j < 3; ++j) {
					const auto [i0, i1] = edges[j];
					const SimplifyVertex &v0 = vertices[i0];
					const SimplifyVertex &v1 = vertices[i1];

					// Border check
					if (preserveBorder) {
						if (v0.border && v1.border)
							continue;
					} else {
						if (v0.border != v1.border)
							continue;
					}

					auto [error, p] = CalculateCollapseError(v0, v1, preserveBorder);
					if (Flipped<false>(p, i0, i1))
						continue;
					if (Flipped<false>(p, i1, i0))
						continue;

					if (t.err[j] < minError) {
						minErrorIndex = j;
						minError = t.err[j];
					}
				}
				if (minErrorIndex != NULL_INDEX) {
					candidateQueue.emplace(i, minErrorIndex);
				}
			}

		};
		tbb::parallel_for(tri_range, build_task, partitioner);

		// Assemble result
		size_t numCandidates = std::min(
			candidateQueue.size(),
			size_t(maxCandidateQueueSize)
		);
		RefVector candidateList(numCandidates);
		for (size_t i = 0; i < numCandidates; ++i) {
			candidateQueue.try_pop(candidateList[i]);
		}

		return candidateList;
	}

	// Compact triangles, compute quadrics, incident, boundary, edge error
	// Returns: candidate list
	void InitIteration(
		const u_int iteration,
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	) {
		if (iteration > 0) {
			// Compact triangles
			decltype(triangles) newTris;
			newTris.reserve(triangles.size());
			auto not_deleted = [](const SimplifyTriangle& t){return !t.deleted;};
			std::ranges::copy_if(triangles, std::back_inserter(newTris), not_deleted);
			std::swap(triangles, newTris);
		}

		// Build per-vertex incident edge tables
		//
		InitIncidentEdges();

		// Init Quadrics by Plane & Edge Errors
		//
		// Identify boundary : vertices[].border=0,1
		//
		// Required at the beginning (iteration == 0)
		//
		if (iteration == 0) {
			InitQuadrics(edgeScreenSize, camera, preserveBorder);
			InitBorders();
		}
		SDL_LOG("Simplify - End initialization");

	}

	// Finally compact mesh before exiting
	void CompactMesh() {
		u_int dst = 0;

		// We assume vertices 'keep' property is set to false (default value)

		// Compress triangles and mark vertices to keep
		auto not_deleted = [](const SimplifyTriangle& t){ return !t.deleted; };
		decltype(triangles) newTriangles;
		newTriangles.reserve(triangles.size());
		for (auto& t: triangles | std::views::filter(not_deleted)) {
			newTriangles.push_back(t);

			vertices[t.v[0]].keep = true;
			vertices[t.v[1]].keep = true;
			vertices[t.v[2]].keep = true;
		}

		// Compress vertices
		decltype(vertices) newVertices;
		auto keep_vertex = [](const SimplifyVertex& v){ return v.keep; };
		for (auto& v_old: vertices | std::views::filter(keep_vertex)) {

			newVertices.push_back(v_old);
			auto& v_new = newVertices.back();
			v_old.newIndex = newVertices.size() - 1;

			v_new.p = v_old.p;
			v_new.norm = v_old.norm;
			v_new.uv = v_old.uv;
			v_new.col = v_old.col;
			v_new.alpha = v_old.alpha;

		}

		// Update triangle vertices with new vertices
		for (auto& t: newTriangles) {
			t.v[0] = vertices[t.v[0]].newIndex;
			t.v[1] = vertices[t.v[1]].newIndex;
			t.v[2] = vertices[t.v[2]].newIndex;
		}

		triangles = newTriangles;
		vertices = newVertices;

	}

};  // ~class Simplify

void SimplifyTriangle::UpdateTriangleError(
	const VertexVector& vertices,
	const float edgeScreenSize,
	const Camera& camera,
	const bool preserveBorder
) {

	for (auto [i, edge]: enumerate(EDGES)) {
		const auto [e1, e2] = edge;
		const auto& v0 = vertices[this->v[e1]];
		const auto& v1 = vertices[this->v[e2]];
		float collapseError = std::get<float>(
			CalculateCollapseError(v0, v1, preserveBorder)
		);
		float screenErrorScale =
			CalculateCollapseScreenErrorScale(v0, v1, edgeScreenSize, camera);
		this->err[i] = collapseError * screenErrorScale;
	}

}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

SimplifyShape::SimplifyShape(const Camera *camera, ExtTriangleMesh *srcMesh,
		const float target, const float edgeScreenSize, const bool preserveBorder) {
	SDL_LOG("Simplify shape " << srcMesh->GetName() << " with target " << target);

	if ((edgeScreenSize > 0.f) && !camera) {
		throw std::runtime_error(
			"The scene camera must be defined in order to enable simplify "
			"edgescreensize option"
		);
	}

	const auto startTime = WallClockTime();

	const u_int targetCount = Max(1u, Floor2UInt(srcMesh->GetTotalTriangleCount() * target));

	/*srcMesh->Save("debug-start.ply");
	ExtTriangleMesh *debugMeshStart = ScreenProjection(*camera, *srcMesh);
	debugMeshStart->Save("debug-start-proj.ply");
	delete debugMeshStart;*/

	Simplify simplify(*srcMesh);
	simplify.Decimate(targetCount, *camera, edgeScreenSize, preserveBorder);
	mesh = simplify.GetExtMesh();

	/*srcMesh->Save("debug-end.ply");
	ExtTriangleMesh *debugMeshEnd = ScreenProjection(*camera, *mesh);
	debugMeshEnd->Save("debug-end-proj.ply");
	delete debugMeshEnd;*/

	SDL_LOG(
		"Simplified shape from "
		<< srcMesh->GetTotalTriangleCount()
		<< " to "
		<< mesh->GetTotalTriangleCount()
		<< " faces"
	);

	// For some debugging
	//mesh->Save("debug.ply");

	const auto endTime = WallClockTime();
	SDL_LOG(std::format("Simplify time: {:3f} secs", endTime - startTime));
}

SimplifyShape::~SimplifyShape() {
	if (!refined)
		delete mesh;
}

ExtTriangleMesh *SimplifyShape::RefineImpl(const Scene *scene) {
	return mesh;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

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

#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_group.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/concurrent_hash_map.h>

#include "luxrays/core/exttrianglemesh.h"
#include "slg/shapes/simplify.h"
#include "slg/scene/scene.h"
#include "slg/utils/harlequincolors.h"

using namespace std;
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

float FLOAT_INFINITY = std::numeric_limits<float>::infinity();

// Enumerate helper (like Python enumerate)
template <typename T,
          typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T && iterable)
{
    struct iterator
    {
        size_t i;
        TIter iter;
        bool operator != (const iterator & other) const { return iter != other.iter; }
        void operator ++ () { ++i; ++iter; }
        auto operator * () const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper
    {
        T iterable;
        auto begin() { return iterator{ 0, std::begin(iterable) }; }
        auto end() { return iterator{ 0, std::end(iterable) }; }
    };
    return iterable_wrapper{ std::forward<T>(iterable) };
}

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

	float operator[](int c) const {
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

// TODO move to point.h
inline void hash_combine(std::size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hash_combine(seed, rest...);
}

template<>
struct std::hash<Point> {
    std::size_t operator()(const Point& p) const noexcept
    {
		std::size_t h=0;
		hash_combine(h, p.x, p.y, p.z);
        return h;
    }
};

struct SimplifyRef {
	u_int tid, tvertex;

	SimplifyRef(u_int p_tid, u_int p_tvertex): tid(p_tid), tvertex(p_tvertex)
	{}
};
using RefPtr = std::shared_ptr<SimplifyRef>;
using RefPtrVector = std::vector< RefPtr >;

struct SimplifyVertex {
	// Core data
	Point p;              // Position
	RefPtrVector refs;       // Incident edges in topology
	SymetricMatrix q;     // Quadric
	bool border;          // Border status

	// Compacting data
	bool keep = false;
	size_t newIndex = 0;

	// LuxCore specific data
	Normal norm;
	UV uv;
	Spectrum col;
	float alpha;

	// Thread sync
	std::mutex mtx;
	void lock() { mtx.lock(); }
	void unlock() { mtx.unlock(); }
	bool tryLock() { return mtx.try_lock(); }

	// Simple constructor
	SimplifyVertex() {}

	// Copy constructor
	SimplifyVertex(const SimplifyVertex& other) {
		p = other.p;
		norm = other.norm;
		uv = other.uv;
		col = other.col;
		alpha = other.alpha;
	}

	// Copy assignment constructor
	SimplifyVertex& operator=(const SimplifyVertex& other) {
		p = other.p;
		norm = other.norm;
		uv = other.uv;
		col = other.col;
		alpha = other.alpha;
		return *this;
	}

};
using VertexVector = std::vector<SimplifyVertex>;

struct SimplifyTriangle {
	std::array<u_int, 3> v;
	Normal geometryN;
	std::array<float, 3> err;
	bool deleted = false;
	bool dirty = false;

	// Update error of the triangle
	void UpdateTriangleError(
		const VertexVector& vertices,
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	);
};
using TriangleVector = std::vector<SimplifyTriangle>;

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
std::tuple<float, Point> CalculateCollapseError(
	const SimplifyVertex& v0,
	const SimplifyVertex& v1,
	const bool preserveBorder
) {

	const SymetricMatrix q = v0.q + v1.q;

	Point pResult;

	// Compute interpolated vertex
	const Point &p1 = v0.p;
	const Point &p2 = v1.p;
	const Point p3 = (p1 + p2) / 2;

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
	return std::tuple(error , pResult);
}

float CalculateCollapseScreenErrorScale(
	const SimplifyVertex& v0,
	const SimplifyVertex& v1,
	float edgeScreenSize,
	const Camera& camera
) {
	const Point& p0 = v0.p;
	const Point& p1 = v1.p;
	if (edgeScreenSize > 0.f) {
		const float notVisibleScale = .5f;

		float p0x, p0y;
		if (!camera.GetSamplePosition(p0, &p0x, &p0y) ||
				!IsValid(p0x) || !IsValid(p0y))
			return notVisibleScale;

		// Normalize
		p0x /= camera.filmWidth;
		p0y /= camera.filmHeight;

		float p1x, p1y;
		if (!camera.GetSamplePosition(p1, &p1x, &p1y) ||
				!IsValid(p1x) || !IsValid(p1y))
			return notVisibleScale;

		// Normalize
		p1x /= camera.filmWidth;
		p1y /= camera.filmHeight;

		const float edge = sqrtf(Sqr(p0x - p1x) + Sqr(p0y - p1y));
		if (edge == 0.f)
			return notVisibleScale;

		return Max(edge / edgeScreenSize, notVisibleScale);
	} else
		return 1.f;
}


class Simplify {
public:
	Simplify(const ExtTriangleMesh &srcMesh) {
		const u_int vertCount = srcMesh.GetTotalVertexCount();
		const u_int triCount = srcMesh.GetTotalTriangleCount();
		const Point *verts = srcMesh.GetVertices();
		const Triangle *tris = srcMesh.GetTriangles();

		vertices.resize(srcMesh.GetTotalVertexCount());

		for (u_int i = 0; i < vertCount; ++i)
			vertices[i].p = verts[i];

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

	Point& getPoint(const RefPtr& ref) {
		auto vertex = triangles[ref->tid].v[ref->tvertex];
		return vertices[vertex].p;
	}

	struct lockException : std::exception {};

	// Partition candidate list into N components (aka "batches")
	//
	// We use k-means algorithm, with k-means++ initialization
	// https://www.geeksforgeeks.org/machine-learning/ml-k-means-algorithm/
	std::vector<RefPtrVector>
	PartitionIndependentEdgeBatches(const RefPtrVector& candidateList) {

		// Settings
		u_int K = tbb::this_task_arena::max_concurrency() * 5;  // Number of clusters (the 'k' of k-means))
		const u_int MAXITERATIONS = 4;

		// Constants
		const u_int NUMCD = candidateList.size();  // Number of candidates
		if (NUMCD < K) K = NUMCD;

		SDL_LOG("Simplify - Partionning " << NUMCD << " candidates");

		// Make a copy of candidates with pointers
		RefPtrVector candidates;
		candidates.reserve(candidateList.size());
		for (auto candidate: candidateList) {
			candidates.push_back(candidate);
		}

		// Centroids
		std::vector<Point> centroids;
		centroids.reserve(K);

		// Init batches (output)
		std::vector<RefPtrVector> batches(K);

		// Prepare points for centroid initialization
		struct ExtPoint: Point {
			float distance = FLOAT_INFINITY;
			ExtPoint(const Point& p): Point(p) {}
			ExtPoint(): Point() {}
		};

		using PointVec = std::vector<ExtPoint>;
		PointVec points(candidates.size());
		tbb::parallel_for(
			tbb::blocked_range<u_int>(0, candidates.size()),
			[&](const tbb::blocked_range<u_int>& r) {
				for (u_int i = r.begin(); i != r.end(); ++i) {
					points[i] = getPoint(candidates[i]);
				}
			}
		);

		SDL_LOG("Simplify - Partionning - Initializing");

		// Initialize the first centroid with a random point
		// For practical reasons, we will take the last
		{
			auto lastPoint = points.back();
			centroids.push_back(lastPoint);
			points.pop_back();
		}

		// Init remaining centroids in k-mean++ fashion
		auto comp = [](const ExtPoint& p0, const ExtPoint& p1) {
			return p0.distance < p1.distance;
		};
		for (u_int i = 0; i < K - 1 ; ++i) {  // for each remaining centroid
			// Compute points distances to the current set of centroids
			// - Nota1: distance is the min distance from point to the cloud
			//   of centroids
			// - Nota2: the cloud of centroids expands at each loop
			// - Nota3: We've already got distances from the points to the cloud
			//	 minus the last added centroid, so we'll use that information
			//   for this iteration
			const auto& lastCentroid = centroids.back();
			tbb::parallel_for(
				tbb::blocked_range<u_int>(0, points.size()),
				[&points, &lastCentroid](const tbb::blocked_range<u_int>& r) {
				for (auto j = r.begin(); j != r.end(); ++j) {
					auto& point = points[j];
					point.distance = std::min(
						point.distance,
						DistanceSquared(point, lastCentroid)
					);
				}
			});

			// Find the farthest candidate to the current cloud of centroids
			// and make it the next centroid
			auto next_centroid_pos = std::max_element(
					std::execution::parallel_policy(), points.begin(), points.end(), comp
			);
			centroids.push_back(*next_centroid_pos);
			std::swap(*next_centroid_pos, points.back());
			points.pop_back();
		}

		SDL_LOG("Simplify - Partition batches - Start iterations");
		const float MINERROR = 3 * K * std::numeric_limits<float>::epsilon();
		u_int iteration = 0;
		while (true) {
			// Assign available candidates to nearest cluster (min distance to centroid)
			// Per-thread local batches:
			tbb::enumerable_thread_specific<std::vector<RefPtrVector>> local_batches(
				[centroids_size=centroids.size()]
				{ return std::vector<RefPtrVector>(centroids_size); }
			);

			tbb::parallel_for(
				tbb::blocked_range<size_t>(0, candidates.size()),
				[&](const tbb::blocked_range<size_t>& r) {
					auto& local = local_batches.local();
					for (size_t idx = r.begin(); idx != r.end(); ++idx) {
						const auto& candidate = candidates[idx];
						const auto& point = getPoint(candidate);

						// Find nearest centroid
						u_int iMin = 0;
						float distMin = FLOAT_INFINITY;
						for (u_int i = 0; i < centroids.size(); ++i) {
							float distance = DistanceSquared(point, centroids[i]);
							if (distance < distMin) {
								distMin = distance;
								iMin = i;
							}
						}
						local[iMin].push_back(candidate);
					}
				}
			);

			// Merge local batches into global batches
			for (auto& local : local_batches) {
				for (size_t i = 0; i < centroids.size(); ++i) {
					batches[i].insert(batches[i].end(), local[i].begin(), local[i].end());
				}
			}

			// Compute new centroids
			decltype(centroids) newCentroids(centroids.size());
			tbb::parallel_for(
				tbb::blocked_range<u_int>(0, K),
				[&](const tbb::blocked_range<u_int>& range) {
					for (u_int i = range.begin(); i != range.end(); ++i) {
						const auto& batch = batches[i];
						Point sum = std::accumulate(
							batch.begin(),
							batch.end(),
							Point(0.f, 0.f, 0.f),
							[this](const Point& p, const RefPtr& r) { return p + getPoint(r); }
						);
						if (!batch.empty())
							newCentroids[i] = sum / float(batch.size());
						else
							newCentroids[i] = Point(0.f, 0.f, 0.f); // or handle empty batch as needed
					}
				}
			);

			// Evaluate error
			float delta = tbb::parallel_reduce(
				tbb::blocked_range<u_int>(0, centroids.size()),
				0.f,
				[&](const tbb::blocked_range<u_int>& r, float local_sum) -> float {
					for (u_int i = r.begin(); i != r.end(); ++i) {
						local_sum += DistanceSquared(newCentroids[i], centroids[i]);
					}
					return local_sum;
				},
				std::plus<float>()
			);

			// Halt condition
			if (delta < MINERROR or iteration > MAXITERATIONS) {
				// Exit loop
				break;
			}

			// Reset centroids and batches
			centroids = newCentroids;
			tbb::parallel_for(
				tbb::blocked_range<size_t>(0, batches.size()),
				[&](const tbb::blocked_range<size_t>& r) {
					for (size_t i = r.begin(); i != r.end(); ++i) {
						batches[i].clear();
					}
				}
			);

			++iteration;
		}  // ~while

		SDL_LOG("Simplify - Partition batches - end");  // TODO
		return batches;
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

			SDL_LOG("Simplify - Start iteration (init) #" << iteration);

			if (startTriangleCount - deletedTriangles <= targetTriangleCount)
				break;

			const u_int initialdeletedTriangles = deletedTriangles;

			// Update mesh constantly
			InitIteration(iteration, edgeScreenSize, camera, preserveBorder);

			// Build candidate list
			SDL_LOG("Simplify - Build candidate list #" << iteration);
			auto candidateList = BuildCandidateList(
				preserveBorder, maxCandidateQueueSize
			);


			// Partition candidates into independent batches
			SDL_LOG("Simplify - Partition candidate list #" << iteration);
			auto batches = PartitionIndependentEdgeBatches(candidateList);

			SDL_LOG("Simplify - Number of batches: " << batches.size());

			using AtomicCounter = std::atomic<u_int>;
			AtomicCounter batchDeleted = 0;
			AtomicCounter missedLocks = 0;

			SDL_LOG("Simplify - Main treatment #" << iteration);
			tbb::parallel_for(
				size_t(0), batches.size(),
				[&](size_t i) {
					u_int localDeleted = 0;
					u_int localMissedLocks = 0;
					for (auto& ref : batches[i]) {
						try {
							localDeleted += CollapseEdge(*ref, edgeScreenSize, camera, preserveBorder);
						} catch (lockException&) {
							++localMissedLocks;
						}
					}
					batchDeleted.fetch_add(localDeleted, std::memory_order_relaxed);
					missedLocks.fetch_add(localMissedLocks, std::memory_order_relaxed);
				}
			);
			deletedTriangles += batchDeleted;

			const u_int iterationDeletedTriangles =
				deletedTriangles - initialdeletedTriangles;

			SDL_LOG(
				"Simplify - Iteration " << iteration
				<< " (" << candidateList.size()
				<< " edge candidates, deleted "
				<< iterationDeletedTriangles
				<< "/" << deletedTriangles
				<< " of " << startTriangleCount << " triangles, "
				<< "missed " << missedLocks << " locks"
				<< ")"
			);
			if (!iterationDeletedTriangles)
				break;
		}

		// Clean up mesh
		CompactMesh();
		SDL_LOG("Simplify - Mesh compacted");
	}

private:

	VertexVector vertices;
	TriangleVector triangles;

	void assert_data(size_t line) {
		for (auto& v: vertices) {
			for (auto& r: v.refs) {
				if(r->tid >= triangles.size()) {
					SDL_LOG("Data error: " << r->tid << " " << triangles.size()
							<< " #" << to_string(line));
					return;
				}
			}
		}
		SDL_LOG("No data error " + to_string(line));
	}

	bool hasNormals, hasUVs, hasColors, hasAlphas;

	// Lock & Neighbor features
	using LockResult = std::tuple<bool, std::mutex&>;
	using LockResults = std::vector<LockResult>;
	using NeighborSet = std::unordered_set<u_int>;


	// Find edge neighbors, ie vertices that could be affected
	// by collapsing the given edge
	NeighborSet edgeNeighbors(const u_int i0, const u_int i1) {
		NeighborSet neighbors;

		neighbors.insert(i0);
		neighbors.insert(i1);
		for (const auto& ref: vertices[i0].refs) {
			for (u_int vertexIndex: triangles[ref->tid].v) {
				neighbors.insert(vertexIndex);
			}
		}
		for (const auto& ref: vertices[i1].refs) {
			for (u_int vertexIndex: triangles[ref->tid].v) {
				neighbors.insert(vertexIndex);
			}
		}
		return neighbors;
	}


	// Lock neighborhood of an edge (including the edge itself)
	// Nota: This is a try-lock
	// Returns: summarized status (ok or not), detailed locks
	std::tuple<bool, LockResults>
	lockEdgeNeighbors(const u_int i0, const u_int i1) {

		// Get edge neighborhood
		auto neighbors = edgeNeighbors(i0, i1);

		// Try to lock
		LockResults tryLockResults;
		for (auto i: neighbors) {
			std::mutex& mtx = vertices[i].mtx;
			//mtx.lock(); bool res = true;
			bool res = mtx.try_lock();
			tryLockResults.emplace_back(res, mtx);
		}

		// Summarize status
		bool status = std::all_of(
			tryLockResults.begin(),
			tryLockResults.end(),
			[](auto& i){ return std::get<bool>(i); }
		);

		return std::tuple(status, tryLockResults);
	}

	// Unlock previously locked neighborhood
	void unlockNeighbors(LockResults& locks) {
		for (auto& [locked, mtx]: locks) {
			if (locked) mtx.unlock();
		}
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

		// Get explicit edge to collapse and lock it
		const u_int startVertexIndex = vertex.tvertex;
		const u_int i0 = t.v[startVertexIndex];
		const u_int i1 = t.v[(startVertexIndex + 1) % 3];

		auto [lockStatus, locks] = lockEdgeNeighbors(i0, i1);
		if (not lockStatus) {
			unlockNeighbors(locks);
			throw lockException();
		}

		// Prepare shortcuts
		SimplifyVertex &v0 = vertices[i0];
		SimplifyVertex &v1 = vertices[i1];

		u_int deletedTriangles = 0;

		// Border check
		if (v0.border != v1.border) {
			SDL_LOG("Simplify - border");  // TODO
			unlockNeighbors(locks);
			return 0;
		}

		// Compute vertex to collapse to
		// Doesn't modify state (neither vertices nor triangles)
		const auto [error, p] = CalculateCollapseError(v0, v1, preserveBorder);

		// Do not collapse edge if it makes a face flip
		// deleted0, deleted1: true/false if the triangles referencing the
		// vertex are deleted
		auto [will_flip0, deleted0] = Flipped(p, i0, i1);
		auto [will_flip1, deleted1] = Flipped(p, i1, i0);
		if (will_flip0 || will_flip1) {
			unlockNeighbors(locks);
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
		const Point triPoint0 = tv0.p;
		const Point triPoint1 = tv1.p;
		const Point triPoint2 = tv2.p;
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
		refs.insert(refs.end(), newRefs0.begin(), newRefs0.end());
		refs.insert(refs.end(), newRefs1.begin(), newRefs1.end());
		deletedTriangles = deletedTriangles0 + deletedTriangles1;

		unlockNeighbors(locks);
		return deletedTriangles;
	}

	// Check if a triangle flips when this edge is removed
	// Returns: check status, deleted status of each ref (vector)
	std::tuple<bool, std::vector<bool> >
	Flipped(const Point &p, const u_int i0, const u_int i1) const {

		const SimplifyVertex &v0 = vertices[i0];

		// Result variable
		auto res = std::make_tuple<bool, std::vector<bool>>
			(false, std::vector<bool>(v0.refs.size()));
		bool& status = std::get<0>(res);
		std::vector<bool>& deleted(std::get<1>(res));

		for (const auto& [k, ref]: enumerate(v0.refs)) {
			const SimplifyTriangle &t = triangles[ref->tid];

			if (t.deleted) continue;

			const u_int s = ref->tvertex;
			const u_int id1 = t.v[(s + 1) % 3];
			const u_int id2 = t.v[(s + 2) % 3];

			// Delete ?
			if (id1 == i1 || id2 == i1) {
				deleted[k] = true;
				continue;
			}

			// Check if the triangle is too narrow
			const Vector d1 = Normalize(vertices[id1].p - p);
			const Vector d2 = Normalize(vertices[id2].p - p);
			if (AbsDot(d1, d2) > .999f) {
				status = true;
				return res;
			}

			// Check if the Normal is changing side
			const Normal geometryN(Normalize(Cross(d1, d2)));
			if (Dot(geometryN, t.geometryN) < .2f) {
				status = true;
				return res;
			}

			deleted[k] = false;
		}

		status = false;
		return res;
	}

	// Update triangle connections and edge error after a edge is collapsed
	// Returns: new incident edges list, number of deleted triangles
	std::tuple<RefPtrVector, u_int> UpdateTriangles(
		const u_int i0,
		const SimplifyVertex &v,  // Collapsed vertex
		const  vector<bool> &deleted,
		const float edgeScreenSize,
		const Camera& camera,
		const bool preserveBorder
	) {
		u_int deletedTriangles = 0;
		RefPtrVector refs;
		refs.reserve(v.refs.size());
		for (const auto& [k, r]: enumerate(v.refs)) {
			SimplifyTriangle &t = triangles[r->tid];

			if (t.deleted)
				continue;

			if (deleted[k]) {
				t.deleted = true;
				deletedTriangles++;
				continue;
			}

			t.v[r->tvertex] = i0;
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

		// Serial accumulation into vertex quadrics
		for (u_int i = 0; i < triangles.size(); ++i) {
			const auto &t = triangles[i];
			vertices[t.v[0]].q += triangleQuadrics[i][0];
			vertices[t.v[1]].q += triangleQuadrics[i][1];
			vertices[t.v[2]].q += triangleQuadrics[i][2];
		}

		// Triangle error update
		for (u_int i = 0; i < triangles.size(); ++i) {
			// Calc Edge Error
			SimplifyTriangle &t = triangles[i];

			t.UpdateTriangleError(vertices, edgeScreenSize, camera, preserveBorder);
		}
	}

	// Build incident edge tables on vertices
	//
	// Modify: vertices
	void InitIncidentEdges() {

		// 1. Parallel clear (optional but clean)
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, vertices.size()),
			[&](const tbb::blocked_range<size_t>& r) {
				for (size_t i = r.begin(); i != r.end(); ++i) {
					vertices[i].refs.clear();
				}
			}
		);

		// 2. Temporary thread-safe concurrent_vectors for each vertex
		std::vector<tbb::concurrent_vector<RefPtr>> tmp_refs(vertices.size());

		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, triangles.size()),
			[&](const tbb::blocked_range<size_t>& r) {
				for (size_t i = r.begin(); i != r.end(); ++i) {
					const auto& t = triangles[i];
					for (size_t j = 0; j < 3; ++j) {
						size_t vertexIndex = t.v[j];
						tmp_refs[vertexIndex].push_back(
							std::make_shared<SimplifyRef>(i, j)
						);
					}
				}
			}
		);

		// 3. Serially move concurrent_vectors to real refs
		for (size_t i = 0; i < vertices.size(); ++i) {
			vertices[i].refs.assign(tmp_refs[i].begin(), tmp_refs[i].end());
		}
	}

	// Init border indicators on vertices
	//
	// Modify: vertices
	void InitBorders() {
		// Set borders to false
		for (auto& v: vertices)
			v.border = false;

		vector<u_int> vcount, vids;
		for (const auto& v: vertices) {
			vcount.clear();
			vids.clear();

			for (const auto& ref: v.refs) {
				auto k = ref->tid;
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
	RefPtrVector BuildCandidateList(
		bool preserveBorder,
		u_int maxCandidateQueueSize
	) const {

		// 1. Thread-safe candidate buffer
		tbb::concurrent_vector<RefPtr> candidateRefs;

		// 2. Parallel candidate search
		tbb::parallel_for(tbb::blocked_range<u_int>(0, triangles.size()),
			[&](const tbb::blocked_range<u_int>& r) {
				for (u_int i = r.begin(); i != r.end(); ++i) {
					const SimplifyTriangle &t = triangles[i];

					u_int minErrorIndex = NULL_INDEX;
					float minError = FLOAT_INFINITY;
					for (u_int j = 0; j < 3; ++j) {
						const u_int i0 = t.v[j];
						const SimplifyVertex &v0 = vertices[i0];
						const u_int i1 = t.v[(j + 1) % 3];
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
						if (std::get<bool>(Flipped(p, i0, i1)))
							continue;
						if (std::get<bool>(Flipped(p, i1, i0)))
							continue;

						if (t.err[j] < minError) {
							minErrorIndex = j;
							minError = t.err[j];
						}
					}
					if (minErrorIndex != NULL_INDEX)
						candidateRefs.push_back(std::make_shared<SimplifyRef>(i, minErrorIndex));
				}
			}
		);

		// 3. Serial step: Build the priority queue with a size cap
		auto refErrorCompare = [&](const RefPtr left, const RefPtr right) {
			auto left_error = triangles[left->tid].err[left->tvertex];
			auto right_error = triangles[right->tid].err[right->tvertex];
			return  left_error < right_error;
		};
		std::priority_queue<RefPtr, RefPtrVector, decltype(refErrorCompare)>
			candidateQueue{ refErrorCompare };

		for (const auto& ref : candidateRefs) {
			if (candidateQueue.size() < maxCandidateQueueSize) {
				candidateQueue.push(ref);
				continue;
			}
			// Compare error for cap logic
			auto top = candidateQueue.top();
			if (refErrorCompare(ref, top)) {
				candidateQueue.pop();
				candidateQueue.push(ref);
			}
		}

		// 4. Serial step: Move priority queue to output structure
		RefPtrVector candidateList;
		candidateList.reserve(candidateQueue.size());
		while (!candidateQueue.empty()) {
			candidateList.push_back(candidateQueue.top());
			candidateQueue.pop();
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
			int dst = 0;
			for (auto& t: triangles)
				if (!t.deleted)
					triangles[dst++] = t;

			triangles.resize(dst);
		}
		// Clear triangles dirty flags
		for (u_int i = 0; i < triangles.size(); ++i)
			triangles[i].dirty = false;

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


};

void SimplifyTriangle::UpdateTriangleError(
	const VertexVector& vertices,
	const float edgeScreenSize,
	const Camera& camera,
	const bool preserveBorder
) {
	using edge_t = std::pair<u_int, u_int>;
	constexpr std::array<edge_t, 3> edges({ {0, 1}, {1, 2}, {2, 0}, });

	for (auto [i, edge]: enumerate(edges)) {
		const auto& v0 = vertices[this->v[edge.first]];
		const auto& v1 = vertices[this->v[edge.second]];
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

	if ((edgeScreenSize > 0.f) && !camera)
		throw runtime_error("The scene camera must be defined in order to enable simplify edgescreensize option");

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
	//std::exit();  // DEBUG - Stop here
}

SimplifyShape::~SimplifyShape() {
	if (!refined)
		delete mesh;
}

ExtTriangleMesh *SimplifyShape::RefineImpl(const Scene *scene) {
	return mesh;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

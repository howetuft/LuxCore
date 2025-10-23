/***************************************************************************
 * Copyright 1998-2025 by authors (see AUTHORS.txt)                        *
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

#include <tbb/tbb.h>
#include <tbb/mutex.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_sort.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/scalable_allocator.h>
#include <tbb/blocked_range2d.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/functional/hash.hpp>

#include "luxrays/core/exttrianglemesh.h"
#include "slg/shapes/simplify.h"
#include "slg/scene/scene.h"
#include "slg/utils/harlequincolors.h"
#include "dset.h"

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

namespace {
// Everything inside this namespace is kept local to this translation
// unit (behaves like static and avoid linker namespace pollution)

// Namespace containing the original code

// Namespace containing the rewriting of the algo, with parallelization
namespace enhanced {


struct alignas(std::hardware_destructive_interference_size) AlignedBool {
	bool _bool = false;
	AlignedBool() = default;
	AlignedBool(const bool& p_bool) : _bool(p_bool) {};
	AlignedBool(bool&& p_bool) : _bool(p_bool) {};
	operator bool() const { return _bool; }
};
static_assert(sizeof(AlignedBool) == std::hardware_destructive_interference_size);

using BoolVector = std::vector<AlignedBool, tbb::cache_aligned_allocator<AlignedBool>>;
using SizeTVector = std::vector<size_t, tbb::cache_aligned_allocator<size_t>>;

// Geometry
using Vector = Eigen::Vector3f;
using Normal = Vector;
using Point = Eigen::Vector4f;

inline luxrays::Normal Eigen2LuxN(const Vector& v) {
	return luxrays::Normal(v[0], v[1], v[2]);
}
inline luxrays::Vector Eigen2LuxV(const Vector& v) {
	return luxrays::Vector(v[0], v[1], v[2]);
}
inline Point Lux2EigenP(const luxrays::Point& p) {
	return Point(p.x, p.y, p.z, 1.f);
}
inline luxrays::Point Eigen2LuxP(const Point& p) {
	return luxrays::Point(p[0], p[1], p[2]);
}
inline Vector Point2Vector(const Point& p) {
	return Vector(p.head<3>());
}
inline Normal TriNormal(const Point& p0, const Point& p1, const Point& p2) {
	return Normal((p1 - p0).head<3>().cross((p2 - p0).head<3>()));
}


constexpr float FLOAT_INFINITY = std::numeric_limits<float>::infinity();

constexpr std::array<std::tuple<size_t, size_t>, 3> EDGES({ {0, 1}, {1, 2}, {2, 0}, });
constexpr std::array<size_t, 3> OPPOSITE{ 2, 0, 1};

// Barycentric coordinates of p in triangle(p0, p1, p2)
// Returns bool status and resulting vector
// Status is false if input data are malformed (non coplanar, p out of
// triangle...)
inline std::tuple<bool, Vector> BaryCoords(
	const Point &p,
	const Point &p0,
	const Point &p1,
	const Point &p2
) {
	const auto error_result = std::tuple(
		false,
		Vector(FLOAT_INFINITY, FLOAT_INFINITY, FLOAT_INFINITY)
	);

	const Normal vCrossW = TriNormal(p0, p2, p);
	const Normal vCrossU = TriNormal(p0, p2, p1);

	if (std::signbit(vCrossW.dot(vCrossU))) return error_result;

	const Vector uCrossW = TriNormal(p0, p1, p);
	const Vector uCrossV = -vCrossU;

	if (std::signbit(uCrossW.dot(uCrossV))) return error_result;

	const float denom = uCrossV.norm();
	if (not denom) return error_result;

	const float s = vCrossW.norm() / denom;
	const float t = uCrossW.norm() / denom;
	const float r = 1.f - s - t;

	auto res = Eigen::Vector3f{r, s, t};

	// All coords should be <= 1.f
	if (not (r <= 1.f and s <= 1.f and t <= 1.f)) return error_result;

	return std::tuple(true, res);
}


// Quadric
using Quadric = Eigen::Matrix4f;

// References (aka edges)
struct
alignas(std::hardware_destructive_interference_size)
SimplifyRef {
	size_t tid = 0;  // Triangle ID
	char tvertex = -1;  // Vertex in triangle, should be in [0;3] (-1: not set)
	float error;  // To prioritize candidates

	SimplifyRef(size_t p_tid, size_t p_tvertex, float p_error=0.f):
		tid(p_tid), tvertex(p_tvertex), error(p_error) {}
	SimplifyRef() {}

	bool initialized() const { return tvertex >= 0; }
};
static_assert(sizeof(SimplifyRef) % std::hardware_destructive_interference_size == 0);

bool RefLess(const SimplifyRef& left, const SimplifyRef& right) {
	return left.error < right.error;
}

using RefVector = std::vector< SimplifyRef, tbb::cache_aligned_allocator<SimplifyRef> >;

// Vertex
struct
alignas(std::hardware_destructive_interference_size)
SimplifyVertex {

	RefVector refs;		  // Incident edges in topology

	// Error computation inputs
	Quadric quad = Quadric::Zero();     // Quadric
	bool border;						// Border status

	// LuxCore specific data
	Normal norm;
	Eigen::Vector2f uv;
	Eigen::Vector3f col;
	float alpha;

	const Point& p() const { return m_p; }
	void setP(const Point p) {
		m_p = p;
		// Update uuid
		m_uuid = 0;
		boost::hash_combine(m_uuid, p[0]);
		boost::hash_combine(m_uuid, p[1]);
		boost::hash_combine(m_uuid, p[2]);
	}

	size_t uuid() const { return m_uuid; }

protected:

	// Position in space
	Point m_p;			  // Position in geometry

	// Unique identifier (for caching)
	size_t m_uuid;

};
static_assert(sizeof(SimplifyVertex) % std::hardware_destructive_interference_size == 0);

using VertexVector = std::vector<
	SimplifyVertex,
	tbb::cache_aligned_allocator<SimplifyVertex>
>;


class TriangleVector;

// Triangle
struct
alignas(std::hardware_destructive_interference_size)
SimplifyTriangle {

	// Static data
	std::array<size_t, 3> v;  // Vertex indices
	Normal geometryN;

	friend class TriangleVector;

	// Constructors
	// The redefinition of the 4-pack is necessary, due to atomics being not
	// movable
	SimplifyTriangle() = default;

	SimplifyTriangle(SimplifyTriangle&& other) :
	v(other.v), geometryN(other.geometryN) {
		m_deleted = other.m_deleted ? true : false;
		m_dirty = other.m_dirty ? true : false;
	}

	SimplifyTriangle(const SimplifyTriangle& other) :
	v(other.v), geometryN(other.geometryN) {
		m_deleted = other.m_deleted ? true : false;
		m_dirty = other.m_dirty ? true : false;
	}


	SimplifyTriangle& operator=(const SimplifyTriangle& other) {
		v = other.v;
		geometryN = other.geometryN;
		m_deleted = other.m_deleted ? true : false;
		m_dirty = other.m_dirty ? true : false;
		return *this;
	}

protected:
	// Dynamic data
	//
	// Triangle dynamic data have been encapsulated to prevent some bugs
	std::atomic<bool> m_deleted = false;
	std::atomic<bool> m_dirty = false;


	// Status handling
	//
	// Triangle accessors have been encapsulated to prevent some bugs.
	// They are only accessible via triangle container (see TriangleVector)
	inline bool deleted() const { return m_deleted; }
	inline bool dirty() const { return m_dirty; }
	inline void set_deleted() { m_deleted = true; }
	inline void set_dirty() { m_dirty = true; }
	inline void clear_deleted() { m_deleted = false; }
	inline void clear_dirty() { m_dirty = false; }

};

static_assert(sizeof(SimplifyTriangle) % std::hardware_destructive_interference_size == 0);

using TriangleVectorBase = std::vector<
	SimplifyTriangle,
	tbb::cache_aligned_allocator<SimplifyTriangle>
>;

class TriangleVector : protected TriangleVectorBase {
public:
	TriangleVector() : TriangleVectorBase() {}
	TriangleVector(size_t p_size) : TriangleVectorBase(p_size) {}

	using TriangleVectorBase::size;
	using TriangleVectorBase::resize;
	using TriangleVectorBase::reserve;
	using TriangleVectorBase::push_back;
	using TriangleVectorBase::insert;
	using TriangleVectorBase::operator[];
	using TriangleVectorBase::begin;
	using TriangleVectorBase::end;

	// Triangle accessors have been encapsulated to prevent some bugs
	inline bool deleted(size_t i) const { return (*this)[i].deleted(); }
	inline bool dirty(size_t i) const { return (*this)[i].dirty(); }
	inline void set_deleted(size_t i) { (*this)[i].set_deleted(); }
	inline void set_dirty(size_t i) { (*this)[i].set_dirty(); }
	inline void clear_deleted(size_t i) { (*this)[i].clear_deleted(); }
	inline void clear_dirty(size_t i) { (*this)[i].clear_dirty(); }
};

// Error between point and Quadric
inline float PointError(const Quadric &q, const Point& p) {
	return fabs(p.transpose() * q * p);
}


// Synchronization
using MutexVector = std::vector<
	std::mutex,
	tbb::cache_aligned_allocator<std::mutex>
>;


// Helper to populate triangles and vertices with input data
template <typename D, typename S, typename T> struct ForEach {
	ForEach(D& p_dest, const S& p_src, T p_treatment) :
		dest(p_dest), src(p_src), treatment(p_treatment) {}
	ForEach(const ForEach&) = default;

	void operator()(const tbb::blocked_range<size_t>& tbbrange) const {
		for (auto i = tbbrange.begin(); i != tbbrange.end(); ++i) {
			treatment(dest[i], src[i]);
		}
	}
	void run(const size_t size) {
		tbb::blocked_range<size_t> range(0, size);
		tbb::parallel_for(range, *this);
	}
	T& treatment;
	D& dest;
	const S& src;
};

#ifndef NDEBUG
#define DBG_SDL_LOG(X) SDL_LOG(X)
#else
#define DBG_SDL_LOG(X) {}
#endif

// Cache feature for candidate list building

// Key is a triangle, identified by its geometric points
struct
alignas(std::hardware_destructive_interference_size)
ErrorCacheKey : std::array<size_t, 3>{};

struct
alignas(std::hardware_destructive_interference_size)
ErrorCacheEntry {
	// vertex index in triangle and error
	uint16_t minIndex;  // 16
	float error; // 32
	uint16_t pad[5];  // 80
};

struct ErrorCacheHash {
	//size_t hash(const ErrorCacheKey& k) {
	static size_t hash(const ErrorCacheKey& k) {
		size_t seed = k[0];
		seed ^= k[1] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		seed ^= k[2] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		return seed;
	}
	static bool equal(const ErrorCacheKey& k0, const ErrorCacheKey& k1) {
		return k0 == k1;
	}
};

using ErrorCachePair = std::pair<const ErrorCacheKey, ErrorCacheEntry>;

using ErrorCacheType = tbb::concurrent_hash_map<
	ErrorCacheKey,
	ErrorCacheEntry,
	ErrorCacheHash,
	tbb::cache_aligned_allocator<ErrorCachePair>
>;

// Objects to handle triangle errors in a thread-safe way
//
class AtomicErrors {
private:
	static constexpr float DEFAULT_VALUE = std::numeric_limits<float>::infinity();
	std::array<std::atomic<float>, 3> m_values{
		DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE
	};
public:
	AtomicErrors() = default;
	AtomicErrors(AtomicErrors&& other) {
		for (int i = 0; i < 3; ++i) {
			m_values[i] = float(other.m_values[i]);
		}
	}
	AtomicErrors(const AtomicErrors& other) {
		for (int i = 0; i < 3; ++i) {
			m_values[i] = float(other.m_values[i]);
		}
	}
	AtomicErrors& operator=(const AtomicErrors& other) {
		for (int i = 0; i < 3; ++i) {
			m_values[i] = float(other.m_values[i]);
		}
		return *this;
	}
	AtomicErrors& operator=(const std::array<float, 3>& other) {
		for (int i = 0; i < 3; ++i) {
			m_values[i] = other[i];
		}
		return *this;
	}
	float operator[](size_t i) const { return float(m_values[i]); }
};
using TriangleErrorVector = std::vector<
	AtomicErrors,
	tbb::cache_aligned_allocator<AtomicErrors>
>;


// The main class
class Simplify {
public:
	Simplify(
		const luxrays::ExtTriangleMesh &srcMesh,
		const slg::Camera& p_camera,
		const float p_edgeScreenSize,
		const bool p_preserveBorder
	) :
		camera(p_camera),
		edgeScreenSize(p_edgeScreenSize),
		preserveBorder(p_preserveBorder),
		vertices(srcMesh.GetTotalVertexCount()),
		vmutexes(srcMesh.GetTotalVertexCount()),
		triangles(srcMesh.GetTotalTriangleCount()),
		ErrorCache(size_t(triangles.size() * 1.25f))
	{
		using SV = SimplifyVertex;


		// Size vertices and triangles containers in accordance to inputs
		size_t vertCount = srcMesh.GetTotalVertexCount();
		size_t triCount = srcMesh.GetTotalTriangleCount();
		vertices.resize(vertCount);
		triangles.resize(triCount);
		trierrors.resize(triCount);

		auto init_vertex = [](SV& vd, const luxrays::Point& vs){
			vd.setP(Lux2EigenP(vs));
			vd.refs.reserve(9);  // Seems reasonable
		};
		ForEach(vertices, srcMesh.GetVertices(), init_vertex).run(vertCount);

		if (srcMesh.HasNormals()) {
			auto init_normal = [](SV& v, const luxrays::Normal& n)
				{ v.norm = Normal(n.x, n.y, n.z); };
			ForEach(vertices, srcMesh.GetNormals(), init_normal).run(vertCount);

			hasNormals = true;
		}

		if (srcMesh.HasUVs(0)) {
			auto init_uv = [](SV& v, const luxrays::UV& u)
				{ v.uv = Eigen::Vector2f(u.u, u.v); };
			ForEach(vertices, srcMesh.GetUVs(0), init_uv).run(vertCount);

			hasUVs = true;
		}

		if (srcMesh.HasColors(0)) {
			auto init_col = [](SV& v, const luxrays::Spectrum& col)
				{ v.col = Vector(col.c); };
			ForEach(vertices, srcMesh.GetColors(0), init_col).run(vertCount);

			hasColors = true;
		}

		if (srcMesh.HasAlphas(0)) {
			auto init_alpha = [](SV& v, const float alpha){ v.alpha = alpha; };
			ForEach(vertices, srcMesh.GetAlphas(0), init_alpha).run(vertCount);

			hasAlphas = true;
		}

		// Init triangles
		auto init_triangle = [](SimplifyTriangle& td, const luxrays::Triangle& ts){
			td.v[0] = ts.v[0];
			td.v[1] = ts.v[1];
			td.v[2] = ts.v[2];
		};
		ForEach(triangles, srcMesh.GetTriangles(), init_triangle).run(triCount);
	}


	// Effectively simplify mesh (reduce triangles)
	void Decimate(const size_t targetTriangleCount) {
		Eigen::initParallel();

		// Work on 10% of all triangles for each iteration
		size_t maxCandidateQueueSize = std::max(
			64u, luxrays::Floor2UInt(triangles.size() * .1f)
		);

		// Main iteration loop
		const size_t startTriangleCount = triangles.size();
		size_t deletedTriangles = 0;
		for (size_t iteration = 0; iteration < 64; ++iteration) {

			if (startTriangleCount - deletedTriangles <= targetTriangleCount) break;

			DBG_SDL_LOG("Simplify - Start iteration #" << iteration);

			// Compute iteration data (including mesh topology)
			DBG_SDL_LOG("Simplify - Initialize data #" << iteration);
			InitIteration(iteration);

			// Build candidate list
			DBG_SDL_LOG("Simplify - Build candidate list #" << iteration);
			auto candidates{BuildCandidateList(maxCandidateQueueSize)};

			// Delete triangles (run batches)
			DBG_SDL_LOG(
				"Simplify - Delete triangles"
				<< " #" << iteration
				);
			const auto iterationDeletedTriangles{DeleteTriangles(candidates)};

			deletedTriangles += iterationDeletedTriangles;

			SDL_LOG(
				"Simplify - End iteration #" << iteration
				<< " - Edge candidates: " << candidates.size() << " - "
				<< " Deleted triangles (current/cumulative/initial): "
				<< iterationDeletedTriangles << "/"
				<< deletedTriangles << "/"
				<< startTriangleCount
			);

			// No more work?
			if (!iterationDeletedTriangles) break;

		}

		// Clean up mesh and cache
		SDL_LOG("Simplify - Finalize computation");
		Finalize();

	}

	// Rebuild an output mesh after simplification
	// Result is written in meshResult class property
	void RebuildExtMesh() {
		const size_t vertCount = vertices.size();
		const size_t triCount = triangles.size();
		using SV = SimplifyVertex;

		luxrays::Point *newVertices =
			luxrays::ExtTriangleMesh::AllocVerticesBuffer(vertCount);
		auto vert_assign = [](luxrays::Point& vd, const SV& vs) {
			vd = Eigen2LuxP(vs.p());
		};
		ForEach(newVertices, vertices, vert_assign).run(vertCount);

		luxrays::Normal *newNorms = nullptr;
		if (hasNormals) {
			newNorms = new luxrays::Normal[vertCount];

			auto norm_assign = [](luxrays::Normal& vd, const SV& vs) {
				vd = Eigen2LuxN(vs.norm);
			};
			ForEach(newNorms, vertices, norm_assign).run(vertCount);
		}

		luxrays::UV *newUVs = nullptr;
		if (hasUVs) {
			newUVs = new luxrays::UV[vertCount];
			auto uv_assign = [](luxrays::UV& vd, const SV& vs) {
				vd.u = vs.uv[0];
				vd.v = vs.uv[1];
			};
			ForEach(newUVs, vertices, uv_assign).run(vertCount);
		}

		luxrays::Spectrum *newCols = nullptr;
		if (hasColors) {
			newCols = new luxrays::Spectrum[vertCount];
			auto col_assign = [](luxrays::Spectrum& vd, const SV& vs) {
				vd.c[0] = vs.col[0];
				vd.c[1] = vs.col[1];
				vd.c[2] = vs.col[2];
			};
			ForEach(newCols, vertices, col_assign).run(vertCount);
		}

		float *newAlphas = nullptr;
		if (hasAlphas) {
			newAlphas = new float[vertCount];
			auto alpha_assign = [](float& vd, const SV& vs) {
				vd = vs.alpha;
			};
			ForEach(newAlphas, vertices, alpha_assign).run(vertCount);
		}

		luxrays::Triangle *newTris =
			luxrays::ExtTriangleMesh::AllocTrianglesBuffer(triCount);

		auto triangle_assign =
			[vertCount](luxrays::Triangle& td, const SimplifyTriangle& ts) {
			assert (ts.v[0] < vertCount);
			assert (ts.v[1] < vertCount);
			assert (ts.v[2] < vertCount);
			td.v[0] = ts.v[0];
			td.v[1] = ts.v[1];
			td.v[2] = ts.v[2];
		};

		ForEach(newTris, triangles, triangle_assign).run(triCount);

		meshResult = new luxrays::ExtTriangleMesh(
				vertCount, triCount, newVertices, newTris,
				newNorms, newUVs, newCols, newAlphas
		);
	}


	// Process simultaneously cache clearing and mesh rebuilding
	void Finalize() {
		tbb::task_group tg;

		// Clear cache
		tg.run([&](){ ErrorCache.clear(); });

		// Compact internal Mesh and Build external mesh
		auto final_task = [&](){
			CompactMesh();
			assert(not meshResult);  // Should be used only once
			RebuildExtMesh();
		};
		tg.run(final_task);

		tg.wait();

	}

	luxrays::ExtTriangleMesh* GetExtMesh() const {
		return meshResult;
	}



private:
	// Main properties (vertices and triangles)
	VertexVector vertices;
	TriangleVector triangles;

	TriangleErrorVector trierrors;

	// Synchronization for vertices
	MutexVector vmutexes;

	// General settings
	const slg::Camera& camera;
	const float edgeScreenSize;
	const bool preserveBorder;

	// LuxCore specific
	bool hasNormals = false;
	bool hasUVs = false;
	bool hasColors = false;
	bool hasAlphas = false;

	// Output
	luxrays::ExtTriangleMesh* meshResult = nullptr;

	ErrorCacheKey makeErrorCacheKey(const SimplifyTriangle& t) const {
		return ErrorCacheKey{
			vertices[t.v[0]].uuid(),
			vertices[t.v[1]].uuid(),
			vertices[t.v[2]].uuid()
		};
	}

	// Cache feature for BuildCandidateList
	// For each triangle, BuildCandidateList computes the best vertex and the
	// associated error, but those computations are both heavy (linear
	// algebra...) and redundant (made more than once per triangle...), so we
	// use a cache. Fundamentally, the key is the triangle, but we do not
	// directly rely on it as the underlying geometrical points can modified.
	// We rather rely on the vertices uuid (which are computed and updated, if
	// needed, for each vertex)
	mutable ErrorCacheType ErrorCache;

	// Iteration initialization
	//
	// Compact triangles, compute quadrics, incidents, borders
	void InitIteration(const size_t iteration) {
		if (iteration > 0) {
			// Compress triangles and mark vertices to keep
			decltype(triangles) newTriangles;
			decltype(trierrors) newTriErrors;
			newTriangles.reserve(triangles.size());
			newTriErrors.reserve(triangles.size());
			for (size_t i = 0; i < triangles.size(); ++i) {
				auto& t = triangles[i];
				if (triangles.deleted(i)) continue;
				auto& e = trierrors[i];
				newTriangles.push_back(std::move(t));
				newTriErrors.push_back(std::move(e));
			}
			triangles = std::move(newTriangles);
			trierrors = std::move(newTriErrors);
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
			InitQuadrics();
			InitBorders();
		}

	}

	// Initialize incident edge tables of vertices
	//
	// Modify: vertices
	void InitIncidentEdges() {

		// Clear previous data
		auto clear_refs = [&](){
			tbb::parallel_for(
				size_t(0), vertices.size(), [&](size_t i){ vertices[i].refs.clear(); }
			);
		};

		// Clear triangle dirty flags
		auto clear_dirty = [&](){
			tbb::parallel_for(
				size_t(0), triangles.size(), [&](size_t i){ triangles.clear_dirty(i); }
			);
		};

		tbb::parallel_invoke(clear_refs, clear_dirty);

		// Build refs
		// Possible race condition: we need mutexes (at vertex grain-scale)
		auto trirange = tbb::blocked_range2d<size_t, size_t>(0, triangles.size(), 0, 3);
		auto build_vertices = [&](tbb::blocked_range2d<size_t, size_t>& r) {
			for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
				const auto& v = triangles[i].v;
				for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
					auto vid = v[j];
					std::lock_guard lock(vmutexes[vid]);
					vertices[vid].refs.emplace_back(i, j);
				}
			}
		};
		tbb::parallel_for(trirange, build_vertices);

		//auto range = tbb::blocked_range<size_t>(0, triangles.size());
		//auto task = [&](tbb::blocked_range<size_t>& r) {
			//for (auto i = r.begin(); i != r.end(); ++i) {
				//const auto& v = triangles[i].v;
				//for (size_t j = 0; j < 3; ++j) {
					//auto vid = v[j];
					//std::lock_guard lock(vmutexes[vid]);
					//vertices[vid].refs.emplace_back(i, j);
				//}
			//}
		//};

		//tbb::parallel_for(range, task);
	}

	// Initialize quadrics on vertices
	//
	// Modify triangles and vertices (q values)
	void InitQuadrics() {

		// Parallel computation of per-triangle quadric contribution
		// And accumulation into vertex quadrics
		MutexVector v_mtx(vertices.size());

		tbb::blocked_range<size_t> triangle_range(0, triangles.size());

		auto quadric_task = [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				SimplifyTriangle &t = triangles[i];

				SimplifyVertex &v0 = vertices[t.v[0]];
				SimplifyVertex &v1 = vertices[t.v[1]];
				SimplifyVertex &v2 = vertices[t.v[2]];

				t.geometryN = TriNormal(v0.p(), v1.p(), v2.p()).normalized();

				const Eigen::Vector4f p(
					t.geometryN[0],
					t.geometryN[1],
					t.geometryN[2],
					-t.geometryN.dot(Point2Vector(v0.p()))
				);
				const Quadric sm(p * p.transpose());

#pragma omp simd
				for (size_t j = 0; j < 3; ++j) {
					auto vertex_index = t.v[j];
					std::scoped_lock lock(v_mtx[vertex_index]);
					vertices[vertex_index].quad.noalias() += sm;
				}
			}
		};
		tbb::parallel_for(triangle_range, quadric_task);

		// Triangle error update
		auto update_task = [&](decltype(triangle_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				trierrors[i] = ComputeTriangleError(triangles[i]);
			}
		};
		tbb::parallel_for(triangle_range, update_task);

	}  // ~InitQuadrics


	// Init border flags on vertices
	//
	// Modify: vertices
	void InitBorders() {
		// vertex border flags are assumed to be initialized to false
		// when vertex is created

		// For each vertex
		tbb::blocked_range<size_t> vertex_range(0, vertices.size());
		auto border_task = [&](const tbb::blocked_range<size_t>& r){
			for (auto i = r.begin() ; i != r.end(); ++i) {
				const auto& v = vertices[i];
				std::map<size_t, size_t> vorders;  // Vertex orders

				// For each triangle incident to the current vertex
				for (const auto& ref: v.refs) {

					// For each vertex of the incident triangle
					for (size_t vid: triangles[ref.tid].v) {
						// Increment incident vertex order
						// If id doesn't exist yet, it will be created (with order=1)
						vorders[vid]++;
					}
				}

				for (auto& p: vorders) {
					if (p.second == 1) {
						// If p.first order is 1, it means that the edge (v, p.first)
						// is referenced by only one triangle, thus it is a border.
						// So we mark p.first to belong to a border
						std::lock_guard lock(vmutexes[p.first]);
						vertices[p.first].border = true;
					}
				}
			}
		};
		tbb::parallel_for(vertex_range, border_task);
	}  // ~InitBorders


	inline float CalculateCollapseScreenErrorScale(
		const SimplifyVertex& v0, const SimplifyVertex& v1
	) const {
		if (edgeScreenSize and not std::signbit(edgeScreenSize)) {
			const Point& p0{v0.p()};
			const Point& p1{v1.p()};
			constexpr float notVisibleScale = .5f;

			float p0x, p0y, p1x, p1y;
			bool notVisibleScale0 = false;
			bool notVisibleScale1 = false;

			auto p0_task = [&] () {
				if (!camera.GetSamplePosition(Eigen2LuxP(p0), &p0x, &p0y) ||
						!luxrays::IsValid(p0x) || !luxrays::IsValid(p0y)) {
					notVisibleScale0 = true;
					return;
				}

				// Normalize
				p0x /= camera.filmWidth;
				p0y /= camera.filmHeight;
			};

			auto p1_task = [&] () {
				if (!camera.GetSamplePosition(Eigen2LuxP(p1), &p1x, &p1y) ||
						!luxrays::IsValid(p1x) || !luxrays::IsValid(p1y)) {
					notVisibleScale1 = true;
					return;
				}

				// Normalize
				p1x /= camera.filmWidth;
				p1y /= camera.filmHeight;
			};

			tbb::parallel_invoke(p0_task, p1_task);
			if (notVisibleScale0 or notVisibleScale1) {
				return notVisibleScale;
			}

			const auto edge = sqrtf(luxrays::Sqr(p0x - p1x) + luxrays::Sqr(p0y - p1y));
			if (not edge) {
				return notVisibleScale;
			}

			return std::max(edge / edgeScreenSize, notVisibleScale);
		} else {
			return 1.f;
		}
	}

	// Calculate collapse point and associated error
	// Returns: error, interpolated point
	inline std::tuple<Point, float> CalculateCollapsePoint(
		const SimplifyVertex& v0, const SimplifyVertex& v1
	) const {
		// Compute resulting quadric
		const Quadric q{v0.quad + v1.quad};

		// Compute interpolated vertex
		const Point &p0 = v0.p();
		const Point &p1 = v1.p();
		const Point p2{(v0.p() + v1.p()) / 2.f};

		// Compute error and associated point

		float error;
		Point pResult;
		if (preserveBorder && v0.border) {
			error = PointError(q, p0);
			pResult = p0;
		} else if (preserveBorder && v1.border) {
			error = PointError(q, p1);
			pResult = p1;
		} else {
			Eigen::Vector3f errors{
				PointError(q, p0),
				PointError(q, p1),
				PointError(q, p2)
			};
			int minIndex;
			error = errors.array().minCoeff(&minIndex);
			switch(minIndex) {
				case 0: pResult = p0; break;
				case 1: pResult = p1; break;
				case 2: pResult = p2; break;
				default: throw std::range_error("Bad point index");
			}
		}

	return std::tuple(std::move(pResult), error);
}

	using CandidateContainer = std::vector<
		SimplifyRef,
		tbb::cache_aligned_allocator<SimplifyRef>
	>;

	// Build candidate list
	//
	CandidateContainer BuildCandidateList(size_t maxCandidateQueueSize) const {

#ifndef NDEBUG
		std::atomic<size_t> cachecalls, cachehits;
#endif


		constexpr char UNDEFINED_INDEX = -1;

		tbb::blocked_range<size_t> tri_range(0, triangles.size());


		auto build_task = [&](const decltype(tri_range)& r, CandidateContainer candidates) {

			candidates.reserve(r.end() - r.begin());
			// Main loop
			for (size_t i = r.begin(); i != r.end(); ++i) {
				const SimplifyTriangle &t = triangles[i];
				char minErrorIndex = UNDEFINED_INDEX;
				float minError = FLOAT_INFINITY;

				// Look into cache whether the triangle has already been computed
				const auto cacheKey(makeErrorCacheKey(t));
#ifndef NDEBUG
				cachecalls++;
#endif
				ErrorCacheType::const_accessor a;
				if (ErrorCache.find(a, cacheKey)) {
					// Hit! --> Just find and unpack...
					minErrorIndex = a->second.minIndex;
					minError = a->second.error;
					a.release();
#ifndef NDEBUG
					cachehits++;
#endif
				} else {
					a.release();
					// No hit: compute and feed cache
					auto& tv0 = t.v[0];
					auto& tv1 = t.v[1];
					auto& tv2 = t.v[2];

#pragma omp simd
					for (size_t j = 0; j < 3; ++j) {
						size_t i0, i1;
						switch(j) {
							case 0: i0 = tv0; i1 = tv1; break;
							case 1: i0 = tv1; i1 = tv2; break;
							case 2: i0 = tv2; i1 = tv0; break;
							default: throw std::logic_error("Bad index");
						}
						const SimplifyVertex &v0 = vertices[i0];
						const SimplifyVertex &v1 = vertices[i1];

						// Border check
						if (preserveBorder) {
							if (v0.border && v1.border) continue;
						} else {
							if (v0.border != v1.border) continue;
						}

						auto&& [p, error] = CalculateCollapsePoint(v0, v1);
						if (Flipped(p, i0, i1)) continue;
						if (Flipped(p, i1, i0)) continue;

						if (trierrors[i][j] < minError) {
							minErrorIndex = j;
							minError = trierrors[i][j];
						}
					}
					ErrorCache.insert(
						std::pair(cacheKey, ErrorCacheEntry(minErrorIndex, minError))
					);
				}

				if (minErrorIndex != UNDEFINED_INDEX) {
					candidates.emplace_back(i, minErrorIndex, minError);
				}
			}
			return candidates;

		};

		auto reduce_task = [&](CandidateContainer a, const CandidateContainer& b) {
			a.reserve(a.size() + b.size());
			std::copy_if(
				std::execution::par,
				b.begin(),
				b.end(),
				std::back_inserter(a),
				[](const SimplifyRef& r){ return r.initialized(); }
			);
			size_t numCandidates = std::min(
					a.size(),
					size_t(maxCandidateQueueSize)
			);

			std::partial_sort(
				std::execution::par,
				a.begin(),
				a.begin() + numCandidates,
				a.end(),
				RefLess
			);
			a.resize(numCandidates);
			return a;
		};

		CandidateContainer identity;

		auto candidates =
			tbb::parallel_reduce(tri_range, identity, build_task, reduce_task);


		DBG_SDL_LOG(
			"Cache stats: hits = " << cachehits
			<< " / total = " << cachecalls
			<< " (size = " << ErrorCache.size() << ")"
		);

		return candidates;

	}  // ~BuildCandidateList


	// Delete triangles
	//
	//
	size_t DeleteTriangles(const CandidateContainer& candidates) {
		tbb::auto_partitioner partitioner;
		tbb::enumerable_thread_specific<size_t> batchDeleted;
		tbb::blocked_range<size_t> batch_range(0, candidates.size());


		auto delete_task = [&](
			const SimplifyRef& candidate,
			tbb::feeder<SimplifyRef>& feeder
		) {
			try {
				// Get explicit edge to collapse
				auto [e0, e1] = EDGES[candidate.tvertex];
				auto e2 = OPPOSITE[candidate.tvertex];
				const size_t i0 = triangles[candidate.tid].v[e0];
				const size_t i1 = triangles[candidate.tid].v[e1];
				const size_t i2 = triangles[candidate.tid].v[e2];

				// Try to lock triangle vertices
				using ulock = std::unique_lock<std::mutex >;
				ulock lk0(vmutexes[i0], std::try_to_lock);
				ulock lk1(vmutexes[i1], std::try_to_lock);
				ulock lk2(vmutexes[i2], std::try_to_lock);
				bool lock_status = bool(lk0) and bool(lk1) and bool(lk2);
				if (not lock_status) throw NoLock();

				auto& v0 = vertices[i0];
				auto& v1 = vertices[i1];

				// Get neighbors (unique, and without edge vertices)
				//
				// Nota: Neighbors are all the vertices that can be affected
				// by given edge collapsing
				RefVector refs;
				refs.insert(refs.end(), v0.refs.begin(), v0.refs.end());
				refs.insert(refs.end(), v1.refs.begin(), v1.refs.end());
				SizeTVector neighbors;
				for (auto& ref : refs) {
					auto vertex_index = triangles[ref.tid].v[ref.tvertex];
					if (
						vertex_index != i0
						and vertex_index != i1
						and vertex_index != i2
					) {
						neighbors.push_back(vertex_index);
					}
				}
				// Compute uniques
				std::sort(neighbors.begin(), neighbors.end());
				auto neighbors_it = std::unique(neighbors.begin(), neighbors.end());
				neighbors.resize(std::distance(neighbors.begin(), neighbors_it));

				batchDeleted.local() += lock_and_collapse(
					candidate,
					neighbors
				);
			} catch(NoLock) {
				// We miss a lock: put the candidate back in the queue
				feeder.add(candidate);
			}
		};
		tbb::parallel_for_each(candidates, delete_task);

		auto deletedTriangles = std::reduce(
			std::execution::par,
			batchDeleted.begin(),
			batchDeleted.end(),
			size_t(0),
			std::plus<size_t>()
		);
		return deletedTriangles;
	}

	// An exception to be raised when we cannot lock the neighborhood
	class NoLock : std::exception {};

	// Lock neighbors and collapse candidate edge
	// In order to benefit from RAII, this function is recursive
	size_t lock_and_collapse (
		const SimplifyRef& candidate,
		SizeTVector& neighbors
	)
	{
		if (not neighbors.empty()) {
			// Get neighbor in the list
			size_t i0 = neighbors.back();
			neighbors.pop_back();

			// Lock current neighbor
			std::unique_lock lock(vmutexes[i0], std::try_to_lock);
			if (not lock) throw NoLock();

			// And call recursively for the next
			size_t res = lock_and_collapse(candidate, neighbors);
			return res;
		} else {
			// Final treatment: collapse
			return CollapseEdge(candidate);
		}
	}



	// Collapse an edge
	// Returns: number of deleted triangles
	// Modifies: triangles, vertices
	size_t CollapseEdge(const SimplifyRef& candidate) {
		// Check triangle
		const size_t triangleIndex = candidate.tid;
		SimplifyTriangle &t = triangles[triangleIndex];

		if (triangles.deleted(triangleIndex) or triangles.dirty(triangleIndex)) {
			return 0;
		}

		// Get explicit edge to collapse
		auto [e1, e2] = EDGES[candidate.tvertex];
		const size_t i0 = t.v[e1];
		const size_t i1 = t.v[e2];

		// Prepare shortcuts
		SimplifyVertex &v0 = vertices[i0];
		SimplifyVertex &v1 = vertices[i1];

		size_t deletedTriangles = 0;

		// Border check
		if (v0.border != v1.border) return 0;

		// Compute vertex' new position (and associated error)
		const auto&& [p, error] = CalculateCollapsePoint(v0, v1);

		// Do not collapse edge if it makes a face flip
		// Get deleted incident triangles
		// deleted0, deleted1: true/false if the triangles referencing the
		// vertex are deleted
		bool flip0, flip1;
		BoolVector deleted0, deleted1;
		tbb::parallel_invoke(
			[&](){ flip0 = Flipped(p, i0, i1); },
			[&](){ flip1 = Flipped(p, i1, i0); },
			[&](){ deleted0 = GetDeletedTriangles(p, i0, i1); },
			[&](){ deleted1 = GetDeletedTriangles(p, i1, i0); }
		);
		if (flip0 or flip1) return 0;

		// Assign new vertex' position and quadric
		v0.setP(p);
		v0.quad += v1.quad;

		// Update LuxCore data
		UpdateLuxData(
			v0, p, vertices[t.v[0]], vertices[t.v[1]], vertices[t.v[2]]
		);

		// Update incident triangles
		RefVector newRefs0, newRefs1;
		size_t numDeletedTriangles0, numDeletedTriangles1;
		tbb::parallel_invoke(
			[&]() {
				std::tie(newRefs0, numDeletedTriangles0) =
					UpdateTriangles(i0, v0, deleted0);
			},
			[&]() {
				std::tie(newRefs1, numDeletedTriangles1) =
					UpdateTriangles(i0, v1, deleted1);
			}
		);

		newRefs0.reserve(newRefs0.size() + newRefs1.size());
		newRefs0.insert(newRefs0.end(), newRefs1.begin(), newRefs1.end());
		v0.refs = std::move(newRefs0);
		deletedTriangles = numDeletedTriangles0 + numDeletedTriangles1;

		return deletedTriangles;
	}

	inline void UpdateLuxData(
		SimplifyVertex& v0,
		const Point& p,
		const SimplifyVertex& tv0,
		const SimplifyVertex& tv1,
		const SimplifyVertex& tv2
	) {
		// Get barycentric coordinates
		auto&& [bcoords_ok, bcoords] = BaryCoords(p, tv0.p(), tv1.p(), tv2.p());

		if (bcoords_ok) {

			if (hasNormals) {
				Eigen::Matrix3f normals;
				normals << tv0.norm, tv1.norm, tv2.norm;
				v0.norm = (normals * bcoords).normalized();
			}
			if (hasUVs) {
				Eigen::Matrix<float, 2, 3> uvs;
				uvs << tv0.uv, tv1.uv, tv2.uv;
				v0.uv = uvs * bcoords;
			}
			if (hasColors) {
				Eigen::Matrix3f colors;
				colors << tv0.col, tv1.col, tv2.col;
				v0.col = colors * bcoords;
			}
			if (hasAlphas) {
				Eigen::Vector3f alphas(tv0.alpha, tv1.alpha, tv2.alpha);
				v0.alpha = alphas.dot(bcoords);
			}
		} else {
			// Must be a malformed triangle
			if (hasNormals) { v0.norm = tv0.norm; }
			if (hasUVs) { v0.uv = tv0.uv; }
			if (hasColors) { v0.col = tv0.col; }
			if (hasAlphas) { v0.alpha = tv0.alpha; }
		}
	}

	// Identify triangles referencing the vertex that have been deleted
	BoolVector GetDeletedTriangles(
		const Point& p,
		const size_t i0,
		const size_t i1
	) const {
		const auto& v0 = vertices[i0];
		BoolVector deleted(v0.refs.size(), false);

		for (size_t k = 0; k < v0.refs.size(); ++k) {
			auto& ref = v0.refs[k];
			const auto& t = triangles[ref.tid];
			if (triangles.deleted(ref.tid)) continue;

			// Compute vertices
			const size_t e0 = ref.tvertex;
			const auto [e1, e2] = EDGES[(e0 + 1) % 3];

			// Mark for deletion?
			deleted[k] = (t.v[e1] == i1 || t.v[e2] == i1);
		}
		return deleted;
	}

	// Check if a triangle flips when this edge is removed
	inline bool Flipped(const Point& p, const size_t i0, const size_t i1) const {

		const auto& v0 = vertices[i0];
		bool res = false;

		for (auto& ref: v0.refs) {
			const auto& t = triangles[ref.tid];

			// Deleted? -> pass
			if (triangles.deleted(ref.tid)) continue;

			const size_t e0 = ref.tvertex;
			const auto [e1, e2] = EDGES[(e0 + 1) % 3];
			const size_t id1 = t.v[e1];
			const size_t id2 = t.v[e2];

			// To be deleted? -> pass
			if (id1 == i1 || id2 == i1) { continue; }

			const auto d1 = Point2Vector(vertices[id1].p() - p);
			const auto d2 = Point2Vector(vertices[id2].p() - p);

			// Check if one of the resulting triangles is too narrow
			{
				// Nota: We use squared norms to avoid square root overhead
				const float dot = d1.dot(d2);
				const float sqrdot = dot * dot;
				const float sqrnorms = d1.squaredNorm() * d2.squaredNorm();
				constexpr float threshold = .999f * .999f;

				if (sqrdot > threshold * sqrnorms) { return true; }
			}

			// Check if one of the Normals is changing side
			{
				// We use squared norms to avoid square root overhead
				const Normal geometryN(d1.cross(d2));
				const float dot = geometryN.dot(t.geometryN);
				const float sqrdot = dot * dot;

				// t.geometryN has already been normalized, so we don't have to
				// multiply by its norm...
				const float sqrnorms = geometryN.squaredNorm();
				constexpr float threshold = .2f * .2f;

				if (std::signbit(dot) or sqrdot < threshold * sqrnorms) {
					return true;
				}
			}

		}  // ~for ref
		return false;
	}  // ~Flipped

	// Compute error for a given triangle
	inline std::array<float, 3> ComputeTriangleError(const SimplifyTriangle& t) const {
		std::array<float, 3> res;
		tbb::parallel_for(
			tbb::blocked_range<uint8_t>(0, 3),
			[&](const tbb::blocked_range<uint8_t>& r) {
				for (auto i = r.begin() ; i != r.end(); ++i) {
					const auto& edge{EDGES[i]};
					const auto [e1, e2] = edge;
					const auto& v0 = vertices[t.v[e1]];
					const auto& v1 = vertices[t.v[e2]];
					auto collapseError = std::get<float>(CalculateCollapsePoint(v0, v1));
					auto screenErrorScale = CalculateCollapseScreenErrorScale(v0, v1);
					res[i] = collapseError * screenErrorScale;
				}
			}
		);
		return res;
	}

	// Update triangle connections and edge error after a edge is collapsed
	// Returns: new incident edges list, number of deleted triangles
	inline std::tuple<RefVector, size_t> UpdateTriangles(
		const size_t i0,
		const SimplifyVertex &v,  // Collapsed vertex
		const BoolVector &deleted
	) {
		size_t deletedTriangles = 0;
		RefVector refs;
		refs.reserve(v.refs.size());
		for (size_t i = 0; i < v.refs.size(); ++i) {
			auto& r = v.refs[i];
			SimplifyTriangle &t = triangles[r.tid];

			if (triangles.deleted(r.tid)) {
				continue;
			}

			if (deleted[i]) {
				triangles.set_deleted(r.tid);
				deletedTriangles++;
				// Update cache (remove triangle)
				auto cachekey = makeErrorCacheKey(t);
				auto status = ErrorCache.erase(cachekey);
				continue;
			}

			// Update cache (remove triangle)
			auto cachekey = makeErrorCacheKey(t);
			ErrorCache.erase(cachekey);

			// We could nearly consider that, as we have locked every vertex
			// in the neigborhood, the triangles are de facto locked too
			t.v[r.tvertex] = i0;
			triangles.set_dirty(r.tid);
			trierrors[r.tid] = ComputeTriangleError(t);

			refs.push_back(r);
		}
		return std::tuple(std::move(refs), std::move(deletedTriangles));
	}

	// Compact mesh before exiting
	void CompactMesh() {


		std::vector<std::atomic_flag, tbb::tbb_allocator<std::atomic_flag>>
			keep(vertices.size());
		tbb::parallel_for(
			size_t(0),
			keep.size(),
			[&keep](size_t i){ keep[i].clear(); }
		);

		// Compress triangles and mark vertices to keep
		auto range = tbb::blocked_range<size_t>(0, triangles.size());
		auto compress_triangles = [&](
			const tbb::blocked_range<size_t>& r, TriangleVector v
		) {
			v.reserve(r.end() - r.begin());

			for (auto i = r.begin(); i != r.end(); ++i) {
				if (triangles.deleted(i)) continue;

				auto& t = triangles[i];
				v.push_back(std::move(t));

				keep[t.v[0]].test_and_set();
				keep[t.v[1]].test_and_set();
				keep[t.v[2]].test_and_set();
			}
			return v;
		};

		TriangleVector newTriangles;

		auto reduce_triangles = [](TriangleVector a, const TriangleVector& b) {
			a.reserve(a.size() + b.size());
			a.insert(a.end(), b.begin(), b.end());
			return a;
		};

		triangles = std::move(
			tbb::parallel_reduce(
				range, newTriangles, compress_triangles, reduce_triangles
			)
		);

		// Compress vertices
		tbb::concurrent_vector<size_t> newIndex, vertMap;
		newIndex.resize(vertices.size());
		vertMap.reserve(vertices.size());

		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, vertices.size()),
			[&](const tbb::blocked_range<size_t>& r) {
				for (size_t i = r.begin(); i != r.end(); ++i) {
					if (not keep[i].test()) {
						continue;
					}
					auto it = vertMap.push_back(i);
					newIndex[i] = it - vertMap.begin();
				}
			}
		);

		VertexVector newVertices(vertMap.size());
		auto range_vertices = tbb::blocked_range<size_t>(0, vertMap.size());
		auto vertex_copy = [&](tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				newVertices[i] = std::move(vertices[vertMap[i]]);
			}
		};
		tbb::parallel_for(range_vertices, vertex_copy);

		vertices = std::move(newVertices);

		// Update triangle vertices with new vertices
		auto range2d = tbb::blocked_range2d<size_t, size_t>(0, triangles.size(), 0, 3);
		auto copy_indices = [&](tbb::blocked_range2d<size_t, size_t>& r) {
			for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
				auto& t = triangles[i];
				for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
					t.v[j] = newIndex[t.v[j]];
					assert(t.v[j] < vertices.size());
				}
			}
		};
		tbb::parallel_for(range2d, copy_indices);

	}

};  // ~class Simplify




}  // ~namespace enhanced
}  // ~namespace (local)

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4


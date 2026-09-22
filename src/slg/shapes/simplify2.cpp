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

// Only compile this file if the feature is enabled

#include <map>
#include <vector>
#include <string>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <cstring> // for memset
#include <functional>

#include <oneapi/tbb.h>
#include <robin_hood.h>

#include <boost/format.hpp>

#include "luxrays/core/trianglemesh.h"
#include "luxrays/usings.h"
#include "luxrays/core/exttrianglemesh.h"
#include "slg/shapes/simplify2.h"
#include "luxrays/utils/buffer.h"
#include "slg/scene/scene.h"
#include "slg/utils/harlequincolors.h"
#include "slg/utils/group_by_equivalence.h"
#include "slg/cameras/camera.h"

using namespace luxrays;

namespace slg {
namespace simplify2 {

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

// SymetricMatrix for quadric error metrics
// The 4x4 symmetric matrix has 10 unique elements:
// [0] = m11, [1] = m12, [2] = m13, [3] = m14,
// [4] = m22, [5] = m23, [6] = m24,
// [7] = m33, [8] = m34,
// [9] = m44

class SymetricMatrix2 {
public:
	// Storage: 10 unique elements of symmetric 4x4 matrix
	float m[10];

	// Default constructor - initialize to zero
	SymetricMatrix2() {
		for (u_int i = 0; i < 10; ++i) {
			m[i] = 0.0f;
		}
	}

	// Constructor with scalar value
	explicit SymetricMatrix2(const float c) {
		for (u_int i = 0; i < 10; ++i) {
			m[i] = c;
		}
	}

	// Constructor with all 10 elements
	SymetricMatrix2(
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

	// Make plane from normal (a,b,c) and distance d: ax+by+cz+d=0
	// This creates the outer product matrix: [a;b;c;d] * [a b c d]
	// For a symmetric matrix, we only store the upper triangular part
	SymetricMatrix2(const float a, const float b, const float c, const float d) {
		// For the plane constructor, SIMD doesn't provide much benefit
		// due to the scattered access pattern. Use scalar operations.
		// This is typically called once per triangle during initialization,
		// not in the hot path.
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

	// Accessor for element
	float operator[](int c) const {
		return m[c];
	}

	// Element accessor for non-const
	float& operator[](int c) {
		return m[c];
	}

	// Determinant of 3x3 submatrix
	// Note: This is for the full 4x4 matrix, but we use specific indices
	float det(
			const u_int a11, const u_int a12, const u_int a13,
			const u_int a21, const u_int a22, const u_int a23,
			const u_int a31, const u_int a32, const u_int a33) const {
		// For a 3x3 submatrix of the 4x4 matrix
		// Using scalar operations as SIMD doesn't help much here
		const float det = m[a11] * m[a22] * m[a33] + m[a13] * m[a21] * m[a32] + m[a12] * m[a23] * m[a31]
				- m[a13] * m[a22] * m[a31] - m[a11] * m[a23] * m[a32] - m[a12] * m[a21] * m[a33];
		return det;
	}

	// Addition
	const SymetricMatrix2 operator+(const SymetricMatrix2 &n) const {
		return SymetricMatrix2(
			m[0] + n[0], m[1] + n[1], m[2] + n[2], m[3] + n[3],
			m[4] + n[4], m[5] + n[5], m[6] + n[6],
			m[7] + n[7], m[8] + n[8],
			m[9] + n[9]);
	}

	// In-place addition
	SymetricMatrix2& operator+=(const SymetricMatrix2& n) {
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
};


class Simplify2 {
public:
	Simplify2(const ExtTriangleMesh &srcMesh) {
		const auto vertCount = srcMesh.GetTotalVertexCount();
		const auto triCount = srcMesh.GetTotalTriangleCount();
		const VertexBuffer verts(srcMesh.GetVertices());
		const TriangleBuffer tris(srcMesh.GetTriangles());

		vertices.resize(vertCount);
		for (u_int i = 0; i < vertCount; ++i)
			vertices[i].p = verts[i];

		if (srcMesh.HasNormals()) {
			const auto& norms = srcMesh.GetNormals();
			for (auto i = 0; i < vertCount; ++i)
				vertices[i].norm = norms[i];

			hasNormals = true;
		} else
			hasNormals = false;

		if (srcMesh.HasUVs(0)) {
			const auto uvs = srcMesh.GetUVs(0);
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].uv = uvs[i];

			hasUVs = true;
		} else
			hasUVs = false;

		if (srcMesh.HasColors(0)) {
			const auto cols = srcMesh.GetColors(0);
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].col = cols[i];

			hasColors = true;
		} else
			hasColors = false;

		if (srcMesh.HasAlphas(0)) {
			const auto alphas = srcMesh.GetAlphas(0);
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

	~Simplify2() {
	}

	ExtTriangleMeshUPtr GetExtMesh() const {
		const u_int vertCount = vertices.size();
		const u_int triCount = triangles.size();

		VertexBuffer newVertices(vertCount);
		for (u_int i = 0; i < vertCount; ++i)
			newVertices[i] = vertices[i].p;

		NormalBuffer newNorms;
		if (hasNormals) {
			newNorms.Allocate(vertCount);
			for (auto i = 0; i < vertCount; ++i)
				newNorms[i] = vertices[i].norm;
		}

		ExtMeshProp<UV>::Layer newUVs = nullptr;
		if (hasUVs) {
			newUVs = std::make_shared<UV[]>(vertCount);
			for (u_int i = 0; i < vertCount; ++i)
				newUVs[i] = vertices[i].uv;
		}

		ExtMeshProp<Spectrum>::Layer newCols = nullptr;
		if (hasColors) {
			newCols = std::make_shared<Spectrum[]>(vertCount);
			for (u_int i = 0; i < vertCount; ++i)
				newCols[i] = vertices[i].col;
		}

		ExtMeshProp<float>::Layer newAlphas = nullptr;
		if (hasAlphas) {
			newAlphas = std::make_shared<float[]>(vertCount);
			for (u_int i = 0; i < vertCount; ++i)
				newAlphas[i] = vertices[i].alpha;
		}

		TriangleBuffer newTris(triCount);
		for (u_int i = 0; i < triCount; ++i) {
			assert (triangles[i].v[0] < vertCount);
			newTris[i].v[0] = triangles[i].v[0];

			assert (triangles[i].v[1] < vertCount);
			newTris[i].v[1] = triangles[i].v[1];

			assert (triangles[i].v[2] < vertCount);
			newTris[i].v[2] = triangles[i].v[2];
		}

		return std::make_unique<ExtTriangleMesh>(
			std::move(newVertices),
			std::move(newTris),
			std::move(newNorms),
			newUVs,
			newCols,
			newAlphas
		);
	}

	void Decimate(const float targetTriangleCount, CameraConstRef scnCamera,
			const float screenSize, const bool border) {
		// TODO: Implement new simplification algorithm here
		// This is where you would put your new implementation

		// For now, just log that we're using the new version
		SDL_LOG("SimplifyShape2: Using experimental simplification algorithm");

		// Call the original algorithm as fallback
		preserveBorder = border;
		camera = &scnCamera;
		edgeScreenSize = screenSize;

		// Work on N% of all triangles for each iteration (keep only N% lowest error candidates)
		// TODO: this should be a parameter (like target), tunable per shape
		const float candidatePercent = 0.3f; // 30%

		// Init
		for (u_int i = 0; i < triangles.size(); ++i)
			triangles[i].deleted = false;

		// Init the screen projection cache (used by UpdateTriangleError when
		// edgeScreenSize > 0)
		vertexScreenX.assign(vertices.size(), 0.f);
		vertexScreenY.assign(vertices.size(), 0.f);
		vertexScreenValid.assign(vertices.size(), false);
		vertexScreenVisible.assign(vertices.size(), false);

		// Main iteration loop
		const u_int startTriangleCount = triangles.size();
		deletedTriangles = 0;
		u_int totalDeletedTriangles = 0;
		for (u_int iteration = 0; iteration < 64; ++iteration) {
			if (startTriangleCount - totalDeletedTriangles <= targetTriangleCount)
				break;

			const double iterationStartTime = WallClockTime();
			double stepStartTime = iterationStartTime;

			// Compact the deleted triangles (iteration > 0), rebuild the vertex
			// references and clear the dirty flags (quadrics, edge errors and
			// border flags are initialized once, at iteration 0)
			UpdateMesh(iteration);
			SDL_LOG("Simplify2: Mesh " << (iteration == 0 ? "initialized" : "updated") << " in "
				<< (boost::format("%.3f") % (WallClockTime() - stepStartTime)) << "secs");

			// Build the edge candidate list and keep only the N% lowest error candidates
			stepStartTime = WallClockTime();
			std::vector<SimplifyRef2> allCandidates;
			allCandidates.reserve(triangles.size());

			// Lambda to compare SimplifyRef2 by error
			auto refCompare = [this](const SimplifyRef2 &a, const SimplifyRef2 &b) {
				return triangles[a.tid].err[a.tvertex] < triangles[b.tid].err[b.tvertex];
			};

			// An empty collapse context: the candidate building only reads
			// the global baseline references (no tail)
			CollapseContext candidateCtx;

			// Evaluate the candidates in parallel: the loop is read-only
			// (CalculateCollapseError and Flipped are const) and each triangle
			// writes only its own slot
			std::vector<u_int> candidateVertexIndex(triangles.size(), NULL_INDEX);
			tbb::parallel_for(size_t(0), triangles.size(),
					[this, &candidateCtx, &candidateVertexIndex](size_t i) {
				const SimplifyTriangle2 &t = triangles[i];

				// Look for the (valid) triangle vertex with the minimum error
				u_int minErrorIndex = NULL_INDEX;
				float minError = std::numeric_limits<float>::infinity();
				for (u_int j = 0; j < 3; ++j) {
					const u_int i0 = t.v[j];
					const SimplifyVertex2 &v0 = vertices[i0];

					const u_int i1 = t.v[(j + 1) % 3];
					const SimplifyVertex2 &v1 = vertices[i1];

					// Border check
					if (preserveBorder) {
						if (v0.border && v1.border)
							continue;
					} else {
						if (v0.border != v1.border)
							continue;
					}

					// Compute vertex to collapse to
					Point p;
					CalculateCollapseError(i0, i1, &p);

					// Don't remove if flipped
					if (Flipped(p, i0, i1, candidateCtx))
						continue;
					if (Flipped(p, i1, i0, candidateCtx))
						continue;

					if (t.err[j] < minError) {
						minErrorIndex = j;
						minError = t.err[j];
					}
				}

				if (minErrorIndex != NULL_INDEX)
					candidateVertexIndex[i] = minErrorIndex;
			});

			// Collect all valid candidates
			for (u_int i = 0; i < triangles.size(); ++i) {
				if (candidateVertexIndex[i] != NULL_INDEX)
					allCandidates.push_back(SimplifyRef2{i, candidateVertexIndex[i]});
			}
			SDL_LOG("Simplify2: Found " << allCandidates.size() << " edge candidates in "
				<< (boost::format("%.3f") % (WallClockTime() - stepStartTime)) << "secs");

			// Keep only the N% lowest error candidates
			const u_int totalCandidateCount = allCandidates.size();
			const u_int nPercentCount = std::max(1u, Floor2UInt(totalCandidateCount * candidatePercent));
			if (allCandidates.size() > nPercentCount) {
				// Select the N% lowest error candidates: nth_element partitions
				// in average O(n) and only the kept prefix needs to be ordered
				// (instead of sorting all the candidates to throw most of them
				// away)
				std::nth_element(allCandidates.begin(), allCandidates.begin() + nPercentCount,
					allCandidates.end(), refCompare);
				allCandidates.resize(nPercentCount);
			}

			// Sort the kept candidates by error (ascending)
			std::sort(allCandidates.begin(), allCandidates.end(), refCompare);
			SDL_LOG("Simplify2: Kept the " << allCandidates.size() << " lowest error candidates ("
				<< (boost::format("%.1f") % (candidatePercent * 100.f)) << "% of " << totalCandidateCount << ")");

			// Copy to candidateList in reverse order (worst first) to match original behavior
			candidateList = allCandidates;
			std::reverse(candidateList.begin(), candidateList.end());

			// Compute candidate neighbourhoods and closures for parallel processing
			stepStartTime = WallClockTime();
			std::vector<robin_hood::unordered_set<u_int>> candidateNeighbourhoods(allCandidates.size());
			tbb::parallel_for(size_t(0), allCandidates.size(),
					[this, &allCandidates, &candidateNeighbourhoods](size_t i) {
				candidateNeighbourhoods[i] = ComputeCandidateNeighbourhood(allCandidates[i]);
			});
			std::vector<std::vector<u_int>> candidateClosures = ComputeCandidateClosures(allCandidates, candidateNeighbourhoods);
			size_t maxClosureSize = 0;
			for (const auto& closure : candidateClosures)
				maxClosureSize = std::max(maxClosureSize, closure.size());
			SDL_LOG("Simplify2: Computed " << candidateClosures.size() << " closures (max size "
				<< maxClosureSize << ") in "
				<< (boost::format("%.3f") % (WallClockTime() - stepStartTime)) << "secs");

			// Process closures in parallel using TBB parallel_reduce
			deletedTriangles = 0;
			stepStartTime = WallClockTime();
			ProcessClosuresParallel(candidateClosures, allCandidates);
			SDL_LOG("Simplify2: Processed " << candidateClosures.size() << " closures in parallel in "
				<< (boost::format("%.3f") % (WallClockTime() - stepStartTime)) << "secs");

			// Note: the closures have disjoint neighbourhoods, so the global
			// triangle flags written by the collapses are race-free and need no
			// merge; only the deleted triangles counter is merged (and the
			// closure disjointness asserted) by applyResult.

			const u_int iterationDeletedTriangles = deletedTriangles;
			totalDeletedTriangles += iterationDeletedTriangles;
			SDL_LOG("Simplify2 iteration " << iteration << " (" << allCandidates.size() << " edge candidates, deleted "
				<< iterationDeletedTriangles << "/" << totalDeletedTriangles << " of " << startTriangleCount
				<< " triangles) in " << (boost::format("%.3f") % (WallClockTime() - iterationStartTime)) << "secs");
			if (iterationDeletedTriangles == 0)
				break;
		}

		// Clean up mesh
		const double compactStartTime = WallClockTime();
		CompactMesh();
		SDL_LOG("Simplify2: Mesh compacted in "
			<< (boost::format("%.3f") % (WallClockTime() - compactStartTime)) << "secs");
	}

private:
	struct SimplifyTriangle2 {
		u_int v[3];
		Normal geometryN;
		float err[3];
		bool deleted, dirty;
	};

	struct SimplifyVertex2 {
		Point p;
		Normal norm;
		UV uv;
		Spectrum col;
		float alpha;

		u_int tstart, tcount;
		SymetricMatrix2 q;

		bool border;
	};

	struct SimplifyRef2 {
		u_int tid, tvertex;
	};

	// Local working state for edge collapses.
	//
	// During the parallel processing of closures, each thread appends the new
	// references to its own tail (read through the global baseline, see
	// GetRef) and counts its own deleted triangles, so that CollapseEdge
	// never mutates the shared reference list. The reference list is rebuilt
	// from scratch by UpdateMesh at each iteration, so the tails are simply
	// dropped at the end of the parallel processing (no merge needed).
	struct CollapseContext {
		std::vector<SimplifyRef2> refsTail;
		u_int deletedCount = 0;
	};

	// Read a reference by logical index: the global baseline plus the tail
	// appended by the collapse context
	const SimplifyRef2 &GetRef(const CollapseContext &ctx, const u_int index) const {
		const u_int baseSize = refs.size();
		return (index < baseSize) ? refs[index] : ctx.refsTail[index - baseSize];
	}

	std::vector<SimplifyTriangle2> triangles;
	std::vector<SimplifyVertex2> vertices;
	std::vector<SimplifyRef2> refs;

	CameraConstPtr camera;
	float edgeScreenSize;

	std::vector<SimplifyRef2> candidateList;

	u_int deletedTriangles;
	bool hasNormals, hasUVs, hasColors, hasAlphas, preserveBorder;

	// Cached screen space projections of the vertices (normalized
	// coordinates), lazily computed and invalidated when a vertex moves.
	// Only used when edgeScreenSize > 0.
	//
	// Race-free during the parallel processing: the entries touched by a
	// closure are all in its neighbourhood, like the other vertex data.
	// The validity/visibility flags are one byte per vertex (not bit packed):
	// the closures have disjoint vertex sets but adjacent vertices can still
	// share a byte, and the bit read-modify-write of e.g. std::vector<bool>
	// would race between closures.
	std::vector<float> vertexScreenX;
	std::vector<float> vertexScreenY;
	std::vector<std::uint8_t> vertexScreenValid;    // the projection has been computed
	std::vector<std::uint8_t> vertexScreenVisible;  // and the vertex is visible

	bool CollapseEdge(const u_int trinagleIndex, const u_int startVertexIndex,
			CollapseContext &ctx, std::vector<bool> &deleted0, std::vector<bool> &deleted1) {
		SimplifyTriangle2 &t = triangles[trinagleIndex];

		if (t.deleted)
			return false;
		if (t.dirty)
			return false;

		const u_int i0 = t.v[startVertexIndex];
		SimplifyVertex2 &v0 = vertices[i0];

		const u_int i1 = t.v[(startVertexIndex + 1) % 3];
		SimplifyVertex2 &v1 = vertices[i1];

		// Border check
		if (v0.border != v1.border)
			return false;

		// Compute vertex to collapse to
		Point p;
		CalculateCollapseError(i0, i1, &p);

		// true/false if the triangles referencing the vertex are deleted
		deleted0.resize(v0.tcount);
		deleted1.resize(v1.tcount);

		// Don't remove if flipped
		if (Flipped(p, i0, i1, ctx, &deleted0))
			return false;
		if (Flipped(p, i1, i0, ctx, &deleted1))
			return false;

		// Save original vertex information
		const Point triPoint0 = vertices[t.v[0]].p;
		const Point triPoint1 = vertices[t.v[1]].p;
		const Point triPoint2 = vertices[t.v[2]].p;

		const Normal triNorm0 = vertices[t.v[0]].norm;
		const Normal triNorm1 = vertices[t.v[1]].norm;
		const Normal triNorm2 = vertices[t.v[2]].norm;

		const UV triUV0 = vertices[t.v[0]].uv;
		const UV triUV1 = vertices[t.v[1]].uv;
		const UV triUV2 = vertices[t.v[2]].uv;

		const Spectrum triCol0 = vertices[t.v[0]].col;
		const Spectrum triCol1 = vertices[t.v[1]].col;
		const Spectrum triCol2 = vertices[t.v[2]].col;

		const float triAlpha0 = vertices[t.v[0]].alpha;
		const float triAlpha1 = vertices[t.v[1]].alpha;
		const float triAlpha2 = vertices[t.v[2]].alpha;

		// Not flipped, so remove edge
		v0.p = p;
		// The vertex moved: invalidate its cached screen projection
		vertexScreenValid[i0] = false;
		v0.q = v1.q + v0.q;

		// Interpolate other vertex attributes
		float b1, b2;
		if (Triangle::GetBaryCoords(
				triPoint0,
				triPoint1,
				triPoint2,
				p, &b1, &b2)) {
			const float b0 = 1.f - b1 - b2;

			if (hasNormals)
				v0.norm = Normalize(b0 * triNorm0 + b1 * triNorm1 + b2 * triNorm2);
			if (hasUVs)
				v0.uv = b0 * triUV0 + b1 * triUV1 + b2 * triUV2;
			if (hasColors)
				v0.col = b0 * triCol0 + b1 * triCol1 + b2 * triCol2;
			if (hasAlphas)
				v0.alpha = b0 * triAlpha0 + b1 * triAlpha1 + b2 * triAlpha2;
		} else {
			// Must be a malformed triangle
			if (hasNormals)
				v0.norm = triNorm0;
			if (hasUVs)
				v0.uv = triUV0;
			if (hasColors)
				v0.col = triCol0;
			if (hasAlphas)
				v0.alpha = triAlpha0;
		}

		const u_int tstart = refs.size() + ctx.refsTail.size();

		UpdateTriangles(i0, v0, deleted0, ctx);
		UpdateTriangles(i0, v1, deleted1, ctx);

		const u_int tcount = (refs.size() + ctx.refsTail.size()) - tstart;

		// Append the new references to the local tail and repoint the vertex.
		// The tail is simply dropped at the end of the parallel processing: the
		// reference list is rebuilt from scratch by UpdateMesh at each
		// iteration, so nothing needs to be merged back.
		v0.tstart = tstart;
		v0.tcount = tcount;

		return true;
	}

	// Check if a triangle flips when this edge is removed
	bool Flipped(const Point &p, const u_int i0, const u_int i1,
			const CollapseContext &ctx,
			std::vector<bool> *deleted = nullptr) const {
		const SimplifyVertex2 &v0 = vertices[i0];

		for (u_int k = 0; k < v0.tcount; ++k) {
			const SimplifyRef2 &ref = GetRef(ctx, v0.tstart + k);
			const SimplifyTriangle2 &t = triangles[ref.tid];

			if (t.deleted)
				continue;

			const u_int s = ref.tvertex;
			const u_int id1 = t.v[(s + 1) % 3];
			const u_int id2 = t.v[(s + 2) % 3];

			// Delete ?
			if (id1 == i1 || id2 == i1) {
				if (deleted)
					(*deleted)[k] = true;
				continue;
			}

			// Check if the triangle is too narrow
			const Vector d1 = Normalize(vertices[id1].p - p);
			const Vector d2 = Normalize(vertices[id2].p - p);
			if (AbsDot(d1, d2) > .999f)
				return true;

			// Check if the Normal is changing side
			const Normal geometryN(Normalize(Cross(d1, d2)));
			if (Dot(geometryN, t.geometryN) < .2f)
				return true;

			if (deleted)
				(*deleted)[k] = false;
		}

		return false;
	}

	// Update triangle connections and edge error after a edge is collapsed
	void UpdateTriangles(const u_int i0, const SimplifyVertex2 &v,
			const std::vector<bool> &deleted, CollapseContext &ctx) {
		for (u_int k = 0; k < v.tcount; ++k) {
			const SimplifyRef2 &r = GetRef(ctx, v.tstart + k);
			SimplifyTriangle2 &t = triangles[r.tid];

			if (t.deleted)
				continue;

			if (deleted[k]) {
				t.deleted = true;
				ctx.deletedCount++;
				continue;
			}

			t.v[r.tvertex] = i0;
			t.dirty = true;
			UpdateTriangleError(t);

			ctx.refsTail.push_back(r);
		}
	}

	// Compact triangles, compute edge error and build reference list
	void UpdateMesh(const u_int iteration) {
		if (iteration > 0) {
			// Compact triangles
			int dst = 0;
			for (u_int i = 0; i < triangles.size(); ++i)
				if (!triangles[i].deleted)
					triangles[dst++] = triangles[i];

			triangles.resize(dst);
		}

		// Init Quadrics by Plane & Edge Errors
		//
		// Required at the beginning (iteration == 0)
		//
		if (iteration == 0) {
			for (u_int i = 0; i < vertices.size(); ++i)
				vertices[i].q = SymetricMatrix2(0.0);

			for (u_int i = 0; i < triangles.size(); ++i) {
				SimplifyTriangle2 &t = triangles[i];

				SimplifyVertex2 &v0 = vertices[t.v[0]];
				SimplifyVertex2 &v1 = vertices[t.v[1]];
				SimplifyVertex2 &v2 = vertices[t.v[2]];

				const Normal geometryN(Normalize(Cross(v1.p - v0.p, v2.p - v0.p)));
				t.geometryN = geometryN;

				// It doesn't matter what vertex I use here because the triangle
				// plane will pass for all 3
				const SymetricMatrix2 sm(geometryN.x, geometryN.y, geometryN.z,
						-Dot(Vector(geometryN), Vector(v0.p)));
				v0.q += sm;
				v1.q += sm;
				v2.q += sm;
			}

			for (u_int i = 0; i < triangles.size(); ++i) {
				// Calc Edge Error
				SimplifyTriangle2 &t = triangles[i];

				UpdateTriangleError(t);
			}
		}

		// Init Reference ID list
		for (u_int i = 0; i < vertices.size(); ++i) {
			vertices[i].tstart = 0;
			vertices[i].tcount = 0;
		}

		for (u_int i = 0; i < triangles.size(); ++i) {
			SimplifyTriangle2 &t = triangles[i];

			vertices[t.v[0]].tcount++;
			vertices[t.v[1]].tcount++;
			vertices[t.v[2]].tcount++;
		}

		u_int tstart = 0;
		for (u_int i = 0; i < vertices.size(); ++i) {
			SimplifyVertex2 &v = vertices[i];

			v.tstart = tstart;
			tstart += v.tcount;
			v.tcount = 0;
		}

		// Write References
		refs.resize(triangles.size() * 3);
		for (u_int i = 0; i < triangles.size(); ++i) {
			SimplifyTriangle2 &t = triangles[i];

			for (u_int j = 0; j < 3; ++j) {
				SimplifyVertex2 &v = vertices[t.v[j]];

				refs[v.tstart + v.tcount].tid = i;
				refs[v.tstart + v.tcount].tvertex = j;

				v.tcount++;
			}
		}

		// Identify boundary : vertices[].border=0,1
		//
		// Required at the beginning (iteration == 0)
		if (iteration == 0) {
			for (u_int i = 0; i < vertices.size(); ++i)
				vertices[i].border = false;

			std::vector<u_int> vcount, vids;
			for (u_int i = 0; i < vertices.size(); ++i) {
				SimplifyVertex2 &v = vertices[i];
				vcount.clear();
				vids.clear();

				for (u_int j = 0; j < v.tcount; ++j) {
					int k = refs[v.tstart + j].tid;
					SimplifyTriangle2 &t = triangles[k];

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

		// Clear dirty flag
		for (u_int i = 0; i < triangles.size(); ++i)
			triangles[i].dirty = false;
	}  // UpdateMesh

	// Finally compact mesh before exiting
	void CompactMesh() {
		u_int dst = 0;

		for (u_int i = 0; i < vertices.size(); ++i)
			vertices[i].tcount = 0;

		for (u_int i = 0; i < triangles.size(); ++i) {
			if (!triangles[i].deleted) {
				const SimplifyTriangle2 &t = triangles[i];
				triangles[dst++] = t;

				vertices[t.v[0]].tcount = 1;
				vertices[t.v[1]].tcount = 1;
				vertices[t.v[2]].tcount = 1;
			}
		}
		triangles.resize(dst);

		dst = 0;
		for (u_int i = 0; i < vertices.size(); ++i) {
			if (vertices[i].tcount) {
				vertices[i].tstart = dst;
				vertices[dst].p = vertices[i].p;

				vertices[dst].norm = vertices[i].norm;
				vertices[dst].uv = vertices[i].uv;
				vertices[dst].col = vertices[i].col;
				vertices[dst].alpha = vertices[i].alpha;

				dst++;
			}
		}

		for (u_int i = 0; i < triangles.size(); ++i) {
			SimplifyTriangle2 &t = triangles[i];

			t.v[0] = vertices[t.v[0]].tstart;
			t.v[1] = vertices[t.v[1]].tstart;
			t.v[2] = vertices[t.v[2]].tstart;
		}
		vertices.resize(dst);
	}

	// Error between vertex and Quadric
	float VertexError(const SymetricMatrix2 &q, const float x, const float y, const float z) const {
		return q[0] * x * x + 2.f * q[1] * x * y + 2.f * q[2] * x * z + 2.f * q[3] * x +
			q[4] * y * y + 2.f * q[5] * y * z + 2.f * q[6] * y +
			q[7] * z * z + 2.f * q[8] * z +
			q[9];
	}

	// Error for one edge
	float CalculateCollapseError(const u_int v1Index, const u_int v2Index,
			Point *pResult = nullptr) const {
		const SymetricMatrix2 q = vertices[v1Index].q + vertices[v2Index].q;

		// Compute interpolated vertex
		const Point &p1 = vertices[v1Index].p;
		const Point &p2 = vertices[v2Index].p;
		const Point p3 = (p1 + p2) / 2;

		// Error can be negative, I add 1 to have screenErrorScale can than
		// work as expected
		const float error1 = VertexError(q, p1.x, p1.y, p1.z) + 1.f;
		const float error2 = VertexError(q, p2.x, p2.y, p2.z) + 1.f;
		const float error3 = VertexError(q, p3.x, p3.y, p3.z) + 1.f;

		float error;
		if (preserveBorder && vertices[v1Index].border) {
			error = error1;
			if (pResult)
				*pResult = p1;
		} else if (preserveBorder && vertices[v2Index].border) {
			error = error2;
			if (pResult)
				*pResult = p2;
		} else {
			error = Min(error1, Min(error2, error3));

			if (pResult) {
				if (error1 == error)
					*pResult = p1;
				if (error2 == error)
					*pResult = p2;
				if (error3 == error)
					*pResult = p3;
			}
		}

		// Adding 1.0 because error have negative values
		return std::max(error + 1.f, 0.f);
	}

	// Get the screen space projection (normalized coordinates) of a vertex,
	// with lazy caching. Returns false if the vertex is not visible (the
	// result is cached anyway: the camera projection is expensive).
	bool GetScreenPosition(const u_int vertexIndex, float * const x, float * const y) {
		if (vertexScreenValid[vertexIndex]) {
			*x = vertexScreenX[vertexIndex];
			*y = vertexScreenY[vertexIndex];
			return vertexScreenVisible[vertexIndex];
		}

		float px, py;
		const bool visible = camera->GetSamplePosition(vertices[vertexIndex].p, &px, &py) &&
				IsValid(px) && IsValid(py);

		if (visible) {
			// Normalize
			px /= camera->filmWidth;
			py /= camera->filmHeight;
		}

		vertexScreenX[vertexIndex] = px;
		vertexScreenY[vertexIndex] = py;
		vertexScreenValid[vertexIndex] = true;
		vertexScreenVisible[vertexIndex] = visible;

		*x = px;
		*y = py;
		return visible;
	}

	// Update the collapse errors of a triangle: quadric error scaled by the
	// screen error scale (one cached camera projection per vertex instead
	// of two per edge)
	void UpdateTriangleError(SimplifyTriangle2 &t) {
		t.err[0] = CalculateCollapseError(t.v[0], t.v[1]);
		t.err[1] = CalculateCollapseError(t.v[1], t.v[2]);
		t.err[2] = CalculateCollapseError(t.v[2], t.v[0]);

		if (edgeScreenSize > 0.f) {
			const float notVisibleScale = .5f;

			float sx[3], sy[3];
			bool visible[3];
			for (u_int j = 0; j < 3; ++j) {
				visible[j] = GetScreenPosition(t.v[j], &sx[j], &sy[j]);
			}

			for (u_int j = 0; j < 3; ++j) {
				const u_int j1 = (j + 1) % 3;

				float scale;
				if (visible[j] && visible[j1]) {
					const float edge = sqrtf(Sqr(sx[j] - sx[j1]) + Sqr(sy[j] - sy[j1]));
					scale = (edge == 0.f) ? notVisibleScale :
							std::max(edge / edgeScreenSize, notVisibleScale);
				} else
					scale = notVisibleScale;

				t.err[j] *= scale;
			}
		}
	}

	// Computes the neighbourhood of a candidate edge collapse
	// The neighbourhood includes all vertices that would be impacted by collapsing
	// the edge (v0, v1) in triangle tid
	robin_hood::unordered_set<u_int> ComputeCandidateNeighbourhood(const SimplifyRef2& candidate) const {
		robin_hood::unordered_set<u_int> neighbourhood;
		const SimplifyTriangle2& t = triangles[candidate.tid];
		const u_int v0_idx = t.v[candidate.tvertex];
		const u_int v1_idx = t.v[(candidate.tvertex + 1) % 3];

		// The two vertices of the edge being collapsed are always in the neighbourhood
		neighbourhood.insert(v0_idx);
		neighbourhood.insert(v1_idx);

		// Add all vertices connected to v0
		for (u_int k = 0; k < vertices[v0_idx].tcount; ++k) {
			const SimplifyRef2& ref = refs[vertices[v0_idx].tstart + k];
			const SimplifyTriangle2& tri = triangles[ref.tid];
			for (u_int j = 0; j < 3; ++j) {
				u_int vid = tri.v[j];
				if (vid != v0_idx && vid != v1_idx) {
					neighbourhood.insert(vid);
				}
			}
		}

		// Add all vertices connected to v1
		for (u_int k = 0; k < vertices[v1_idx].tcount; ++k) {
			const SimplifyRef2& ref = refs[vertices[v1_idx].tstart + k];
			const SimplifyTriangle2& tri = triangles[ref.tid];
			for (u_int j = 0; j < 3; ++j) {
				u_int vid = tri.v[j];
				if (vid != v0_idx && vid != v1_idx) {
					neighbourhood.insert(vid);
				}
			}
		}

		return neighbourhood;
	}

	// Computes candidate closures (connected components in the neighbour graph)
	//
	// Two candidates are connected if their neighbourhoods share a vertex.
	// Instead of testing all the O(n^2) candidate pairs, a vertex -> candidates
	// reverse index is built: all the candidates in the same bucket share a
	// vertex, i.e. they are all neighbour-connected, so the (chained) buckets
	// define the equivalence relation given to GroupByEquivalence (which
	// relies on a parallel Union-Find).
	std::vector<std::vector<u_int>> ComputeCandidateClosures(const std::vector<SimplifyRef2>& candidates,
			const std::vector<robin_hood::unordered_set<u_int>>& candidateNeighbourhoods) {
		const u_int candidateCount = candidates.size();
		if (candidateCount == 0) {
			return {};
		}

		// Build the vertex -> candidates reverse index as a CSR structure
		// (bucket count, prefix sum, bucket fill on flat arrays): no hashing
		// and no per bucket allocation.
		// (Kept serial: the count/fill updates of a vertex would be shared by
		// all the candidates reading it, so the parallel version would need
		// contended atomics.)
		const u_int vertexCount = vertices.size();
		std::vector<u_int> bucketStart(vertexCount + 1, 0);
		for (u_int i = 0; i < candidateCount; ++i) {
			for (const u_int v : candidateNeighbourhoods[i]) {
				++bucketStart[v + 1];
			}
		}
		for (u_int v = 0; v < vertexCount; ++v) {
			bucketStart[v + 1] += bucketStart[v];
		}

		std::vector<u_int> bucketEntries(bucketStart[vertexCount]);
		{
			std::vector<u_int> bucketCursor(bucketStart.begin(), bucketStart.end() - 1);
			for (u_int i = 0; i < candidateCount; ++i) {
				for (const u_int v : candidateNeighbourhoods[i]) {
					bucketEntries[bucketCursor[v]++] = i;
				}
			}
		}

		// Build the equivalence relation: all the candidates in a bucket are
		// neighbour-connected (chaining each bucket is enough to express it)
		std::vector<Relation> relations;
		relations.reserve(bucketEntries.size());
		for (u_int v = 0; v < vertexCount; ++v) {
			for (u_int k = bucketStart[v] + 1; k < bucketStart[v + 1]; ++k) {
				relations.push_back(Relation(bucketEntries[k - 1], bucketEntries[k]));
			}
		}

		if (relations.empty()) {
			// No connections at all: each candidate is its own closure
			std::vector<std::vector<u_int>> closures;
			closures.reserve(candidateCount);
			for (u_int i = 0; i < candidateCount; ++i)
				closures.push_back(std::vector<u_int>{i});

			return closures;
		}

		// Group the connected candidates with the parallel Union-Find
		const Classes classes = GroupByEquivalence(candidateCount, RelationSpan(relations));

		// Convert the classes to closures (the GroupByEquivalence classes come
		// with their members in ascending order, i.e. ascending error: the
		// greedy processing order of the collapses)
		std::vector<std::vector<u_int>> closures;
		closures.reserve(classes.size());
		for (const auto& indices : classes)
			closures.push_back(std::vector<u_int>(indices.begin(), indices.end()));

		return closures;
	}

	// Class for processing closures in parallel using TBB parallel_reduce.
	//
	// The closures have disjoint neighbourhoods, so the triangle and vertex
	// updates performed by CollapseEdge (including the global deleted/dirty
	// flags) are race-free and need no merge: each body only appends
	// thread-local references (CollapseContext, dropped at the end: the
	// reference list is rebuilt at each iteration) and counts its deleted
	// triangles.
	class ParallelClosureProcessor {
		Simplify2& simplify;
		const std::vector<std::vector<u_int>>& closures;
		const std::vector<SimplifyRef2>& allCandidates;

		// Local state: appended refs tail and deleted triangles counter
		CollapseContext ctx;

		// Candidate triangles deleted by this body (for the disjointness check)
		std::vector<u_int> deletedCandidates;

	public:
		// Constructor for the master thread
		ParallelClosureProcessor(Simplify2& s,
				const std::vector<std::vector<u_int>>& c,
				const std::vector<SimplifyRef2>& a)
			: simplify(s), closures(c), allCandidates(a) {
			ctx.deletedCount = s.deletedTriangles;
		}

		// Split constructor for TBB
		ParallelClosureProcessor(ParallelClosureProcessor& other, tbb::split)
			: simplify(other.simplify), closures(other.closures), allCandidates(other.allCandidates) {}

		// Process a range of closures
		void operator()(const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i < r.end(); ++i) {
				ProcessClosure(closures[i]);
			}
		}

		// Join (reduce step): merge the sibling's counter and disjointness data
		void join(ParallelClosureProcessor& other) {
			ctx.deletedCount += other.ctx.deletedCount;
			deletedCandidates.insert(deletedCandidates.end(),
					other.deletedCandidates.begin(), other.deletedCandidates.end());
		}

		// Check the closure disjointness and merge the deleted triangles
		// counter. The global triangle flags are already up to date: they are
		// written by the collapses themselves.
		void applyResult() {
			// Assertion: a candidate triangle cannot be deleted in two
			// different closure processes
			std::sort(deletedCandidates.begin(), deletedCandidates.end());
			for (size_t i = 1; i < deletedCandidates.size(); ++i) {
				if (deletedCandidates[i] == deletedCandidates[i - 1]) {
					SDL_LOG("ERROR: Triangle " << deletedCandidates[i] << " was deleted in multiple closures!");
					SDL_LOG("  This indicates a bug in closure computation - neighbourhoods overlap.");
					assert(false && "Triangle deleted in multiple closures - neighbourhoods overlap!");
				}
			}

			simplify.deletedTriangles = ctx.deletedCount;
		}

	private:
		// Process a single closure
		void ProcessClosure(const std::vector<u_int>& closureIndices) {
			std::vector<bool> deleted0, deleted1;

			// Process each candidate in the closure in order
			for (u_int idx : closureIndices) {
				const SimplifyRef2& candidate = allCandidates[idx];

				// Skip if the triangle was already deleted (e.g. by an
				// earlier collapse in this closure)
				if (simplify.triangles[candidate.tid].deleted) {
					continue;
				}

				// Try to collapse this edge
				const bool success = simplify.CollapseEdge(candidate.tid, candidate.tvertex, ctx, deleted0, deleted1);

				if (success) {
					// Record the collapsed candidate for the disjointness check
					deletedCandidates.push_back(candidate.tid);
				}
			}
		}
	};

	// Process all closures using TBB parallel_reduce
	void ProcessClosuresParallel(const std::vector<std::vector<u_int>>& closures,
			const std::vector<SimplifyRef2>& allCandidates) {
		if (closures.empty()) {
			return;
		}

		// Process the largest closures first to limit the load imbalance: the
		// candidates of a closure interact, so a closure is processed serially
		// and the biggest ones must start as early as possible
		std::vector<std::vector<u_int>> sortedClosures(closures.begin(), closures.end());
		std::sort(sortedClosures.begin(), sortedClosures.end(),
				[](const std::vector<u_int>& a, const std::vector<u_int>& b) {
					return a.size() > b.size();
				});

		// Use parallel_reduce to process closures in parallel.
		//
		// The closures have disjoint neighbourhoods so the updates of triangles
		// and vertices performed by CollapseEdge (including the global
		// deleted/dirty flags) are race-free and need no merge.
		ParallelClosureProcessor processor(*this, sortedClosures, allCandidates);

		// Grain size: process at least 1 closure per thread
		const size_t grain_size = std::max<size_t>(1, sortedClosures.size() / tbb::this_task_arena::max_concurrency());

		SDL_LOG("Simplify2: Processing " << sortedClosures.size() << " closures on "
			<< tbb::this_task_arena::max_concurrency() << " threads (grain size " << grain_size << ")");

		tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, sortedClosures.size(), grain_size),
			processor
		);

		// Check the closure disjointness and merge the deleted triangles counter
		processor.applyResult();
	}

};

} // namespace simplify2

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

} // namespace slg

namespace slg {

using namespace luxrays;
using namespace slg;

SimplifyShape2::SimplifyShape2(CameraConstPtr camera, ExtTriangleMeshRef srcMesh,
		const float target, const float edgeScreenSize, const bool preserveBorder) {
	SDL_LOG("SimplifyShape2: Creating simplified shape " << srcMesh.GetName() << " with target " << target);

	if ((edgeScreenSize > 0.f) && !camera)
		throw std::runtime_error("The scene.GetCamera() must be defined in order to enable simplify edgescreensize option");

	const double startTime = WallClockTime();

	const u_int targetCount = std::max(1u, Floor2UInt(srcMesh.GetTotalTriangleCount() * target));

	simplify2::Simplify2 simplify(srcMesh);
	simplify.Decimate(targetCount, *camera, edgeScreenSize, preserveBorder);
	mesh = simplify.GetExtMesh();

	SDL_LOG("SimplifyShape2: Simplified shape from " << srcMesh.GetTotalTriangleCount() << " to " << mesh->GetTotalTriangleCount() << " faces");

	const double endTime = WallClockTime();
	SDL_LOG("SimplifyShape2 time: " << (boost::format("%.3f") % (endTime - startTime)) << "secs");
}

SimplifyShape2::~SimplifyShape2() {
}

ExtTriangleMeshUPtr SimplifyShape2::RefineImpl(SceneConstRef scene) {
	return std::move(mesh);
}

// Feature flag function
bool IsSimplify2Enabled() {
	return true;
}

} // namespace slg

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

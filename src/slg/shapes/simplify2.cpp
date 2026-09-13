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

// This is the EXPERIMENTAL version of Simplify shape
// Enable with LUXCORE_ENABLE_SIMPLIFY2 CMake option

// Only compile this file if the feature is enabled
#if LUXCORE_SIMPLIFY2_ENABLED

#include <map>
#include <vector>
#include <string>
#include <queue>
#include <limits>
#include <cstring> // for memset

#include <boost/format.hpp>

// SIMD intrinsics
#if defined(__SSE__)
#include <xmmintrin.h>  // SSE
#elif defined(__AVX__)
#include <immintrin.h>  // AVX
#endif

#include "luxrays/core/trianglemesh.h"
#include "luxrays/usings.h"
#include "luxrays/core/exttrianglemesh.h"
#include "slg/shapes/simplify2.h"
#include "luxrays/utils/buffer.h"
#include "slg/scene/scene.h"
#include "slg/utils/harlequincolors.h"
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

// Check if SIMD is available
#if defined(__SSE__) || defined(__AVX__) || defined(__ARM_NEON)
    #define SIMPLIFY2_USE_SIMD 1
#endif

// SIMD-optimized SymetricMatrix for quadric error metrics
// Uses SoA (Structure of Arrays) layout for better SIMD utilization
// The 4x4 symmetric matrix has 10 unique elements:
// [0] = m11, [1] = m12, [2] = m13, [3] = m14,
// [4] = m22, [5] = m23, [6] = m24,
// [7] = m33, [8] = m34,
// [9] = m44

class SymetricMatrix2 {
public:
	// Storage: 10 unique elements of symmetric 4x4 matrix
	// Layout optimized for cache locality and SIMD access
	float m[10];

	// Default constructor - initialize to zero
	SymetricMatrix2() {
		// Use memset for faster zero-initialization
		#ifdef SIMPLIFY2_USE_SIMD
			// Zero 3 x 4 = 12 floats (covers 10 elements)
			__m128* ptr = reinterpret_cast<__m128*>(m);
			ptr[0] = _mm_setzero_ps();
			ptr[1] = _mm_setzero_ps();
			// Zero remaining 2 floats
			m[8] = 0.0f;
			m[9] = 0.0f;
		#else
			for (u_int i = 0; i < 10; ++i) {
				m[i] = 0.0f;
			}
		#endif
	}

	// Constructor with scalar value
	explicit SymetricMatrix2(const float c) {
		#ifdef SIMPLIFY2_USE_SIMD
			// Broadcast scalar to all 10 elements
			const __m128 scalar = _mm_set1_ps(c);
			__m128* ptr = reinterpret_cast<__m128*>(m);
			ptr[0] = scalar;
			ptr[1] = scalar;
			m[8] = c;
			m[9] = c;
		#else
			for (u_int i = 0; i < 10; ++i) {
				m[i] = c;
			}
		#endif
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

	// SIMD-optimized addition
	const SymetricMatrix2 operator+(const SymetricMatrix2 &n) const {
		#ifdef SIMPLIFY2_USE_SIMD
			SymetricMatrix2 result;
			// Process 4 elements at a time with SIMD
			__m128* src1 = reinterpret_cast<const __m128*>(m);
			__m128* src2 = reinterpret_cast<const __m128*>(n.m);
			__m128* dst = reinterpret_cast<__m128*>(result.m);
			
			// Process first 8 elements (2 x 4-vector SIMD)
			dst[0] = _mm_add_ps(src1[0], src2[0]);
			dst[1] = _mm_add_ps(src1[1], src2[1]);
			
			// Process remaining 2 elements
			result.m[8] = m[8] + n.m[8];
			result.m[9] = m[9] + n.m[9];
			
			return result;
		#else
			return SymetricMatrix2(
				m[0] + n[0], m[1] + n[1], m[2] + n[2], m[3] + n[3],
				m[4] + n[4], m[5] + n[5], m[6] + n[6],
				m[7] + n[7], m[8] + n[8],
				m[9] + n[9]);
		#endif
	}

	// SIMD-optimized in-place addition
	SymetricMatrix2& operator+=(const SymetricMatrix2& n) {
		#ifdef SIMPLIFY2_USE_SIMD
			// Process 4 elements at a time with SIMD
			__m128* src = reinterpret_cast<const __m128*>(n.m);
			__m128* dst = reinterpret_cast<__m128*>(m);
			
			// Process first 8 elements (2 x 4-vector SIMD)
			dst[0] = _mm_add_ps(dst[0], src[0]);
			dst[1] = _mm_add_ps(dst[1], src[1]);
			
			// Process remaining 2 elements
			m[8] += n.m[8];
			m[9] += n.m[9];
		#else
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
		#endif
		
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

		// Work on 10% of all triangles for each iteration
		maxCandidateQueueSize = std::max(64u, Floor2UInt(triangles.size() * .1f));

		// Init
		for (u_int i = 0; i < triangles.size(); ++i)
			triangles[i].deleted = false;

		// Main iteration loop
		const u_int startTriangleCount = triangles.size();
		deletedTriangles = 0;
		std::vector<bool> deleted0, deleted1;
		for (u_int iteration = 0; iteration < 64; ++iteration) {
			if (startTriangleCount - deletedTriangles <= targetTriangleCount)
				break;

			const u_int initialdeletedTriangles = deletedTriangles;

			// Update mesh constantly
			UpdateMesh(iteration);

			// Remove vertices & mark deleted triangles
			for (u_int i = 0; i < candidateList.size(); ++i)
				CollapseEdge(candidateList[i].tid, candidateList[i].tvertex, deleted0, deleted1);

			const u_int iterationDeletedTriangles = deletedTriangles - initialdeletedTriangles;
			SDL_LOG("Simplify2 iteration " << iteration << " (" << candidateList.size() << " edge candidates, deleted " << iterationDeletedTriangles << "/" << deletedTriangles << " of " << startTriangleCount << " triangles)");
			if (iterationDeletedTriangles == 0)
				break;
		}

		// Clean up mesh
		CompactMesh();
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

	class SimplifyRefErrCompare2 {
	public:
		SimplifyRefErrCompare2(const Simplify2 &s) : simplify(s) { }

		bool operator()(const SimplifyRef2 &sr1, const SimplifyRef2 &sr2) const {
			return simplify.triangles[sr1.tid].err[sr1.tvertex] < simplify.triangles[sr2.tid].err[sr2.tvertex];
		}

	private:
		const Simplify2 &simplify;
	};

	std::vector<SimplifyTriangle2> triangles;
	std::vector<SimplifyVertex2> vertices;
	std::vector<SimplifyRef2> refs;

	CameraConstPtr camera;
	float edgeScreenSize;

	u_int maxCandidateQueueSize;
	std::vector<SimplifyRef2> candidateList;

	u_int deletedTriangles;
	bool hasNormals, hasUVs, hasColors, hasAlphas, preserveBorder;

	bool CollapseEdge(const u_int trinagleIndex, const u_int startVertexIndex,
			std::vector<bool> &deleted0, std::vector<bool> &deleted1) {
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
		if (Flipped(p, i0, i1, &deleted0))
			return false;
		if (Flipped(p, i1, i0, &deleted1))
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

		const u_int tstart = refs.size();

		UpdateTriangles(i0, v0, deleted0);
		UpdateTriangles(i0, v1, deleted1);

		const u_int tcount = refs.size() - tstart;

		if (tcount <= v0.tcount) {
			// Save ram
			if (tcount)
				std::copy(&refs[tstart], &refs[tstart] + tcount, &refs[v0.tstart]);
		} else
			// Append
			v0.tstart = tstart;

		v0.tcount = tcount;

		return true;
	}

	// Check if a triangle flips when this edge is removed
	bool Flipped(const Point &p, const u_int i0, const u_int i1,
			std::vector<bool> *deleted = nullptr) const {
		const SimplifyVertex2 &v0 = vertices[i0];

		for (u_int k = 0; k < v0.tcount; ++k) {
			const SimplifyTriangle2 &t = triangles[refs[v0.tstart + k].tid];

			if (t.deleted)
				continue;

			const u_int s = refs[v0.tstart + k].tvertex;
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
			const std::vector<bool> &deleted) {
		for (u_int k = 0; k < v.tcount; ++k) {
			const SimplifyRef2 &r = refs[v.tstart + k];
			SimplifyTriangle2 &t = triangles[r.tid];

			if (t.deleted)
				continue;

			if (deleted[k]) {
				t.deleted = true;
				deletedTriangles++;
				continue;
			}

			t.v[r.tvertex] = i0;
			t.dirty = true;
			UpdateTriangleError(t);

			refs.push_back(r);
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
				}			}
		}

		// Build the edge candidate queue
		std::priority_queue<SimplifyRef2, std::vector<SimplifyRef2>, SimplifyRefErrCompare2>
			candidateQueue{ SimplifyRefErrCompare2(*this) };
		for (u_int i = 0; i < triangles.size(); ++i) {
			const SimplifyTriangle2 &t = triangles[i];

			// Look for the (valid) triangle vertex with the minimum error
			u_int minErrorIndex = NULL_INDEX;
			float minError = std::numeric_limits<float>::infinity();
			for (u_int j = 0; j < 3; ++j) {
				const u_int i0 = t.v[j];
				SimplifyVertex2 &v0 = vertices[i0];

				const u_int i1 = t.v[(j + 1) % 3];
				SimplifyVertex2 &v1 = vertices[i1];

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
				if (Flipped(p, i0, i1))
					continue;
				if (Flipped(p, i1, i0))
					continue;

				if (t.err[j] < minError) {
					minErrorIndex = j;
					minError = t.err[j];
				}
			}
				
			if (minErrorIndex == NULL_INDEX)
				continue;
			
			if (candidateQueue.size() < maxCandidateQueueSize) {
				candidateQueue.push(SimplifyRef2{i, minErrorIndex});
				continue;
			}

			const SimplifyRef2 &top = candidateQueue.top();
			if (t.err[minErrorIndex] < triangles[top.tid].err[top.tvertex]) {
				candidateQueue.pop();
				candidateQueue.push(SimplifyRef2{i, minErrorIndex});
			}
		}

		if (candidateQueue.size() > 0) {
			candidateList.resize(candidateQueue.size());
			for (u_int i = candidateList.size() - 1;;) {
				candidateList[i] = candidateQueue.top();
				candidateQueue.pop();

				if (i == 0)
					break;
				--i;
			}

			/*for (u_int i = 0; i < Min<u_int>(candidateList.size(), 10u); ++i) {
				const SimplifyTriangle2 &t = triangles[candidateList[i].tid];

				SDL_LOG("#" << i << " Min. error: " << fixed << setprecision(10) << t.err[candidateList[i].tvertex] << " (triangle " << candidateList[i].tid << ")");

				const u_int i0 = t.v[candidateList[i].tvertex];
				SimplifyVertex2 &v0 = vertices[i0];
				
				const u_int i1 = t.v[(candidateList[i].tvertex + 1) % 3];
				SimplifyVertex2 &v1 = vertices[i1];

				SDL_LOG("#" << i << " Collapse screen error scale: " << fixed << setprecision(10) <<  CalculateCollapseScreenErrorScale(v0.p, v1.p));
				SDL_LOG("#" << i << " Triangle " << candidateList[i].tid << " border: " <<
						vertices[t.v[candidateList[i].tvertex]].border << " " <<
						vertices[t.v[(candidateList[i].tvertex + 1) % 3]].border);
			}*/
			
			/*ExtTriangleMeshBuilder meshBuilder;
			for (u_int i = 0; i < candidateList.size(); ++i) {
				const SimplifyTriangle2 &t = triangles[candidateList[i].tid];

				const u_int index = meshBuilder.vertices.size();
				meshBuilder.AddVertex(vertices[t.v[candidateList[i].tvertex]].p);
				meshBuilder.AddVertex(vertices[t.v[(candidateList[i].tvertex + 1) % 3]].p);
				meshBuilder.AddVertex(vertices[t.v[(candidateList[i].tvertex + 2) % 3]].p);

				meshBuilder.AddTriangle(Triangle(index, index + 1, index +2));
			}
			ExtTriangleMesh *debugMesh = meshBuilder.GetExtTriangleMesh();
			debugMesh->Save("debug-candidates.ply");
			delete debugMesh;*/
		}

		// Clear dirty flag
		for (u_int i = 0; i < triangles.size(); ++i)
			triangles[i].dirty = false;
	}

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

	float CalculateCollapseScreenErrorScale(const Point &v0, const Point &v1) const {
		if (edgeScreenSize > 0.f) {
			const float notVisibleScale = .5f;

			float v0x, v0y;
			if (!camera->GetSamplePosition(v0, &v0x, &v0y) ||
					!IsValid(v0x) || !IsValid(v0y))
				return notVisibleScale;

			// Normalize
			v0x /= camera->filmWidth;
			v0y /= camera->filmHeight;

			float v1x, v1y;
			if (!camera->GetSamplePosition(v1, &v1x, &v1y) ||
					!IsValid(v1x) || !IsValid(v1y))
				return notVisibleScale;

			// Normalize
			v1x /= camera->filmWidth;
			v1y /= camera->filmHeight;

			const float edge = sqrtf(Sqr(v0x - v1x) + Sqr(v0y - v1y));
			if (edge == 0.f)
				return notVisibleScale;

			return std::max(edge / edgeScreenSize, notVisibleScale);
		} else
			return 1.f;
	}
	
	void UpdateTriangleError(SimplifyTriangle2 &t) const {
		t.err[0] = CalculateCollapseError(t.v[0], t.v[1]) *
				CalculateCollapseScreenErrorScale(vertices[t.v[0]].p, vertices[t.v[1]].p);

		t.err[1] = CalculateCollapseError(t.v[1], t.v[2]) *
				CalculateCollapseScreenErrorScale(vertices[t.v[1]].p, vertices[t.v[2]].p);

		t.err[2] = CalculateCollapseError(t.v[2], t.v[0]) *
				CalculateCollapseScreenErrorScale(vertices[t.v[2]].p, vertices[t.v[0]].p);
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
		throw runtime_error("The scene.GetCamera() must be defined in order to enable simplify edgescreensize option");

	const float startTime = WallClockTime();

	const u_int targetCount = std::max(1u, Floor2UInt(srcMesh.GetTotalTriangleCount() * target));

	simplify2::Simplify2 simplify(srcMesh);
	simplify.Decimate(targetCount, *camera, edgeScreenSize, preserveBorder);
	mesh = simplify.GetExtMesh();

	SDL_LOG("SimplifyShape2: Simplified shape from " << srcMesh.GetTotalTriangleCount() << " to " << mesh->GetTotalTriangleCount() << " faces");

	const float endTime = WallClockTime();
	SDL_LOG("SimplifyShape2 time: " << (boost::format("%.3f") % (endTime - startTime)) << "secs");
}

SimplifyShape2::~SimplifyShape2() {
}

ExtTriangleMeshUPtr SimplifyShape2::RefineImpl(SceneConstRef scene) {
	return std::move(mesh);
}

// Feature flag function
bool IsSimplify2Enabled() {
	// This will be controlled by CMake option LUXCORE_ENABLE_SIMPLIFY2
	return true;
}

} // namespace slg

#endif // LUXCORE_SIMPLIFY2_ENABLED

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

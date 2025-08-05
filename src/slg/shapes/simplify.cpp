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
namespace simple {

using namespace std;
using namespace luxrays;
using namespace slg;

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


class Simplify {
public:
	Simplify(const ExtTriangleMesh &srcMesh) {
		const u_int vertCount = srcMesh.GetTotalVertexCount();
		const u_int triCount = srcMesh.GetTotalTriangleCount();
		const Point *verts = srcMesh.GetVertices();
		const Triangle *tris = srcMesh.GetTriangles();

		vertices.resize(vertCount);
		for (u_int i = 0; i < vertCount; ++i)
			vertices[i].p = verts[i];
		
		if (srcMesh.HasNormals()) {
			const Normal *norms = srcMesh.GetNormals();
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].norm = norms[i];

			hasNormals = true;
		} else
			hasNormals = false;
		
		if (srcMesh.HasUVs(0)) {
			const UV *uvs = srcMesh.GetUVs(0);
			for (u_int i = 0; i < vertCount; ++i)
				vertices[i].uv = uvs[i];

			hasUVs = true;
		} else
			hasUVs = false;
		
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

	~Simplify() {
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

	void Decimate(const float targetTriangleCount, const Camera& scnCamera,
			const float screenSize, const bool border) {
		preserveBorder = border;
		camera = &scnCamera;
		edgeScreenSize = screenSize;

		// Work on 10% of all triangles for each iteration
		maxCandidateQueueSize = Max(64u, Floor2UInt(triangles.size() * .1f));

		// Init
		for (u_int i = 0; i < triangles.size(); ++i)
			triangles[i].deleted = false;

		// Main iteration loop
		const u_int startTriangleCount = triangles.size();
		deletedTriangles = 0;
		vector<bool> deleted0, deleted1;
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
			SDL_LOG("Simplify iteration " << iteration << " (" << candidateList.size() << " edge candidates, deleted " << iterationDeletedTriangles << "/" << deletedTriangles << " of " << startTriangleCount << " triangles)");
			if (iterationDeletedTriangles == 0)
				break;
		}

		// Clean up mesh
		CompactMesh();
	}

private:
	struct SimplifyTriangle {
		u_int v[3];
		Normal geometryN;
		float err[3];
		bool deleted, dirty;
	};

	struct SimplifyVertex {
		Point p;
		Normal norm;
		UV uv;
		Spectrum col;
		float alpha;

		u_int tstart, tcount;
		SymetricMatrix q;

		bool border;

	};

	struct SimplifyRef {
		u_int tid, tvertex;
	};
	class SimplifyRefErrCompare {
	public:
		SimplifyRefErrCompare(const Simplify &s) : simplify(s) { }

		bool operator()(const SimplifyRef &sr1, const SimplifyRef &sr2) const {
			return simplify.triangles[sr1.tid].err[sr1.tvertex] < simplify.triangles[sr2.tid].err[sr2.tvertex];
		}

	private:
		const Simplify &simplify;
	};

	vector<SimplifyTriangle> triangles;
	vector<SimplifyVertex> vertices;
	vector<SimplifyRef> refs;

	const Camera *camera;
	float edgeScreenSize;

	u_int maxCandidateQueueSize;
	vector<SimplifyRef> candidateList;

	u_int deletedTriangles;
	bool hasNormals, hasUVs, hasColors, hasAlphas, preserveBorder;

	bool CollapseEdge(const u_int trinagleIndex, const u_int startVertexIndex,
			vector<bool> &deleted0, vector<bool> &deleted1) {
		SimplifyTriangle &t = triangles[trinagleIndex];

		if (t.deleted)
			return false;
		if (t.dirty)
			return false;

		const u_int i0 = t.v[startVertexIndex];
		SimplifyVertex &v0 = vertices[i0];

		const u_int i1 = t.v[(startVertexIndex + 1) % 3];
		SimplifyVertex &v1 = vertices[i1];

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
				copy(&refs[tstart], &refs[tstart] + tcount, &refs[v0.tstart]);
		} else
			// Append
			v0.tstart = tstart;

		v0.tcount = tcount;

		return true;
	}

	// Check if a triangle flips when this edge is removed
	bool Flipped(const Point &p, const u_int i0, const u_int i1,
			vector<bool> *deleted = nullptr) const {
		const SimplifyVertex &v0 = vertices[i0];

		for (u_int k = 0; k < v0.tcount; ++k) {
			const SimplifyTriangle &t = triangles[refs[v0.tstart + k].tid];

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
	void UpdateTriangles(const u_int i0, const SimplifyVertex &v,
			const  vector<bool> &deleted) {
		for (u_int k = 0; k < v.tcount; ++k) {
			const SimplifyRef &r = refs[v.tstart + k];
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
				vertices[i].q = SymetricMatrix(0.0);

			for (u_int i = 0; i < triangles.size(); ++i) {
				SimplifyTriangle &t = triangles[i];

				SimplifyVertex &v0 = vertices[t.v[0]];
				SimplifyVertex &v1 = vertices[t.v[1]];
				SimplifyVertex &v2 = vertices[t.v[2]];

				const Normal geometryN(Normalize(Cross(v1.p - v0.p, v2.p - v0.p)));
				t.geometryN = geometryN;

				// It doesn't matter what vertex I use here because the triangle
				// plane will pass for all 3
				const SymetricMatrix sm(geometryN.x, geometryN.y, geometryN.z,
						-Dot(Vector(geometryN), Vector(v0.p)));
				v0.q += sm;
				v1.q += sm;
				v2.q += sm;
			}

			for (u_int i = 0; i < triangles.size(); ++i) {
				// Calc Edge Error
				SimplifyTriangle &t = triangles[i];

				UpdateTriangleError(t);
			}
		}

		// Init Reference ID list
		for (u_int i = 0; i < vertices.size(); ++i) {
			vertices[i].tstart = 0;
			vertices[i].tcount = 0;
		}

		for (u_int i = 0; i < triangles.size(); ++i) {
			SimplifyTriangle &t = triangles[i];

			vertices[t.v[0]].tcount++;
			vertices[t.v[1]].tcount++;
			vertices[t.v[2]].tcount++;
		}

		u_int tstart = 0;
		for (u_int i = 0; i < vertices.size(); ++i) {
			SimplifyVertex &v = vertices[i];

			v.tstart = tstart;
			tstart += v.tcount;
			v.tcount = 0;
		}

		// Write References
		refs.resize(triangles.size() * 3);
		for (u_int i = 0; i < triangles.size(); ++i) {
			SimplifyTriangle &t = triangles[i];

			for (u_int j = 0; j < 3; ++j) {
				SimplifyVertex &v = vertices[t.v[j]];

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

			vector<u_int> vcount, vids;
			for (u_int i = 0; i < vertices.size(); ++i) {
				SimplifyVertex &v = vertices[i];
				vcount.clear();
				vids.clear();

				for (u_int j = 0; j < v.tcount; ++j) {
					int k = refs[v.tstart + j].tid;
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

		// Build the edge candidate queue
		priority_queue<SimplifyRef, vector<SimplifyRef>, SimplifyRefErrCompare>
			candidateQueue{ SimplifyRefErrCompare(*this) };
		for (u_int i = 0; i < triangles.size(); ++i) {
			const SimplifyTriangle &t = triangles[i];

			// Look for the (valid) triangle vertex with the minimum error
			u_int minErrorIndex = NULL_INDEX;
			float minError = numeric_limits<float>::infinity();
			for (u_int j = 0; j < 3; ++j) {
				const u_int i0 = t.v[j];
				SimplifyVertex &v0 = vertices[i0];

				const u_int i1 = t.v[(j + 1) % 3];
				SimplifyVertex &v1 = vertices[i1];

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
				candidateQueue.push(SimplifyRef{i, minErrorIndex});
				continue;
			}

			const SimplifyRef &top = candidateQueue.top();
			if (t.err[minErrorIndex] < triangles[top.tid].err[top.tvertex]) {
				candidateQueue.pop();
				candidateQueue.push(SimplifyRef{i, minErrorIndex});
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
				const SimplifyTriangle &t = triangles[candidateList[i].tid];

				SDL_LOG("#" << i << " Min. error: " << fixed << setprecision(10) << t.err[candidateList[i].tvertex] << " (triangle " << candidateList[i].tid << ")");

				const u_int i0 = t.v[candidateList[i].tvertex];
				SimplifyVertex &v0 = vertices[i0];
			
				const u_int i1 = t.v[(candidateList[i].tvertex + 1) % 3];
				SimplifyVertex &v1 = vertices[i1];

				SDL_LOG("#" << i << " Collapse screen error scale: " << fixed << setprecision(10) <<  CalculateCollapseScreenErrorScale(v0.p, v1.p));
				SDL_LOG("#" << i << " Triangle " << candidateList[i].tid << " border: " <<
						vertices[t.v[candidateList[i].tvertex]].border << " " <<
						vertices[t.v[(candidateList[i].tvertex + 1) % 3]].border);
			}*/
			
			/*ExtTriangleMeshBuilder meshBuilder;
			for (u_int i = 0; i < candidateList.size(); ++i) {
				const SimplifyTriangle &t = triangles[candidateList[i].tid];

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
				const SimplifyTriangle &t = triangles[i];
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
			SimplifyTriangle &t = triangles[i];

			t.v[0] = vertices[t.v[0]].tstart;
			t.v[1] = vertices[t.v[1]].tstart;
			t.v[2] = vertices[t.v[2]].tstart;
		}
		vertices.resize(dst);
	}

	// Error between vertex and Quadric
	float VertexError(const SymetricMatrix &q, const float x, const float y, const float z) const {
		return q[0] * x * x + 2.f * q[1] * x * y + 2.f * q[2] * x * z + 2.f * q[3] * x +
				q[4] * y * y + 2.f * q[5] * y * z + 2.f * q[6] * y +
				q[7] * z * z + 2.f * q[8] * z +
				q[9];
	}

	// Error for one edge
	float CalculateCollapseError(const u_int v1Index, const u_int v2Index,
			Point *pResult = nullptr) const {
		const SymetricMatrix q = vertices[v1Index].q + vertices[v2Index].q;

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
		return Max(error + 1.f, 0.f);
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

			return Max(edge / edgeScreenSize, notVisibleScale);
		} else
			return 1.f;
	}
	
	void UpdateTriangleError(SimplifyTriangle &t) const {
		t.err[0] = CalculateCollapseError(t.v[0], t.v[1]) *
				CalculateCollapseScreenErrorScale(vertices[t.v[0]].p, vertices[t.v[1]].p);

		t.err[1] = CalculateCollapseError(t.v[1], t.v[2]) *
				CalculateCollapseScreenErrorScale(vertices[t.v[1]].p, vertices[t.v[2]].p);

		t.err[2] = CalculateCollapseError(t.v[2], t.v[0]) *
				CalculateCollapseScreenErrorScale(vertices[t.v[2]].p, vertices[t.v[0]].p);
	}
};

}

// Namespace containing the rewriting of the algo
namespace enhanced {


using BoolVector = std::vector<bool, tbb::cache_aligned_allocator<bool>>;
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


// Quadric and References
using Quadric = Eigen::Matrix4f;
const auto Upper = Eigen::UpLoType::Upper;

struct
alignas(std::hardware_destructive_interference_size)
SimplifyRef {

	SimplifyRef(size_t p_tid, size_t p_tvertex, float p_error=0.f):
		tid(p_tid), tvertex(p_tvertex), error(p_error) {}
	SimplifyRef() = default;

	bool initialized() const { return tvertex >= 0; }

	size_t getTid() const { return tid; }
	int8_t getTVertex() const { return tvertex; }
	float getError() const { return error; }

private:
	size_t tid = 0;  // Triangle ID
	int8_t tvertex = -1;  // Vertex in triangle, should be in [0;3] (-1: not set)
	float error;  // To prioritize candidates
};

// TODO remove
//bool RefLess(const SimplifyRef& left, const SimplifyRef& right) {
	//return left.getError() < right.getError();
//}

//TODO
//using RefVector = std::vector< SimplifyRef, tbb::cache_aligned_allocator<SimplifyRef> >;
using RefVector = tbb::concurrent_vector< SimplifyRef, tbb::cache_aligned_allocator<SimplifyRef> >;


// Vertex
struct
alignas(std::hardware_destructive_interference_size)
SimplifyVertex {

	// Constructor
	SimplifyVertex() = default;
	SimplifyVertex(
		const Point& p_point,
		const Quadric& p_quad,
		const bool p_border
	) : quad(p_quad), border(p_border)
	{
		setP(p_point);
	}

	// Ref handlers
	inline const RefVector& getRefs() const { return refs; }
	inline const SimplifyRef& getRef(size_t i) const { return refs[i]; }
	inline void setRefs(const RefVector& p_refs) { refs = p_refs; }
	inline void setRefs(RefVector&& p_refs) { refs = p_refs; }
	inline void reserveRefs(size_t size) {
		refs.reserve(size);
	}
	inline void clearRefs() {
		refs.clear();
	}
	inline void emplaceBackRefs(size_t p_tid, int8_t p_tvertex) {
		refs.emplace_back(p_tid, p_tvertex);
	}
	inline void pushBackRefs(const SimplifyRef& ref) {
		refs.push_back(ref);
	}
	inline size_t getSizeRefs() const { return refs.size(); }

	// Other core accessors
	inline const Quadric& getQuad() const { return quad; }
	inline void addQuad(const Quadric& other) { quad += other; }
	inline bool getBorder() const { return border; }
	inline void setBorder(bool p_border) { border = p_border; }

	inline const Point& getP() const { return m_p; }
	inline void setP(const Point p) {
		m_p = p;
		// Update uuid
		m_uuid = 0;
		boost::hash_combine(m_uuid, p[0]);
		boost::hash_combine(m_uuid, p[1]);
		boost::hash_combine(m_uuid, p[2]);
	}

	inline size_t uuid() const { return m_uuid; }

	// Lux accessors
	inline const Normal& getNormal() const { return norm; }
	inline void setNormal(const Normal& p_norm) { norm = p_norm; }
	inline void setNormal(Normal&& p_norm) { norm = p_norm; }

	inline const Eigen::Vector2f& getUV() const { return uv; }
	inline void setUV(const Eigen::Vector2f& p_uv) { uv = p_uv; }
	inline void setUV(Eigen::Vector2f&& p_uv) { uv = p_uv; }

	inline const Eigen::Vector3f& getCol() const { return col; }
	inline void setCol(const Eigen::Vector3f& p_col) { col = p_col; }
	inline void setCol(Eigen::Vector3f&& p_col) { col = p_col; }

	inline float getAlpha() const { return alpha; }
	inline void setAlpha(float p_alpha) { alpha = p_alpha; }

	inline bool operator==(const SimplifyVertex& other) {
		return m_p == other.m_p;
	}

protected:
	RefVector refs;		  // Incident edges in topology

	// Error computation inputs
	Quadric quad = Quadric::Zero();     // Quadric
	bool border;						// Border status

	// LuxCore specific data
	Normal norm;
	Eigen::Vector2f uv;
	Eigen::Vector3f col;
	float alpha;

	// Position in space
	Point m_p;			  // Position in geometry

	// Unique identifier
	size_t m_uuid;

};

using VertexVector = std::vector<
	SimplifyVertex,
	tbb::cache_aligned_allocator<SimplifyVertex>
>;


// Error between vertex and Quadric
inline float VertexError(const Quadric &q, const Point& p) {
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

// Candidate for decimation
struct alignas(128) Candidate {

	Candidate() = default;
	Candidate(
		bool p_valid,
		const SimplifyRef& p_ref,
		Point p_point,
		float p_error
	) : valid(p_valid), ref(p_ref), collapsePoint(p_point), error(p_error)
	{}

	bool getValid() const { return valid; }
	const SimplifyRef& getRef() const { return ref; }
	const Point& getCollapsePoint() const { return collapsePoint; }
	float getError() const { return error; }

private:
	bool valid = false;  // Will be set to true if this candidate is a valid one
	SimplifyRef ref;
	Point collapsePoint;
	float error = std::numeric_limits<float>::max();
};

bool CandidateLess(const Candidate& left, const Candidate& right) {
	return left.getError() < right.getError();
}

// Triangle
struct
alignas(std::hardware_destructive_interference_size)
SimplifyTriangle {
	// Static data
	std::array<size_t, 3> v;  // Vertex indices
	Normal geometryN;

	// Dynamic data
	bool deleted = false;
	bool dirty = false;
	std::array<float, 3> errors{0.f, 0.f, 0.f};


	const std::optional<Candidate>& getCandidate(
		size_t p_uuid0, size_t p_uuid1, size_t p_uuid2
	) const {
		if (p_uuid0 == uuid0 and p_uuid1 == uuid1 and p_uuid2 == uuid2) {
			return optCandidate;
		} else {
			return NullCandidate;
		}
	}

	void setCandidate(
		const Candidate& candidate,
		size_t p_uuid0,
		size_t p_uuid1,
		size_t p_uuid2
	) const {
		uuid0 = p_uuid0;
		uuid1 = p_uuid1;
		uuid2 = p_uuid2;
		optCandidate.emplace(candidate);
	}

	private:
	// Cache data
	mutable size_t uuid0, uuid1, uuid2;  // Vertex uuid
	mutable std::optional<Candidate> optCandidate;  // Associated candidate
	static constexpr std::optional<Candidate> NullCandidate{};

};

using TriangleVector = std::vector<
	SimplifyTriangle,
	tbb::cache_aligned_allocator<SimplifyTriangle>
>;

using TriangleIterator = TriangleVector::iterator;

// TODO Rename into CandidateCache

// Cache feature for candidate building

// Key is a triangle, identified by its geometric points
struct alignas(128) ErrorCacheKey {
	const Point p0;
	const Point p1;
	const Point p2;
	const Normal n;

	ErrorCacheKey(
		const Point& p_p0,
		const Point& p_p1,
		const Point& p_p2,
		const Normal& p_n
	) : p0(p_p0), p1(p_p1), p2(p_p2), n(p_n)
	{}



};

using ErrorCacheEntry = bool;

//struct alignas(128) ErrorCacheEntry {
	//// vertex index in triangle and error
	//uint16_t minIndex;  // 16
	//float error; // 32
	//uint16_t pad[5];  // 80
//};

struct ErrorCacheHash {
	//size_t hash(const ErrorCacheKey& k) {
	static size_t hash(const ErrorCacheKey& k) {
		size_t seed = hashPoint(k.p0);
		seed ^= hashPoint(k.p1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		seed ^= hashPoint(k.p2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		seed ^= hashNormal(k.n) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		return seed;
	}
	static bool equal(const ErrorCacheKey& k0, const ErrorCacheKey& k1) {
		return (
			k0.p0 == k1.p0 and k0.p1 == k1.p1 and k0.p2 == k1.p2 and k0.n == k0.n
		);
	}

	private:
	static size_t hashPoint(const Point& p) {
		size_t seed = std::hash<float>{}(p[0]);
		seed ^= std::hash<float>{}(p[1]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		seed ^= std::hash<float>{}(p[2]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		return seed;
	}
	static size_t hashNormal(const Normal& n) {
		size_t seed = std::hash<float>{}(n[0]);
		seed ^= std::hash<float>{}(n[1]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		seed ^= std::hash<float>{}(n[2]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		return seed;
	}
};

//struct alignas(128) ErrorCachePair : public std::pair<const ErrorCacheKey, ErrorCacheEntry> {};
using ErrorCachePair = std::pair<const ErrorCacheKey, ErrorCacheEntry>;

using ErrorCacheType = tbb::concurrent_hash_map<
	ErrorCacheKey,
	ErrorCacheEntry,
	ErrorCacheHash,
	tbb::cache_aligned_allocator<ErrorCachePair>
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
		ErrorCache(size_t(vertices.size() * 2))
	{
		using SV = SimplifyVertex;


		// Size vertices and triangles containers in accordance to inputs
		size_t vertCount = srcMesh.GetTotalVertexCount();
		size_t triCount = srcMesh.GetTotalTriangleCount();

		auto init_vertex = [](SV& vd, const luxrays::Point& vs){
			vd.setP(Lux2EigenP(vs));
			vd.reserveRefs(9);  // Seems reasonable
		};
		ForEach(vertices, srcMesh.GetVertices(), init_vertex).run(vertCount);

		if (srcMesh.HasNormals()) {
			auto init_normal = [](SV& v, const luxrays::Normal& n)
				{ v.setNormal(Normal(n.x, n.y, n.z)); };
			ForEach(vertices, srcMesh.GetNormals(), init_normal).run(vertCount);

			hasNormals = true;
		}

		if (srcMesh.HasUVs(0)) {
			auto init_uv = [](SV& v, const luxrays::UV& u)
				{ v.setUV(Eigen::Vector2f(u.u, u.v)); };
			ForEach(vertices, srcMesh.GetUVs(0), init_uv).run(vertCount);

			hasUVs = true;
		}

		if (srcMesh.HasColors(0)) {
			auto init_col = [](SV& v, const luxrays::Spectrum& col)
				{ v.setCol(Vector(col.c)); };
			ForEach(vertices, srcMesh.GetColors(0), init_col).run(vertCount);

			hasColors = true;
		}

		if (srcMesh.HasAlphas(0)) {
			auto init_alpha = [](SV& v, const float alpha){ v.setAlpha(alpha); };
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
		//Eigen::initParallel();

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
			auto candidates = BuildCandidateList(maxCandidateQueueSize);

			// Delete triangles (run batches)
			DBG_SDL_LOG(
				"Simplify - Delete triangles"
				<< " #" << iteration
				);
			const auto iterationDeletedTriangles = DeleteTriangles2(candidates);

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

		// TODO Debug
		for (auto& t: triangles) {
			assert(t.v[0] < vertices.size());
			assert(t.v[1] < vertices.size());
			assert(t.v[2] < vertices.size());
		}
		DBG_SDL_LOG("Vertices.size: " << vertices.size());

	}

	// Rebuild an output mesh after simplification
	// Result is written in meshResult class property
	void RebuildExtMesh() {
		const size_t vertCount = vertices.size();
		const size_t triCount = triangles.size();
		using SV = SimplifyVertex;

		// TODO Debug
		for (auto& t: triangles) {
			assert(t.v[0] < vertices.size());
			assert(t.v[1] < vertices.size());
			assert(t.v[2] < vertices.size());
		}

		DBG_SDL_LOG("Vertices.size: " && vertices.size());
		luxrays::Point *newVertices =
			luxrays::ExtTriangleMesh::AllocVerticesBuffer(vertCount);
		auto vert_assign = [](luxrays::Point& vd, const SV& vs) {
			vd = Eigen2LuxP(vs.getP());
		};
		ForEach(newVertices, vertices, vert_assign).run(vertCount);

		luxrays::Normal *newNorms = nullptr;
		if (hasNormals) {
			newNorms = new luxrays::Normal[vertCount];

			auto norm_assign = [](luxrays::Normal& vd, const SV& vs) {
				vd = Eigen2LuxN(vs.getNormal());
			};
			ForEach(newNorms, vertices, norm_assign).run(vertCount);
		}

		luxrays::UV *newUVs = nullptr;
		if (hasUVs) {
			newUVs = new luxrays::UV[vertCount];
			auto uv_assign = [](luxrays::UV& vd, const SV& vs) {
				vd.u = vs.getUV()[0];
				vd.v = vs.getUV()[1];
			};
			ForEach(newUVs, vertices, uv_assign).run(vertCount);
		}

		luxrays::Spectrum *newCols = nullptr;
		if (hasColors) {
			newCols = new luxrays::Spectrum[vertCount];
			auto col_assign = [](luxrays::Spectrum& vd, const SV& vs) {
				vd.c[0] = vs.getCol()[0];
				vd.c[1] = vs.getCol()[1];
				vd.c[2] = vs.getCol()[2];
			};
			ForEach(newCols, vertices, col_assign).run(vertCount);
		}

		float *newAlphas = nullptr;
		if (hasAlphas) {
			newAlphas = new float[vertCount];
			auto alpha_assign = [](float& vd, const SV& vs) {
				vd = vs.getAlpha();
			};
			ForEach(newAlphas, vertices, alpha_assign).run(vertCount);
		}

		luxrays::Triangle *newTris =
			luxrays::ExtTriangleMesh::AllocTrianglesBuffer(triCount);


		auto triangle_assign =
			[](luxrays::Triangle& td, const SimplifyTriangle& ts) {
			if (not ts.deleted) {
				td.v[0] = ts.v[0];
				td.v[1] = ts.v[1];
				td.v[2] = ts.v[2];
			}
		};

		ForEach(newTris, triangles, triangle_assign).run(triCount);

		// TODO
		for (auto& t: triangles) {
			assert(t.v[0] < vertCount);
			assert(t.v[1] < vertCount);
			assert(t.v[2] < vertCount);
		}

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

			// TODO
			DBG_SDL_LOG("Check consistence triangles/vertices");
			for (auto& t: triangles) {
				assert(t.v[0] < vertices.size());
				assert(t.v[1] < vertices.size());
				assert(t.v[2] < vertices.size());
			}

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

	// Synchronization & multithreading
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
#if 0
		if (iteration > 0) {
			// Compress triangles
			auto not_deleted = [](const SimplifyTriangle& t){ return !t.deleted; };
			decltype(triangles) newTriangles;
			newTriangles.reserve(triangles.size());
			//for (auto& t: triangles | std::views::filter(not_deleted)) {
			for (size_t i = 0; i < triangles.size(); ++i) {
				auto& t = triangles[i];
				if (t.deleted) continue;
				newTriangles.push_back(t);
			}
			triangles = std::move(newTriangles);
		}
#endif



		// Build per-vertex incident edge tables
		//

		// Init Quadrics by Plane & Edge Errors
		//
		// Identify boundary : vertices[].border=0,1
		//
		// Required at the beginning (iteration == 0)
		//
		if (iteration == 0) {
			InitIncidentEdges();
			InitQuadrics();
			InitBorders();
		}

	}

	// Initialize incident edge tables of vertices
	//
	// Modify: vertices
	void InitIncidentEdges() {

		// Clear previous data
		for (auto& v: vertices) {
			v.clearRefs();
		}

		// Resize mutexes
		MutexVector newVmutexes(vertices.size());
		vmutexes = std::move(newVmutexes);

		// Build refs
		// Possible race condition: we need mutexes (at vertex grain-scale)
		auto range = tbb::blocked_range<size_t>(0, triangles.size());
		auto task = [&](tbb::blocked_range<size_t>& r) {
			for (auto tid = r.begin(); tid != r.end(); ++tid) {
				if (triangles[tid].deleted) continue;
				triangles[tid].dirty = false;  // Clear triangle dirty flags, by the way
				const auto& v = triangles[tid].v;
				for (size_t tvertex = 0; tvertex < 3; ++tvertex) {
					auto vid = v[tvertex];
					std::lock_guard lock(vmutexes[vid]);
					vertices[vid].emplaceBackRefs(tid, tvertex);
				}
			}
		};

		tbb::parallel_for(range, task);
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

				t.geometryN = TriNormal(v0.getP(), v1.getP(), v2.getP()).normalized();

				const Eigen::Vector4f p(
					t.geometryN[0],
					t.geometryN[1],
					t.geometryN[2],
					-t.geometryN.dot(Point2Vector(v0.getP()))
				);
				const Quadric sm(p * p.transpose());

				for (size_t j = 0; j < 3; ++j) {
					auto vertex_index = t.v[j];
					std::scoped_lock lock(v_mtx[vertex_index]);
					vertices[vertex_index].addQuad(sm);
				}
			}
		};
		tbb::parallel_for(triangle_range, quadric_task);

		// Triangle error update
		auto update_task = [&](decltype(triangle_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				triangles[i].errors = ComputeTriangleError(triangles[i]);
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
				for (const auto& ref: v.getRefs()) {

					// For each vertex of the incident triangle
					for (size_t vid: triangles[ref.getTid()].v) {
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
						vertices[p.first].setBorder(true);
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
			const Point& p0 = v0.getP();
			const Point& p1 = v1.getP();
			constexpr float notVisibleScale = .5f;

			float p0x, p0y;
			if (!camera.GetSamplePosition(Eigen2LuxP(p0), &p0x, &p0y) ||
					!luxrays::IsValid(p0x) || !luxrays::IsValid(p0y)) {
				return notVisibleScale;
			}

			// Normalize
			p0x /= camera.filmWidth;
			p0y /= camera.filmHeight;

			float p1x, p1y;
			if (!camera.GetSamplePosition(Eigen2LuxP(p1), &p1x, &p1y) ||
					!luxrays::IsValid(p1x) || !luxrays::IsValid(p1y)) {
				return notVisibleScale;
			}

			// Normalize
			p1x /= camera.filmWidth;
			p1y /= camera.filmHeight;

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
		const Quadric q = v0.getQuad() + v1.getQuad();

		// Compute interpolated vertex
		const Point &p0 = v0.getP();
		const Point &p1 = v1.getP();
		const Point p2 = (v0.getP() + v1.getP()) / 2.f;

		// Compute error and associated point

		float error;
		Point pResult;
		if (preserveBorder && v0.getBorder()) {
			error = VertexError(q, p0);
			pResult = p0;
		} else if (preserveBorder && v1.getBorder()) {
			error = VertexError(q, p1);
			pResult = p1;
		} else {
			Eigen::Vector3f errors{
				VertexError(q, p0),
				VertexError(q, p1),
				VertexError(q, p2)
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

	return std::tuple(pResult, error);
}

	using CandidateContainer = std::vector<
		Candidate,
		tbb::cache_aligned_allocator<Candidate>
	>;

	// Build candidate list
	//
	CandidateContainer BuildCandidateList(size_t maxCandidateQueueSize) const {

#ifndef NDEBUG
		std::atomic<size_t> cachecalls, cachehits;
#endif


		constexpr char UNDEFINED_INDEX = -1;
		CandidateContainer candidates;
		candidates.reserve(triangles.size());


		auto tri_range = tbb::blocked_range( triangles.begin(), triangles.end() );


		auto build_task = [&](const decltype(tri_range)& r, CandidateContainer v) {
			v.reserve(r.end() - r.begin());
			// Loop into triangles
			for (auto i = r.begin(); i != r.end(); ++i) {
				const SimplifyTriangle &t = *i;
				if (t.deleted) continue;

				int8_t minErrorIndex = UNDEFINED_INDEX;
				float minError = FLOAT_INFINITY;
				Point minP;

				auto& tv0 = t.v[0];
				auto& tv1 = t.v[1];
				auto& tv2 = t.v[2];

				// Vertex uuids
				auto uuid0 = vertices[tv0].uuid();
				auto uuid1 = vertices[tv1].uuid();
				auto uuid2 = vertices[tv2].uuid();

				// Check whether triangle has not already been computed
				auto& optCandidate = t.getCandidate(uuid0, uuid1, uuid2);
				if (optCandidate.has_value()) {
					v.push_back(*optCandidate);
					continue;
				}


				for (int8_t j = 0; j < 3; ++j) {
					size_t i0, i1;
					int8_t k;
					switch(j) {
						case 0: i0 = tv0; i1 = tv1; k = 1; break;
						case 1: i0 = tv1; i1 = tv2; k = 2; break;
						case 2: i0 = tv2; i1 = tv0; k = 0; break;
						default: throw std::logic_error("Bad index");
					}
					const SimplifyVertex& v0 = vertices[i0];
					const SimplifyVertex& v1 = vertices[i1];

					// Border check
					if (preserveBorder) {
						if (v0.getBorder() && v1.getBorder()) continue;
					} else {
						if (v0.getBorder() != v1.getBorder()) continue;
					}

					auto [p, error] = CalculateCollapsePoint(v0, v1);
					if (Flipped(p, v0, v1)) continue;
					if (Flipped(p, v1, v0)) continue;

					if (t.errors[j] < minError) {
						minErrorIndex = j;
						// TODO this error or the one from CalculateCollapsePoint?
						minError = i->errors[j];
						minP = p;
					}
				}  // ~for j

				// Candidate is valid only if it is relinked to a vertex...
				if (minErrorIndex != UNDEFINED_INDEX) {;
					size_t tid = i - triangles.begin();
					Candidate candidate{
						true,  // valid
						SimplifyRef(tid , minErrorIndex, minError),  // TODO remove minError
						minP,
						minError
					};
					v.push_back(candidate);
					t.setCandidate(candidate, uuid0, uuid1, uuid2);

				}  // ~if minError

			}  // ~for i
			tbb::parallel_sort(v.begin(), v.end(), CandidateLess);
			return v;

		};  // ~build_task


		auto reduce_task = [&](CandidateContainer a, const CandidateContainer& b) {
			size_t numCandidates = std::min(
					a.size() + b.size(),
					size_t(float(triangles.size()) * 0.1)
					//size_t(maxCandidateQueueSize)
			);
			CandidateContainer r(a.size() + b.size());
			std::merge(a.begin(), a.end(), b.begin(), b.end(), r.begin(), CandidateLess);
			r.resize(numCandidates);
			return r;
		};

		CandidateContainer identity;

		candidates = parallel_reduce(
			tri_range,
			identity,
			build_task,
			reduce_task
		);

		// Take only the n first elements (resize)
		//candidates.resize(numCandidates);

		SDL_LOG("Simplify - Selected " << candidates.size() << " candidates");

		DBG_SDL_LOG(
			"Cache stats: hits = " << cachehits
			<< " / total = " << cachecalls
			<< " (size = " << ErrorCache.size() << ")"
		);

		return candidates;

	}  // ~BuildCandidateList


	// Check if a triangle flips when this edge is removed
	//inline bool Flipped(const Point& p, const size_t i0, const size_t i1) const {
	inline bool Flipped(
		const Point& p,
		const SimplifyVertex& v0,
		const SimplifyVertex& v1
	) const {

		//const auto& v0 = vertices[i0];
		bool res = false;

		for (auto& ref: v0.getRefs()) {
			const auto& t = triangles[ref.getTid()];

			// Deleted? -> pass
			if (t.deleted) continue;

			auto& normal = t.geometryN;

			const size_t e0 = ref.getTVertex();
			int8_t e1, e2;
			switch(e0) {
				case 0: e1 = 1; e2 = 2; break;
				case 1: e1 = 2; e2 = 0; break;
				case 2: e1 = 0; e2 = 1; break;
			}
			//const auto [e1, e2] = EDGES[(e0 + 1) % 3];
			const SimplifyVertex& ve1 = vertices[t.v[e1]];
			const SimplifyVertex& ve2 = vertices[t.v[e2]];

			// To be deleted? -> pass
			if (ve1.uuid() == v1.uuid() || ve2.uuid() == v1.uuid()) { continue; }

			const Point& p1(ve1.getP());
			const Point& p2(ve2.getP());

			ErrorCacheKey key{ p, p1, p2, normal };

			bool hit;
			bool assertTriangle;
			{  // Scope for accessor
				ErrorCacheType::const_accessor a;
				hit = ErrorCache.find(a, key);
				if (hit) { assertTriangle = a->second; }
			}
			if (not hit) {
				assertTriangle = AssertTriangle(p, p1, p2, normal);
				ErrorCache.insert(std::pair(key, assertTriangle));
			}
			if (not assertTriangle) {
				return true;
			}

		}  // ~for ref
		return false;
	}  // ~Flipped

	static bool AssertTriangle(
		const Point& p,
		const Point& p1,
		const Point& p2,
		const Normal& tNormal  // Triangle normal
	) {
		const auto d1 = Point2Vector(p1 - p);
		const auto d2 = Point2Vector(p2 - p);

		// Check if one of the resulting triangles is too narrow
		{
			// Nota: We use squared norms to avoid square root overhead
			const float dot = d1.dot(d2);
			const float sqrdot = dot * dot;
			const float sqrnorms = d1.squaredNorm() * d2.squaredNorm();
			constexpr float threshold = .999f * .999f;

			if (sqrdot > threshold * sqrnorms) { return false; }
		}

		// Check if one of the Normals is changing side
		{
			// We use squared norms to avoid square root overhead
			const Normal geometryN(d1.cross(d2));
			const float dot = geometryN.dot(tNormal);
			const float sqrdot = dot * dot;

			// t.geometryN has already been normalized, so we don't have to
			// multiply by its norm...
			const float sqrnorms = geometryN.squaredNorm();
			constexpr float threshold = .2f * .2f;

			if (std::signbit(dot) or sqrdot < threshold * sqrnorms) {
				return false;
			}
		}
		return true;
	}

	// Compute error for a given triangle
	inline std::array<float, 3> ComputeTriangleError(const SimplifyTriangle& t) const {
		std::array<float, 3> res;
		for (auto [i, edge]: enumerate(EDGES)) {
			const auto [e1, e2] = edge;
			const auto& v0 = vertices[t.v[e1]];
			const auto& v1 = vertices[t.v[e2]];
			auto collapseError = std::get<float>(CalculateCollapsePoint(v0, v1));
			auto screenErrorScale = CalculateCollapseScreenErrorScale(v0, v1);
			res[i] = collapseError * screenErrorScale;
		}
		return res;
	}


	// CollapsingResult result
	struct CollapsingResult {
		SimplifyVertex newVertex;
		size_t oldVertexIdx0, oldVertexIdx1;  // Collapsed edge
		SizeTVector deletedTriangles;

		CollapsingResult(

			const SimplifyVertex& p_newVertex,
			size_t p_oldVertexIdx0,
			size_t p_oldVertexIdx1,
			const SizeTVector& p_deletedTriangles
		) :
			newVertex(p_newVertex),
			oldVertexIdx0(p_oldVertexIdx0),
			oldVertexIdx1(p_oldVertexIdx1),
			deletedTriangles(p_deletedTriangles)
		{}

		CollapsingResult() = default;
	};

	// Delete triangles
	//
	//
	size_t DeleteTriangles2(const CandidateContainer& candidates) {

		size_t numDeletedTriangles = 0;

		// Step #1 - Get list of collapsing vertices (parallel-reduce approach)
		using CandidateIterator = CandidateContainer::const_iterator;
		using CandidateBRange = tbb::blocked_range<CandidateIterator>;
		using CollapsingResultVector = std::vector<
			CollapsingResult,
			tbb::scalable_allocator<CollapsingResult>
		>;

		assert(
			std::all_of(
				candidates.begin(),
				candidates.end(),
				[&](const Candidate& c)
				{ return not triangles[c.getRef().getTid()].deleted; }
			)
		);

		CandidateBRange candidateRange{candidates.begin(), candidates.end()};

		struct CollapseTask {
			CollapseTask(const Simplify& p_simplify) : simplify(p_simplify) {}

			CollapsingResultVector operator()(
				const CandidateBRange& r, CollapsingResultVector v
			) const {  // Do not forget const!
				v.reserve(v.size() + std::distance(r.begin(), r.end()));
				for (auto i = r.begin(); i != r.end(); ++i) {
					std::optional<CollapsingResult>
						collapsing{simplify.CollapseCandidate(*i)};
					if (collapsing.has_value()) {
						v.push_back(*collapsing);
					}
				}
				return v;
			}
		private:
			const Simplify& simplify;

		};

		struct ReduceTask {
			ReduceTask() = default;

			CollapsingResultVector operator()(
				CollapsingResultVector a,
				const CollapsingResultVector& b
			) const {  // Do not forget const!
				a.reserve(a.size() + b.size());
				a.insert(a.end(), b.begin(), b.end());
				return a;
			}

		};


		CollapsingResultVector identity;

		auto collapsingOutput = tbb::parallel_reduce (
			// Range
			candidateRange,

			// Identity
			identity,

			// Parallel computation on private list
			CollapseTask(*this),

			// Parallel reduction
			ReduceTask{}
		);

		// Step #2 - Apply effectively (sequential)
		std::vector< size_t, tbb::tbb_allocator<size_t>> trianglesToUpdate;
		trianglesToUpdate.reserve(triangles.size() / 10);
		vertices.reserve(vertices.size() + collapsingOutput.size());

		for (const auto& collapsing: collapsingOutput) {

			// Check whether incident triangles have not been already deleted
			bool anyDeletedTriangle =
				std::any_of(
					//std::execution::par,
					collapsing.deletedTriangles.begin(),
					collapsing.deletedTriangles.end(),
					[&](size_t i){
						return triangles[i].deleted;
					}
				)
				or
				std::any_of(
					//std::execution::par,
					collapsing.newVertex.getRefs().begin(),
					collapsing.newVertex.getRefs().end(),
					[&](const SimplifyRef& r) {
						return triangles[r.getTid()].deleted;
					}
				);
			if (anyDeletedTriangle) continue;

			// Add new vertex
			vertices.push_back(collapsing.newVertex);
			auto inew = vertices.size() - 1;  // Index of new vertex
			assert(vertices[inew] == collapsing.newVertex);
			assert(vertices[inew].getSizeRefs() > 0);

			// Call for triangle updates and remove refs of old vertices
			// (but don't remove them, otherwise it will mess triangle-vertices
			// correspondance)
			auto& oldv0 = vertices[collapsing.oldVertexIdx0];
			auto& oldv1 = vertices[collapsing.oldVertexIdx0];
			// Call for triangle updates
			for (auto& ref: oldv0.getRefs()) trianglesToUpdate.push_back(ref.getTid());
			for (auto& ref: oldv1.getRefs()) trianglesToUpdate.push_back(ref.getTid());
			oldv0.clearRefs();
			oldv1.clearRefs();

#ifndef NDEBUG
			for (auto& ref: collapsing.newVertex.getRefs()) {
				auto& t = triangles[ref.getTid()];
				assert(not t.deleted);
			}
#endif

			// Modify affected triangle vertices ("mend" the mesh)
			// SERIAL
			for (auto& ref: collapsing.newVertex.getRefs()) {
				auto& t = triangles[ref.getTid()];
				assert(not t.deleted);

				// Update triangle vertex
				t.v[ref.getTVertex()] = inew;

				// Register for update
				trianglesToUpdate.push_back(ref.getTid());

			}

			// Mark deleted triangles and erase cache accordingly
			std::for_each(
				std::execution::par,
				collapsing.deletedTriangles.begin(),
				collapsing.deletedTriangles.end(),
				[&](size_t i){
					auto& t = triangles[i];
					t.deleted = true;
				}
			);

			// Update counter
			numDeletedTriangles += collapsing.deletedTriangles.size();


		}  // ~for collapsing

		// Step #3 - Update triangles

		// Get unique incident triangles to update
		tbb::parallel_sort(trianglesToUpdate.begin(), trianglesToUpdate.end());
		auto last = std::unique(trianglesToUpdate.begin(), trianglesToUpdate.end());
		trianglesToUpdate.erase(last, trianglesToUpdate.end());

		// Update incident triangles
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, trianglesToUpdate.size()),
			[&](const tbb::blocked_range<size_t>& r){
				for (size_t i = r.begin(); i != r.end(); ++i) {
					auto tid = trianglesToUpdate[i];
					auto& t = triangles[tid];

					// Update triangle
					t.errors = ComputeTriangleError(t);
					t.geometryN = TriNormal(
						vertices[t.v[0]].getP(),
						vertices[t.v[1]].getP(),
						vertices[t.v[2]].getP()
					).normalized();


				}  // ~for
			}  // ~lambda
		);

		SDL_LOG("Terminating");
		return numDeletedTriangles;

	}  // ~DeleteTriangles2


	// Collapse an edge
	// Returns: a CollapsingResult object, or nullopt
	std::optional<CollapsingResult>
	CollapseCandidate(const Candidate& candidate) const {
		// Check triangle
		const size_t triangleIndex = candidate.getRef().getTid();
		const SimplifyTriangle &t = triangles[triangleIndex];

		assert(not t.deleted);

		// Get explicit edge to collapse
		assert(candidate.getRef().getTVertex() >= 0 and candidate.getRef().getTVertex() <= 3);
		auto& [e0, e1] = EDGES[candidate.getRef().getTVertex()];
		const size_t i0 = t.v[e0];
		const size_t i1 = t.v[e1];

		// Prepare shortcuts to vertices
		const SimplifyVertex& v0 = vertices[i0];
		const SimplifyVertex& v1 = vertices[i1];

		// Border check
		// TODO should have been checked in candidate contest
		if (v0.getBorder() != v1.getBorder()) return std::nullopt;

		// Compute vertex new position (and associated error)
		// TODO should have been computed in candidate contest
		//const auto [p, error] = CalculateCollapsePoint(v0, v1);
		Point p = candidate.getCollapsePoint();
		float error = candidate.getError();

		// Do not collapse edge if it makes a face flip
		// This should have been checked when building candidate list
		assert(not Flipped(p, v0, v1));
		assert(not Flipped(p, v1, v0));

		// Get deleted incident triangles
		auto deleted0 = GetDeletedTriangles2(i0, i1);
		auto deleted1 = GetDeletedTriangles2(i1, i0);

		// Create new vertex
		SimplifyVertex vnew{
			p,
			v0.getQuad() + v1.getQuad(),
			v0.getBorder() or v1.getBorder()
		};

		// Interpolate other vertex attributes
		const auto& tv0 = vertices[t.v[0]];
		const auto& tv1 = vertices[t.v[1]];
		const auto& tv2 = vertices[t.v[2]];
		auto [bcoords_ok, bcoords] = BaryCoords(p, tv0.getP(), tv1.getP(), tv2.getP());

		if (bcoords_ok) {

			if (hasNormals) {
				Eigen::Matrix3f normals;
				normals << tv0.getNormal(), tv1.getNormal(), tv2.getNormal();
				vnew.setNormal((normals * bcoords).normalized());
			}
			if (hasUVs) {
				Eigen::Matrix<float, 2, 3> uvs;
				uvs << tv0.getUV(), tv1.getUV(), tv2.getUV();
				vnew.setUV(uvs * bcoords);
			}
			if (hasColors) {
				Eigen::Matrix3f colors;
				colors << tv0.getCol(), tv1.getCol(), tv2.getCol();
				vnew.setCol(colors * bcoords);
			}
			if (hasAlphas) {
				Eigen::Vector3f alphas(tv0.getAlpha(), tv1.getAlpha(), tv2.getAlpha());
				vnew.setAlpha(alphas.dot(bcoords));
			}
		} else {
			// Must be a malformed triangle
			if (hasNormals) { vnew.setNormal(tv0.getNormal()); }
			if (hasUVs) { vnew.setUV(tv0.getUV()); }
			if (hasColors) { vnew.setCol(tv0.getCol()); }
			if (hasAlphas) { vnew.setAlpha(tv0.getAlpha()); }
		}

		// Update incident triangles
		//RefVector newRefs;
		SizeTVector deletedTriangles;
		//newRefs.reserve(v0.refs.size() + v1.refs.size());
		vnew.reserveRefs(v0.getSizeRefs() + v1.getSizeRefs());

		// Record triangles to delete and make list unique
		deletedTriangles.insert(
			deletedTriangles.end(), deleted0.begin(), deleted0.end()
		);
		deletedTriangles.insert(
			deletedTriangles.end(), deleted1.begin(), deleted1.end()
		);
		std::sort(deletedTriangles.begin(), deletedTriangles.end());
		auto last = std::unique(deletedTriangles.begin(), deletedTriangles.end());
		deletedTriangles.erase(last, deletedTriangles.end());

		// Add refs to vnew
		for (size_t j = 0; j < v0.getSizeRefs(); ++j) {
			const auto& ref = v0.getRef(j);
			const auto& t = triangles[ref.getTid()];
			if (t.deleted) continue;
			vnew.pushBackRefs(ref);
		}

		for (size_t j = 0; j < v1.getSizeRefs(); ++j) {
			const auto& ref = v1.getRef(j);
			const auto& t = triangles[ref.getTid()];
			if (t.deleted) continue;
			vnew.pushBackRefs(ref);
		}


		return std::make_optional<CollapsingResult>(
			vnew,
			i0,
			i1,
			deletedTriangles
		);
	}

	SizeTVector GetDeletedTriangles2(
		const size_t i0,
		const size_t i1
	) const {
		SizeTVector result;

		const auto& v0 = vertices[i0];

		for (size_t k = 0; k < v0.getSizeRefs(); ++k) {
			auto& ref = v0.getRef(k);
			auto& t = triangles[ref.getTid()];
			if (t.deleted) continue;

			// Compute vertices
			const size_t e0 = ref.getTVertex();
			const auto [e1, e2] = EDGES[(e0 + 1) % 3];

			//// Mark for deletion?

			if(t.v[e1] == i1 || t.v[e2] == i1) {
				assert(ref.getTid() < triangles.size());
				result.push_back(ref.getTid());
			}
		}
		return result;
	}

	// Compact mesh before exiting
	void CompactMesh() {

		const auto max_concurrency = tbb::this_task_arena::max_concurrency();

		// Compress triangles and mark vertices to keep
		std::vector<std::atomic_flag> keep(vertices.size());
		for (auto& k: keep) k.clear();

		//auto not_deleted = [](const SimplifyTriangle& t){ return !t.deleted; };
		tbb::enumerable_thread_specific<decltype(triangles)> newTrianglesETS;
		for (auto& e: newTrianglesETS) e.reserve(triangles.size() / max_concurrency);

		auto range = tbb::blocked_range<size_t>(0, triangles.size());
		auto compress_triangles = [&](tbb::blocked_range<size_t>& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				auto& t = triangles[i];
				if (t.deleted) continue;
				newTrianglesETS.local().push_back(t);

				keep[t.v[0]].test_and_set();
				keep[t.v[1]].test_and_set();
				keep[t.v[2]].test_and_set();
			}
		};
		tbb::parallel_for(range, compress_triangles);

		TriangleVector newTriangles;
		newTriangles.reserve(triangles.size());
		for (auto& l: newTrianglesETS) {
			newTriangles.insert(
				newTriangles.end(),
				std::make_move_iterator(l.begin()),
				std::make_move_iterator(l.end())
			);
		}
		triangles = std::move(newTriangles);

		// Compress vertices
		// This part must be partly sequential
		std::vector<size_t> vertMap;
		std::unordered_map<size_t, size_t> newIndex;
		vertMap.reserve(vertices.size());

		for (size_t i = 0; i < vertices.size(); ++i) {
			if (not keep[i].test()) {
				continue;
			}
			newIndex[i] = vertMap.size();
			vertMap.push_back(i);
		}
		DBG_SDL_LOG(
			"Simplify - Compressing from "
			<< vertices.size() << " to "
			<< vertMap.size() << " points"
		);

		decltype(vertices) newVertices(vertMap.size());
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
				for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
					auto& t = triangles[i];
					assert(newIndex.contains(t.v[j]));
					t.v[j] = newIndex[t.v[j]];
					assert(t.v[j] < vertices.size());
				}
			}
		};
		tbb::parallel_for(range2d, copy_indices);

		for (auto& t: triangles) {
			assert(t.v[0] < vertices.size());
			assert(t.v[1] < vertices.size());
			assert(t.v[2] < vertices.size());
		}

	}

};  // ~class Simplify




}  // ~namespace enhanced
}  // ~namespace (local)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

slg::SimplifyShape::SimplifyShape(
	const slg::Camera *camera,
	luxrays::ExtTriangleMesh *srcMesh,
	const float target,
	const float edgeScreenSize,
	const bool preserveBorder,
	const bool simplifyEnhanced
) {
	SDL_LOG("Simplify shape " << srcMesh->GetName() << " with target " << target);

	if ((edgeScreenSize > 0.f) && !camera) {
		throw std::runtime_error(
			"The scene camera must be defined in order to enable simplify "
			"edgescreensize option"
		);
	}

	const auto startTime = luxrays::WallClockTime();

	const size_t targetCount = std::max(
		1u, luxrays::Floor2UInt(srcMesh->GetTotalTriangleCount() * target)
	);

	/*srcMesh->Save("debug-start.ply");
	ExtTriangleMesh *debugMeshStart = ScreenProjection(*camera, *srcMesh);
	debugMeshStart->Save("debug-start-proj.ply");
	delete debugMeshStart;*/


	if (simplifyEnhanced) {
		enhanced::Simplify simplify(*srcMesh, *camera, edgeScreenSize, preserveBorder);
		simplify.Decimate(targetCount);
		mesh = simplify.GetExtMesh();
	} else {
		simple::Simplify simplify(*srcMesh);
		simplify.Decimate(targetCount, *camera, edgeScreenSize, preserveBorder);
		mesh = simplify.GetExtMesh();
	}

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

	const auto endTime = luxrays::WallClockTime();
	SDL_LOG(std::format("Simplify time: {:3f} secs", endTime - startTime));

	//std::exit(0);  // DEBUG - Stop here
}

slg::SimplifyShape::~SimplifyShape() {
	if (!refined)
		delete mesh;
}

luxrays::ExtTriangleMesh *slg::SimplifyShape::RefineImpl(const slg::Scene *scene) {
	return mesh;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

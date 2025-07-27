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
SimplifyRef {
	size_t tid = 0;  // Triangle ID
	char tvertex = -1;  // Vertex in triangle, should be in [0;3] (-1: not set)
	float error;  // To prioritize candidates

	SimplifyRef(size_t p_tid, size_t p_tvertex, float p_error=0.f):
		tid(p_tid), tvertex(p_tvertex), error(p_error) {}
	SimplifyRef() {}

	bool initialized() const { return tvertex >= 0; }
};

bool RefLess(const SimplifyRef& left, const SimplifyRef& right) {
	return left.error < right.error;
}

using RefVector = std::vector< SimplifyRef, tbb::cache_aligned_allocator<SimplifyRef> >;

// Vertex
struct SimplifyVertex {

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

	// Unique identifier
	size_t m_uuid;

};

using VertexVector = std::vector<
	SimplifyVertex,
	tbb::cache_aligned_allocator<SimplifyVertex>
>;

// Triangle
struct SimplifyTriangle {
	// Static data
	std::array<size_t, 3> v;  // Vertex indices
	Normal geometryN;

	// Dynamic data
	bool deleted = false;
	bool dirty = false;
};

using TriangleVector = std::vector<
	SimplifyTriangle,
	tbb::cache_aligned_allocator<SimplifyTriangle>
>;

// Error between vertex and Quadric
inline float VertexError(const Quadric &q, const Point& p) {
	return p.transpose() * q * p;
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
		preserveBorder(p_preserveBorder)
	{
		using SV = SimplifyVertex;


		// Size vertices and triangles containers in accordance to inputs
		size_t vertCount = srcMesh.GetTotalVertexCount();
		size_t triCount = srcMesh.GetTotalTriangleCount();
		vertices.resize(vertCount);
		triangles.resize(triCount);
		trierrors.resize(triCount, {.0f, .0f, .0f});

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
			auto candidates = BuildCandidateList(maxCandidateQueueSize);

			// Delete triangles (run batches)
			DBG_SDL_LOG(
				"Simplify - Delete triangles"
				<< " #" << iteration
				);
			const auto iterationDeletedTriangles = DeleteTriangles(candidates);

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
	std::vector<
		std::array<float, 3>,
		tbb::cache_aligned_allocator<std::array<float, 3>>
	> trierrors;

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

	// Key is a triangle, identified by its geometric points
	using ErrorCacheKey = std::array<size_t, 3>;

	ErrorCacheKey makeErrorCacheKey(const SimplifyTriangle& t) const {
		return ErrorCacheKey{
			vertices[t.v[0]].uuid(),
			vertices[t.v[1]].uuid(),
			vertices[t.v[2]].uuid()
		};
	}

	using ErrorCacheEntry = std::tuple<char, float>;  // vertex index in triangle and error

	struct ErrorCacheHash{
		inline size_t hash(const ErrorCacheKey& k) const noexcept {
			size_t seed = 0;
			seed ^= k[0] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= k[1] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= k[2] + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			return seed;
		}
		inline bool equal(const ErrorCacheKey& k0, const ErrorCacheKey& k1) const {
			return k0 == k1;
		}
	};

	using ErrorCacheType = tbb::concurrent_hash_map<
		ErrorCacheKey,
		ErrorCacheEntry,
		ErrorCacheHash,
		tbb::cache_aligned_allocator<std::pair<const ErrorCacheKey, ErrorCacheEntry>>
	>;

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
			auto not_deleted = [](const SimplifyTriangle& t){ return !t.deleted; };
			decltype(triangles) newTriangles;
			decltype(trierrors) newTriErrors;
			newTriangles.reserve(triangles.size());
			newTriErrors.reserve(triangles.size());
			//for (auto& t: triangles | std::views::filter(not_deleted)) {
			for (size_t i = 0; i < triangles.size(); ++i) {
				auto& t = triangles[i];
				if (t.deleted) continue;
				auto& e = trierrors[i];
				newTriangles.push_back(t);
				newTriErrors.push_back(e);
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
			ErrorCache.rehash(triangles.size() * 1.5);
		}

	}

	// Initialize incident edge tables of vertices
	//
	// Modify: vertices
	void InitIncidentEdges() {

		// Clear previous data
		for (auto& v: vertices) {
			v.refs.clear();
		}

		// Build refs
		// Possible race condition: we need mutexes (at vertex grain-scale)
		auto range = tbb::blocked_range<size_t>(0, triangles.size());
		MutexVector vertex_mutexes(vertices.size());
		auto task = [&](tbb::blocked_range<size_t>& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				triangles[i].dirty = false;  // Clear triangle dirty flags, by the way
				const auto& v = triangles[i].v;
				for (size_t j = 0; j < 3; ++j) {
					auto vid = v[j];
					std::lock_guard lock(vertex_mutexes[vid]);
					vertices[vid].refs.emplace_back(i, j);
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

				t.geometryN = TriNormal(v0.p(), v1.p(), v2.p()).normalized();

				const Eigen::Vector4f p(
					t.geometryN[0],
					t.geometryN[1],
					t.geometryN[2],
					-t.geometryN.dot(Point2Vector(v0.p()))
				);
				const Quadric sm(p * p.transpose());

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

		MutexVector mutexes(vertices.size());

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
						std::lock_guard lock(mutexes[p.first]);
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
			const Point& p0 = v0.p();
			const Point& p1 = v1.p();
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
		const Quadric q = v0.quad + v1.quad;

		// Compute interpolated vertex
		const Point &p0 = v0.p();
		const Point &p1 = v1.p();
		const Point p2 = (v0.p() + v1.p()) / 2.f;

		// Compute error and associated point

		float error;
		Point pResult;
		if (preserveBorder && v0.border) {
			error = VertexError(q, p0) + 1.f;
			pResult = p0;
		} else if (preserveBorder && v1.border) {
			error = VertexError(q, p1) + 1.f;
			pResult = p1;
		} else {
			Eigen::Vector3f errors{
				VertexError(q, p0),
				VertexError(q, p1),
				VertexError(q, p2)
			};
			// Error can be negative, I add 1 to have screenErrorScale to work as
			// expected
			errors += Eigen::Vector3f::Ones();
			const std::array<const Point*, 3> points{&p0, &p1, &p2};
			int minIndex;
			error = errors.array().minCoeff(&minIndex);
			pResult = *points[minIndex];
		}

	// Adding 1.0 because error have negative values
	error = std::max(error + 1.f, 0.f);
	return std::tuple(std::move(pResult), error);
}

	using CandidateContainer = std::vector<
		SimplifyRef,
		tbb::scalable_allocator<SimplifyRef>
	>;

	// Build candidate list
	//
	CandidateContainer BuildCandidateList(size_t maxCandidateQueueSize) const {

#ifndef NDEBUG
		std::atomic<size_t> cachecalls, cachehits;
#endif


		constexpr char UNDEFINED_INDEX = -1;
		CandidateContainer candidates(triangles.size());
		tbb::blocked_range<size_t> tri_range(0, triangles.size());


		auto build_task = [&](const decltype(tri_range)& r) {
			// Main loop
			for (size_t i = r.begin(); i != r.end(); ++i) {
				const SimplifyTriangle &t = triangles[i];
				char minErrorIndex = UNDEFINED_INDEX;
				float minError = FLOAT_INFINITY;

				// Look into cache whether the triangle has already been computed
				const auto cacheKey = makeErrorCacheKey(t);
				ErrorCacheType::accessor a;
				auto res = ErrorCache.find(a, cacheKey);
#ifndef NDEBUG
				cachecalls++;
#endif
				if (res) {
					// Hit! --> Just unpack...
					std::tie(minErrorIndex, minError) = a->second;
#ifndef NDEBUG
					cachehits++;
#endif
				} else {
					// No hit: compute and feed cache
					auto& tv0 = t.v[0];
					auto& tv1 = t.v[1];
					auto& tv2 = t.v[2];
					const std::array<std::tuple<size_t, size_t>, 3> edges(
						{
							{tv0, tv1},
							{tv1, tv2},
							{tv2, tv0},
						}
					);

					for (size_t j = 0; j < 3; ++j) {
						const auto [i0, i1] = edges[j];
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
						std::pair(cacheKey, std::tuple(minErrorIndex, minError))
					);
				}

				if (minErrorIndex != UNDEFINED_INDEX) {
					candidates[i] = SimplifyRef(i, minErrorIndex, minError);
				}
			}

		};
		tbb::parallel_for(tri_range, build_task);
		// Remove unitialized
		{
			decltype(candidates) candidates2;
			candidates2.reserve(candidates.size());
			std::copy_if(
				std::execution::par,
				candidates.begin(),
				candidates.end(),
				std::back_inserter(candidates2),
				[](const SimplifyRef& r){ return r.initialized(); }
			);
			candidates = std::move(candidates2);
		}

		// Sort
		size_t numCandidates = std::min({
				candidates.size(),
				size_t(maxCandidateQueueSize)
		});
		std::partial_sort(
			std::execution::par,
			candidates.begin(),
			candidates.begin() + numCandidates,
			candidates.end(),
			RefLess
		);

		// Take only the n first elements (resize)
		candidates.resize(numCandidates);

		DBG_SDL_LOG(
			"Cache stats: hits = " << cachehits
			<< " / total = " << cachecalls
			<< " (size = " << ErrorCache.size() << ")"
		);

		return candidates;

	}  // ~BuildCandidateList


	// Lock neighbors and collapse candidate edge
	// In order to benefit from RAII, this function is recursive
	size_t lock_and_collapse (
		const SimplifyRef& candidate,
		SizeTVector& neighbors,
		MutexVector& mutexes
	)
	{
		if (not neighbors.empty()) {
			// Get neighbor in the list
			size_t i0 = neighbors.back();
			neighbors.pop_back();

			// Lock current neighbor
			std::unique_lock lock(mutexes[i0], std::try_to_lock);
			if (not lock) throw NoLock();

			// And call recursively for the next
			size_t res = lock_and_collapse(candidate, neighbors, mutexes);
			return res;
		} else {
			// Final treatment: collapse
			return CollapseEdge(candidate);
		}
	}

	class NoLock : std::exception {};

	// Delete triangles
	//
	//
	size_t DeleteTriangles(const CandidateContainer& candidates) {
		tbb::auto_partitioner partitioner;
		tbb::enumerable_thread_specific<size_t> batchDeleted;
		tbb::blocked_range<size_t> batch_range(0, candidates.size());

		// Vertex mutexes
		MutexVector mutexes(vertices.size());
		//class triangleFeeder {
			//void add(
		//}

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
				ulock lk0(mutexes[i0], std::try_to_lock);
				ulock lk1(mutexes[i1], std::try_to_lock);
				ulock lk2(mutexes[i2], std::try_to_lock);
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
					neighbors,
					mutexes
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


	// Collapse an edge
	// Returns: number of deleted triangles
	// Modifies: triangles, vertices
	size_t CollapseEdge(const SimplifyRef& candidate) {
		// Check triangle
		const size_t triangleIndex = candidate.tid;
		SimplifyTriangle &t = triangles[triangleIndex];

		if (t.deleted or t.dirty) return 0;

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
		// deleted0, deleted1: true/false if the triangles referencing the
		// vertex are deleted
		if (Flipped(p, i0, i1)) return 0;
		if (Flipped(p, i1, i0)) return 0;

		// At this stage, no triangle flip is to fear anymore,
		// so we can collapse edge

		// Get deleted incident triangles
		auto&& deleted0 = GetDeletedTriangles(p, i0, i1);
		auto&& deleted1 = GetDeletedTriangles(p, i1, i0);

		// Assign new vertex' position and quadric
		v0.setP(p);
		v0.quad += v1.quad;

		// Interpolate other vertex attributes
		const auto& tv0 = vertices[t.v[0]];
		const auto& tv1 = vertices[t.v[1]];
		const auto& tv2 = vertices[t.v[2]];

		// Get barycentric coordinates
		auto&& [bcoords_ok, bcoords] = BaryCoords(p, tv0.p(), tv1.p(), tv2.p());

		if (bcoords_ok) {

			if (hasNormals) {
				Eigen::Matrix3f normals;
				normals << tv0.norm, tv1.norm, tv2.norm;
				v0.norm = (normals * bcoords).normalized();
				// TODO
				//const auto triNorm0 = tv0.norm;
				//const auto triNorm1 = tv1.norm;
				//const auto triNorm2 = tv2.norm;
				//v0.norm = (b0 * triNorm0 + b1 * triNorm1 + b2 * triNorm2).normalized();
			}
			if (hasUVs) {
				Eigen::Matrix<float, 2, 3> uvs;
				uvs << tv0.uv, tv1.uv, tv2.uv;
				v0.uv = uvs * bcoords;
				// TODO
				//const luxrays::UV triUV0 = tv0.uv;
				//const luxrays::UV triUV1 = tv1.uv;
				//const luxrays::UV triUV2 = tv2.uv;
				//v0.uv = b0 * triUV0 + b1 * triUV1 + b2 * triUV2;
			}
			if (hasColors) {
				Eigen::Matrix3f colors;
				colors << tv0.col, tv1.col, tv2.col;
				v0.col = colors * bcoords;
				// TODO
				//const luxrays::Spectrum triCol0 = tv0.col;
				//const luxrays::Spectrum triCol1 = tv1.col;
				//const luxrays::Spectrum triCol2 = tv2.col;
				//v0.col = b0 * triCol0 + b1 * triCol1 + b2 * triCol2;
			}
			if (hasAlphas) {
				Eigen::Vector3f alphas(tv0.alpha, tv1.alpha, tv2.alpha);
				v0.alpha = alphas.dot(bcoords);
				//TODO
				//const float triAlpha0 = tv0.alpha;
				//const float triAlpha1 = tv1.alpha;
				//const float triAlpha2 = tv2.alpha;
				//v0.alpha = b0 * triAlpha0 + b1 * triAlpha1 + b2 * triAlpha2;
			}
		} else {
			// Must be a malformed triangle
			if (hasNormals) { v0.norm = tv0.norm; }
			if (hasUVs) { v0.uv = tv0.uv; }
			if (hasColors) { v0.col = tv0.col; }
			if (hasAlphas) { v0.alpha = tv0.alpha; }
		}

		// Update incident triangles
		auto&& [newRefs0, deletedTriangles0] = UpdateTriangles(i0, v0, deleted0);
		auto&& [newRefs1, deletedTriangles1] = UpdateTriangles(i0, v1, deleted1);
		newRefs0.reserve(newRefs0.size() + newRefs1.size());
		newRefs0.insert(newRefs0.end(), newRefs1.begin(), newRefs1.end());
		v0.refs = std::move(newRefs0);
		deletedTriangles = deletedTriangles0 + deletedTriangles1;

		return deletedTriangles;
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
			if (t.deleted) continue;

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
			if (t.deleted) continue;

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
		for (const auto& [k, r]: enumerate(v.refs)) {
			SimplifyTriangle &t = triangles[r.tid];

			if (t.deleted)
				continue;

			if (deleted[k]) {
				t.deleted = true;
				deletedTriangles++;
				// Update cache (remove triangle)
				auto cachekey = makeErrorCacheKey(t);
				auto status = ErrorCache.erase(cachekey);
				continue;
			}

			// Update cache (remove triangle)
			auto cachekey = makeErrorCacheKey(t);
			ErrorCache.erase(cachekey);

			t.v[r.tvertex] = i0;
			t.dirty = true;
			trierrors[r.tid] = ComputeTriangleError(t);

			refs.push_back(r);
		}
		return std::tuple(std::move(refs), std::move(deletedTriangles));
	}

	// Compact mesh before exiting
	void CompactMesh() {

		const auto max_concurrency = tbb::this_task_arena::max_concurrency();

		// Compress triangles and mark vertices to keep
		std::vector<std::atomic_flag> keep(vertices.size());
		for (auto& k: keep) k.clear();

		auto not_deleted = [](const SimplifyTriangle& t){ return !t.deleted; };
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
		std::vector<size_t> newIndex, vertMap;
		newIndex.resize(vertices.size());
		vertMap.reserve(vertices.size());

		for (size_t i = 0; i < vertices.size(); ++i) {
			if (not keep[i].test()) {
				continue;
			}
			newIndex[i] = vertMap.size();
			vertMap.push_back(i);
		}

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

	std::exit(0);  // DEBUG - Stop here
}

slg::SimplifyShape::~SimplifyShape() {
	if (!refined)
		delete mesh;
}

luxrays::ExtTriangleMesh *slg::SimplifyShape::RefineImpl(const slg::Scene *scene) {
	return mesh;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

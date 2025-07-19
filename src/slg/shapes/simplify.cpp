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

#include <tbb/mutex.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_sort.h>

#include <Eigen/Dense>

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
	auto v0 = (p1 - p0).head<3>();
	auto v1 = (p2 - p0).head<3>();
	return v0.cross(v1);
}
inline bool GetBaryCoords(
	const Point &p0,
	const Point &p1,
	const Point &p2,
	const Point &hitPoint,
	float *b1,
	float *b2
) {
	const Normal vCrossW = TriNormal(p0, p2, hitPoint);
	const Normal vCrossU = TriNormal(p0, p2, p1);

	if (vCrossW.dot(vCrossU) < 0.f)
		return false;

	const Vector uCrossW = TriNormal(p0, p1, hitPoint);
	const Vector uCrossV = TriNormal(p0, p1, p2);

	if (uCrossW.dot(uCrossV) < 0.f)
		return false;

	const float denom = uCrossV.norm();
	const float r = vCrossW.norm() / denom;
	const float t = uCrossW.norm() / denom;

	*b1 = r;
	*b2 = t;

	return ((r <= 1.f) && (t <= 1.f) && (r + t <= 1.f));
}

constexpr float FLOAT_INFINITY = std::numeric_limits<float>::infinity();

constexpr std::array<std::tuple<size_t, size_t>, 3> EDGES({ {0, 1}, {1, 2}, {2, 0}, });

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

using SymetricMatrix = Eigen::Matrix4f;


struct SimplifyRef {
	size_t tid = 0;
	size_t tvertex = std::numeric_limits<size_t>::infinity();
	bool initialized = false;

	SimplifyRef(size_t p_tid, size_t p_tvertex):
		tid(p_tid), tvertex(p_tvertex), initialized(true) {}
	SimplifyRef() {}
};

using RefVector = std::vector< SimplifyRef, tbb::cache_aligned_allocator<SimplifyRef> >;

using BatchVector = std::vector<RefVector>;

struct SimplifyVertex {
	// Core data
	Point p;			  // Position
	RefVector refs;				  // Incident edges in topology

	SymetricMatrix q;     // Quadric
	bool border;          // Border status

	// LuxCore specific data
	Normal norm;
	luxrays::UV uv;
	luxrays::Spectrum col;
	float alpha;

};

using PlainVertexVector = std::vector<
	SimplifyVertex,
	tbb::cache_aligned_allocator<SimplifyVertex>
>;

// Batch design is supposed to guarantee that, in main treatment, each
// vertex is accessed by only one thread at most. However, we can check it enabling
// CHECK_VERTEX_RACE
#ifndef CHECK_VERTEX_RACE
// Normal form
using VertexVector = PlainVertexVector;
#else
// Overloaded form for checking
class VertexVector: public PlainVertexVector {
// Call `clear` to initiate recording and `check` just after
public:
	reference operator[]( size_type pos ) {
		_threads.emplace(pos, std::this_thread::get_id());
		return PlainVertexVector::operator[](pos);
	}
	const_reference operator[]( size_type pos ) const {
		_threads.emplace(pos, std::this_thread::get_id());
		return PlainVertexVector::operator[](pos);
	}
	void clear() {
		_threads.clear();
	}
	void check() const {
		for (size_t i = 0; i < size(); ++i) {
			auto [b, e] = _threads.equal_range(i);
			std::vector<std::thread::id> values;
			for (auto i = b; i != e; ++i) values.push_back(i->second);
			auto last = std::unique(values.begin(), values.end());
			values.erase(last, values.end());
			auto count = values.size();
			//auto count = std::ranges::count(r);
			if (count > 1) {
				SDL_LOG("Simplify - Warning: vertex " << i << " handled by " << count << " threads.");
				for (auto t: values) SDL_LOG("Thread " << t);
			}
		}
	}
private:
	mutable tbb::concurrent_multimap<size_t, std::thread::id> _threads;

};
#endif

struct SimplifyTriangle {
	std::array<size_t, 3> v;  // Vertex indices
	Normal geometryN;
	std::array<float, 3> err;
	bool deleted = false;
	bool dirty = false;

	// Update error of the triangle
	void UpdateTriangleError(
		const VertexVector& vertices,
		const float edgeScreenSize,
		const slg::Camera& camera,
		const bool preserveBorder
	);
};

using TriangleVector = std::vector<
	SimplifyTriangle,
	tbb::cache_aligned_allocator<SimplifyTriangle>
>;

// Error between vertex and Quadric
inline float VertexError(
	const SymetricMatrix &q,
	const Point& p
) {
	return p.transpose() * q * p;
}


// Error for one edge
// Returns: error, interpolated point
inline std::tuple<float, Point> CalculateCollapseError(
	const SimplifyVertex& v0,
	const SimplifyVertex& v1,
	const bool preserveBorder
) {

	const SymetricMatrix q = v0.q + v1.q;

	Point pResult;

	// Compute interpolated vertex
	const Point &p1 = v0.p;
	const Point &p2 = v1.p;
	const Point p3 = (v0.p + v1.p) / 2.f;

	// Error can be negative, I add 1 to have screenErrorScale can than
	// work as expected
	const float error1 = VertexError(q, p1) + 1.f;
	const float error2 = VertexError(q, p2) + 1.f;
	const float error3 = VertexError(q, p3) + 1.f;

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

inline float CalculateCollapseScreenErrorScale(
	const SimplifyVertex& v0,
	const SimplifyVertex& v1,
	float edgeScreenSize,
	const slg::Camera& camera
) {
	if (edgeScreenSize > 0.f) {
		const Point& p0 = v0.p;
		const Point& p1 = v1.p;
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

		const float edge = sqrtf(luxrays::Sqr(p0x - p1x) + luxrays::Sqr(p0y - p1y));
		if (edge == 0.f) {
			return notVisibleScale;
		}

		return std::max(edge / edgeScreenSize, notVisibleScale);
	} else {
		return 1.f;
	}
}

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
	Simplify(const luxrays::ExtTriangleMesh &srcMesh) {
		using SV = SimplifyVertex;

		// Size vertices and triangles containers in accordance to inputs
		size_t vertCount = srcMesh.GetTotalVertexCount();
		size_t triCount = srcMesh.GetTotalTriangleCount();
		vertices.resize(vertCount);
		triangles.resize(triCount);

		auto init_vertex = [](SV& vd, const luxrays::Point& vs){
			vd.p = Lux2EigenP(vs);
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
			auto init_uv = [](SV& v, const luxrays::UV& u){ v.uv = u; };
			ForEach(vertices, srcMesh.GetUVs(0), init_uv).run(vertCount);

			hasUVs = true;
		}

		if (srcMesh.HasColors(0)) {
			auto init_col = [](SV& v, const luxrays::Spectrum& col){ v.col = col; };
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
	void Decimate(
		const size_t targetTriangleCount,
		const slg::Camera& camera,
		const float edgeScreenSize,
		const bool preserveBorder
	) {
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
			InitIteration(iteration, edgeScreenSize, camera, preserveBorder);

			// Build candidate list
			DBG_SDL_LOG("Simplify - Build candidate list #" << iteration);
			auto candidateList = BuildCandidateList(
				preserveBorder, maxCandidateQueueSize
			);

			// Partition candidates into independent batches
			DBG_SDL_LOG("Simplify - Partition candidate list #" << iteration);
			auto batches = PartitionIndependentEdgeBatches(candidateList);

			// Delete triangles (run batches)
			DBG_SDL_LOG(
				"Simplify - Delete triangles ("
				<< batches.size() << " batches)"
				<< " #" << iteration
				);
			const size_t iterationDeletedTriangles = DeleteTriangles(
				batches, edgeScreenSize, camera, preserveBorder
			);

			deletedTriangles += iterationDeletedTriangles;

			SDL_LOG(
				"Simplify - End iteration #" << iteration
				<< " - Edge candidates: " << candidateList.size() << " - "
				<< " Deleted triangles (current/cumulative/initial): "
				<< iterationDeletedTriangles << "/"
				<< deletedTriangles << "/"
				<< startTriangleCount
			);

			// No more work?
			if (!iterationDeletedTriangles) break;

			//std::exit(0);  // DEBUG - Stop here
		}

		// Clean up mesh
		SDL_LOG("Simplify - Compact mesh");
		CompactMesh();

	//std::exit(0);  // DEBUG - Stop here
	}


	// Rebuild an output mesh after simplification
	luxrays::ExtTriangleMesh *GetExtMesh() const {
		const size_t vertCount = vertices.size();
		const size_t triCount = triangles.size();
		using SV = SimplifyVertex;

		luxrays::Point *newVertices = luxrays::ExtTriangleMesh::AllocVerticesBuffer(vertCount);
		auto vert_assign = [](luxrays::Point& vd, const SV& vs) {
			vd = Eigen2LuxP(vs.p);
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
				vd = vs.uv;
			};
			ForEach(newUVs, vertices, uv_assign).run(vertCount);
		}

		luxrays::Spectrum *newCols = nullptr;
		if (hasColors) {
			newCols = new luxrays::Spectrum[vertCount];
			auto col_assign = [](luxrays::Spectrum& vd, const SV& vs) {
				vd = vs.col;
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
		auto triangle_assign = [vertCount](luxrays::Triangle& td, const SimplifyTriangle& ts) {
			assert (ts.v[0] < vertCount);
			assert (ts.v[1] < vertCount);
			assert (ts.v[2] < vertCount);
			td.v[0] = ts.v[0];
			td.v[1] = ts.v[1];
			td.v[2] = ts.v[2];
		};
		ForEach(newTris, triangles, triangle_assign).run(triCount);

		return new luxrays::ExtTriangleMesh(
				vertCount, triCount, newVertices, newTris,
				newNorms, newUVs, newCols, newAlphas
		);
	}



private:
	// Main properties (verticies and triangles)
	VertexVector vertices;
	TriangleVector triangles;

	// LuxCore specific
	bool hasNormals = false;
	bool hasUVs = false;
	bool hasColors = false;
	bool hasAlphas = false;

	// Iteration initialization
	//
	// Compact triangles, compute quadrics, incidents, borders
	void InitIteration(
		const size_t iteration,
		const float edgeScreenSize,
		const slg::Camera& camera,
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
		// Race condition: we need mutexes (at vertex grain-scale)
		auto range = tbb::blocked_range<size_t>(0, triangles.size());
		std::vector<std::mutex> vertex_mutexes(vertices.size());
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
	void InitQuadrics(
		const float edgeScreenSize,
		const slg::Camera& camera,
		const bool preserveBorder
	) {
		// Starting values
		tbb::blocked_range<size_t> vertex_range(0, vertices.size());
		auto initv_task = [&](const tbb::blocked_range<size_t>& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				//vertices[i].q = SymetricMatrix(0.0);
				vertices[i].q = Eigen::Matrix4f::Zero();
			}
		};
		tbb::parallel_for(vertex_range, initv_task);

		// Parallel computation of per-triangle quadric contributions
		std::vector<std::array<SymetricMatrix, 3>> triangleQuadrics(triangles.size());

		tbb::blocked_range<size_t> triangle_range(0, triangles.size());

		auto quadric_task = [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				SimplifyTriangle &t = triangles[i];

				SimplifyVertex &v0 = vertices[t.v[0]];
				SimplifyVertex &v1 = vertices[t.v[1]];
				SimplifyVertex &v2 = vertices[t.v[2]];

				const Normal geometryN = TriNormal(v0.p, v1.p, v2.p).normalized();

				t.geometryN = geometryN;

				const Eigen::Vector4f p(
					geometryN[0],
					geometryN[1],
					geometryN[2],
					-geometryN.dot(Point2Vector(v0.p))
				);
				const SymetricMatrix sm = p * p.transpose();

				triangleQuadrics[i][0] = sm;
				triangleQuadrics[i][1] = sm;
				triangleQuadrics[i][2] = sm;
			}
		};
		tbb::parallel_for(triangle_range, quadric_task);

		// Accumulation into vertex quadrics
		std::vector<tbb::mutex> v_mtx(vertices.size());
		auto accumulate_task = [&](decltype(triangle_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				const auto &t = triangles[i];
				for (size_t j = 0; j < 3; ++j) {
					auto vertex_index = t.v[j];
					tbb::mutex::scoped_lock lock(v_mtx[vertex_index]);
					vertices[vertex_index].q.noalias() += triangleQuadrics[i][j];
				}
			}
		};
		tbb::parallel_for(triangle_range, accumulate_task);

		// Triangle error update
		auto update_task = [&](decltype(triangle_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				triangles[i].UpdateTriangleError(
					vertices, edgeScreenSize, camera, preserveBorder
				);
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

		std::vector<std::mutex> mutexes(vertices.size());

		// For each vertex
		tbb::blocked_range<size_t> vertex_range(0, vertices.size());
		auto border_task = [&](const tbb::blocked_range<size_t>& r){
			for (auto i = r.begin() ; i != r.end(); ++i) {
				const auto v = vertices[i];
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

	// Build candidate list
	//
	RefVector BuildCandidateList(
		bool preserveBorder,
		size_t maxCandidateQueueSize
	) const {

		RefVector candidates(triangles.size());
		tbb::blocked_range<size_t> tri_range(0, triangles.size());

		auto build_task = [&](const decltype(tri_range)& r) {
			// Main loop
			for (size_t i = r.begin(); i != r.end(); ++i) {
				const SimplifyTriangle &t = triangles[i];
				auto tv0 = t.v[0];
				auto tv1 = t.v[1];
				auto tv2 = t.v[2];
				const std::array<std::tuple<size_t, size_t>, 3> edges(
					{
						{tv0, tv1},
						{tv1, tv2},
						{tv2, tv0},
					}
				);

				size_t minErrorIndex = NULL_INDEX;
				float minError = FLOAT_INFINITY;
				for (size_t j = 0; j < 3; ++j) {
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
					candidates[i] = SimplifyRef(i, minErrorIndex);
				}
			}

		};
		tbb::parallel_for(tri_range, build_task);

		// Sort
		auto refErrorCompare = [&](const SimplifyRef& left, const SimplifyRef& right) {
			if (not left.initialized) return false;  // Unitialized is always greater
			if (not right.initialized) return true;  // than anything else
			const auto left_error = triangles[left.tid].err[left.tvertex];
			const auto right_error = triangles[right.tid].err[right.tvertex];
			return left_error < right_error;
		};
		tbb::parallel_sort(candidates, refErrorCompare);

		// Take only the n first elements (resize)
		SimplifyRef uninitializedRef;
		auto firstUninitialized =
			std::ranges::lower_bound(candidates, uninitializedRef, refErrorCompare)
			- candidates.begin();
		size_t numCandidates = std::min({
			candidates.size(),
			size_t(maxCandidateQueueSize),
			size_t(firstUninitialized)
		});
		candidates.resize(numCandidates);

		return candidates;

	}  // ~BuildCandidateList


	// Partition candidate list into components (aka "batches")
	//
	// We use parallelized connected components algorithm
	//
	//
	BatchVector
	PartitionIndependentEdgeBatches(const RefVector& candidateList) const {
		// Step 1: Build closure set (= candidate vertices + their neighborhoods)
		tbb::enumerable_thread_specific<std::unordered_set<size_t>> closures;
		tbb::blocked_range<size_t> candidate_range(0, candidateList.size());
		auto closure_task = [&](const decltype(candidate_range)& r) {
			for (auto i = r.begin(); i != r.end(); ++i) {
				auto& c = candidateList[i];
				const auto& t = triangles[c.tid];

				// Get explicit edge to collapse
				auto [e1, e2] = EDGES[c.tvertex];
				const size_t i0 = t.v[e1];
				const size_t i1 = t.v[e2];

				// Get neghbors and insert
				auto neighbors = edgeNeighbors(i0, i1);
				closures.local().insert(neighbors.begin(), neighbors.end());
			}
		};
		tbb::parallel_for(candidate_range, closure_task);

		// Reduce
		std::unordered_set<size_t> closure;
		for (auto& local: closures) {
			closure.merge(local);
		}

		// Step 2: Build connected components (union-find)
		DisjointSets unionFind(vertices.size());
		tbb::blocked_range<size_t> triangle_range(0, triangles.size());
		auto build_connected = [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				const auto& t = triangles[i];
				for (const auto e: EDGES) {
					auto [e0, e1] = e;
					size_t i0 = t.v[e0];
					size_t i1 = t.v[e1];
					if (closure.contains(i0) and closure.contains(i1)) {
						unionFind.unite(i0, i1);
					}
				}
			}
		};
		tbb::parallel_for(triangle_range, build_connected);

		// Build batches (sequential)
		std::unordered_map<size_t, RefVector> batches;
		for (const auto& c: candidateList) {
			size_t i = triangles[c.tid].v[c.tvertex];  // Vertex index
			size_t batchIndex = unionFind.find(i);
			batches[batchIndex].push_back(c);
		}
		BatchVector res;
		res.reserve(batches.size());
		for (const auto& batch: batches) {
			res.push_back(batch.second);
		}

		return res;
	}

	// Delete triangles, running batches
	//
	//
	size_t DeleteTriangles(
		const BatchVector& batches,
		const float edgeScreenSize,
		const slg::Camera& camera,
		const bool preserveBorder
	) {
		// Batch design guarantees that each vertex is accessed by only one thread
		// in the following treatment.
		// One can check it by enabling CHECK_VERTEX_RACE
#ifdef CHECK_VERTEX_RACE
		vertices.clear();
#endif

		tbb::enumerable_thread_specific<size_t> batchDeleted;
		tbb::blocked_range<size_t> batch_range(0, batches.size());
		auto delete_task = [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				for (auto& ref : batches[i]) {
					batchDeleted.local() += CollapseEdge(
						ref, edgeScreenSize, camera, preserveBorder
					);
				}
			}
		};
		tbb::parallel_for(batch_range, delete_task);

#ifdef CHECK_VERTEX_RACE
		vertices.check();
#endif
		auto deletedTriangles = std::accumulate(
			batchDeleted.begin(),
			batchDeleted.end(),
			size_t(0),
			std::plus<size_t>()
		);
		return deletedTriangles;
	}

	// Neighbor features
	using NeighborSet = std::unordered_set<size_t>;


	// Find edge neighbors, ie vertices that could be affected
	// by collapsing the given edge
	NeighborSet edgeNeighbors(const size_t i0, const size_t i1) const {
		NeighborSet neighbors;

		neighbors.insert(i0);
		neighbors.insert(i1);
		for (const auto& ref: vertices[i0].refs) {
			for (size_t vertexIndex: triangles[ref.tid].v) {
				neighbors.insert(vertexIndex);
			}
		}
		for (const auto& ref: vertices[i1].refs) {
			for (size_t vertexIndex: triangles[ref.tid].v) {
				neighbors.insert(vertexIndex);
			}
		}
		return neighbors;
	}

	// Collapse an edge
	// Returns: number of deleted triangles
	// Modifies: triangles, vertices
	size_t CollapseEdge(
		const SimplifyRef& vertex,  /* Candidate vertex to collapse */
		const float edgeScreenSize,
		const slg::Camera& camera,
		const bool preserveBorder
	) {
		// Check triangle
		const size_t triangleIndex = vertex.tid;
		SimplifyTriangle &t = triangles[triangleIndex];

		if (t.deleted)
			return 0;
		if (t.dirty)
			return 0;

		// Get explicit edge to collapse
		auto [e1, e2] = EDGES[vertex.tvertex];
		const size_t i0 = t.v[e1];
		const size_t i1 = t.v[e2];

		// Prepare shortcuts
		SimplifyVertex &v0 = vertices[i0];
		SimplifyVertex &v1 = vertices[i1];

		size_t deletedTriangles = 0;

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
		v0.q.noalias() = v1.q + v0.q;

		// Interpolate other vertex attributes
		const auto& tv0 = vertices[t.v[0]];
		const auto& tv1 = vertices[t.v[1]];
		const auto& tv2 = vertices[t.v[2]];
		const auto& triPoint0 = tv0.p;
		const auto& triPoint1 = tv1.p;
		const auto& triPoint2 = tv2.p;
		float b1, b2;
		if (GetBaryCoords( triPoint0, triPoint1, triPoint2, p, &b1, &b2)) {
			const float b0 = 1.f - b1 - b2;

			if (hasNormals) {
				const auto triNorm0 = tv0.norm;
				const auto triNorm1 = tv1.norm;
				const auto triNorm2 = tv2.norm;
				v0.norm = (b0 * triNorm0 + b1 * triNorm1 + b2 * triNorm2).normalized();
			}
			if (hasUVs) {
				const luxrays::UV triUV0 = tv0.uv;
				const luxrays::UV triUV1 = tv1.uv;
				const luxrays::UV triUV2 = tv2.uv;
				v0.uv = b0 * triUV0 + b1 * triUV1 + b2 * triUV2;
			}
			if (hasColors) {
				const luxrays::Spectrum triCol0 = tv0.col;
				const luxrays::Spectrum triCol1 = tv1.col;
				const luxrays::Spectrum triCol2 = tv2.col;
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
				const luxrays::UV triUV0 = tv0.uv;
				v0.uv = triUV0;
			}
			if (hasColors) {
				const luxrays::Spectrum triCol0 = tv0.col;
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
		newRefs0.insert(newRefs0.end(), newRefs1.begin(), newRefs1.end());
		refs.swap(newRefs0);
		deletedTriangles = deletedTriangles0 + deletedTriangles1;

		return deletedTriangles;
	}

	// Check if a triangle flips when this edge is removed
	// Returns: check status, deleted status of each ref (vector)
	// This template contains 2 versions (modelled on the original
	// code...): one will only compute flip status, the second will
	// also compute deleted.
	// (templatization has been used to avoid code duplication)
	using FlippedFullReturn = std::tuple<bool, std::vector<bool>>;

	template<bool F=true>
	inline
	std::conditional<F, FlippedFullReturn, bool>::type
	Flipped(const Point& p, const size_t i0, const size_t i1) const {

		const SimplifyVertex &v0 = vertices[i0];
		std::vector<bool> deleted;

		if constexpr(F) {
			deleted.resize(v0.refs.size());
		}

		for (size_t k = 0; k < v0.refs.size(); ++k) {
			auto& ref = v0.refs[k];
			const SimplifyTriangle &t = triangles[ref.tid];

			if (t.deleted) continue;

			const size_t e0 = ref.tvertex;
			const auto [e1, e2] = EDGES[(e0 + 1) % 3];
			const size_t id1 = t.v[e1];
			const size_t id2 = t.v[e2];

			// Delete ?
			if (id1 == i1 || id2 == i1) {
				if constexpr(F) {
					deleted[k] = true;
				}
				continue;
			}

			// Check if the triangle is too narrow
			const auto d1 = Point2Vector(vertices[id1].p - p).normalized();
			const auto d2 = Point2Vector(vertices[id2].p - p).normalized();

			const float absdot = fabs(d1.dot(d2));
			if (absdot > .999f) {
				if constexpr (F) {
					return FlippedFullReturn(true, deleted);
				} else {
					return true;
				}
			}

			// Check if the Normal is changing side
			const Normal geometryN(d1.cross(d2).normalized());
			if (geometryN.dot(t.geometryN) < .2f) {
				if constexpr(F) {
					return FlippedFullReturn(true, deleted);
				} else {
					return true;
				}
			}

			//// Check if the Normal is changing side
			//// (avoiding sqrt function)  TODO
			////auto rawnormal = Vector(d1).cross(Vector(d2)); TODO
			//auto rawnormal = TriNormal(p, vertices[id1].p, vertices[id2].p);
			//const float sqrlen_rawnormal = rawnormal.dot(rawnormal);
			//const float normdot = rawnormal.dot(t.geometryN);
			//const float sqrnormdot = normdot * normdot;
			//constexpr float sqrthreshold2 = .2f * .2f;
			//if  (std::signbit(normdot)  or sqrnormdot < sqrthreshold2 * sqrlen_rawnormal) {
				//if constexpr(F) {
					//return FlippedFullReturn(true, deleted);
				//} else {
					//return true;
				//}
			//}

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
	std::tuple<RefVector, size_t> UpdateTriangles(
		const size_t i0,
		const SimplifyVertex &v,  // Collapsed vertex
		const std::vector<bool> &deleted,
		const float edgeScreenSize,
		const slg::Camera& camera,
		const bool preserveBorder
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
				continue;
			}

			t.v[r.tvertex] = i0;
			t.dirty = true;
			t.UpdateTriangleError(vertices, edgeScreenSize, camera, preserveBorder);

			refs.push_back(r);
		}
		return std::tuple(refs, deletedTriangles);
	}

	// Finally compact mesh before exiting
	void CompactMesh() {
		size_t dst = 0;

		std::vector<bool> keep(vertices.size(), false);
		std::vector<size_t> newIndex(vertices.size(), -1);

		// We assume vertices 'keep' property is set to false (default value)

		// Compress triangles and mark vertices to keep
		auto not_deleted = [](const SimplifyTriangle& t){ return !t.deleted; };
		decltype(triangles) newTriangles;
		newTriangles.reserve(triangles.size());
		for (auto& t: triangles | std::views::filter(not_deleted)) {
			newTriangles.push_back(t);

			keep[t.v[0]] = true;
			keep[t.v[1]] = true;
			keep[t.v[2]] = true;
		}

		// Compress vertices
		decltype(vertices) newVertices;
		for (size_t i = 0; i < vertices.size(); ++i) {

			if (not keep[i]) {
				continue;
			}
			auto v_old = vertices[i];

			newVertices.push_back(v_old);
			auto& v_new = newVertices.back();
			newIndex[i] = newVertices.size() - 1;

			v_new.p = v_old.p;
			v_new.norm = v_old.norm;
			v_new.uv = v_old.uv;
			v_new.col = v_old.col;
			v_new.alpha = v_old.alpha;

		}

		// Update triangle vertices with new vertices
		for (auto& t: newTriangles) {
			t.v[0] = newIndex[t.v[0]];
			t.v[1] = newIndex[t.v[1]];
			t.v[2] = newIndex[t.v[2]];
		}

		triangles = newTriangles;
		vertices = newVertices;
	}

};  // ~class Simplify


void SimplifyTriangle::UpdateTriangleError(
	const VertexVector& vertices,
	const float edgeScreenSize,
	const slg::Camera& camera,
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
		enhanced::Simplify simplify(*srcMesh);
		simplify.Decimate(targetCount, *camera, edgeScreenSize, preserveBorder);
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
}

slg::SimplifyShape::~SimplifyShape() {
	if (!refined)
		delete mesh;
}

luxrays::ExtTriangleMesh *slg::SimplifyShape::RefineImpl(const slg::Scene *scene) {
	return mesh;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

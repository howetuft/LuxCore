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

#include <cassert>
#include <deque>

#include <boost/serialization/shared_ptr.hpp>

#include "luxrays/core/trianglemesh.h"
#include "luxrays/core/exttrianglemesh.h"

using namespace std;
using namespace luxrays;

//------------------------------------------------------------------------------
// VertexBuffer
//------------------------------------------------------------------------------

VertexBuffer::VertexBuffer(size_t size) {
	Allocate(size);
}

VertexBuffer::VertexBuffer(std::span<const Point> points) {

	Allocate(points.size());
	std::copy(points.begin(), points.end(), AsPoints().begin());
}

void VertexBuffer::Allocate(size_t meshVertCount) {

	// Embree requires a float padding field at the end
	byte_size = sizeof(Point) * meshVertCount + sizeof(float);
	data = std::make_unique<std::byte[]>(byte_size);

	// This is a trick so I can check if the buffer has been really allocated
	// with AllocVerticesBuffer() or not. It is useful for debugging LuxCore
	// applications.
	AsFloats().back() = 1234.1234f;
}

std::span<float> VertexBuffer::AsFloats() const {
	// Compute size
	assert(byte_size % sizeof(float) == 0);
	size_t size = byte_size / sizeof(float);

	// Make span
	auto * ptr = reinterpret_cast<float*>(data.get());
	return std::span<float>(ptr, size);
}

std::span<Point> VertexBuffer::AsPoints() const {
	static_assert(sizeof(Point) == 3 * sizeof(float), "Point must be exactly 3 floats");
	// Compute size
	assert(byte_size % sizeof(Point) == sizeof(float));  // Padded
	size_t size = byte_size / sizeof(Point);

	// Make span
	auto * ptr = reinterpret_cast<Point*>(data.get());
	return std::span<Point>(ptr, size);
}

std::span<std::byte> VertexBuffer::AsBytes() const {
	return std::span<std::byte>(data.get(), byte_size);
}


//------------------------------------------------------------------------------
// TriangleMesh
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::Mesh)

//------------------------------------------------------------------------------
// TriangleMesh
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::TriangleMesh)

TriangleMesh::TriangleMesh(
		const u_int meshTriCount, VertexBuffer&& meshVertices,
		Triangle *meshTris) {
	const u_int meshVertCount = meshVertices.GetVertCount();
	assert (meshVertCount > 0);
	assert (meshTriCount > 0);
	assert (meshTris != NULL);

	appliedTransSwapsHandedness = false;

	// Check if the buffer has been really allocated with AllocVerticesBuffer() or not.
	auto vertBuff = meshVertices.AsFloats();
	if (vertBuff.back() != 1234.1234f) {
		throw runtime_error(
			"luxrays::TriangleMesh() used with a vertex buffer not allocated with luxrays::TriangleMesh::AllocVerticesBuffer()");
	}

	vertCount = meshVertCount;
	triCount = meshTriCount;
	vertices = std::move(meshVertices);
	tris = meshTris;

	Preprocess();
}

void TriangleMesh::Preprocess() {
	// Compute mesh area
	float local_area = 0.f;

	#pragma omp parallel for reduction(+:local_area)
	for (int i = 0; i < static_cast<int>(triCount); ++i)
		local_area += tris[i].Area(vertices);

	area = local_area;
	cachedBBoxValid = false;
}

BBox TriangleMesh::GetBBox() const {
	if (!cachedBBoxValid) {
		BBox bbox;
		for (u_int i = 0; i < vertCount; ++i)
			bbox = Union(bbox, vertices[i]);
		cachedBBox = bbox;
		
		cachedBBoxValid = true;
	}

	return cachedBBox;
}

void TriangleMesh::ApplyTransform(const Transform &trans) {
	appliedTrans = appliedTrans * trans;
	appliedTransSwapsHandedness = appliedTrans.SwapsHandedness();

	for (u_int i = 0; i < vertCount; ++i)
		vertices[i] *= trans;

	Preprocess();
}

TriangleMeshUPtr TriangleMesh::Merge(
	const deque<const Mesh *> &meshes,
	TriangleMeshID **preprocessedMeshIDs,
	TriangleID **preprocessedMeshTriangleIDs) {
	u_int totalVertexCount = 0;
	u_int totalTriangleCount = 0;

	for (deque<const Mesh *>::const_iterator m = meshes.begin(); m < meshes.end(); m++) {
		totalVertexCount += (*m)->GetTotalVertexCount();
		totalTriangleCount += (*m)->GetTotalTriangleCount();
	}

	assert (totalVertexCount > 0);
	assert (totalTriangleCount > 0);
	assert (meshes.size() > 0);

	VertexBuffer v(totalVertexCount);
	Triangle *i = AllocTrianglesBuffer(totalTriangleCount);

	if (preprocessedMeshIDs)
		*preprocessedMeshIDs = new TriangleMeshID[totalTriangleCount];
	if (preprocessedMeshTriangleIDs)
		*preprocessedMeshTriangleIDs = new TriangleID[totalTriangleCount];

	u_int vIndex = 0;
	u_int iIndex = 0;
	TriangleMeshID currentID = 0;
	for (auto m = meshes.cbegin(); m != meshes.cend(); ++m) {
		// Copy the mesh vertices
		//memcpy(&v[vIndex], (*m)->GetVertices(), sizeof(Point) * (*m)->GetTotalVertexCount());
		auto dest = v.Subset(vIndex);
		auto orig = (*m)->GetVertices();
		std::copy(orig.begin(), orig.end(), dest.begin());

		const Triangle *tris = (*m)->GetTriangles();

		// Translate mesh indices
		for (u_int j = 0; j < (*m)->GetTotalTriangleCount(); j++) {
			i[iIndex].v[0] = tris[j].v[0] + vIndex;
			i[iIndex].v[1] = tris[j].v[1] + vIndex;
			i[iIndex].v[2] = tris[j].v[2] + vIndex;

			if (preprocessedMeshIDs)
				(*preprocessedMeshIDs)[iIndex] = currentID;
			if (preprocessedMeshTriangleIDs)
				(*preprocessedMeshTriangleIDs)[iIndex] = j;

			++iIndex;
		}

		vIndex += (*m)->GetTotalVertexCount();
		if (preprocessedMeshIDs) {
			// To avoid compiler warning
			currentID = currentID + 1;
		}
	}

	return std::make_unique<TriangleMesh>(totalTriangleCount, std::move(v), i);
}

u_int TriangleMesh::GetUniqueVerticesMapping(
		vector<u_int> &uniqueVertices,
		bool (*CompareVertices)(const TriangleMesh& mesh,
				const u_int vertIndex1, const u_int vertIndex2)) const {
	const u_int originalVertCount = GetTotalVertexCount();

	// Find duplicate vertices

	vector<u_int> sortedVertIndices(originalVertCount);
	for (u_int i = 0; i < originalVertCount; ++i)
		sortedVertIndices[i] = i;

	auto compareVerts = [this](const u_int vert_idx_a, const u_int vert_idx_b) {
		const Point vert_a = this->GetVertex(Transform::TRANS_IDENTITY, vert_idx_a);
		const Point vert_b = this->GetVertex(Transform::TRANS_IDENTITY, vert_idx_b);
		if (vert_a == vert_b) {
			// Special case for doubles, so we ensure ordering.
			return vert_idx_a > vert_idx_b;
		}
		const float x1 = vert_a.x + vert_a.y + vert_a.z;
		const float x2 = vert_b.x + vert_b.y + vert_b.z;

		return x1 < x2;
	};
	sort(sortedVertIndices.begin(), sortedVertIndices.end(), compareVerts);

	// This array stores index of the original vertex for the given vertex index.
	uniqueVertices.resize(originalVertCount);
	u_int uniqueVertCount = 0;

	for (u_int sortedVertIndex = 0; sortedVertIndex < originalVertCount; ++sortedVertIndex) {
		const int vertIndex = sortedVertIndices[sortedVertIndex];
		const Point vertex = GetVertex(Transform::TRANS_IDENTITY, vertIndex);
		bool isDuplicate = false;

		for (u_int otherSortedVertIndex = sortedVertIndex + 1; otherSortedVertIndex < originalVertCount;
				++otherSortedVertIndex) {
			const u_int otherVertIndex = sortedVertIndices[otherSortedVertIndex];
			const Point otherVertex = GetVertex(Transform::TRANS_IDENTITY, otherVertIndex);

			if ((otherVertex.x + otherVertex.y + otherVertex.z)
				- (vertex.x + vertex.y + vertex.z) > 3 * FLT_EPSILON) {
				// We are too far away now, we wouldn't have a duplicate.
				break;
			}

			if (CompareVertices(*this, vertIndex, otherVertIndex)) {
				isDuplicate = true;
				uniqueVertices[vertIndex] = otherVertIndex;
				break;
			}
		}

		if (!isDuplicate) {
			uniqueVertices[vertIndex] = vertIndex;
			++uniqueVertCount;
		}
	}

	// Make sure we always points to the very first orig vertex.
	for (u_int i = 0; i < originalVertCount; ++i) {
		u_int origIndex = uniqueVertices[i];
		while (origIndex != uniqueVertices[origIndex]) {
			origIndex = uniqueVertices[origIndex];
		}
		uniqueVertices[i] = origIndex;
	}
	
	return uniqueVertCount;
}

//------------------------------------------------------------------------------
// InstanceTriangleMesh
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::InstanceTriangleMesh)

InstanceTriangleMesh::InstanceTriangleMesh(TriangleMeshRef m, const Transform &t) {

	trans = t;
	transSwapsHandedness = t.SwapsHandedness();
	mesh = &m;

	// The mesh area is compute on demand and cached
	cachedArea = -1.f;

	cachedBBoxValid = false;
}

BBox InstanceTriangleMesh::GetBBox() const {
	if (!cachedBBoxValid) {
		cachedBBox = trans * mesh->GetBBox();
		cachedBBoxValid = true;
	}

	return cachedBBox;
}

//------------------------------------------------------------------------------
// MotionTriangleMesh
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::MotionTriangleMesh)

MotionTriangleMesh::MotionTriangleMesh(TriangleMeshRef m, const MotionSystem &ms) {
	
	motionSystem = ms;
	mesh = &m;

	// The mesh area is compute on demand and cached
	cachedArea = -1.f;

	cachedBBoxValid = false;
}

BBox MotionTriangleMesh::GetBBox() const {
	if (!cachedBBoxValid) {
		cachedBBox = motionSystem.Bound(mesh->GetBBox(), true);
		cachedBBoxValid = true;
	}

	return cachedBBox;
}

void MotionTriangleMesh::ApplyTransform(const Transform &trans) {
	motionSystem.ApplyTransform(trans);

	// Invalidate the cached result
	cachedArea = -1.f;

	cachedBBoxValid = false;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

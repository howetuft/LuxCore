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

#include <iostream>
#include <fstream>
#include <cstring>
#include <execution>

#include <boost/format.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <filesystem>

#include "luxrays/core/exttrianglemesh.h"
#include "luxrays/utils/ply/rply.h"
#include "luxrays/utils/serializationutils.h"
#include "luxrays/utils/strutils.h"

using namespace std;
using namespace luxrays;

//------------------------------------------------------------------------------
// ExtMesh
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::ExtMesh)

void ExtMesh::GetDifferentials(const Transform &local2World,
		const u_int triIndex, const Normal &shadeNormal, const u_int dataIndex,
        Vector *dpdu, Vector *dpdv,
        Normal *dndu, Normal *dndv) const {
    // Compute triangle partial derivatives
    const Triangle &tri = GetTriangles()[triIndex];
	const u_int v0Index = tri.v[0];
	const u_int v1Index = tri.v[1];
	const u_int v2Index = tri.v[2];

    UV uv0, uv1, uv2;
    if (HasUVs(dataIndex)) {
        uv0 = GetUV(v0Index, dataIndex);
        uv1 = GetUV(v1Index, dataIndex);
        uv2 = GetUV(v2Index, dataIndex);
    } else {
		uv0 = UV(.5f, .5f);
		uv1 = UV(.5f, .5f);
		uv2 = UV(.5f, .5f);
	}

    // Compute deltas for triangle partial derivatives
	const float du1 = uv0.u - uv2.u;
	const float du2 = uv1.u - uv2.u;
	const float dv1 = uv0.v - uv2.v;
	const float dv2 = uv1.v - uv2.v;
	const float determinant = du1 * dv2 - dv1 * du2;

	if (determinant == 0.f) {
		// Handle 0 determinant for triangle partial derivative matrix
		CoordinateSystem(Vector(shadeNormal), dpdu, dpdv);
		*dndu = Normal();
		*dndv = Normal();
	} else {
		const float invdet = 1.f / determinant;

		// Using localToWorld in order to do all computation relative to
		// the global coordinate system
		const Point p0 = GetVertex(local2World, v0Index);
		const Point p1 = GetVertex(local2World, v1Index);
		const Point p2 = GetVertex(local2World, v2Index);

		const Vector dp1 = p0 - p2;
		const Vector dp2 = p1 - p2;

		const Vector geometryDpDu = ( dv2 * dp1 - dv1 * dp2) * invdet;
		const Vector geometryDpDv = (-du2 * dp1 + du1 * dp2) * invdet;

		*dpdu = Cross(shadeNormal, Cross(geometryDpDu, shadeNormal));
		*dpdv = Cross(shadeNormal, Cross(geometryDpDv, shadeNormal));

		if (HasNormals()) {
			// Using localToWorld in order to do all computation relative to
			// the global coordinate system
			const Normal n0 = Normalize(GetShadeNormal(local2World, v0Index));
			const Normal n1 = Normalize(GetShadeNormal(local2World, v1Index));
			const Normal n2 = Normalize(GetShadeNormal(local2World, v2Index));

			const Normal dn1 = n0 - n2;
			const Normal dn2 = n1 - n2;
			*dndu = ( dv2 * dn1 - dv1 * dn2) * invdet;
			*dndv = (-du2 * dn1 + du1 * dn2) * invdet;
		} else {
			*dndu = Normal();
			*dndv = Normal();
		}
	}
}

//------------------------------------------------------------------------------
// ExtTriangleMesh
//------------------------------------------------------------------------------

// This is a workaround to a GCC bug described here:
//  https://svn.boost.org/trac10/ticket/3730
//  https://marc.info/?l=boost&m=126496738227673&w=2
namespace boost{
template<>
struct is_virtual_base_of<luxrays::TriangleMesh, luxrays::ExtTriangleMesh>: public mpl::true_ {};
}

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::ExtTriangleMesh)

ExtTriangleMesh::ExtTriangleMesh(
	const u_int meshVertCount,
	const u_int meshTriCount,
	Buffer<Point>&& meshVertices,
	Buffer<Triangle>&& meshTris,
	Optionals<Normal>&& meshNormals,
	Optionals<UV>&& mUVs,
	Optionals<Spectrum>&& mCols,
	Optionals<float>&& mAlphas,
	const float bRadius
) :
	TriangleMesh(meshVertCount, meshTriCount, std::move(meshVertices), std::move(meshTris))
{
	// Nota: 'normals' are vertex normals. They are optional: if not present,
	// triangle normals will be used.

	bevelRadius = bRadius;
	bevelCylinders = nullptr;
	bevelBoundingCylinders = nullptr;
	bevelBVHArrayNodes = nullptr;

	std::optional<ArrayOfOptionals<UV>> meshUVs;
	if (mUVs)
		(*meshUVs)[0] = std::move(mUVs);

	std::optional<ArrayOfOptionals<Spectrum>> meshCols;
	if (mCols)
		(*meshCols)[0] = std::move(mCols);

	std::optional<ArrayOfOptionals<float>> meshAlphas;
	if (mAlphas)
		(*meshAlphas)[0] = std::move(mAlphas);

	Init(
		std::move(meshNormals),
		std::move(meshUVs),
		std::move(meshCols),
		std::move(meshAlphas)
	);
}

ExtTriangleMesh::ExtTriangleMesh(
		const u_int meshVertCount, const u_int meshTriCount,
		Buffer<Point>&& meshVertices,
		Buffer<Triangle>&& meshTris,
		Optionals<Normal>&& meshNormals,
		std::optional<ArrayOfOptionals<UV>>&& meshUVs,
		std::optional<ArrayOfOptionals<Spectrum>>&& meshCols,
		std::optional<ArrayOfOptionals<float>>&& meshAlphas,
		const float bRadius
) :
		TriangleMesh(meshVertCount, meshTriCount, std::move(meshVertices), std::move(meshTris)) {
	bevelRadius = bRadius;
	bevelCylinders = nullptr;
	bevelBoundingCylinders = nullptr;
	bevelBVHArrayNodes = nullptr;

	Init(
		std::move(meshNormals),
		std::move(meshUVs),
		std::move(meshCols),
		std::move(meshAlphas)
	);
}

void ExtTriangleMesh::Init(
	Optionals<Normal>&& meshNormals,
	std::optional<ArrayOfOptionals<UV>>&& meshUVs,
	std::optional<ArrayOfOptionals<Spectrum>>&& meshCols,
	std::optional<ArrayOfOptionals<float>>&& meshAlphas
) {
	if (meshUVs && (meshUVs->size() > EXTMESH_MAX_DATA_COUNT)) {
		throw runtime_error("Error in ExtTriangleMesh::ExtTriangleMesh(): trying to define more (" +
				ToString(meshUVs->size()) + ") UV sets than EXTMESH_MAX_DATA_COUNT");
	}
	if (meshCols && (meshCols->size() > EXTMESH_MAX_DATA_COUNT)) {
		throw runtime_error("Error in ExtTriangleMesh::ExtTriangleMesh(): trying to define more (" +
				ToString(meshCols->size()) + ") Color sets than EXTMESH_MAX_DATA_COUNT");
	}
	if (meshAlphas && (meshAlphas->size() > EXTMESH_MAX_DATA_COUNT)) {
		throw runtime_error("Error in ExtTriangleMesh::ExtTriangleMesh(): trying to define more (" +
				ToString(meshAlphas->size()) + ") Alpha sets than EXTMESH_MAX_DATA_COUNT");
	}

	normals = std::move(meshNormals);
	triNormals.resize(triCount);

	if (meshUVs)
		uvs = std::move(*meshUVs);
	if (meshCols)
		cols = std::move(*meshCols);
	if (meshAlphas)
		alphas = std::move(*meshAlphas);

	// Compute triangle normals and bevel
	Preprocess();
}

void ExtTriangleMesh::Preprocess() {
	// Compute all triangle normals
	#pragma omp parallel for
	for (long long i = 0; i < triCount; ++i)
		triNormals[i] = tris[i].GetGeometryNormal(vertices);

	PreprocessBevel();
}

void ExtTriangleMesh::Delete() {
	//delete[] vertices;
	//delete[] tris;

	//delete[] normals;
	//delete[] triNormals;

	//for (UV *uv : uvs)
		//delete[] uv;
	//for (Spectrum *c : cols)
		//delete[] c;
	//for (float *a : alphas)
		//delete[] a;
	//for (float *v : vertAOV)
		//delete[] v;
	//for (float *t : triAOV)
		//delete[] t;

	delete[] bevelCylinders;
	delete[] bevelBoundingCylinders;
	delete[] bevelBVHArrayNodes;
}

Optionals<Normal> ExtTriangleMesh::ComputeNormals() {
	bool allocated;
	if (not normals.has_value()) {
		allocated = true;
		normals = Optionals<Normal>(vertCount);
	} else
		allocated = false;

	auto& n = normals.value();  // From this stage, normals have values...

	for (u_int i = 0; i < vertCount; ++i)
		n[i] =  Normal(0.f, 0.f, 0.f);
	for (u_int i = 0; i < triCount; ++i) {
		const Vector e1 = vertices[tris[i].v[1]] - vertices[tris[i].v[0]];
		const Vector e2 = vertices[tris[i].v[2]] - vertices[tris[i].v[0]];
		const Normal N = Normal(Normalize(Cross(e1, e2)));
		n[tris[i].v[0]] += N;
		n[tris[i].v[1]] += N;
		n[tris[i].v[2]] += N;
	}
	//int printedWarning = 0;
	for (u_int i = 0; i < vertCount; ++i) {
		n[i] = Normalize(n[i]);
		// Check for degenerate triangles/normals, they can freeze the GPU
		if (isnan(n[i].x) || isnan(n[i].y) || isnan(n[i].z)) {
			/*if (printedWarning < 15) {
				SDL_LOG("The model contains a degenerate normal (index " << i << ")");
				++printedWarning;
			} else if (printedWarning == 15) {
				SDL_LOG("The model contains more degenerate normals");
				++printedWarning;
			}*/
			n[i] = Normal(0.f, 0.f, 1.f);
		}
	}

	return allocated ? std::move(normals) : std::nullopt;
}

void ExtTriangleMesh::ApplyTransform(const Transform &trans) {
	TriangleMesh::ApplyTransform(trans);

	if (normals.has_value()) {
		auto& n = normals.value();
		for (u_int i = 0; i < vertCount; ++i) {
			n[i] *= trans;
			n[i] = Normalize(n[i]);
		}
	}

	Preprocess();
}

void ExtTriangleMesh::CopyAOV(ExtTriangleMeshPtr destMesh) const {
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (HasVertexAOV(i)) {
			auto& vaov = vertAOV[i].value();
			Buffer<float> out(vertCount);
			std::copy(std::execution::par, vaov.begin(), vaov.end(), out.begin());

			destMesh->DeleteVertexAOV(i);
			destMesh->SetVertexAOV(i, std::move(out));
		}

		if (HasTriAOV(i)) {
			auto& taov = triAOV[i].value();
			Buffer<float> out(triCount);
			std::copy(std::execution::par, taov.begin(), taov.end(), out.begin());

			destMesh->DeleteTriAOV(i);
			destMesh->SetTriAOV(i, (Buffer<float>&&) std::move(out));
		}
	}
}

ExtTriangleMeshPtr ExtTriangleMesh::CopyExt(
		Optionals<Point>&& meshVertices,
		Optionals<Triangle>&& meshTris,
		Optionals<Normal>&& meshNormals,
		std::optional<ArrayOfOptionals<UV>>&& meshUVs,
		std::optional<ArrayOfOptionals<Spectrum>>&& meshCols,
		std::optional<ArrayOfOptionals<float>>&& meshAlphas,
		const float bRadius
) const {
	// Copy/Replace
	// Copy from argument when given, otherwise copy from this

	Optionals<Point> vs;

	if (meshVertices.has_value()) {
		vs = std::move(meshVertices);
	} else {
		vs.resize(vertCount);  // Cannot be emplaced, because of ending checksum
		std::copy(std::execution::par, vertices.begin(), vertices.end(), vs->begin());
	}

	Optionals<Triangle> ts;
	if (meshTris.has_value()) {
		ts = std::move(meshTris);
	} else {
		ts.resize(triCount);
		std::copy(std::execution::par, tris.begin(), tris.end(), ts->begin());
	}

	Optionals<Normal> ns;
	if (!meshNormals && HasNormals()) {
		// Copy buffer
		Buffer<Normal> buffer(normals.buffer());
		ns = std::move(Optionals<Normal>(std::move(buffer)));
	} else {
		ns = std::move(meshNormals);
	}


	ArrayOfOptionals<UV> us;
	ArrayOfOptionals<Spectrum> cs;
	ArrayOfOptionals<float> as;
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (HasUVs(i) && (!meshUVs || (*meshUVs)[i])) {
			us[i].resize(vertCount);
			std::copy(std::execution::par, uvs[i]->begin(), uvs[i]->end(), us[i]->begin());
		} else {
			us[i] = meshUVs ? std::move((*meshUVs)[i]) : std::nullopt;
		}

		if (HasColors(i) && (!meshCols || (*meshCols)[i])) {
			cs[i].resize(vertCount);
			std::copy(std::execution::par, cols[i]->begin(), cols[i]->end(), cs[i]->begin());
		} else {
			cs[i] = meshCols ? std::move((*meshCols)[i]) : std::nullopt;
		}

		if (HasAlphas(i) && (!meshAlphas || (*meshAlphas)[i])) {
			as[i].resize(vertCount);
			std::copy(std::execution::par, alphas[i]->begin(), alphas[i]->end(), as[i]->begin());
		} else {
			as[i] = meshAlphas ? std::move((*meshAlphas)[i]) : std::nullopt;
		}
	}

	auto m = std::make_shared<ExtTriangleMesh>(
		vertCount,
		triCount,
		std::move(*vs),
		std::move(*ts),
		std::move(ns),
		std::move(us),
		std::move(cs),
		std::move(as),
		bRadius
	);
	m->SetLocal2World(appliedTrans);

	// Copy AOV too
	CopyAOV(m);

	return m;
}

ExtTriangleMeshPtr ExtTriangleMesh::Copy(
	Optionals<Point>&& meshVertices,
	Optionals<Triangle>&& meshTris,
	Optionals<Normal>&& meshNormals,
	Optionals<UV>&& mUVs,
	Optionals<Spectrum>&& mCols,
	Optionals<float>&& mAlphas,
	const float bRadius
) const {
	ArrayOfOptionals<UV> meshUVs;
	std::fill(meshUVs.begin(), meshUVs.end(), std::nullopt);
	if (mUVs)
		meshUVs[0] = std::move(mUVs);

	ArrayOfOptionals<Spectrum> meshCols;
	std::fill(meshCols.begin(), meshCols.end(), std::nullopt);
	if (mCols)
		meshCols[0] = std::move(mCols);

	ArrayOfOptionals<float> meshAlphas;
	std::fill(meshAlphas.begin(), meshAlphas.end(), std::nullopt);
	if (mAlphas)
		meshAlphas[0] = std::move(mAlphas);

	return CopyExt(
		std::move(meshVertices),
		std::move(meshTris),
		std::move(meshNormals),
		std::make_optional(std::move(meshUVs)),
		std::make_optional(std::move(meshCols)),
		std::make_optional(std::move(meshAlphas)),
		bRadius
	);
}

ExtTriangleMeshPtr ExtTriangleMesh::Merge(
	const vector<ExtTriangleMeshConstPtr> &meshes,
	const vector<Transform> *trans
) {
	u_int totalVertexCount = 0;
	u_int totalTriangleCount = 0;

	for (auto mesh : meshes) {
		totalVertexCount += mesh->GetTotalVertexCount();
		totalTriangleCount += mesh->GetTotalTriangleCount();
	}

	assert (totalVertexCount > 0);
	assert (totalTriangleCount > 0);
	assert (meshes.size() > 0);

	Buffer<Point> meshVertices{totalVertexCount};
	Buffer<Triangle> meshTris{totalTriangleCount};

	Optionals<Normal> meshNormals;
	ArrayOfOptionals<UV> meshUVs;
	ArrayOfOptionals<Spectrum> meshCols;
	ArrayOfOptionals<float> meshAlphas;
	ArrayOfOptionals<float> meshVertAOV;
	ArrayOfOptionals<float> meshTriAOV;

	if (meshes[0]->HasNormals())
		meshNormals = Optionals<Normal>(totalVertexCount);

	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; i++) {
		if (meshes[0]->HasUVs(i))
			meshUVs[i].emplace(totalVertexCount);
		else
			meshUVs[i] = std::nullopt;

		if (meshes[0]->HasColors(i))
			meshCols[i].emplace(totalVertexCount);
		else
			meshCols[i] = std::nullopt;

		if (meshes[0]->HasAlphas(i))
			meshAlphas[i].emplace(totalVertexCount);
		else
			meshAlphas[i] = std::nullopt;

		if (meshes[0]->HasVertexAOV(i))
			meshVertAOV[i].emplace(totalVertexCount);
		else
			meshVertAOV[i] = std::nullopt;

		if (meshes[0]->HasTriAOV(i))
			meshTriAOV[i].emplace(totalTriangleCount);
		else
			meshTriAOV[i] = std::nullopt;
	}

	u_int vIndex = 0;
	u_int iIndex = 0;
	for (u_int meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
		ExtTriangleMeshConstPtr mesh = meshes[meshIndex];
		const Transform *transformation = trans ? &((*trans)[meshIndex]) : nullptr;

		// It is a ExtTriangleMesh so I can use Transform::TRANS_IDENTITY everywhere
		// in the following code instead of local2World

		// Copy the mesh vertices
		if (transformation) {
			for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
				meshVertices[i + vIndex] = (*transformation) * mesh->GetVertex(Transform::TRANS_IDENTITY, i);
		} else {
			for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
				meshVertices[i + vIndex] = mesh->GetVertex(Transform::TRANS_IDENTITY, i);
		}

		// Copy the mesh normals
		if (meshes[0]->HasNormals() != mesh->HasNormals())
			throw runtime_error("Error in ExtTriangleMesh::Merge(): trying to merge meshes with different type of normal definitions");
		if (meshes[0]->HasNormals()) {
			if (transformation) {
				for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
					meshNormals.value()[i + vIndex] = Normalize((*transformation) * mesh->GetShadeNormal(Transform::TRANS_IDENTITY, i));
			} else {
				for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
					meshNormals.value()[i + vIndex] = mesh->GetShadeNormal(Transform::TRANS_IDENTITY, i);
			}
		}

		for (u_int dataIndex = 0; dataIndex < EXTMESH_MAX_DATA_COUNT; dataIndex++) {
			// Copy the mesh uvs
			if (meshes[0]->HasUVs(dataIndex) != mesh->HasUVs(dataIndex))
				throw runtime_error("Error in ExtTriangleMesh::Merge(): trying to merge meshes with different type of UV definitions");
			if (meshes[0]->HasUVs(dataIndex)) {
				for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
					meshUVs[dataIndex].value()[i + vIndex] = mesh->GetUV(i, dataIndex);
			}

			// Copy the mesh colors
			if (meshes[0]->HasColors(dataIndex) != mesh->HasColors(dataIndex))
				throw runtime_error("Error in ExtTriangleMesh::Merge(): trying to merge meshes with different type of color definitions");
			if (meshes[0]->HasColors(dataIndex)) {
				for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
					meshCols[dataIndex].value()[i + vIndex] = mesh->GetColor(i, dataIndex);
			}

			// Copy the mesh alphas
			if (meshes[0]->HasAlphas(dataIndex) != mesh->HasAlphas(dataIndex))
				throw runtime_error("Error in ExtTriangleMesh::Merge(): trying to merge meshes with different type of alpha definitions");
			if (meshes[0]->HasAlphas(dataIndex)) {
				for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
					meshAlphas[dataIndex].value()[i + vIndex] = mesh->GetAlpha(i, dataIndex);
			}

			// Copy the mesh vertex AOV
			if (meshes[0]->HasVertexAOV(dataIndex) != mesh->HasVertexAOV(dataIndex))
				throw runtime_error("Error in ExtTriangleMesh::Merge(): trying to merge meshes with different type of vertex AOV definitions");
			if (meshes[0]->HasVertexAOV(dataIndex)) {
				for (u_int i = 0; i < mesh->GetTotalVertexCount(); ++i)
					meshVertAOV[dataIndex].value()[i + vIndex] = mesh->GetVertexAOV(i, dataIndex);
			}

			// Copy the mesh triangle AOV
			if (meshes[0]->HasTriAOV(dataIndex) != mesh->HasTriAOV(dataIndex))
				throw runtime_error("Error in ExtTriangleMesh::Merge(): trying to merge meshes with different type of triangle AOV definitions");
			if (meshes[0]->HasTriAOV(dataIndex)) {
				for (u_int i = 0; i < mesh->GetTotalTriangleCount(); ++i)
					meshTriAOV[dataIndex].value()[i + vIndex] = mesh->GetTriAOV(i, dataIndex);
			}
		}

		// Translate mesh indices
		const auto& tris = mesh->GetTriangles();
		for (u_int j = 0; j < mesh->GetTotalTriangleCount(); j++) {
			meshTris[iIndex].v[0] = tris[j].v[0] + vIndex;
			meshTris[iIndex].v[1] = tris[j].v[1] + vIndex;
			meshTris[iIndex].v[2] = tris[j].v[2] + vIndex;

			++iIndex;
		}


		vIndex += mesh->GetTotalVertexCount();
	}

	auto newMesh = std::make_shared<ExtTriangleMesh>(
		totalVertexCount,
		totalTriangleCount,
		std::move(meshVertices),
		std::move(meshTris),
		std::move(meshNormals),
		std::move(meshUVs),
		std::move(meshCols),
		std::move(meshAlphas)
	);
	for (u_int dataIndex = 0; dataIndex < EXTMESH_MAX_DATA_COUNT; dataIndex++) {
		newMesh->SetVertexAOV(dataIndex, std::move(meshVertAOV[dataIndex]));
		newMesh->SetTriAOV(dataIndex, std::move(meshTriAOV[dataIndex]));
	}

	return newMesh;
}

// For some reason, LoadSerialized() and SaveSerialized() must be in the same
// file of BOOST_CLASS_EXPORT_IMPLEMENT()

ExtTriangleMeshPtr ExtTriangleMesh::LoadSerialized(const string &fileName) {
	SerializationInputFile sif(fileName);

	ExtTriangleMeshPtr mesh;
	sif.GetArchive() >> mesh;

	if (!sif.IsGood())
		throw runtime_error("Error while loading serialized scene: " + fileName);

	return mesh;
}

void ExtTriangleMesh::SaveSerialized(const string &fileName) const {
	SerializationOutputFile sof(fileName);

	//const ExtTriangleMesh *mesh = this;
	sof.GetArchive() << this;

	if (!sof.IsGood())
		throw runtime_error("Error while saving serialized mesh: " + fileName);

	sof.Flush();
}

//------------------------------------------------------------------------------
// ExtInstanceTriangleMesh
//------------------------------------------------------------------------------

// This is a workaround to a GCC bug described here:
//  https://svn.boost.org/trac10/ticket/3730
//  https://marc.info/?l=boost&m=126496738227673&w=2
namespace boost{
template<>
struct is_virtual_base_of<luxrays::InstanceTriangleMesh, luxrays::ExtInstanceTriangleMesh>: public mpl::true_ {};
}

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::ExtInstanceTriangleMesh)

void ExtInstanceTriangleMesh::UpdateMeshReferences(ExtTriangleMeshPtr oldMesh, ExtTriangleMeshPtr newMesh) {
	if (static_pointer_cast<ExtTriangleMesh>(mesh) == oldMesh) {
		mesh = newMesh;
		cachedArea = false;
	}
}

//------------------------------------------------------------------------------
// ExtMotionTriangleMesh
//------------------------------------------------------------------------------

// This is a workaround to a GCC bug described here:
//  https://svn.boost.org/trac10/ticket/3730
//  https://marc.info/?l=boost&m=126496738227673&w=2
namespace boost{
template<>
struct is_virtual_base_of<luxrays::MotionTriangleMesh, luxrays::ExtMotionTriangleMesh>: public mpl::true_ {};
}

BOOST_CLASS_EXPORT_IMPLEMENT(luxrays::ExtMotionTriangleMesh)

void ExtMotionTriangleMesh::UpdateMeshReferences(ExtTriangleMeshPtr oldMesh, ExtTriangleMeshPtr newMesh) {
	if (static_pointer_cast<ExtTriangleMesh>(mesh) == oldMesh) {
		mesh = oldMesh;
		cachedArea = false;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

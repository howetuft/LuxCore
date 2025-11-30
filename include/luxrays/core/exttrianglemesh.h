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

#ifndef _LUXRAYS_EXTTRIANGLEMESH_H
#define	_LUXRAYS_EXTTRIANGLEMESH_H

#include <cassert>
#include <cstdlib>

#include <boost/lexical_cast.hpp>
#include <boost/serialization/vector.hpp>

#include "luxrays/luxrays.h"
#include "luxrays/usings.h"
#include "luxrays/core/bvh/bvhbuild.h"
#include "luxrays/core/color/color.h"
#include "luxrays/core/geometry/uv.h"
#include "luxrays/core/geometry/triangle.h"
#include "luxrays/core/geometry/frame.h"
#include "luxrays/core/geometry/motionsystem.h"
#include "luxrays/core/trianglemesh.h"
#include "luxrays/core/namedobject.h"
#include "luxrays/utils/properties.h"
#include "luxrays/utils/serializationutils.h"

namespace luxrays {

// OpenCL data types
namespace ocl {
#include "luxrays/core/exttrianglemesh_types.cl"
}

class Ray;
class RayHit;

/*
 * The inheritance scheme used here:
 *
 *         | =>    TriangleMesh      => |
 * Mesh => |                            | => ExtTriangleMesh
 *         | =>       ExtMesh        => |
 *
 *         | => InstanceTriangleMesh => |
 * Mesh => |                            | => ExtInstanceTriangleMesh
 *         | =>       ExtMesh        => |
 *
 *         | => MotionTriangleMesh   => |
 * Mesh => |                            | => ExtMotionTriangleMesh
 *         | =>       ExtMesh        => |
 */


class ExtMesh : virtual public Mesh, public NamedObject {
public:
	ExtMesh() : bevelRadius(0.f) { }
	virtual ~ExtMesh() { }

	virtual float GetBevelRadius() const { return bevelRadius; }
	virtual bool IntersectBevel(const luxrays::Ray &ray, const luxrays::RayHit &rayHit,
			bool &continueToTrace, float &rayHitT,
			luxrays::Point &p, luxrays::Normal &n) const {
		continueToTrace = false;
		return false;
	}

	virtual bool HasNormals() const = 0;
	virtual bool HasUVs(const u_int dataIndex) const = 0;
	virtual bool HasColors(const u_int dataIndex) const = 0;
	virtual bool HasAlphas(const u_int dataIndex) const = 0;

	virtual bool HasVertexAOV(const u_int dataIndex) const = 0;
	virtual bool HasTriAOV(const u_int dataIndex) const = 0;

	virtual Normal GetGeometryNormal(const luxrays::Transform &local2World, const u_int triIndex) const = 0;
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int triIndex, const u_int vertIndex) const = 0;
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int vertIndex) const = 0;

	virtual UV GetUV(const u_int vertIndex, const u_int dataIndex) const = 0;
	virtual Spectrum GetColor(const u_int vertIndex, const u_int dataIndex) const = 0;
	virtual float GetAlpha(const u_int vertIndex, const u_int dataIndex) const = 0;

	virtual float GetVertexAOV(const u_int vertIndex, const u_int dataIndex) const = 0;
	virtual float GetTriAOV(const u_int triIndex, const u_int dataIndex) const = 0;

	virtual bool GetTriBaryCoords(const luxrays::Transform &local2World, const u_int triIndex, const Point &hitPoint, float *b1, float *b2) const = 0;
    virtual void GetDifferentials(const luxrays::Transform &local2World,
			const u_int triIndex, const Normal &shadeNormal, const u_int dataIndex,
			Vector *dpdu, Vector *dpdv,
			Normal *dndu, Normal *dndv) const;

	virtual Normal InterpolateTriNormal(const luxrays::Transform &local2World,
			const u_int triIndex, const float b1, const float b2) const = 0;
	virtual UV InterpolateTriUV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const = 0;
	virtual Spectrum InterpolateTriColor(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const = 0;
	virtual float InterpolateTriAlpha(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const = 0;

	virtual float InterpolateTriVertexAOV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const = 0;

	virtual void Delete() = 0;
	virtual void Save(const std::string &fileName) const = 0;

	friend class boost::serialization::access;

protected:
	float bevelRadius;

private:
	template<class Archive> void serialize(Archive &ar, const u_int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Mesh);
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(NamedObject);

		ar & bevelRadius;
	}
};



class ExtTriangleMesh : public TriangleMesh, public ExtMesh {
public:
	ExtTriangleMesh(
		const u_int meshVertCount,
		const u_int meshTriCount,
		Buffer<Point>&& meshVertices,
		Buffer<Triangle>&& meshTris,
		Optionals<Normal>&& meshNormals = std::nullopt,
		Optionals<UV>&& meshUVs = std::nullopt,
		Optionals<Spectrum>&& meshCols = std::nullopt,
		Optionals<float>&& meshAlphas = std::nullopt,
		const float bRadius = 0.f
	);
	ExtTriangleMesh(
		const u_int meshVertCount,
		const u_int meshTriCount,
		Buffer<Point>&& meshVertices,
		Buffer<Triangle>&& meshTris,
		Optionals<Normal>&& meshNormals,
		std::optional<ArrayOfOptionals<UV>>&& meshUVs,
		std::optional<ArrayOfOptionals<Spectrum>>&& meshCols,
		std::optional<ArrayOfOptionals<float>>&& meshAlphas,
		const float bRadius = 0.f
	);
	~ExtTriangleMesh() { };
	virtual void Delete();

	void SetVertexAOV(const u_int dataIndex, Optionals<float>&& opt) {
		vertAOV[dataIndex] = std::move(opt);
	}
	void SetVertexAOV(const u_int dataIndex, Buffer<float>&& buf) {
		vertAOV[dataIndex] = Optionals<float>(std::move(buf));
	}
	void DeleteVertexAOV(const u_int dataIndex) {
		vertAOV[dataIndex].reset();
	}
	auto& GetVertexAOVs(const u_int dataIndex) const { return vertAOV[dataIndex]; }

	void SetTriAOV(const u_int dataIndex, Optionals<float>&& opt) {
		triAOV[dataIndex] = std::move(opt);
	}
	void SetTriAOV(const u_int dataIndex, Buffer<float>&& buf) {
		triAOV[dataIndex] = Optionals<float>(std::move(buf));
	}
	void DeleteTriAOV(const u_int dataIndex) {
		triAOV[dataIndex].reset();
	}
	auto& GetTriAOVs(const u_int dataIndex) const { return triAOV[dataIndex]; }

	auto& GetNormals() const { return normals; }
	auto& GetTriNormals() const { return triNormals; }

	void SetUVs(const u_int dataIndex, Buffer<UV>&& data) {
		uvs[dataIndex] = Optionals<UV>(std::move(data));
	}
	void DeleteUVs(const u_int dataIndex) {
		uvs[dataIndex].reset();
	}
	auto& GetUVs(const u_int dataIndex) const { return uvs[dataIndex]; }

	void SetColors(const u_int dataIndex, Buffer<Spectrum>&& data) {
		cols[dataIndex] = Optionals<Spectrum>(std::move(data));
	}
	void DeleteColors(const u_int dataIndex) {
		cols[dataIndex].reset();
	}
	auto& GetColors(const u_int dataIndex) const { return cols[dataIndex]; }

	void SetAlphas(const u_int dataIndex, Buffer<float>&& data) {
		alphas[dataIndex] = Optionals<float>(std::move(data));
	}
	void DeleteAlphas(const u_int dataIndex) {
		alphas[dataIndex].reset();
	}
	auto& GetAlphas(const u_int dataIndex) const { return alphas[dataIndex]; }

	//const std::array<UV *, EXTMESH_MAX_DATA_COUNT> &GetAllUVs() const { return uvs; }
	//const std::array<Spectrum *, EXTMESH_MAX_DATA_COUNT> &GetAllColors() const { return cols; }
	//const std::array<float *, EXTMESH_MAX_DATA_COUNT> &GetAllAlphas() const { return alphas; }
	const auto & GetAllUVs() const { return uvs; }
	const auto & GetAllColors() const { return cols; }
	const auto & GetAllAlphas() const { return alphas; }
	const auto & GetAllVertexAOVs() const { return vertAOV; }
	const auto & GetAllTriangleAOVs() const { return triAOV; }

	Optionals<Normal> ComputeNormals();

	virtual MeshType GetType() const { return TYPE_EXT_TRIANGLE; }

	virtual bool HasNormals() const { return normals.has_value(); }
	virtual bool HasUVs(const u_int dataIndex) const { return uvs[dataIndex].has_value(); }
	virtual bool HasColors(const u_int dataIndex) const { return cols[dataIndex].has_value(); }
	virtual bool HasAlphas(const u_int dataIndex) const { return alphas[dataIndex].has_value(); }

	virtual bool HasVertexAOV(const u_int dataIndex) const { return vertAOV[dataIndex].has_value(); }
	virtual bool HasTriAOV(const u_int dataIndex) const { return triAOV[dataIndex].has_value(); }

	virtual Normal GetGeometryNormal(const luxrays::Transform &local2World, const u_int triIndex) const {
		// Pre-computed geometry normals already factor appliedTransSwapsHandedness
		return triNormals[triIndex];
	}
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int triIndex, const u_int vertIndex) const {
		const auto& val = *normals;  // Get optional value
		return (appliedTransSwapsHandedness ? -1.f : 1.f) * val[tris[triIndex].v[vertIndex]];
	}
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int vertIndex) const {
		auto& val = *normals;  // Get optional value
		return (appliedTransSwapsHandedness ? -1.f : 1.f) * val[vertIndex];
	}

	virtual UV GetUV(const u_int vertIndex, const u_int dataIndex) const {
		auto& val = *uvs[dataIndex];  // Get optional value
		return val[vertIndex];
	}
	virtual Spectrum GetColor(const u_int vertIndex, const u_int dataIndex) const {
		auto& val = *cols[dataIndex];  // Get optional value
		return val[vertIndex];
	}
	virtual float GetAlpha(const u_int vertIndex, const u_int dataIndex) const {
		auto& val = *alphas[dataIndex];  // Get optional value
		return val[vertIndex]; }

	virtual float GetVertexAOV(const u_int vertIndex, const u_int dataIndex) const {
		if (HasTriAOV(dataIndex)) {
			auto& val = *vertAOV[dataIndex];  // Get optional value
			return val[vertIndex];
		}
		else
			return 0.f;
	}
	virtual float GetTriAOV(const u_int triIndex, const u_int dataIndex) const {
		if (HasTriAOV(dataIndex)) {
			auto& val = *triAOV[dataIndex];  // Get optional value
			return val[triIndex];
		}
		else
			return 0.f;
	}

	virtual bool GetTriBaryCoords(const luxrays::Transform &local2World, const u_int triIndex, const Point &hitPoint, float *b1, float *b2) const {
		const Triangle &tri = tris[triIndex];
		return tri.GetBaryCoords(vertices, hitPoint, b1, b2);
	}
	void SetLocal2World(const luxrays::Transform &t) {
		appliedTrans = t;
		appliedTransSwapsHandedness = appliedTrans.SwapsHandedness();
	}

	virtual void ApplyTransform(const Transform &trans);

	virtual Normal InterpolateTriNormal(const luxrays::Transform &local2World, const u_int triIndex,
			const float b1, const float b2) const {
		if (not normals.has_value())
			return GetGeometryNormal(local2World, triIndex);
		auto& value = *normals;  // Optional value
		const Triangle &tri = tris[triIndex];
		const float b0 = 1.f - b1 - b2;
		return (appliedTransSwapsHandedness ? -1.f : 1.f) * Normalize(b0 * value[tri.v[0]] + b1 * value[tri.v[1]] + b2 * value[tri.v[2]]);
	}

	virtual UV InterpolateTriUV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		if (HasUVs(dataIndex)) {
			auto& value = *uvs[dataIndex];  // Optional value
			auto &tri = tris[triIndex];
			const float b0 = 1.f - b1 - b2;
			return b0 * value[tri.v[0]] + b1 * value[tri.v[1]] + b2 * value[tri.v[2]];
		} else
			return UV(0.f, 0.f);
	}

	virtual Spectrum InterpolateTriColor(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		if (HasColors(dataIndex)) {
			auto& value = *cols[dataIndex];  // Optional value
			const Triangle &tri = tris[triIndex];
			const float b0 = 1.f - b1 - b2;
			return b0 * value[tri.v[0]] + b1 * value[tri.v[1]] + b2 * value[tri.v[2]];
		} else
			return Spectrum(1.f);
	}

	virtual float InterpolateTriAlpha(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		if (HasAlphas(dataIndex)) {
			auto& value = *alphas[dataIndex];  // Optional value
			auto& tri = tris[triIndex];
			const float b0 = 1.f - b1 - b2;
			return b0 * value[tri.v[0]] + b1 * value[tri.v[1]] + b2 * value[tri.v[2]];
		} else
			return 1.f;
	}

	virtual float InterpolateTriVertexAOV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		if (HasVertexAOV(dataIndex)) {
			auto& value = *vertAOV[dataIndex];  // Optional value
			auto& tri = tris[triIndex];
			const float b0 = 1.f - b1 - b2;
			return b0 * value[tri.v[0]] + b1 * value[tri.v[1]] + b2 * value[tri.v[2]];
		} else
			return 0.f;
	}

	virtual void Save(const std::string &fileName) const;

	void CopyAOV(ExtTriangleMeshPtr destMesh) const;
	ExtTriangleMeshPtr CopyExt(
			Optionals<Point>&& meshVertices,
			Optionals<Triangle>&& meshTris,
			Optionals<Normal>&& meshNormals,
			std::optional<ArrayOfOptionals<UV>>&& meshUVs,
			std::optional<ArrayOfOptionals<Spectrum>>&& meshCols,
			std::optional<ArrayOfOptionals<float>>&& meshAlphas,
			const float bRadius = 0.f
	) const;
	ExtTriangleMeshPtr Copy(
			Optionals<Point>&& meshVertices,
			Optionals<Triangle>&& meshTris,
			Optionals<Normal>&& meshNormals,
			Optionals<UV>&& mUVs,
			Optionals<Spectrum>&& mCols,
			Optionals<float>&& mAlphas,
			const float bRadius = 0.f
	) const;
	ExtTriangleMeshPtr Copy(const float bRadius = 0.f) const {
		return CopyExt(
			std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, bRadius
		);
	}

	virtual bool IntersectBevel(const luxrays::Ray &ray, const luxrays::RayHit &rayHit,
			bool &continueToTrace, float &rayHitT,
			luxrays::Point &p, luxrays::Normal &n) const;

	static ExtTriangleMeshPtr Load(const std::string &fileName);
	static ExtTriangleMeshPtr Merge(const std::vector<ExtTriangleMeshConstPtr > &meshes,
			const std::vector<luxrays::Transform> *trans = nullptr);

	friend class ExtInstanceTriangleMesh;
	friend class ExtMotionTriangleMesh;
	friend class boost::serialization::access;

public:
	class BevelCylinder {
	public:
		BevelCylinder() { }
		BevelCylinder(const luxrays::Point &cv0, const luxrays::Point &cv1) {
			v0 = cv0;
			v1 = cv1;
		}

		float Intersect(const luxrays::Ray &ray, const float bevelRadius) const;
		void IntersectNormal(const luxrays::Point &pos, const float bevelRadius,
				luxrays::Normal &n) const;

		luxrays::Point v0, v1;
	};

	class BevelBoundingCylinder {
	public:
		BevelBoundingCylinder() { }
		BevelBoundingCylinder(const luxrays::Point &cv0, const luxrays::Point &cv1,
		const float r) {
			v0 = cv0;
			v1 = cv1;
			radius = r;
		}

		luxrays::BBox GetBBox() const;
		bool IsInside(const luxrays::Point &p) const;
		float Intersect(const luxrays::Ray &ray, const float bevelRadius) const;
		void IntersectNormal(const luxrays::Point &pos, const float bevelRadius,
				luxrays::Normal &n) const;

		luxrays::Point v0, v1;
		float radius;
	};

	static ExtTriangleMeshPtr LoadPly(const std::string &fileName);
	static ExtTriangleMeshPtr LoadSerialized(const std::string &fileName);

	// Used by serialization
	ExtTriangleMesh() {
	}

	void Init(
		Optionals<Normal>&& meshNormals,
		std::optional<ArrayOfOptionals<UV>>&& meshUVs,
		std::optional<ArrayOfOptionals<Spectrum>>&& meshCols,
		std::optional<ArrayOfOptionals<float>>&& meshAlphas
	);

	void Preprocess();
	void PreprocessBevel();

	virtual void SavePly(const std::string &fileName) const;
	virtual void SaveSerialized(const std::string &fileName) const;

	template<class Archive> void save(Archive &ar, const unsigned int version) const {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(TriangleMesh);
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ExtMesh);

		const bool hasNormals = HasNormals();
		ar & hasNormals;
		if (HasNormals())
			for (u_int i = 0; i < vertCount; ++i)
				ar & normals.value()[i];

		for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; i++) {
			const bool hasUVs = HasUVs(i);
			ar & hasUVs;
			if (hasUVs)
				ar & uvs[i].value();

			const bool hasColors = HasColors(i);
			ar & hasColors;
			if (hasColors)
				ar & cols[i].value();

			const bool hasAlphas = HasAlphas(i);
			ar & hasAlphas;
			if (hasAlphas)
				ar & alphas[i].value();

			const bool hasVertexAOV = HasVertexAOV(i);
			ar & hasVertexAOV;
			if (hasVertexAOV)
				ar & vertAOV[i].value();

			const bool hasTriangleAOV = HasTriAOV(i);
			ar & hasTriangleAOV;
			if (hasTriangleAOV)
				ar & triAOV[i].value();
		}
	}

	template<class Archive>	void load(Archive &ar, const unsigned int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(TriangleMesh);
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ExtMesh);

		bool hasNormals;
		ar & hasNormals;
		if (hasNormals) {
			normals = Optionals<Normal>(vertCount);
			for (u_int i = 0; i < vertCount; ++i)
				ar & normals.value()[i];
		} else
			normals = std::nullopt;
		triNormals = Buffer<Normal>(triCount);

		for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; i++) {
			bool hasUVs;
			ar & hasUVs;
			if (hasUVs) {
				// TODO
				uvs[i] = Optionals<UV>(vertCount);
				//ar & boost::serialization::make_array<UV>(uvs[i], vertCount);
				ar & uvs[i].value();
			} else
				uvs[i] = std::nullopt;

			bool hasColors;
			ar & hasColors;
			if (hasColors) {
				cols[i] = Optionals<Spectrum>(vertCount);
				//ar & boost::serialization::make_array<Spectrum>(cols[i], vertCount);
				ar & cols[i].value();
			} else
				cols[i] = std::nullopt;

			bool hasAlphas;
			ar & hasAlphas;
			if (hasAlphas) {
				alphas[i] = Optionals<float>(vertCount);
				//ar & boost::serialization::make_array<float>(alphas[i], vertCount);
				ar & alphas[i].value();
			} else
				alphas[i] = std::nullopt;

			bool hasVertexAOV;
			ar & hasVertexAOV;
			if (hasVertexAOV) {
				vertAOV[i] = Optionals<float>(vertCount);
				//ar & boost::serialization::make_array<float>(vertAOV[i], vertCount);
				ar & vertAOV[i].value();
			} else
				vertAOV[i] = std::nullopt;

			bool hasTriangleAOV;
			ar & hasTriangleAOV;
			if (hasTriangleAOV) {
				triAOV[i] = Optionals<float>(triCount);
				//ar & boost::serialization::make_array<float>(triAOV[i], triCount);
				ar & triAOV[i].value();
			} else
				triAOV[i] = std::nullopt;
		}

		bevelCylinders = nullptr;
		bevelBoundingCylinders = nullptr;
		bevelBVHArrayNodes = nullptr;

		Preprocess();
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER()


	Buffer<Normal> triNormals; // Triangle normals (computed)
	Optionals<Normal> normals; // Vertices normals

	ArrayOfOptionals<UV> uvs; // Vertex uvs (optional)
	ArrayOfOptionals<Spectrum> cols; // Vertex colors (optional)
	ArrayOfOptionals<float> alphas; // Vertex alphas (optional)

	ArrayOfOptionals<float> vertAOV; // Vertex AOV (optional)
	ArrayOfOptionals<float> triAOV; // Triangle AOV (optional)

	BevelCylinder *bevelCylinders;
	BevelBoundingCylinder *bevelBoundingCylinders;
	luxrays::ocl::IndexBVHArrayNode *bevelBVHArrayNodes;
};

class ExtInstanceTriangleMesh : public InstanceTriangleMesh, public ExtMesh {
public:
	ExtInstanceTriangleMesh(ExtTriangleMeshPtr m, const Transform &t) : 
		InstanceTriangleMesh(m, t) { }
	~ExtInstanceTriangleMesh() { };
	virtual void Delete() {	}

	virtual MeshType GetType() const { return TYPE_EXT_TRIANGLE_INSTANCE; }

	virtual float GetBevelRadius() const { return static_pointer_cast<ExtTriangleMesh>(mesh)->GetBevelRadius(); }

	virtual bool HasNormals() const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasNormals(); }
	virtual bool HasUVs(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasUVs(dataIndex); }
	virtual bool HasColors(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasColors(dataIndex); }
	virtual bool HasAlphas(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasAlphas(dataIndex); }

	virtual bool HasVertexAOV(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasVertexAOV(dataIndex); }
	virtual bool HasTriAOV(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasTriAOV(dataIndex); }

	virtual Normal GetGeometryNormal(const luxrays::Transform &local2World, const u_int triIndex) const {
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(local2World * static_pointer_cast<ExtTriangleMesh>(mesh)->GetGeometryNormal(Transform::TRANS_IDENTITY, triIndex));
	}
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int triIndex, const u_int vertIndex) const {
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(local2World * static_pointer_cast<ExtTriangleMesh>(mesh)->GetShadeNormal(Transform::TRANS_IDENTITY, triIndex, vertIndex));
	}
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int vertIndex) const {
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(local2World * static_pointer_cast<ExtTriangleMesh>(mesh)->GetShadeNormal(Transform::TRANS_IDENTITY, vertIndex));
	}
	virtual UV GetUV(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetUV(vertIndex, dataIndex);
	}
	virtual Spectrum GetColor(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetColor(vertIndex, dataIndex);
	}
	virtual float GetAlpha(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetAlpha(vertIndex, dataIndex);
	}

	virtual float GetVertexAOV(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetVertexAOV(vertIndex, dataIndex);
	}
	virtual float GetTriAOV(const unsigned triIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetTriAOV(triIndex, dataIndex);
	}

	virtual bool GetTriBaryCoords(const luxrays::Transform &local2World, const u_int triIndex,
			const Point &hitPoint, float *b1, float *b2) const {
		const Triangle &tri = mesh->GetTriangles()[triIndex];

		return Triangle::GetBaryCoords(
				GetVertex(local2World, tri.v[0]),
				GetVertex(local2World, tri.v[1]),
				GetVertex(local2World, tri.v[2]),
				hitPoint, b1, b2);
	}

	virtual Normal InterpolateTriNormal(const luxrays::Transform &local2World,
			const u_int triIndex, const float b1, const float b2) const {
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(trans * static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriNormal(
				Transform::TRANS_IDENTITY, triIndex, b1, b2));
	}

	virtual UV InterpolateTriUV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriUV(triIndex,
				b1, b2, dataIndex);
	}

	virtual Spectrum InterpolateTriColor(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriColor(triIndex,
				b1, b2, dataIndex);
	}

	virtual float InterpolateTriAlpha(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriAlpha(triIndex,
				b1, b2, dataIndex);
	}

	virtual float InterpolateTriVertexAOV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriVertexAOV(triIndex,
				b1, b2, dataIndex);
	}

	virtual bool IntersectBevel(const luxrays::Ray &ray, const luxrays::RayHit &rayHit,
			bool &continueToTrace, float &rayHitT,
			luxrays::Point &p, luxrays::Normal &n) const;

	virtual void Save(const std::string &fileName) const { static_pointer_cast<ExtTriangleMesh>(mesh)->Save(fileName); }

	const Transform &GetTransformation() const { return trans; }
	ExtTriangleMeshPtr GetExtTriangleMesh() const { return static_pointer_cast<ExtTriangleMesh>(mesh); };

	void UpdateMeshReferences(ExtTriangleMeshPtr oldMesh, ExtTriangleMeshPtr newMesh);

	friend class boost::serialization::access;


private:
	// Used by serialization
	ExtInstanceTriangleMesh() { }

	template<class Archive> void save(Archive &ar, const unsigned int version) const {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(InstanceTriangleMesh);
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ExtMesh);
	}

	template<class Archive>	void load(Archive &ar, const unsigned int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(InstanceTriangleMesh);
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ExtMesh);
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER()
};

class ExtMotionTriangleMesh : public MotionTriangleMesh, public ExtMesh {
public:
	ExtMotionTriangleMesh(ExtTriangleMeshPtr m, const MotionSystem &ms) :
		MotionTriangleMesh(m, ms) { }
	~ExtMotionTriangleMesh() { }
	virtual void Delete() {	}

	virtual MeshType GetType() const { return TYPE_EXT_TRIANGLE_MOTION; }

	virtual float GetBevelRadius() const { return static_pointer_cast<ExtTriangleMesh>(mesh)->GetBevelRadius(); }

	virtual bool HasNormals() const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasNormals(); }
	virtual bool HasUVs(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasUVs(dataIndex); }
	virtual bool HasColors(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasColors(dataIndex); }
	virtual bool HasAlphas(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasAlphas(dataIndex); }

	virtual bool HasVertexAOV(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasVertexAOV(dataIndex); }
	virtual bool HasTriAOV(const u_int dataIndex) const { return static_pointer_cast<ExtTriangleMesh>(mesh)->HasTriAOV(dataIndex); }

	virtual Normal GetGeometryNormal(const luxrays::Transform &local2World, const u_int triIndex) const {
		const bool transSwapsHandedness = local2World.SwapsHandedness();
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(local2World * static_pointer_cast<ExtTriangleMesh>(mesh)->GetGeometryNormal(local2World, triIndex));
	}
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int triIndex, const u_int vertIndex) const {
		const bool transSwapsHandedness = local2World.SwapsHandedness();
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(local2World * static_pointer_cast<ExtTriangleMesh>(mesh)->GetShadeNormal(local2World, triIndex, vertIndex));
	}
	virtual Normal GetShadeNormal(const luxrays::Transform &local2World, const u_int vertIndex) const {
		const bool transSwapsHandedness = local2World.SwapsHandedness();
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(local2World * static_pointer_cast<ExtTriangleMesh>(mesh)->GetShadeNormal(local2World, vertIndex));
	}
	virtual UV GetUV(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetUV(vertIndex, dataIndex);
	}
	virtual Spectrum GetColor(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetColor(vertIndex, dataIndex);
	}
	virtual float GetAlpha(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetAlpha(vertIndex, dataIndex);
	}

	virtual float GetVertexAOV(const unsigned vertIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetVertexAOV(vertIndex, dataIndex);
	}
	virtual float GetTriAOV(const unsigned triIndex, const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->GetTriAOV(triIndex, dataIndex);
	}

	virtual bool GetTriBaryCoords(const luxrays::Transform &local2World, const u_int triIndex,
			const Point &hitPoint, float *b1, float *b2) const {
		const Triangle &tri = mesh->GetTriangles()[triIndex];

		return Triangle::GetBaryCoords(GetVertex(local2World, tri.v[0]),
				GetVertex(local2World, tri.v[1]), GetVertex(local2World, tri.v[2]),
				hitPoint, b1, b2);
	}

	virtual Normal InterpolateTriNormal(const luxrays::Transform &local2World,
			const u_int triIndex, const float b1, const float b2) const {
		const bool transSwapsHandedness = local2World.SwapsHandedness();
		return (transSwapsHandedness ? -1.f : 1.f) * Normalize(local2World * static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriNormal(
				local2World, triIndex, b1, b2));
	}

	virtual UV InterpolateTriUV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriUV(triIndex,
				b1, b2, dataIndex);
	}
	
	virtual Spectrum InterpolateTriColor(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriColor(triIndex,
				b1, b2, dataIndex);
	}
	
	virtual float InterpolateTriAlpha(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriAlpha(triIndex,
				b1, b2, dataIndex);
	}

	virtual float InterpolateTriVertexAOV(const u_int triIndex, const float b1, const float b2,
			const u_int dataIndex) const {
		return static_pointer_cast<ExtTriangleMesh>(mesh)->InterpolateTriVertexAOV(triIndex,
				b1, b2, dataIndex);
	}

	virtual void Save(const std::string &fileName) const { static_pointer_cast<ExtTriangleMesh>(mesh)->Save(fileName); }

	const MotionSystem &GetMotionSystem() const { return motionSystem; }
	ExtTriangleMeshPtr GetExtTriangleMesh() const { return static_pointer_cast<ExtTriangleMesh>(mesh); };

	void UpdateMeshReferences(ExtTriangleMeshPtr oldMesh, ExtTriangleMeshPtr newMesh);

	friend class boost::serialization::access;

private:
	// Used by serialization
	ExtMotionTriangleMesh() {
	}

	template<class Archive> void save(Archive &ar, const unsigned int version) const {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(MotionTriangleMesh);
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ExtMesh);
	}

	template<class Archive>	void load(Archive &ar, const unsigned int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(MotionTriangleMesh);
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ExtMesh);
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER()
};

}

BOOST_SERIALIZATION_ASSUME_ABSTRACT(luxrays::ExtMesh)

BOOST_CLASS_VERSION(luxrays::ExtTriangleMesh, 4)
BOOST_CLASS_VERSION(luxrays::ExtInstanceTriangleMesh, 4)
BOOST_CLASS_VERSION(luxrays::ExtMotionTriangleMesh, 4)

BOOST_CLASS_EXPORT_KEY(luxrays::ExtTriangleMesh)
BOOST_CLASS_EXPORT_KEY(luxrays::ExtInstanceTriangleMesh)
BOOST_CLASS_EXPORT_KEY(luxrays::ExtMotionTriangleMesh)

#endif	/* _LUXRAYS_EXTTRIANGLEMESH_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

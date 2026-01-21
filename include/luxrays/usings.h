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

// This file is intended to gather all usings (pointers, refs etc.) for
// luxrays classes

#pragma once

#include <memory>
#include <optional>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_free.hpp>

namespace std {
	class jthread;
}



// OptionalPtr is a template to build a pointer with optional value from a given type.
// OptionalPtr differs from a raw pointer in at least two ways:
// - The absence of a pointed object is represented by std::nullopt. Dereferencing
// the pointer in this case will raise a std::bad_optional_access exception, rather
// than the infamous segmentation fault.
// - By convention, OptionalPtr is explicitely a *non-owning pointer*.
//
template<typename T>
struct OptionalPtr : public std::optional<std::reference_wrapper<T>> {
	using parent = std::optional<std::reference_wrapper<T>>;

	using parent::operator=;
	using parent::parent;

	const T& operator*() const {
		return this->value().get();
	}
	T& operator*() {
		return this->value().get();
	}

	T* operator->(void) const {
		return std::pointer_traits<T*>::pointer_to(this->value().get());
	}

	inline bool operator==(const OptionalPtr<T>& other) const {
		if (bool(*this) xor bool(other)) return false;
		if (!*this and !other) return true;

		T& lhs = this->value().get();
		T& rhs = other.value().get();

		return lhs == rhs;  // Rely on underlying type operator
	}
	inline bool operator==(const T& other) const {
		if (not this->has_value()) return false;

		T& lhs = this->value().get();
		T& rhs = other;
		return lhs == rhs;
	}


	T* ptr(void) {
		return &this->value().get();
	}
	const T* ptr(void) const {
		return &this->value().get();
	}

	// Boost serialization plumbing
	template<class Archive>
    void save(Archive & ar, const unsigned int file_version) const {
		bool value_state = this->has_value();
		ar & value_state;

		if (value_state) {
			ar & this->ptr();
		}
    }
	template<class Archive>
    void load(Archive & ar, const unsigned int file_version) {
		bool value_state;
		ar & value_state;

		if (value_state) {
			T* ptr = nullptr;
			ar & ptr;
			this->value() = *ptr;
		}
    }
	friend class boost::serialization::access;
	BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template<typename T>
struct std::hash<OptionalPtr<const T>> {
	size_t operator()(const OptionalPtr<const T&> opt) const {
		if (!opt) return 0;
		auto& ref = opt();
		const auto * ptr = &ref;
		return std::hash<size_t>()(reinterpret_cast<size_t>(ptr));
	}
};

namespace luxrays {

class Accelerator;
using AcceleratorPtr = std::shared_ptr<Accelerator>;
using AcceleratorConstPtr = std::shared_ptr<const Accelerator>;
using AcceleratorUPtr = std::unique_ptr<Accelerator>;
using AcceleratorConstUPtr = std::unique_ptr<const Accelerator>;

class BBox;
using BBoxPtr = std::shared_ptr<BBox>;
using BBoxConstPtr = std::shared_ptr<const BBox>;
using BBoxUPtr = std::unique_ptr<BBox>;
using BBoxConstUPtr = std::unique_ptr<const BBox>;

class Context;
using ContextPtr = std::shared_ptr<Context>;
using ContextConstPtr = std::shared_ptr<const Context>;
using ContextUPtr = std::unique_ptr<Context>;
using ContextConstUPtr = std::unique_ptr<const Context>;

class DataSet;
using DataSetPtr = std::shared_ptr<DataSet>;
using DataSetConstPtr = std::shared_ptr<const DataSet>;
using DataSetUPtr = std::unique_ptr<DataSet>;
using DataSetConstUPtr = std::unique_ptr<const DataSet>;

class Device;
using DevicePtr = std::shared_ptr<Device>;
using DeviceConstPtr = std::shared_ptr<const Device>;
using DeviceUPtr = std::unique_ptr<Device>;
using DeviceConstUPtr = std::unique_ptr<const Device>;

class DeviceDescription;
using DeviceDescriptionPtr = std::shared_ptr<DeviceDescription>;
using DeviceDescriptionConstPtr = std::shared_ptr<const DeviceDescription>;
using DeviceDescriptionUPtr = std::unique_ptr<DeviceDescription>;
using DeviceDescriptionConstUPtr = std::unique_ptr<const DeviceDescription>;

class HardwareDevice;
using HardwareDevicePtr = std::shared_ptr<HardwareDevice>;
using HardwareDeviceConstPtr = std::shared_ptr<const HardwareDevice>;
using HardwareDeviceUPtr = std::unique_ptr<HardwareDevice>;
using HardwareDeviceConstUPtr = std::unique_ptr<const HardwareDevice>;

class IntersectionDevice;
using IntersectionDevicePtr = std::shared_ptr<IntersectionDevice>;
using IntersectionDeviceConstPtr = std::shared_ptr<const IntersectionDevice>;
using IntersectionDeviceUPtr = std::unique_ptr<IntersectionDevice>;
using IntersectionDeviceConstUPtr = std::unique_ptr<const IntersectionDevice>;

class Mesh;
//using MeshPtr = std::shared_ptr<Mesh>;
//using MeshConstPtr = std::shared_ptr<const Mesh>;
using MeshUPtr = std::unique_ptr<Mesh>;
using MeshConstUPtr = std::unique_ptr<const Mesh>;
using MeshRef = Mesh&;
using MeshConstRef = const Mesh &;

class ExtMesh;
//using ExtMeshPtr = std::shared_ptr<ExtMesh>;
//using ExtMeshConstPtr = std::shared_ptr<const ExtMesh>;
using ExtMeshUPtr = std::unique_ptr<ExtMesh>;
using ExtMeshConstUPtr = std::unique_ptr<const ExtMesh>;
using ExtMeshRef = ExtMesh&;
using ExtMeshConstRef = const ExtMesh&;

class ExtMesh;
//using ExtMeshConstPtr = std::shared_ptr<const ExtMesh>;
//using ExtMeshPtr = std::shared_ptr<ExtMesh>;
using ExtMeshConstRef = const ExtMesh&;
using ExtMeshRef = ExtMesh&;

class ExtTriangleMesh;
using ExtTriangleMeshUPtr = std::unique_ptr<ExtTriangleMesh>;
using ExtTriangleMeshConstUPtr = std::unique_ptr<const ExtTriangleMesh>;
using ExtTriangleMeshConstRef = const ExtTriangleMesh&;
using ExtTriangleMeshRef = ExtTriangleMesh&;

class ExtInstanceTriangleMesh;
using ExtInstanceTriangleMeshUPtr = std::unique_ptr<ExtInstanceTriangleMesh>;
using ExtInstanceTriangleMeshConstUPtr = std::unique_ptr<const ExtInstanceTriangleMesh>;
using ExtInstanceTriangleMeshConstRef = const ExtInstanceTriangleMesh&;
using ExtInstanceTriangleMeshRef = ExtInstanceTriangleMesh&;

class ExtMotionTriangleMesh;
using ExtMotionTriangleMeshUPtr = std::unique_ptr<ExtMotionTriangleMesh>;
using ExtMotionTriangleMeshConstUPtr = std::unique_ptr<const ExtMotionTriangleMesh>;
using ExtMotionTriangleMeshConstRef = const ExtMotionTriangleMesh&;
using ExtMotionTriangleMeshRef = ExtMotionTriangleMesh&;

class Matrix4x4;
using Matrix4x4Ptr = std::shared_ptr<Matrix4x4>;
using Matrix4x4ConstPtr = std::shared_ptr<Matrix4x4>;
using Matrix4x4UPtr = std::unique_ptr<Matrix4x4>;
using Matrix4x4ConstUPtr = std::unique_ptr<Matrix4x4>;

class NamedObject;
using NamedObjectPtr = std::shared_ptr<NamedObject>;
using NamedObjectConstPtr = std::shared_ptr<const NamedObject>;
using NamedObjectUPtr = std::unique_ptr<NamedObject>;
using NamedObjectConstUPtr = std::unique_ptr<const NamedObject>;
using NamedObjectRef = NamedObject&;
using NamedObjectConstRef = const NamedObject&;

class Normal;
using NormalPtr = std::shared_ptr<Normal>;
using NormalConstPtr = std::shared_ptr<Normal>;
using NormalUPtr = std::unique_ptr<Normal>;
using NormalConstUPtr = std::unique_ptr<Normal>;

class Point;
using PointPtr = std::shared_ptr<Point>;
using PointConstPtr = std::shared_ptr<const Point>;
using PointUPtr = std::unique_ptr<Point>;
using PointConstUPtr = std::unique_ptr<const Point>;

class Ray;
using RayPtr = std::shared_ptr<Ray>;
using RayConstPtr = std::shared_ptr<const Ray>;
using RayUPtr = std::unique_ptr<Ray>;
using RayConstUPtr = std::unique_ptr<const Ray>;

class RayHit;
using RayHitPtr = std::shared_ptr<RayHit>;
using RayHitConstPtr = std::shared_ptr<const RayHit>;
using RayHitUPtr = std::unique_ptr<RayHit>;
using RayHitConstUPtr = std::unique_ptr<const RayHit>;

class RGBColor;
using RGBColorPtr = std::shared_ptr<RGBColor>;
using RGBColorConstPtr = std::shared_ptr<const RGBColor>;
using RGBColorUPtr = std::unique_ptr<RGBColor>;
using RGBColorConstUPtr = std::unique_ptr<const RGBColor>;

class Triangle;
//using TrianglePtr = std::shared_ptr<Triangle>;
//using TriangleConstPtr = std::shared_ptr<const Triangle>;
using TriangleUPtr = std::unique_ptr<Triangle>;
using TriangleConstUPtr = std::unique_ptr<const Triangle>;

class TriangleMesh;
//using TriangleMeshPtr = std::shared_ptr<TriangleMesh>;
//using TriangleMeshConstPtr = std::shared_ptr<const TriangleMesh>;
using TriangleMeshUPtr = std::unique_ptr<TriangleMesh>;
using TriangleMeshConstUPtr = std::unique_ptr<const TriangleMesh>;
using TriangleMeshRef = TriangleMesh&;
using TriangleMeshConstRef = const TriangleMesh&;

class UV;
using UVPtr = std::shared_ptr<UV>;
using UVConstPtr = std::shared_ptr<const UV>;
using UVUPtr = std::unique_ptr<UV>;
using UVConstUPtr = std::unique_ptr<const UV>;

class Vector;
using VectorPtr = std::shared_ptr<Vector>;
using VectorConstPtr = std::shared_ptr<const Vector>;
using VectorUPtr = std::unique_ptr<Vector>;
using VectorConstUPtr = std::unique_ptr<const Vector>;

class Properties;
using PropertiesPtr = const std::unique_ptr<Properties> &;
using PropertiesConstPtr = const std::unique_ptr<const Properties> &;
using PropertiesUPtr = std::unique_ptr<Properties>;
using PropertiesRef = Properties &;
using PropertiesConstRef = const Properties &;

using JThread = std::jthread;
using JThreadPtr = std::unique_ptr<std::jthread>;

class RandomGenerator;
using RandomGeneratorUPtr = std::unique_ptr<RandomGenerator>;

}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

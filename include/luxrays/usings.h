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
#include <thread>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_free.hpp>


// Macro to define useful associated types
#define DECLARE_SUBTYPES(T) \
class T; \
using T##Ref = T&; \
using T##ConstRef = const T&; \
using T##UPtr = std::unique_ptr<T>; \
using T##ConstUPtr = std::unique_ptr<const T>; \
using T##Ptr = const std::unique_ptr<T> &; \
using T##ConstPtr = const std::unique_ptr<const T> &; \
using T##SPtr = std::shared_ptr<T>; \
using T##ConstSPtr = std::shared_ptr<const T>;



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

	// Allow boolean context usage (e.g., if (optPtr) ...)
	explicit operator bool() const {
		return this->has_value();
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

DECLARE_SUBTYPES(Accelerator);
DECLARE_SUBTYPES(BBox);
DECLARE_SUBTYPES(BSphere);
DECLARE_SUBTYPES(Context);
DECLARE_SUBTYPES(DataSet);
DECLARE_SUBTYPES(Device);
DECLARE_SUBTYPES(DeviceDescription);
DECLARE_SUBTYPES(HardwareDevice);
DECLARE_SUBTYPES(IntersectionDevice)
DECLARE_SUBTYPES(Mesh);
DECLARE_SUBTYPES(ExtMesh);
DECLARE_SUBTYPES(ExtMesh);
DECLARE_SUBTYPES(ExtTriangleMesh);
DECLARE_SUBTYPES(ExtInstanceTriangleMesh);
DECLARE_SUBTYPES(ExtMotionTriangleMesh);
DECLARE_SUBTYPES(Matrix4x4);
DECLARE_SUBTYPES(NamedObject);
DECLARE_SUBTYPES(Normal);
DECLARE_SUBTYPES(Point);
DECLARE_SUBTYPES(Ray);
DECLARE_SUBTYPES(RayHit);
DECLARE_SUBTYPES(RGBColor);
DECLARE_SUBTYPES(Triangle);
DECLARE_SUBTYPES(TriangleMesh);
DECLARE_SUBTYPES(UV);
DECLARE_SUBTYPES(Vector);
DECLARE_SUBTYPES(Properties);
DECLARE_SUBTYPES(RandomGenerator);

using JThread = std::jthread;
using JThreadUPtr = std::unique_ptr<std::jthread>;


}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

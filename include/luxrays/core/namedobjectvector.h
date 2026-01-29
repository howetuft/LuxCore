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

#ifndef _LUXRAYS_NAMEDOBJECTVECTOR_H
#define	_LUXRAYS_NAMEDOBJECTVECTOR_H

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <ostream>
#include <ranges>

#include <boost/bimap.hpp>
#include <boost/bimap/unordered_set_of.hpp>

#include "luxrays/luxrays.h"
#include "luxrays/usings.h"
#include "luxrays/core/namedobject.h"

namespace slg {
	class ExtMeshCache;
}

namespace luxrays {
using NamedObjectRefWrapper = std::reference_wrapper<const luxrays::NamedObject>;
}

namespace luxrays {

inline std::size_t hash_value(const luxrays::NamedObjectRefWrapper & no)
{
	boost::hash<std::string> hasher;
	return hasher(no.get().GetName());
}

inline bool operator==(
	luxrays::NamedObjectRefWrapper const& lhs,
	luxrays::NamedObjectRefWrapper const& rhs
) {
	return std::addressof(lhs.get()) == std::addressof(rhs.get());
}

struct UndefinedNamedObjectError : public std::runtime_error {
	UndefinedNamedObjectError(const std::string& msg) : std::runtime_error(msg) {}
};




class NamedObjectVector {
public:
	NamedObjectVector();
	virtual ~NamedObjectVector() = default;

	/// DefineObj allows to transfer the given named object to the object container.
	/// The container takes ownership of the transfered object.
	///
	/// If an object with the same name already exists in the container, it is
	/// swapped with the new transfered object and returned to caller
	///
	/// Return: a tuple containing:
	/// - a reference to the transfered object, now in the container,
	/// - and the old object (as a std::unique_ptr), if there was one,
	///	  or nullptr in other cases
	///
	///	See also specialization for NamedObject
	template<typename T>
	std::tuple< T&, std::unique_ptr<T> > DefineObj(std::unique_ptr<T>&& obj) {
		// Run base class method
		auto [newObjRef, oldObjPtr] = DefineObj<NamedObject>(std::move(obj));

		// Cast result to derived and return
		auto& newDerivedRef = dynamic_cast<T&>(newObjRef);
		auto oldDerivedPtr = oldObjPtr ?
			dynamic_uptr_cast<T>(std::move(oldObjPtr)) :
			std::unique_ptr<T>();

		return std::make_tuple(std::ref(newDerivedRef), std::move(oldDerivedPtr));
	}


	bool IsObjDefined(const std::string &name) const;

	NamedObjectConstRef GetObj(const std::string &name) const;
	NamedObjectRef GetObj(const std::string &name);
	NamedObjectConstRef GetObj(const u_int index) const;
	NamedObjectRef GetObj(const u_int index);

	u_int GetIndex(const std::string &name) const;
	u_int GetIndex(NamedObjectConstRef o) const;

	const std::string &GetName(const u_int index) const;
	const std::string &GetName(NamedObjectConstRef o) const;

	u_int GetSize()const;
	auto GetNames() const {
		// Returns a view of references to the object names
		return objs | std::views::transform([](const auto& obj) -> const std::string & {
			return obj->GetName();
		});
	}
	//void GetNames(std::vector<std::string> &names) const;
	auto GetObjs() {
		// Returns a view of references to the objects
		return objs | std::views::transform([](const auto& obj) -> NamedObjectRef {
			return *obj;
		});
	}

	void DeleteObj(const std::string &name);
	void DeleteObjs(const std::vector<std::string> &names);

	std::string ToString() const;

private:
	friend class slg::ExtMeshCache;

	using Name2IndexType = boost::bimap<
		boost::bimaps::unordered_set_of<std::string>,
		boost::bimaps::unordered_set_of<u_int>
	> ;

	using Index2ObjType = boost::bimap<
		boost::bimaps::unordered_set_of<u_int>,
		boost::bimaps::unordered_set_of<luxrays::NamedObjectRefWrapper>
	>;

	// Declaration order matters for construction/destruction!
	// Please do not modify
	std::vector<NamedObjectUPtr> objs;
	Name2IndexType name2index;
	Index2ObjType index2obj;


};

// Specialization (declaration) of DefineObj for NamedObject class, so that
// derived classes can rely on it
template<>
std::tuple< NamedObject&, std::unique_ptr<NamedObject> >
NamedObjectVector::DefineObj(std::unique_ptr<NamedObject>&& obj);

}

#endif	/* _LUXRAYS_NAMEDOBJECTVECTOR_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

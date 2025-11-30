// Copyright 2025 - 2025, by Authors (see AUTHORS.txt)
// SPDX-License-Identifier: Apache-2.0


/***************************************************************************
 * Copyright 2025-2025 by Authors (see AUTHORS.txt)                        *
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


#pragma once

#include <vector>
#include <execution>
#include <array>
#include <optional>
#include <span>
#include <utility>
#include <variant>

#include <boost/serialization/vector.hpp>

#include "luxrays/core/geometry/point.h"

constexpr unsigned EXTMESH_MAX_DATA_COUNT = 8;


namespace luxrays {


// A buffer class, for elaborate types, like Points, Triangles etc.
// with an underlying storage in integral types (like float, bytes etc.)
//
// We ensure the buffer is always padded with an additional last element, to
// meet Embree requirements. In practice, it means we ensure
// capacity >= size + 1
//
// Content can be large collections, thus we favorize move semantics and make
// copy explicit by declaring copy constructors private.
// T is front type
// I is integral underlying type
template<
	typename T,
	typename I = std::conditional<std::is_same<T, Triangle>::value, unsigned, float>::type
>
class Buffer : public std::span<T> {
public:
	// Internal types
	using data_t = std::vector<I>;
	using span_t = std::span<T>;


	// TODO concepts (check alignement, size etc.)

	static constexpr size_t DATA_RATIO = sizeof(T) / sizeof(I);

	// Constructors
	// Default
	Buffer() : m_data() {
		ensure_padding();
		rebuild_span();
	}

	explicit Buffer(size_t size) : m_data(size * DATA_RATIO) {
		ensure_padding();
		rebuild_span();
	}

	// Copy constructor (made explicit)
	explicit Buffer(const Buffer<T, I>& other) :
		m_data(other.m_data)
	{
		ensure_padding();
		rebuild_span();
	}


	// Move
	Buffer(Buffer<T, I>&&) = default;
	Buffer<T, I>& operator=(Buffer<T, I>&&) = default;

	// Copy from existing buffer
	Buffer(const T* raw, size_t p_size) :
		m_data(
			reinterpret_cast<const I*>(raw),
			reinterpret_cast<const I*>(raw + p_size)
		)
	{
		ensure_padding();
		rebuild_span();
	}

	template<typename InputIt>
	Buffer(InputIt p_first, InputIt p_last) :
		Buffer(std::distance(p_first, p_last))
	{
		std::copy(std::execution::par, p_first, p_last, this->begin());
	}


	// Construct (copy) from shared_ptr of internal type
	Buffer(const std::shared_ptr<I[]> p, size_t n)
	{
		auto dataspan = std::span<I>(p.get(), n);
		m_data = std::vector<I>(dataspan.begin(), dataspan.end());

		ensure_padding();
		rebuild_span();
	}

	// Iterator
	using span_t::iterator;
	using span_t::begin;
	using span_t::end;

	// Element access
	using span_t::front;
	using span_t::back;
	using span_t::operator[];

	// Observers
	using span_t::size;
	using span_t::size_bytes;
	using span_t::empty;
	operator bool() const {
		return not empty();
	}

	// Subviews
	using span_t::first;
	using span_t::last;

	// Underlying buffer
	I* underlying() {
		return m_data.data();
	}
	const I* underlying() const {
		return m_data.data();
	}

	// Capacity
	void resize(size_t count) {
		auto new_data_size = count * sizeof(T) / sizeof(I);
		// We reserve one more for padding
		m_data.reserve(new_data_size + 1);

		// Resize and rebuild span
		m_data.resize(new_data_size);
		ensure_padding();
		rebuild_span();
	}

	// Modifiers
	void clear() {
		m_data.clear();
		ensure_padding();  // Maybe useless
		rebuild_span();
	}
	void swap(Buffer<T, I>& other) {
		std::swap(*this, other);
	}

	void copy(const Buffer<T, I>& other) {
		Buffer<T> copy_of_other(other);  // Make a copy with private constructor
		this->swap(copy_of_other);
		ensure_padding();
		rebuild_span();
	}

	Buffer<T>& operator+=(const Buffer<T>& other) {  // Concatenation
		auto this_size = this->size();
		auto other_size = other.size();

		this->resize(this_size + other_size);

		std::copy(
			std::execution::par,
			other.begin(),
			other.end(),
			this->begin() + this_size
		);

		ensure_padding();
		rebuild_span();
		return *this;
	}

	void push_back(const T& value) {
		this->resize(this->size() + 1);
		ensure_padding();
		rebuild_span();
		this->back() = value;
	}
	void push_back(T&& value) {
		this->resize(this->size() + 1);
		ensure_padding();
		rebuild_span();
		this->back() = std::move(value);
	}

	// Serialization
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const u_int version) {
		ar & m_data;
		// Repad and reconstruct the span base class after serialization
		ensure_padding();
        rebuild_span();
	}

private:
	data_t m_data;  // Underlying data buffer

	// Reinterpreter
	inline T* cast_data() {
		return reinterpret_cast<T*>(m_data.data());
	}

	// Sentinel for padding
	static constexpr std::tuple<float, unsigned> SENTINEL = {1234.1234f, -1};

	// Ensure data are padded (embree requirements)
	// (caveat: it does not call rebuild_span)
	inline void ensure_padding() {
		auto data_size = m_data.size();
		if (not (m_data.capacity() > data_size)) {
			// Add padding
			m_data.reserve(data_size + 1);
		}
		// Stamp sentinel
		*(m_data.data() + data_size) = std::get<I>(SENTINEL);
	}

	// Rebuild span (after data iterators have been invalidated)
	inline void rebuild_span() {
		// Assert padding
		assert(m_data.capacity() > m_data.size());

		// Compute span size
		auto span_size = m_data.size() * sizeof(I) / sizeof(T);

		// Assert underlying data integrity
		assert(m_data.size() * sizeof(I) % sizeof(T) == 0);

        this->std::span<T>::operator=(span_t(cast_data(), span_size));
	}


	// Copy assignment (made private to avoid silent copy)
	Buffer<T, I>& operator=(const Buffer<T, I>&) = default;

};


// Optional buffer

template<typename T>
class Optionals : public std::optional<Buffer<T>> {
public:
	using optbase_t = std::optional<Buffer<T>>;
	using buffer_t = Buffer<T>;

	Optionals(const std::nullopt_t opt = std::nullopt) {}

	// Move construct from existing buffer
	explicit Optionals(Buffer<T>&& buf) :
		optbase_t(buf ? std::make_optional(std::move(buf)) : std::nullopt)
	{}

	// Copy from raw buffer
	Optionals(const T* rawbuf, size_t size) : optbase_t(buffer_t(rawbuf, size))
	{}

	// Copy from smart pointer buffer
	template <typename S>  // Source type (could be float, for instance)
	Optionals(const std::shared_ptr<S[]> shared, size_t size) :
		optbase_t(
			shared.use_count() ?
			std::make_optional<Buffer<T>>(shared, size) : std::nullopt
		)
	{}

	// Initialize with n default elements
	explicit Optionals(size_t n) :
		std::optional<Buffer<T>>(n)
	{}

	// Move constructor and assignment
	Optionals(Optionals<T>&&) = default;
	Optionals<T>& operator=(Optionals<T>&&) = default;

	// Move construct from buffer base type
	Optionals(Buffer<T>::span_t&& vec) :
		optbase_t(std::forward<typename Buffer<T>::span_t>(vec)) {}

	void resize(size_t size) {
		if (not size or not this->has_value()) return;
		this->value().resize(size);
	}

	const Buffer<T>& buffer() const { return this->value(); }
	Buffer<T>& buffer() { return this->value(); }

	// Concatenate two optionals
	Optionals& operator+=(const Optionals& other) {
		if (not this->has_value() or not other.has_value()) return (*this);

		auto& this_buffer = this->value();
		const auto& other_buffer = other.buffer();
		this_buffer += other_buffer;
		return *this;
	}

	const T * get() const {
		return buffer().data();
	}
	T * get() { return buffer().data(); }
	const void * getvoid() const {
		return reinterpret_cast<const void *>(buffer().data());
	}
	void clear() { this->reset(); }

	size_t size() const { return buffer().size(); }

	T& operator[]( std::ptrdiff_t idx ) { return this->value()[idx]; }
	const T& operator[]( std::ptrdiff_t idx ) const { return this->value()[idx]; }

	// Copy other into this
	void copy(const Optionals<T>& other) {
		if (not other.has_value()) {
			this->reset();
			return;
		}

		// Make a copy of the buffer
		Buffer<T> buffer;
		buffer.copy(other.buffer());
		Optionals<T> newOptional{std::move(buffer)};
		this->swap(newOptional);
	}



private:
	// Restrict access to copy constructors, in order to avoid
	// unwanted copy invocation (move is preferred)
	Optionals(const Optionals<T>& other) = default;
	Optionals<T>& operator=(const Optionals<T>& other) = delete;

};


// Array of buffers
template<typename T>
struct ArrayOfBuffers :
	std::array<Buffer<T>, EXTMESH_MAX_DATA_COUNT>
{
	ArrayOfBuffers() {};

	ArrayOfBuffers(std::array<T *, EXTMESH_MAX_DATA_COUNT> * rawarray, size_t size) {
		if (not rawarray) return;
		for (size_t i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
			auto src = (*rawarray)[i];
			if (src) {
				(*this)[i] = Optionals<T>(src, size);
			}
		}
	}

	operator bool() const {
		return std::any_of(this->begin(), this->end(), std::identity());
	}
};


// Array of optional buffers
template<typename T>
struct ArrayOfOptionals :
	std::array<Optionals<T>, EXTMESH_MAX_DATA_COUNT>
{
	ArrayOfOptionals() {};

	ArrayOfOptionals(std::array<T *, EXTMESH_MAX_DATA_COUNT> * rawarray, size_t size) {
		if (not rawarray) return;
		for (size_t i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
			auto src = (*rawarray)[i];
			if (src) {
				(*this)[i] = Optionals<T>(src, size);
			}
		}
	}

	ArrayOfOptionals(ArrayOfBuffers<T>&& buffers) : ArrayOfOptionals() {
		if (not buffers) return;  // None of the buffers contains data

		for(size_t i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
			if (not buffers[i]) continue;
			(*this)[i] = Optionals<T>(std::move(buffers[i]));
		}
	}
};


// Make an array from a shared_ptr
template <typename T>
std::vector<T> make_vector_from_shared_ptr(const std::shared_ptr<T[]>& shared_array, size_t size) {
    return std::vector<T>(shared_array.get(), shared_array.get() + size);
}

} // namespace luxrays
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

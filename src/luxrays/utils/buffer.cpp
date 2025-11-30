
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

#include <luxrays/utils/buffer.h>

namespace luxrays {

// Point specialization

const Point POINT_SENTINEL{1234.1234f, 1234.1234f, 1234.1234f};

template <>
void Buffer<Point>::allocate(size_t size) {
	// Embree requires a float padding field at the end
	// This is a trick so I can check if the buffer has been really allocated
	// with AllocVerticesBuffer() or not. It is useful for debugging LuxCore
	// applications.
	Buffer newBuf(size + 1);
	newBuf[size] = POINT_SENTINEL;
	this->swap(newBuf);

}

template <>
Buffer<Point>::iterator Buffer<Point>::end() { return std::prev(base_t::end()); }

template <>
Buffer<Point>::const_iterator Buffer<Point>::end() const {
	return std::prev(base_t::end());
}

template <>
Buffer<Point>::reference Buffer<Point>::back() {
	return *std::prev(this->end());
}
template <>
Buffer<Point>::const_reference Buffer<Point>::back() const {
	return *std::prev(this->end());
}

template <>
size_t Buffer<Point>::size() const {
	assert(base_t::back() == POINT_SENTINEL);
	return base_t::size() - 1;
}

template <>
void Buffer<Point>::resize(size_t size) {
	base_t::pop_back(); // Remove sentinel
	base_t::resize(size + 1);
	base_t::back() = POINT_SENTINEL;
}


template <>
void Buffer<Point>::push_back( const Point& value ) {
	assert(base_t::back() == POINT_SENTINEL);
	base_t::pop_back();
	base_t::push_back(value);
	base_t::push_back(POINT_SENTINEL);
}
template <>
void Buffer<Point>::push_back( Point&& value ) {
	assert(base_t::back() == POINT_SENTINEL);
	base_t::pop_back();
	base_t::push_back(value);
	base_t::push_back(POINT_SENTINEL);
}

template <>
Buffer<Point> Buffer<Point>::operator+=(const Buffer<Point>& other) {

	auto this_size = this->size();
	auto other_size = other.size();

	this->pop_back();  // Remove sentinel

	base_t::resize(this_size + other_size - 1);

	std::copy(
		std::execution::par,
		other.begin(),
		other.end(),
		this->begin() + this_size
	);
	assert(base_t::back() == POINT_SENTINEL);
	return *this;
}

} // namespace luxrays
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

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

#include <iostream>

void __verify(const char * condition, const char * file, int line) {
    std::cerr
        << "Warning! Condition failed: '"
        << std::string(condition) <<
        "', in file " <<file << "line " <<  line << ".\n";
}

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <new>
#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>
namespace luxmem {
	using ::scalable_malloc;
	using ::scalable_free;
}

// Throwing versions
#if 0
void* operator new(std::size_t sz)
{
    if (sz == 0)
        ++sz; // avoid std::malloc(0) which may return nullptr on success

    if (void *ptr = luxmem::scalable_malloc(sz))
        return ptr;

    throw std::bad_alloc{}; // required by [new.delete.single]/3
}

void* operator new[](std::size_t sz)
{
    if (sz == 0)
        ++sz; // avoid std::malloc(0) which may return nullptr on success

    if (void *ptr = luxmem::scalable_malloc(sz))
        return ptr;

    throw std::bad_alloc{}; // required by [new.delete.single]/3
}

void operator delete (void* ptr) noexcept
{
  if (ptr != 0) luxmem::scalable_free (ptr);
}

void operator delete[] (void* ptr) noexcept
{
  operator delete (ptr);
}

void operator delete (void* ptr, std::size_t size) noexcept
{
  if (ptr != 0) luxmem::scalable_free (ptr);
}

// Non throwing

void* operator new (size_t size, const std::nothrow_t&) noexcept
{
  assert(false);
  if (size == 0) size = 1;
  if (void* ptr = luxmem::scalable_malloc (size))
    return ptr;
  return NULL;
}

void* operator new[] (size_t size, const std::nothrow_t&) noexcept
{
  return operator new (size, std::nothrow);
}

void operator delete (void* ptr, const std::nothrow_t&) noexcept
{
  if (ptr != 0) luxmem::scalable_free (ptr);
}

void operator delete[] (void* ptr, const std::nothrow_t&) noexcept
{
  operator delete (ptr, std::nothrow);
}
#endif
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

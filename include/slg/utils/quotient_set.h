/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
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

#include <memory>
#include <vector>
#include <utility>

namespace slg {

// Forward declaration
class UnionFind;

using EquivalenceRelation = std::vector<std::pair<size_t, size_t>>;

// QuotientSet represents a partition of a set of elements into equivalence classes
// (quotient set).

class QuotientSet {
public:
	QuotientSet(size_t elementCount, const EquivalenceRelation& relation);

private:
	std::unique_ptr<UnionFind> dsu;
};

} // namespace slg

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

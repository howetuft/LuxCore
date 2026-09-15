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

#include <vector>
#include <utility>
#include <ranges>

namespace slg {

using Classes = std::vector<std::vector<size_t>>;

// GroupByEquivalence computes the quotient set (equivalence classes) from an
// equivalence relation.
// Given a set of elements [0, numElements) and a relation (pairs of equivalent
// elements), it returns a Classes object where each inner vector contains all
// elements that are equivalent under the transitive closure of the relation.
//
// In other words, if relation contains pairs indicating which elements are
// equivalent, GroupByEquivalence groups all elements into disjoint classes
// where each class contains all elements that are transitively equivalent.
//
// The function relies on a parallel implementation of the Union-Find algorithm
// (using TBB) and should be fast for large equivalence relations.
//
// The function accepts any range of std::pair<size_t, size_t> as the relation
// parameter, including std::vector, std::span, and lazy views like
// std::views::transform.
//
// Usage examples:
//   // With std::vector
//   std::vector<std::pair<size_t, size_t>> pairs = {{0,1}, {2,3}};
//   auto clusters = GroupByEquivalence(n, pairs);
//
//   // With std::span
//   auto clusters = GroupByEquivalence(n, std::span(pairs));
//
//   // With std::views::transform (lazy evaluation)
//   auto view = std::views::iota(0u, m)
//       | std::views::transform([](size_t i) {
//           return std::make_pair(i, i+1);
//       });
//   auto clusters = GroupByEquivalence(n, view);

template<std::ranges::range Range>
    requires std::same_as<std::ranges::range_value_t<Range>,
        std::pair<size_t, size_t>>
Classes GroupByEquivalence(size_t numElements, Range&& relation);

} // namespace slg

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

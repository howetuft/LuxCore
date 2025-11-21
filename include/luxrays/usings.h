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

namespace luxrays {

class Accelerator;
using AcceleratorPtr = std::shared_ptr<Accelerator>;
using AcceleratorConstPtr = std::shared_ptr<const Accelerator>;

class DataSet;
using DataSetPtr = std::shared_ptr<DataSet>;
using DataSetConstPtr = std::shared_ptr<const DataSet>;

}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

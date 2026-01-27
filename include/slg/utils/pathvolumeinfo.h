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

#ifndef _SLG_VOLUMEINFO_H
#define	_SLG_VOLUMEINFO_H

#include <ostream>

#include "luxrays/usings.h"
#include "slg/slg.h"

namespace slg {

// OpenCL data types
namespace ocl {
#include "slg/utils/pathvolumeinfo_types.cl"
}

//------------------------------------------------------------------------------
// PathVolumeInfo
//
// A class used to store volume related information on the on going path
//------------------------------------------------------------------------------

#define PATHVOLUMEINFO_SIZE 8

class Volume;

class PathVolumeInfo {
public:
	PathVolumeInfo();

	VolumeConstRef GetCurrentVolume() const { return *currentVolume; }
	bool HasCurrentVolume() const { return bool(currentVolume); }
	VolumeConstRef GetVolume(const u_int i) const { return *volumeList[i]; }
	const u_int GetListSize() const { return volumeListSize; }

	void AddVolume(VolumeConstOPtr vol);
	void AddVolume(VolumeConstRef vol);
	void RemoveVolume(VolumeConstOPtr vol);
	void SetCurrentVolume(VolumeConstOPtr vol) { currentVolume = vol; }
	void SetCurrentVolume(VolumeConstRef vol) { currentVolume.reset(std::addressof(vol)); }
	void SetVolume(const u_int i, VolumeConstOPtr vol) { volumeList[i] = vol; }

	VolumeConstOPtr SimulateRemoveVolume(VolumeConstOPtr vol) const;
	VolumeConstOPtr SimulateAddVolume(VolumeConstOPtr) const;

	void SetScatteredStart(const bool v) { scatteredStart = v; }
	bool IsScatteredStart() const { return scatteredStart; }

	void Update(const BSDFEvent eventType, const BSDF &bsdf);
	bool ContinueToTrace(const BSDF &bsdf) const;

	void SetHitPointVolumes(
		HitPoint &hitPoint,
		VolumeConstOPtr matInteriorVolume,
		VolumeConstOPtr matExteriorVolume,
		VolumeConstOPtr defaultWorldVolume
	) const;

private:
	static bool CompareVolumePriorities(
		VolumeConstOPtr vol1,
		VolumeConstOPtr vol2
	);

	VolumeConstOPtr currentVolume;
	// Using a fixed array here mostly to have the same code as the OpenCL implementation
	std::array<VolumeConstOPtr, PATHVOLUMEINFO_SIZE> volumeList;
	u_int volumeListSize;

	bool scatteredStart;
};

extern std::ostream &operator<<(std::ostream &os, const PathVolumeInfo &pvi);

}

#endif	/* _SLG_VOLUMEINFO_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

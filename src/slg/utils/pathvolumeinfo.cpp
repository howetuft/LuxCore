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

#include <cstddef>
#include <stdexcept>

#include "luxrays/usings.h"
#include "slg/bsdf/bsdf.h"
#include "slg/usings.h"
#include "slg/volumes/volume.h"
#include "slg/utils/pathvolumeinfo.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PathVolumeInfo
//------------------------------------------------------------------------------

PathVolumeInfo::PathVolumeInfo() {
	currentVolume = std::nullopt;
	volumeListSize = 0;

	scatteredStart = false;
}

void PathVolumeInfo::AddVolume(OptionalPtr<const Volume> vol) {
	if (!vol || volumeListSize == PATHVOLUMEINFO_SIZE) {
		// Out of space, I just ignore the volume
		return;
	}


	// Update the current volume. ">=" because I want to catch the last added volume.
	if (!HasCurrentVolume() || (vol->GetPriority() >= GetCurrentVolume().GetPriority())) {
		currentVolume = vol;
	}

	// Add the volume to the list
	volumeList[volumeListSize++] = vol;
}

void PathVolumeInfo::RemoveVolume(OptionalPtr<const Volume> vol) {
	if (!vol || volumeListSize == 0) {
		// empty volume list
		return;
	}

	// Update the current volume and the list
	bool found = false;
	currentVolume = std::nullopt;
	for (u_int i = 0; i < volumeListSize; ++i) {
		if (found) {
			// Re-compact the list
			SetVolume(i - 1, GetVolume(i));
		} else if (volumeList[i] == vol) {
			// Found the volume to remove
			found = true;
			continue;
		}

		// Update currentVolume. ">=" because I want to catch the last added volume.
		if (
			not HasCurrentVolume()
			or GetVolume(i).GetPriority() >= GetCurrentVolume().GetPriority()
		) {
			SetCurrentVolume(GetVolume(i));
		}
	}

	// Update the list size
	--volumeListSize;
}

OptionalPtr<const Volume> PathVolumeInfo::SimulateAddVolume(OptionalPtr<const Volume> vol) const {
	// A volume wins over current if and only if it is the same volume or has an
	// higher priority

	if (HasCurrentVolume()) {
		if (vol) {
			auto curPriority = GetCurrentVolume().GetPriority();
			auto volPriority = vol->GetPriority();
			return
				curPriority > volPriority ?
				OptionalPtr<const Volume>(GetCurrentVolume()) :
				vol;
		} else {
			return OptionalPtr<const Volume>(GetCurrentVolume());
		}
	} else return vol;
}

OptionalPtr<const Volume> PathVolumeInfo::SimulateRemoveVolume(OptionalPtr<const Volume> vol) const {


	if (not vol || volumeListSize == 0) {
		// NULL volume or empty volume list
		return HasCurrentVolume() ? OptionalPtr<const Volume>(GetCurrentVolume()) : std::nullopt;
	}

	// Update the current volume
	bool found = false;
	OptionalPtr<const Volume> newCurrentVolume = std::nullopt;
	for (u_int i = 0; i < volumeListSize; ++i) {
		if (!found && GetVolume(i) == vol) {
			// Found the volume to remove
			found = true;
			continue;
		}

		// Update newCurrentVolume. ">=" because I want to catch the last added volume.
		if (!newCurrentVolume || (GetVolume(i).GetPriority() >= VolumeConstRef(newCurrentVolume).GetPriority())) {
			newCurrentVolume = GetVolume(i);
		}
	}

	return newCurrentVolume;
}

void PathVolumeInfo::Update(const BSDFEvent eventType, const BSDF &bsdf) {
	// Update only if it isn't a volume scattering and the material can TRANSMIT
	if (bsdf.IsVolume())
		scatteredStart = true;
	else {
		scatteredStart = false;

		if(eventType & TRANSMIT) {
			if (bsdf.hitPoint.intoObject)
				AddVolume(bsdf.GetMaterialInteriorVolume());
			else
				RemoveVolume(bsdf.GetMaterialInteriorVolume());
		}
	}
}

bool PathVolumeInfo::CompareVolumePriorities(
	OptionalPtr<const Volume> vol1,
	OptionalPtr<const Volume> vol2
) {
	// Special cases: one or both are empty
	if (not vol2) return true;
	if (not vol1) return false;

	// A volume wins over another if and only if it is the same volume or has an
	// higher priority

	if (vol1 == vol2)
		return true;
	else
		return (vol1->GetPriority() > vol2->GetPriority());
}

bool PathVolumeInfo::ContinueToTrace(const BSDF &bsdf) const {
	// Check if the volume priority system has to be applied
	if (bsdf.GetEventTypes() & TRANSMIT) {
		// Ok, the surface can transmit so check if volume priority
		// system is telling me to continue to trace the ray

		// I have to continue to trace the ray if:
		//
		// 1) I'm entering an object and the interior volume has a
		// lower priority than the current one (or is the same volume).
		//
		// 2) I'm exiting an object, the material is NULL and I'm not leaving
		// the current volume.

		OptionalPtr<const Volume> bsdfInteriorVol = bsdf.GetMaterialInteriorVolume();

		// Condition #1
		if (bsdf.hitPoint.intoObject && CompareVolumePriorities(currentVolume, bsdfInteriorVol))
			return true;

		// Condition #2
		//
		// I have to calculate the potentially new currentVolume in order
		// to check if I'm leaving the current one
		if ((!bsdf.hitPoint.intoObject) &&
				(bsdf.GetMaterialType() == NULLMAT) &&
				currentVolume && (SimulateRemoveVolume(bsdfInteriorVol) == currentVolume))
			return true;
	}

	return false;
}

void  PathVolumeInfo::SetHitPointVolumes(HitPoint &hitPoint,
		OptionalPtr<const Volume> matInteriorVolume,
		OptionalPtr<const Volume> matExteriorVolume,
		OptionalPtr<const Volume> defaultWorldVolume) const {
	// Set interior and exterior volumes

	if (hitPoint.intoObject) {
		// From outside to inside the object

		if (matInteriorVolume) {
			hitPoint.interiorVolume = SimulateAddVolume(matInteriorVolume);
		} else {
			hitPoint.interiorVolume = std::nullopt;
		}

		if (not HasCurrentVolume())
			hitPoint.exteriorVolume = matExteriorVolume;
		else {
			// if (!material->GetExteriorVolume()) there may be conflict here
			// between the material definition and the currentVolume value.
			// The currentVolume value wins.
			hitPoint.exteriorVolume = GetCurrentVolume();
		}

		if (not hitPoint.exteriorVolume) {
			// No volume information, I use the default volume
			hitPoint.exteriorVolume = defaultWorldVolume;
		}
	} else {
		// From inside to outside the object

		if (!HasCurrentVolume())
			hitPoint.interiorVolume = matInteriorVolume;
		else {
			// if (!material->GetInteriorVolume()) there may be conflict here
			// between the material definition and the currentVolume value.
			// The currentVolume value wins.
			hitPoint.interiorVolume = GetCurrentVolume();
		}

		if (!hitPoint.interiorVolume) {
			// No volume information, I use the default volume
			hitPoint.interiorVolume = defaultWorldVolume;
		}
		if (matInteriorVolume) {
			hitPoint.exteriorVolume = SimulateRemoveVolume(matInteriorVolume);
		}
	}
}

namespace slg {

ostream &operator<<(ostream &os, const PathVolumeInfo &pvi) {
	os << "PathVolumeInfo[" << (pvi.HasCurrentVolume() ? pvi.GetCurrentVolume().GetName() : "NULL") << ", ";

	for (u_int i = 0; i < pvi.GetListSize(); ++i)
		os << "#" << i << " => " << (pvi.HasCurrentVolume() ? pvi.GetCurrentVolume().GetName() : "NULL") << ", ";
	
	os << pvi.IsScatteredStart() << "]";

	return os;
}

}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

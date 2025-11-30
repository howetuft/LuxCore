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

#ifndef _SLG_STRANDSSHAPE_H
#define	_SLG_STRANDSSHAPE_H

#include <string>
#include <vector>

#include "luxrays/utils/cyhair/cyHairFile.h"

#include "slg/shapes/shape.h"

namespace slg {

class StrendsShape : public Shape {
public:
	typedef enum {
		TESSEL_RIBBON, TESSEL_RIBBON_ADAPTIVE,
		TESSEL_SOLID, TESSEL_SOLID_ADAPTIVE
	} TessellationType;

	StrendsShape(SceneConstRef scene,
			const luxrays::cyHairFile *hairFile, const TessellationType tesselType,
			const u_int adaptiveMaxDepth, const float adaptiveError, 
			const u_int solidSideCount, const bool solidCapBottom, const bool solidCapTop,
			const bool useCameraPosition);
	virtual ~StrendsShape();

	virtual ShapeType GetType() const { return STRANDS; }

protected:
	virtual luxrays::ExtTriangleMeshPtr RefineImpl(SceneConstRef scene);
	
	void TessellateRibbon(
		SceneConstRef scene,
		const luxrays::Buffer<luxrays::Point> &hairPoints,
		const luxrays::Buffer<float> &hairSizes, const luxrays::Buffer<luxrays::Spectrum> &hairCols,
		const luxrays::Buffer<luxrays::UV> &hairUVs, const luxrays::Buffer<float> &hairTransps,
		luxrays::Buffer<luxrays::Point> &meshVerts, luxrays::Buffer<luxrays::Normal> &meshNorms,
		luxrays::Buffer<luxrays::Triangle> &meshTris, luxrays::Buffer<luxrays::UV> &meshUVs, luxrays::Buffer<luxrays::Spectrum> &meshCols,
		luxrays::Buffer<float> &meshTransps) const;
	void TessellateAdaptive(SceneConstRef scene,
		const bool solid, const luxrays::Buffer<luxrays::Point> &hairPoints,
		const luxrays::Buffer<float> &hairSizes, const luxrays::Buffer<luxrays::Spectrum> &hairCols,
		const luxrays::Buffer<luxrays::UV> &hairUVs, const luxrays::Buffer<float> &hairTransps,
		luxrays::Buffer<luxrays::Point> &meshVerts, luxrays::Buffer<luxrays::Normal> &meshNorms,
		luxrays::Buffer<luxrays::Triangle> &meshTris, luxrays::Buffer<luxrays::UV> &meshUVs, luxrays::Buffer<luxrays::Spectrum> &meshCols,
		luxrays::Buffer<float> &meshTransps) const;
	void TessellateSolid(SceneConstRef scene,
		const luxrays::Buffer<luxrays::Point> &hairPoints,
		const luxrays::Buffer<float> &hairSizes, const luxrays::Buffer<luxrays::Spectrum> &hairCols,
		const luxrays::Buffer<luxrays::UV> &hairUVs, const luxrays::Buffer<float> &hairTransps,
		luxrays::Buffer<luxrays::Point> &meshVerts, luxrays::Buffer<luxrays::Normal> &meshNorms,
		luxrays::Buffer<luxrays::Triangle> &meshTris, luxrays::Buffer<luxrays::UV> &meshUVs, luxrays::Buffer<luxrays::Spectrum> &meshCols,
		luxrays::Buffer<float> &meshTransps) const;

	// Tessellation options
	u_int adaptiveMaxDepth;
	float adaptiveError;
	u_int solidSideCount;
	bool solidCapBottom, solidCapTop;
	bool useCameraPosition;

	luxrays::ExtTriangleMeshPtr mesh;
};

}

#endif	/* _SLG_STRANDSSHAPE_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

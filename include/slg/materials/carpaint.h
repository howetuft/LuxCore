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

#ifndef _SLG_CARPAINTMAT_H
#define	_SLG_CARPAINTMAT_H

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// CarPaint material
//------------------------------------------------------------------------------

class CarPaintMaterial : public Material {
public:
	CarPaintMaterial(TextureConstOPtr frontTransp, TextureConstOPtr backTransp,
			TextureConstOPtr emitted, TextureConstOPtr bump,
			TextureConstOPtr kd, TextureConstOPtr ks1, TextureConstOPtr ks2, TextureConstOPtr ks3,
			TextureConstOPtr m1, TextureConstOPtr m2, TextureConstOPtr m3,
			TextureConstOPtr r1, TextureConstOPtr r2, TextureConstOPtr r3, TextureConstOPtr ka, TextureConstOPtr d);

	virtual MaterialType GetType() const { return CARPAINT; }
	virtual BSDFEvent GetEventTypes() const { return GLOSSY | REFLECT; };

	virtual luxrays::Spectrum Albedo(const HitPoint &hitPoint) const;

	virtual luxrays::Spectrum Evaluate(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW = NULL, float *reversePdfW = NULL) const;
	virtual luxrays::Spectrum Sample(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, luxrays::Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const;
	virtual void Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const;

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexsreferencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	struct CarPaintData {
		std::string name;
		float kd[COLOR_SAMPLES];
		float ks1[COLOR_SAMPLES];
		float ks2[COLOR_SAMPLES];
		float ks3[COLOR_SAMPLES];
		float r1, r2, r3;
		float m1, m2, m3;
	};
	static const struct CarPaintData data[8];
	static int NbPresets() { return 8; }

	TextureConstOPtr Kd;
	TextureConstOPtr Ks1;
	TextureConstOPtr Ks2;
	TextureConstOPtr Ks3;
	TextureConstOPtr M1;
	TextureConstOPtr M2;
	TextureConstOPtr M3;
	TextureConstOPtr R1;
	TextureConstOPtr R2;
	TextureConstOPtr R3;
	TextureConstOPtr Ka;
	TextureConstOPtr depth;
};

}

#endif	/* _SLG_CARPAINTMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

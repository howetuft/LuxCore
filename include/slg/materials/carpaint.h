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
	CarPaintMaterial(OptionalPtr<const Texture> frontTransp, OptionalPtr<const Texture> backTransp,
			OptionalPtr<const Texture> emitted, OptionalPtr<const Texture> bump,
			OptionalPtr<const Texture> kd, OptionalPtr<const Texture> ks1, OptionalPtr<const Texture> ks2, OptionalPtr<const Texture> ks3,
			OptionalPtr<const Texture> m1, OptionalPtr<const Texture> m2, OptionalPtr<const Texture> m3,
			OptionalPtr<const Texture> r1, OptionalPtr<const Texture> r2, OptionalPtr<const Texture> r3, OptionalPtr<const Texture> ka, OptionalPtr<const Texture> d);

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

	OptionalPtr<const Texture> Kd;
	OptionalPtr<const Texture> Ks1;
	OptionalPtr<const Texture> Ks2;
	OptionalPtr<const Texture> Ks3;
	OptionalPtr<const Texture> M1;
	OptionalPtr<const Texture> M2;
	OptionalPtr<const Texture> M3;
	OptionalPtr<const Texture> R1;
	OptionalPtr<const Texture> R2;
	OptionalPtr<const Texture> R3;
	OptionalPtr<const Texture> Ka;
	OptionalPtr<const Texture> depth;
};

}

#endif	/* _SLG_CARPAINTMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

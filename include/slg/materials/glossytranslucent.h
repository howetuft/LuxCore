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

#ifndef _SLG_GLOSSYTRANSLUCENTMAT_H
#define	_SLG_GLOSSYTRANSLUCENTMAT_H

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// Glossy Translucent material
//------------------------------------------------------------------------------

class GlossyTranslucentMaterial : public Material {
public:
	GlossyTranslucentMaterial(TextureConstOPtr frontTransp, TextureConstOPtr backTransp,
			TextureConstOPtr emitted, TextureConstOPtr bump,
			TextureConstOPtr kd, TextureConstOPtr kt, TextureConstOPtr ks, TextureConstOPtr ks2,
			TextureConstOPtr u, TextureConstOPtr u2, TextureConstOPtr v, TextureConstOPtr v2,
			TextureConstOPtr ka, TextureConstOPtr ka2, TextureConstOPtr d, TextureConstOPtr d2,
			TextureConstOPtr i, TextureConstOPtr i2, const bool mbounce, const bool mbounce2);

	virtual MaterialType GetType() const { return GLOSSYTRANSLUCENT; }
	virtual BSDFEvent GetEventTypes() const { return GLOSSY | REFLECT | TRANSMIT; };

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

	TextureConstOPtr GetKd() const { return Kd; }
	TextureConstOPtr GetKt() const { return Kt; }
	TextureConstOPtr GetKs() const { return Ks; }
	TextureConstOPtr GetKs_bf() const { return Ks_bf; }
	TextureConstOPtr GetNu() const { return nu; }
	TextureConstOPtr GetNu_bf() const { return nu_bf; }
	TextureConstOPtr GetNv() const { return nv; }
	TextureConstOPtr GetNv_bf() const { return nv_bf; }
	TextureConstOPtr GetKa() const { return Ka; }
	TextureConstOPtr GetKa_bf() const { return Ka_bf; }
	TextureConstOPtr GetDepth() const { return depth; }
	TextureConstOPtr GetDepth_bf() const { return depth_bf; }
	TextureConstOPtr GetIndex() const { return index; }
	TextureConstOPtr GetIndex_bf() const { return index_bf; }
	const bool IsMultibounce() const { return multibounce; }
	const bool IsMultibounce_bf() const { return multibounce_bf; }

private:
	TextureConstOPtr Kd;
	TextureConstOPtr Kt;
	TextureConstOPtr Ks;
	TextureConstOPtr Ks_bf;
	TextureConstOPtr nu;
	TextureConstOPtr nu_bf;
	TextureConstOPtr nv;
	TextureConstOPtr nv_bf;
	TextureConstOPtr Ka;
	TextureConstOPtr Ka_bf;
	TextureConstOPtr depth;
	TextureConstOPtr depth_bf;
	TextureConstOPtr index;
	TextureConstOPtr index_bf;
	const bool multibounce;
	const bool multibounce_bf;
};

}

#endif	/* _SLG_GLOSSYTRANSLUCENTMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

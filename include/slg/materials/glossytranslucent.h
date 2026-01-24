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
	GlossyTranslucentMaterial(OptionalPtr<const Texture> frontTransp, OptionalPtr<const Texture> backTransp,
			OptionalPtr<const Texture> emitted, OptionalPtr<const Texture> bump,
			OptionalPtr<const Texture> kd, OptionalPtr<const Texture> kt, OptionalPtr<const Texture> ks, OptionalPtr<const Texture> ks2,
			OptionalPtr<const Texture> u, OptionalPtr<const Texture> u2, OptionalPtr<const Texture> v, OptionalPtr<const Texture> v2,
			OptionalPtr<const Texture> ka, OptionalPtr<const Texture> ka2, OptionalPtr<const Texture> d, OptionalPtr<const Texture> d2,
			OptionalPtr<const Texture> i, OptionalPtr<const Texture> i2, const bool mbounce, const bool mbounce2);

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

	OptionalPtr<const Texture> GetKd() const { return Kd; }
	OptionalPtr<const Texture> GetKt() const { return Kt; }
	OptionalPtr<const Texture> GetKs() const { return Ks; }
	OptionalPtr<const Texture> GetKs_bf() const { return Ks_bf; }
	OptionalPtr<const Texture> GetNu() const { return nu; }
	OptionalPtr<const Texture> GetNu_bf() const { return nu_bf; }
	OptionalPtr<const Texture> GetNv() const { return nv; }
	OptionalPtr<const Texture> GetNv_bf() const { return nv_bf; }
	OptionalPtr<const Texture> GetKa() const { return Ka; }
	OptionalPtr<const Texture> GetKa_bf() const { return Ka_bf; }
	OptionalPtr<const Texture> GetDepth() const { return depth; }
	OptionalPtr<const Texture> GetDepth_bf() const { return depth_bf; }
	OptionalPtr<const Texture> GetIndex() const { return index; }
	OptionalPtr<const Texture> GetIndex_bf() const { return index_bf; }
	const bool IsMultibounce() const { return multibounce; }
	const bool IsMultibounce_bf() const { return multibounce_bf; }

private:
	OptionalPtr<const Texture> Kd;
	OptionalPtr<const Texture> Kt;
	OptionalPtr<const Texture> Ks;
	OptionalPtr<const Texture> Ks_bf;
	OptionalPtr<const Texture> nu;
	OptionalPtr<const Texture> nu_bf;
	OptionalPtr<const Texture> nv;
	OptionalPtr<const Texture> nv_bf;
	OptionalPtr<const Texture> Ka;
	OptionalPtr<const Texture> Ka_bf;
	OptionalPtr<const Texture> depth;
	OptionalPtr<const Texture> depth_bf;
	OptionalPtr<const Texture> index;
	OptionalPtr<const Texture> index_bf;
	const bool multibounce;
	const bool multibounce_bf;
};

}

#endif	/* _SLG_GLOSSYTRANSLUCENTMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

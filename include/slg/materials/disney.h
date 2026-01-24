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

#ifndef _SLG_DISNEYMAT_H
#define	_SLG_DISNEYMAT_H

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
//							Disney BRDF
// Based on "Physically Based Shading at Disney" presentet SIGGRAPH 2012 
//------------------------------------------------------------------------------

class DisneyMaterial : public Material {
public:
	DisneyMaterial(
		OptionalPtr<const Texture> frontTransp,
		OptionalPtr<const Texture> backTransp,
		OptionalPtr<const Texture> emitted,
		OptionalPtr<const Texture> bump,
		OptionalPtr<const Texture> baseColor,
		OptionalPtr<const Texture> subsurface,
		OptionalPtr<const Texture> roughness,
		OptionalPtr<const Texture> metallic,
		OptionalPtr<const Texture> specular,
		OptionalPtr<const Texture> specularTint,
		OptionalPtr<const Texture> clearcoat,
		OptionalPtr<const Texture> clearcoatGloss,
		OptionalPtr<const Texture> anisotropic,
		OptionalPtr<const Texture> sheen,
		OptionalPtr<const Texture> sheenTint,
		OptionalPtr<const Texture> filmAmount, 
		OptionalPtr<const Texture> filmThickness, 
		OptionalPtr<const Texture> filmIor
	);

	virtual MaterialType GetType() const { return DISNEY; }
	virtual BSDFEvent GetEventTypes() const { return GLOSSY | REFLECT; };

	virtual luxrays::Spectrum Albedo(
		const HitPoint &hitPoint
	) const;

	virtual luxrays::Spectrum Evaluate(
		const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, 
		const luxrays::Vector &localEyeDir, 
		BSDFEvent *event,
		float *directPdfW = NULL, 
		float *reversePdfW = NULL
	) const;

	virtual luxrays::Spectrum Sample(
		const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, 
		luxrays::Vector *localSampledDir,
		const float u0, 
		const float u1, 
		const float passThroughEvent,
		float *pdfW,
		BSDFEvent *event
	) const;

	virtual void Pdf(
		const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, 
		const luxrays::Vector &localEyeDir,
		float *directPdfW, 
		float *reversePdfW
	) const;

	luxrays::PropertiesUPtr ToProperties(
		const ImageMapCache &imgMapCache, 
		const bool useRealFileName
	) const;

	void UpdateTextureReferences(
		TextureConstRef  oldTex, 
		TextureRef  newTex
	);

	void AddReferencedTextures(std::unordered_set<const Texture *>& referencedTexs) const;

	OptionalPtr<const Texture> GetBaseColor() const { return BaseColor; };
	OptionalPtr<const Texture> GetSubsurface() const { return Subsurface; };
	OptionalPtr<const Texture> GetRoughness() const { return Roughness; };
	OptionalPtr<const Texture> GetMetallic() const { return Metallic; };
	OptionalPtr<const Texture> GetSpecular() const { return Specular; };
	OptionalPtr<const Texture> GetSpecularTint() const { return SpecularTint; };
	OptionalPtr<const Texture> GetClearcoat() const { return Clearcoat; };
	OptionalPtr<const Texture> GetClearcoatGloss() const { return ClearcoatGloss; };
	OptionalPtr<const Texture> GetAnisotropic() const { return Anisotropic; };
	OptionalPtr<const Texture> GetSheen() const { return Sheen; };
	OptionalPtr<const Texture> GetSheenTint() const { return SheenTint; };
	OptionalPtr<const Texture> GetFilmAmount() const { return filmAmount; }
	OptionalPtr<const Texture> GetFilmThickness() const { return filmThickness; }
	OptionalPtr<const Texture> GetFilmIOR() const { return filmIor; }

private:
	OptionalPtr<const Texture> BaseColor;
	OptionalPtr<const Texture> Subsurface;
	OptionalPtr<const Texture> Roughness;
	OptionalPtr<const Texture> Metallic;
	OptionalPtr<const Texture> Specular;
	OptionalPtr<const Texture> SpecularTint;
	OptionalPtr<const Texture> Clearcoat;
	OptionalPtr<const Texture> ClearcoatGloss;
	OptionalPtr<const Texture> Anisotropic;
	OptionalPtr<const Texture> Sheen;
	OptionalPtr<const Texture> SheenTint;
	OptionalPtr<const Texture> filmAmount;
	OptionalPtr<const Texture> filmThickness;
	OptionalPtr<const Texture> filmIor;

	void UpdateGlossiness();

	luxrays::Spectrum CalculateTint(const luxrays::Spectrum &color) const;

	float GTR1(const float NdotH, const float a) const;
	float GTR2_Aniso(const float NdotH, const float HdotX, const float HdotY,
			const float ax, const float ay) const;
	float SmithG_GGX_Aniso(const float NdotV, const float VdotX, const float VdotY,
			const float ax, const float ay) const;
	float SmithG_GGX(const float NdotV, const float alphaG) const;
	float Schlick_Weight(const float cosi) const;
	void Anisotropic_Params(const float anisotropic, const float roughness, float &ax, float &ay) const;
	void ComputeRatio(const float metallic, const float clearcoat,
			float &RatioGlossy, float &diffuseWeight, float &RatioClearcoat) const;

	luxrays::Spectrum DisneyDiffuse(const luxrays::Spectrum &color, const float roughness,
			const float NdotL, const float NdotV, const float LdotH) const;
	luxrays::Spectrum DisneySubsurface(const luxrays::Spectrum &color, const float roughness,
			const float NdotL, const float NdotV, const float LdotH) const;
	luxrays::Spectrum DisneyMetallic(const luxrays::Spectrum &color, const float specular,
			const float specularTint, const float metallic,
			const float anisotropic, const float roughness,
			const float NdotL, const float NdotV, const float NdotH,
			const float LdotH, const float VdotH,
			const luxrays::Vector &wi, const luxrays::Vector &wo, const luxrays::Vector &H) const;
	float DisneyClearCoat(const float clearcoat, const float clearcoatGloss,
			const float NdotL, const float NdotV, const float NdotH, const float LdotH) const;
	luxrays::Spectrum DisneySheen(const luxrays::Spectrum &color, const float sheen,
			const float sheenTint, const float LdotH) const;
	
	luxrays::Spectrum DisneyEvaluate(const bool fromLight, 
		const luxrays::Spectrum &color,
		const float subsurface, const float roughness,
		const float metallic, const float specular, const float specularTint,
		const float clearcoat, const float clearcoatGloss, const float anisotropicGloss,
		const float sheen, const float sheenTint, const float localFilmAmount, const float localFilmThickness,
		const float localFilmIor, const Vector &localLightDir, const Vector &localEyeDir, 
		BSDFEvent *event, float *directPdfW, float *reversePdfW) const;

	luxrays::Vector DisneyDiffuseSample(const luxrays::Vector &wo, float u0, float u1) const;
	luxrays::Vector DisneyMetallicSample(const float anisotropic, const float roughness,
			const luxrays::Vector &wo, float u0, float u1) const;
	luxrays::Vector DisneyClearcoatSample(const float clearcoatGloss,
			const luxrays::Vector &wo, float u0, float u1) const;

	void DisneyPdf(const bool fromLight, const float roughness, const float metallic,
			const float clearcoat, const float clearcoatGloss, const float anisotropic,
			const Vector &localLightDir, const Vector &localEyeDir,
			float *directPdfW, float *reversePdfW) const;
	void DiffusePdf(const bool fromLight,
			const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
			float *directPdfW, float *reversePdfW) const;
	void MetallicPdf(const bool fromLight, const float anisotropic, const float roughness,
			const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
			float *directPdfW, float *reversePdfW) const;
	void ClearcoatPdf(const bool fromLight, const float clearcoatGloss,
			const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
			float *directPdfW, float *reversePdfW) const;
};

}

#endif	/* _SLG_DISNEYMAT_H */// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

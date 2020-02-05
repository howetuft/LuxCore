#line 2 "texture_eval_funcs.cl"

/***************************************************************************
 * Copyright 1998-2018 by authors (see AUTHORS.txt)                        *
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

//#define DEBUG_PRINTF_KERNEL_NAME 1
//#define DEBUG_PRINTF_TEXTURE_EVAL 1

//------------------------------------------------------------------------------
// Texture evaluation functions
//------------------------------------------------------------------------------

#define EvalStack_PushUInt(a) { ((__global uint *)evalStack)[*evalStackOffset] = a; *evalStackOffset = *evalStackOffset + 1; }
#define EvalStack_PopUInt(a) { *evalStackOffset = *evalStackOffset - 1; a = ((__global uint *)evalStack)[*evalStackOffset]; }

#define EvalStack_PushFloat(a) { evalStack[*evalStackOffset] = a; *evalStackOffset = *evalStackOffset + 1; }
#define EvalStack_PushFloat2(a) { EvalStack_PushFloat(a.s0); EvalStack_PushFloat(a.s1); }
#define EvalStack_PushFloat3(a) { EvalStack_PushFloat(a.s0); EvalStack_PushFloat(a.s1); EvalStack_PushFloat(a.s2); }
#define EvalStack_PopFloat(a) { *evalStackOffset = *evalStackOffset - 1; a = evalStack[*evalStackOffset]; }
#define EvalStack_PopFloat2(a) { EvalStack_PopFloat(a.s1); EvalStack_PopFloat(a.s0); }
#define EvalStack_PopFloat3(a) { EvalStack_PopFloat(a.s2); EvalStack_PopFloat(a.s1); EvalStack_PopFloat(a.s0); }
#define EvalStack_Read(x) (evalStack[(*evalStackOffset) + x])
#define EvalStack_Read2(x) ((float2)(evalStack[(*evalStackOffset) + x], evalStack[(*evalStackOffset) + x + 1]))
#define EvalStack_Read3(x) ((float3)(evalStack[(*evalStackOffset) + x], evalStack[(*evalStackOffset) + x + 1], evalStack[(*evalStackOffset) + x + 2]))

OPENCL_FORCE_NOT_INLINE void Texture_EvalOpGenericBumpOffsetU(
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint,
		const float sampleDistance) {
	const float3 origP = VLOAD3F(&hitPoint->p.x);
	const float3 origShadeN = VLOAD3F(&hitPoint->shadeN.x);
	const float2 origUV = VLOAD2F(&hitPoint->uv[0].u);

	// Save original P
	EvalStack_PushFloat3(origP);
	// Save original shadeN
	EvalStack_PushFloat3(origShadeN);
	// Save original UV
	EvalStack_PushFloat2(origUV);

	// Update HitPoint
	__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
	const float3 dpdu = VLOAD3F(&hitPointTmp->dpdu.x);
	const float3 dndu = VLOAD3F(&hitPoint->dndu.x);
	// Shift hitPointTmp.du in the u direction and calculate value
	const float uu = sampleDistance / length(dpdu);
	VSTORE3F(origP + uu * dpdu, &hitPointTmp->p.x);
	hitPointTmp->uv[0].u = origUV.s0 + uu;
	hitPointTmp->uv[0].v = origUV.s1;
	VSTORE3F(normalize(origShadeN + uu * dndu), &hitPointTmp->shadeN.x);
}

OPENCL_FORCE_NOT_INLINE void Texture_EvalOpGenericBumpOffsetV(
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint,
		const float sampleDistance) {
	// -1 is for result of EVAL_BUMP_GENERIC_OFFSET_U
	const float3 origP = EvalStack_Read3(-8 - 1);
	const float3 origShadeN = EvalStack_Read3(-5 - 1);
	const float2 origUV = EvalStack_Read2(-2 - 1);

	// Update HitPoint
	__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
	const float3 dpdv = VLOAD3F(&hitPointTmp->dpdv.x);
	const float3 dndv = VLOAD3F(&hitPoint->dndv.x);
	// Shift hitPointTmp.dv in the v direction and calculate value
	const float vv = sampleDistance / length(dpdv);
	VSTORE3F(origP + vv * dpdv, &hitPointTmp->p.x);
	hitPointTmp->uv[0].u = origUV.s0;
	hitPointTmp->uv[0].v = origUV.s1 + vv;
	VSTORE3F(normalize(origShadeN + vv * dndv), &hitPointTmp->shadeN.x);
}

OPENCL_FORCE_NOT_INLINE void Texture_EvalOpGenericBump(
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint,
		const float sampleDistance) {
	float evalFloatTexBase, evalFloatTexOffsetU, evalFloatTexOffsetV;
	EvalStack_PopFloat(evalFloatTexOffsetV);
	EvalStack_PopFloat(evalFloatTexOffsetU);

	float3 origP, origShadeN;
	float2 origUV;
	EvalStack_PopFloat2(origUV);
	EvalStack_PopFloat3(origShadeN);
	EvalStack_PopFloat3(origP);

	// evalFloatTexBase is the very first evaluation done so
	// it is the last for a stack pop
	EvalStack_PopFloat(evalFloatTexBase);

	// Restore original P, shadeN and UV
	__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
	VSTORE3F(origP, &hitPointTmp->p.x);
	VSTORE3F(origShadeN, &hitPointTmp->shadeN.x);
	hitPointTmp->uv[0].u = origUV.s0;
	hitPointTmp->uv[0].v = origUV.s1;

	const float3 shadeN = GenericTexture_Bump(hitPoint, sampleDistance,
			evalFloatTexBase, evalFloatTexOffsetU, evalFloatTexOffsetV);
	EvalStack_PushFloat3(shadeN);
}

OPENCL_FORCE_NOT_INLINE void Texture_EvalOp(
		__global const TextureEvalOp* restrict evalOp,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint,
		const float sampleDistance
		TEXTURES_PARAM_DECL) {
	__global const Texture* restrict tex = &texs[evalOp->texIndex];

#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("EvalOp tex index=%d type=%d evalType=%d *evalStackOffset=%d\n", evalOp->texIndex, tex->type, evalOp->evalType, *evalStackOffset);
#endif

	const TextureEvalOpType evalType = evalOp->evalType;
	switch (tex->type) {
		//----------------------------------------------------------------------
		// CONST_FLOAT
		//----------------------------------------------------------------------
		case CONST_FLOAT: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = ConstFloatTexture_ConstEvaluateFloat(tex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = ConstFloatTexture_ConstEvaluateSpectrum(tex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					const float3 shadeN = ConstTexture_Bump(hitPoint);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// CONST_FLOAT3
		//----------------------------------------------------------------------
		case CONST_FLOAT3: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = ConstFloat3Texture_ConstEvaluateFloat(tex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = ConstFloat3Texture_ConstEvaluateSpectrum(tex);
					EvalStack_PushFloat3(eval);
					break;
				}	
				case EVAL_BUMP: {
					const float3 shadeN = ConstTexture_Bump(hitPoint);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// IMAGEMAP
		//----------------------------------------------------------------------
		case IMAGEMAP: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = ImageMapTexture_ConstEvaluateFloat(tex, hitPoint IMAGEMAPS_PARAM);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = ImageMapTexture_ConstEvaluateSpectrum(tex, hitPoint IMAGEMAPS_PARAM);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					const float3 shadeN = ImageMapTexture_Bump(tex, hitPoint IMAGEMAPS_PARAM);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// SCALE_TEX
		//----------------------------------------------------------------------
		case SCALE_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = ScaleTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = ScaleTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					float evalFloatTex1, evalFloatTex2;

					EvalStack_PopFloat(evalFloatTex2);
					EvalStack_PopFloat(evalFloatTex1);

					float3 bumbNTex1, bumbNTex2;

					EvalStack_PopFloat3(bumbNTex2);
					EvalStack_PopFloat3(bumbNTex1);

					const float3 shadeN = ScaleTexture_Bump(hitPoint, bumbNTex1, bumbNTex2, evalFloatTex1, evalFloatTex2);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// FRESNEL_APPROX_N
		//----------------------------------------------------------------------
		case FRESNEL_APPROX_N: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1;
					EvalStack_PopFloat(tex1);

					const float eval = FresnelApproxNTexture_ConstEvaluateFloat(tex1);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					EvalStack_PopFloat3(tex1);

					const float3 eval = FresnelApproxNTexture_ConstEvaluateSpectrum(tex1);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// FRESNEL_APPROX_K
		//----------------------------------------------------------------------
		case FRESNEL_APPROX_K: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1;
					EvalStack_PopFloat(tex1);

					const float eval = FresnelApproxKTexture_ConstEvaluateFloat(tex1);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					EvalStack_PopFloat3(tex1);

					const float3 eval = FresnelApproxKTexture_ConstEvaluateSpectrum(tex1);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// MIX_TEX
		//----------------------------------------------------------------------
		case MIX_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float amt, tex1, tex2;
					EvalStack_PopFloat(amt);
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = MixTexture_ConstEvaluateFloat(tex1, tex2, amt);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float amt;
					float3 tex1, tex2;
					EvalStack_PopFloat(amt);
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = MixTexture_ConstEvaluateSpectrum(tex1, tex2, amt);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					float evalFloatTex1, evalFloatTex2, evalFloatAmount;

					EvalStack_PopFloat(evalFloatAmount);
					EvalStack_PopFloat(evalFloatTex2);
					EvalStack_PopFloat(evalFloatTex1);

					float3 bumbNTex1, bumbNTex2, bumbNAmount;

					EvalStack_PopFloat3(bumbNAmount);
					EvalStack_PopFloat3(bumbNTex2);
					EvalStack_PopFloat3(bumbNTex1);

					const float3 shadeN = MixTexture_Bump(hitPoint,
							bumbNTex1, bumbNTex2, bumbNAmount,
							evalFloatTex1, evalFloatTex2, evalFloatAmount);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// ADD_TEX
		//----------------------------------------------------------------------
		case ADD_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = AddTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = AddTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					float3 bumbNTex1, bumbNTex2;

					EvalStack_PopFloat3(bumbNTex2);
					EvalStack_PopFloat3(bumbNTex1);

					const float3 shadeN = AddTexture_Bump(hitPoint, bumbNTex1, bumbNTex2);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// SUBTRACT_TEX
		//----------------------------------------------------------------------
		case SUBTRACT_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = SubtractTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = SubtractTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					float3 bumbNTex1, bumbNTex2;

					EvalStack_PopFloat3(bumbNTex2);
					EvalStack_PopFloat3(bumbNTex1);

					const float3 shadeN = SubtractTexture_Bump(hitPoint, bumbNTex1, bumbNTex2);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// HITPOINTCOLOR
		//----------------------------------------------------------------------
		case HITPOINTCOLOR: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = HitPointColorTexture_ConstEvaluateFloat(hitPoint,
							tex->hitPointColor.dataIndex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = HitPointColorTexture_ConstEvaluateSpectrum(hitPoint,
							tex->hitPointColor.dataIndex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// HITPOINTALPHA
		//----------------------------------------------------------------------
		case HITPOINTALPHA: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = HitPointAlphaTexture_ConstEvaluateFloat(hitPoint,
							tex->hitPointAlpha.dataIndex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = HitPointAlphaTexture_ConstEvaluateSpectrum(hitPoint,
							tex->hitPointAlpha.dataIndex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// HITPOINTGREY
		//----------------------------------------------------------------------
		case HITPOINTGREY: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = HitPointGreyTexture_ConstEvaluateFloat(hitPoint,
							tex->hitPointGrey.dataIndex, tex->hitPointGrey.channelIndex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = HitPointGreyTexture_ConstEvaluateSpectrum(hitPoint,
							tex->hitPointGrey.dataIndex, tex->hitPointGrey.channelIndex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// NORMALMAP_TEX
		//----------------------------------------------------------------------
		case NORMALMAP_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = NormalMapTexture_ConstEvaluateFloat();
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = NormalMapTexture_ConstEvaluateSpectrum();
					EvalStack_PushFloat3(eval);
					break;
				}	
				case EVAL_BUMP: {
					float3 evalSpectrumTex;
					EvalStack_PopFloat3(evalSpectrumTex);

					const float3 shadeN = NormalMapTexture_Bump(tex, hitPoint, evalSpectrumTex);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLACKBODY_TEX
		//----------------------------------------------------------------------
		case BLACKBODY_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlackBodyTexture_ConstEvaluateFloat(VLOAD3F(tex->blackBody.rgb.c));
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlackBodyTexture_ConstEvaluateSpectrum(VLOAD3F(tex->blackBody.rgb.c));
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// IRREGULARDATA_TEX
		//----------------------------------------------------------------------
		case IRREGULARDATA_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = IrregularDataTexture_ConstEvaluateFloat(VLOAD3F(tex->irregularData.rgb.c));
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = IrregularDataTexture_ConstEvaluateSpectrum(VLOAD3F(tex->irregularData.rgb.c));
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// DENSITYGRID_TEX
		//----------------------------------------------------------------------
		case DENSITYGRID_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = DensityGridTexture_ConstEvaluateFloat(hitPoint,
							tex->densityGrid.nx, tex->densityGrid.ny, tex->densityGrid.nz,
							tex->densityGrid.imageMapIndex, &tex->densityGrid.mapping
							IMAGEMAPS_PARAM);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = DensityGridTexture_ConstEvaluateSpectrum(hitPoint,
							tex->densityGrid.nx, tex->densityGrid.ny, tex->densityGrid.nz,
							tex->densityGrid.imageMapIndex, &tex->densityGrid.mapping
							IMAGEMAPS_PARAM);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// ABS_TEX
		//----------------------------------------------------------------------
		case ABS_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1;
					EvalStack_PopFloat(tex1);

					const float eval = AbsTexture_ConstEvaluateFloat(tex1);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					EvalStack_PopFloat3(tex1);

					const float3 eval = AbsTexture_ConstEvaluateSpectrum(tex1);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// CLAMP_TEX
		//----------------------------------------------------------------------
		case CLAMP_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1;
					EvalStack_PopFloat(tex1);

					const float eval = ClampTexture_ConstEvaluateFloat(tex1,
							tex->clampTex.minVal, tex->clampTex.maxVal);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					EvalStack_PopFloat3(tex1);

					const float3 eval = ClampTexture_ConstEvaluateSpectrum(tex1,
							tex->clampTex.minVal, tex->clampTex.maxVal);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BILERP_TEX
		//----------------------------------------------------------------------
		case BILERP_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex00, tex01, tex10, tex11;
					EvalStack_PopFloat(tex11);
					EvalStack_PopFloat(tex10);
					EvalStack_PopFloat(tex01);
					EvalStack_PopFloat(tex00);
					

					const float eval = BilerpTexture_ConstEvaluateFloat(hitPoint,
							tex00, tex01, tex10, tex11);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex00, tex01, tex10, tex11;
					EvalStack_PopFloat3(tex11);
					EvalStack_PopFloat3(tex10);
					EvalStack_PopFloat3(tex01);
					EvalStack_PopFloat3(tex00);

					const float3 eval = BilerpTexture_ConstEvaluateSpectrum(hitPoint,
							tex00, tex01, tex10, tex11);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// COLORDEPTH_TEX
		//----------------------------------------------------------------------
		case COLORDEPTH_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1;
					EvalStack_PopFloat(tex1);

					const float eval = ColorDepthTexture_ConstEvaluateFloat(tex->colorDepthTex.dVal, tex1);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					EvalStack_PopFloat3(tex1);

					const float3 eval = ColorDepthTexture_ConstEvaluateSpectrum(tex->colorDepthTex.dVal, tex1);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// HSV_TEX
		//----------------------------------------------------------------------
		case HSV_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, hueTex, satTex, valTex;
					EvalStack_PopFloat(valTex);
					EvalStack_PopFloat(satTex);
					EvalStack_PopFloat(hueTex);
					EvalStack_PopFloat(tex1);
					

					const float eval = HsvTexture_ConstEvaluateFloat(tex1, hueTex, satTex, valTex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					float hueTex, satTex, valTex;
					EvalStack_PopFloat(valTex);
					EvalStack_PopFloat(satTex);
					EvalStack_PopFloat(hueTex);
					EvalStack_PopFloat3(tex1);

					const float3 eval = HsvTexture_ConstEvaluateSpectrum(tex1, hueTex, satTex, valTex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// DIVIDE_TEX
		//----------------------------------------------------------------------
		case DIVIDE_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = DivideTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = DivideTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// REMAP_TEX
		//----------------------------------------------------------------------
		case REMAP_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float valueTex, sourceMinTex, sourceMaxTex, targetMinTex, targetMaxTex;
					EvalStack_PopFloat(targetMaxTex);
					EvalStack_PopFloat(targetMinTex);
					EvalStack_PopFloat(sourceMaxTex);
					EvalStack_PopFloat(sourceMinTex);
					EvalStack_PopFloat(valueTex);

					const float eval = RemapTexture_ConstEvaluateFloat(valueTex,
							sourceMinTex, sourceMaxTex, targetMinTex, targetMaxTex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 valueTex;
					float sourceMinTex, sourceMaxTex, targetMinTex, targetMaxTex;
					EvalStack_PopFloat(targetMaxTex);
					EvalStack_PopFloat(targetMinTex);
					EvalStack_PopFloat(sourceMaxTex);
					EvalStack_PopFloat(sourceMinTex);
					EvalStack_PopFloat3(valueTex);

					const float3 eval = RemapTexture_ConstEvaluateSpectrum(valueTex,
							sourceMinTex, sourceMaxTex, targetMinTex, targetMaxTex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// OBJECTID_TEX
		//----------------------------------------------------------------------
		case OBJECTID_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = ObjectIDTexture_ConstEvaluateFloat(hitPoint);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = ObjectIDTexture_ConstEvaluateSpectrum(hitPoint);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					const float3 shadeN = ConstTexture_Bump(hitPoint);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// OBJECTID_COLOR_TEX
		//----------------------------------------------------------------------
		case OBJECTID_COLOR_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = ObjectIDColorTexture_ConstEvaluateFloat(hitPoint);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = ObjectIDColorTexture_ConstEvaluateSpectrum(hitPoint);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					const float3 shadeN = ConstTexture_Bump(hitPoint);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// OBJECTID_NORMALIZED_TEX
		//----------------------------------------------------------------------
		case OBJECTID_NORMALIZED_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = ObjectIDNormalizedTexture_ConstEvaluateFloat(hitPoint);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = ObjectIDNormalizedTexture_ConstEvaluateSpectrum(hitPoint);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					const float3 shadeN = ConstTexture_Bump(hitPoint);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// DOT_PRODUCT_TEX
		//----------------------------------------------------------------------
		case DOT_PRODUCT_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = DotProductTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = DotProductTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// POWER_TEX
		//----------------------------------------------------------------------
		case POWER_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = PowerTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float3 eval = PowerTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// LESS_THAN_TEX
		//----------------------------------------------------------------------
		case LESS_THAN_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = LessThanTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float3 eval = LessThanTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// GREATER_THAN_TEX
		//----------------------------------------------------------------------
		case GREATER_THAN_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = GreaterThanTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float3 eval = GreaterThanTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// ROUNDING_TEX
		//----------------------------------------------------------------------
		case ROUNDING_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = RoundingTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float3 eval = RoundingTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// MODULO_TEX
		//----------------------------------------------------------------------
		case MODULO_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = ModuloTexture_ConstEvaluateFloat(tex1, tex2);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float3 eval = ModuloTexture_ConstEvaluateSpectrum(tex1, tex2);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// SHADING_NORMAL_TEX
		//----------------------------------------------------------------------
		case SHADING_NORMAL_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = ShadingNormalTexture_ConstEvaluateFloat(hitPoint);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = ShadingNormalTexture_ConstEvaluateSpectrum(hitPoint);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// POSITION_TEX
		//----------------------------------------------------------------------
		case POSITION_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = PositionTexture_ConstEvaluateFloat(hitPoint);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = PositionTexture_ConstEvaluateSpectrum(hitPoint);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// SPLIT_FLOAT3
		//----------------------------------------------------------------------
		case SPLIT_FLOAT3: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float3 tex1;
					EvalStack_PopFloat3(tex1);

					const float eval = SplitFloat3Texture_ConstEvaluateFloat(tex1,
							tex->splitFloat3Tex.channelIndex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					EvalStack_PopFloat(tex1);

					const float3 eval = SplitFloat3Texture_ConstEvaluateSpectrum(tex1,
							tex->splitFloat3Tex.channelIndex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// MAKE_FLOAT3
		//----------------------------------------------------------------------
		case MAKE_FLOAT3: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2, tex3;
					EvalStack_PopFloat(tex3);
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = MakeFloat3Texture_ConstEvaluateFloat(tex1,
							tex2, tex3);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float tex1, tex2, tex3;
					EvalStack_PopFloat(tex3);
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float3 eval = MakeFloat3Texture_ConstEvaluateSpectrum(tex1,
							tex2, tex3);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BRIGHT_CONTRAST_TEX
		//----------------------------------------------------------------------
		case BRIGHT_CONTRAST_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, brightnessTex, contrastTex;
					EvalStack_PopFloat(contrastTex);
					EvalStack_PopFloat(brightnessTex);
					EvalStack_PopFloat(tex1);

					const float eval = BrightContrastTexture_ConstEvaluateFloat(tex1,
							brightnessTex, contrastTex);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1;
					float brightnessTex, contrastTex;
					EvalStack_PopFloat(contrastTex);
					EvalStack_PopFloat(brightnessTex);
					EvalStack_PopFloat3(tex1);

					const float3 eval = BrightContrastTexture_ConstEvaluateSpectrum(tex1,
							brightnessTex, contrastTex);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_BLEND
		//----------------------------------------------------------------------
		case BLENDER_BLEND: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderBlendTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderBlend.type, tex->blenderBlend.direction,
							tex->blenderBlend.contrast, tex->blenderBlend.bright,
							&tex->blenderBlend.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderBlendTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderBlend.type, tex->blenderBlend.direction,
							tex->blenderBlend.contrast, tex->blenderBlend.bright,
							&tex->blenderBlend.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_CLOUDS
		//----------------------------------------------------------------------
		case BLENDER_CLOUDS: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderCloudsTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderClouds.noisebasis, tex->blenderClouds.noisesize,
							tex->blenderClouds.noisedepth, tex->blenderClouds.contrast,
							tex->blenderClouds.bright, tex->blenderClouds.hard,
							&tex->blenderClouds.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderCloudsTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderClouds.noisebasis, tex->blenderClouds.noisesize,
							tex->blenderClouds.noisedepth, tex->blenderClouds.contrast,
							tex->blenderClouds.bright, tex->blenderClouds.hard,
							&tex->blenderClouds.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_DISTORTED_NOISE
		//----------------------------------------------------------------------
		case BLENDER_DISTORTED_NOISE: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderDistortedNoiseTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderDistortedNoise.noisedistortion, tex->blenderDistortedNoise.noisebasis,
							tex->blenderDistortedNoise.distortion, tex->blenderDistortedNoise.noisesize,
							tex->blenderDistortedNoise.contrast, tex->blenderDistortedNoise.bright,
							&tex->blenderDistortedNoise.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderDistortedNoiseTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderDistortedNoise.noisedistortion, tex->blenderDistortedNoise.noisebasis,
							tex->blenderDistortedNoise.distortion, tex->blenderDistortedNoise.noisesize,
							tex->blenderDistortedNoise.contrast, tex->blenderDistortedNoise.bright,
							&tex->blenderDistortedNoise.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_MAGIC
		//----------------------------------------------------------------------
		case BLENDER_MAGIC: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderMagicTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderMagic.noisedepth, tex->blenderMagic.turbulence,
							tex->blenderMagic.contrast, tex->blenderMagic.bright,
							&tex->blenderMagic.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderMagicTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderMagic.noisedepth, tex->blenderMagic.turbulence,
							tex->blenderMagic.contrast, tex->blenderMagic.bright,
							&tex->blenderMagic.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_MARBLE
		//----------------------------------------------------------------------
		case BLENDER_MARBLE: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderMarbleTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderMarble.type, tex->blenderMarble.noisebasis,
							tex->blenderMarble.noisebasis2, tex->blenderMarble.noisesize,
							tex->blenderMarble.turbulence, tex->blenderMarble.noisedepth,
							tex->blenderMarble.contrast, tex->blenderMarble.bright,
							tex->blenderMarble.hard,
							&tex->blenderMagic.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderMarbleTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderMarble.type, tex->blenderMarble.noisebasis,
							tex->blenderMarble.noisebasis2, tex->blenderMarble.noisesize,
							tex->blenderMarble.turbulence, tex->blenderMarble.noisedepth,
							tex->blenderMarble.contrast, tex->blenderMarble.bright,
							tex->blenderMarble.hard,
							&tex->blenderMagic.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_MUSGRAVE
		//----------------------------------------------------------------------
		case BLENDER_MUSGRAVE: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderMusgraveTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderMusgrave.type, tex->blenderMusgrave.noisebasis,
							tex->blenderMusgrave.dimension, tex->blenderMusgrave.intensity,
							tex->blenderMusgrave.lacunarity, tex->blenderMusgrave.offset,
							tex->blenderMusgrave.gain, tex->blenderMusgrave.octaves,
							tex->blenderMusgrave.noisesize, tex->blenderMusgrave.contrast,
							tex->blenderMusgrave.bright,
							&tex->blenderMusgrave.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderMusgraveTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderMusgrave.type, tex->blenderMusgrave.noisebasis,
							tex->blenderMusgrave.dimension, tex->blenderMusgrave.intensity,
							tex->blenderMusgrave.lacunarity, tex->blenderMusgrave.offset,
							tex->blenderMusgrave.gain, tex->blenderMusgrave.octaves,
							tex->blenderMusgrave.noisesize, tex->blenderMusgrave.contrast,
							tex->blenderMusgrave.bright,
							&tex->blenderMusgrave.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_NOISE
		//----------------------------------------------------------------------
		case BLENDER_NOISE: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderNoiseTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderNoise.noisedepth, tex->blenderNoise.bright,
							tex->blenderNoise.contrast);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderNoiseTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderNoise.noisedepth, tex->blenderNoise.bright,
							tex->blenderNoise.contrast);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_STUCCI
		//----------------------------------------------------------------------
		case BLENDER_STUCCI: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderStucciTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderStucci.type, tex->blenderStucci.noisebasis,
							tex->blenderStucci.noisesize, tex->blenderStucci.turbulence,
							tex->blenderStucci.contrast, tex->blenderStucci.bright,
							tex->blenderStucci.hard,
							&tex->blenderStucci.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderStucciTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderStucci.type, tex->blenderStucci.noisebasis,
							tex->blenderStucci.noisesize, tex->blenderStucci.turbulence,
							tex->blenderStucci.contrast, tex->blenderStucci.bright,
							tex->blenderStucci.hard,
							&tex->blenderStucci.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_WOOD
		//----------------------------------------------------------------------
		case BLENDER_WOOD: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderWoodTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderWood.type, tex->blenderWood.noisebasis2,
							tex->blenderWood.noisebasis, tex->blenderWood.noisesize,
							tex->blenderWood.turbulence, tex->blenderWood.contrast,
							tex->blenderWood.bright, tex->blenderWood.hard,
							&tex->blenderWood.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderWoodTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderWood.type, tex->blenderWood.noisebasis2,
							tex->blenderWood.noisebasis, tex->blenderWood.noisesize,
							tex->blenderWood.turbulence, tex->blenderWood.contrast,
							tex->blenderWood.bright, tex->blenderWood.hard,
							&tex->blenderWood.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BLENDER_VORONOI
		//----------------------------------------------------------------------
		case BLENDER_VORONOI: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = BlenderVoronoiTexture_ConstEvaluateFloat(hitPoint,
							tex->blenderVoronoi.distancemetric, tex->blenderVoronoi.feature_weight1,
							tex->blenderVoronoi.feature_weight2, tex->blenderVoronoi.feature_weight3,
							tex->blenderVoronoi.feature_weight4, tex->blenderVoronoi.noisesize,
							tex->blenderVoronoi.intensity, tex->blenderVoronoi.exponent,
							tex->blenderVoronoi.contrast, tex->blenderVoronoi.bright,
							&tex->blenderWood.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = BlenderVoronoiTexture_ConstEvaluateSpectrum(hitPoint,
							tex->blenderVoronoi.distancemetric, tex->blenderVoronoi.feature_weight1,
							tex->blenderVoronoi.feature_weight2, tex->blenderVoronoi.feature_weight3,
							tex->blenderVoronoi.feature_weight4, tex->blenderVoronoi.noisesize,
							tex->blenderVoronoi.intensity, tex->blenderVoronoi.exponent,
							tex->blenderVoronoi.contrast, tex->blenderVoronoi.bright,
							&tex->blenderWood.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// CHECKERBOARD2D
		//----------------------------------------------------------------------
		case CHECKERBOARD2D: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = CheckerBoard2DTexture_ConstEvaluateFloat(hitPoint,
							tex1, tex2, &tex->checkerBoard2D.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = CheckerBoard2DTexture_ConstEvaluateSpectrum(hitPoint,
							tex1, tex2, &tex->checkerBoard2D.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// CHECKERBOARD3D
		//----------------------------------------------------------------------
		case CHECKERBOARD3D: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = CheckerBoard3DTexture_ConstEvaluateFloat(hitPoint,
							tex1, tex2, &tex->checkerBoard3D.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = CheckerBoard3DTexture_ConstEvaluateSpectrum(hitPoint,
							tex1, tex2, &tex->checkerBoard3D.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// CLOUD_TEX
		//----------------------------------------------------------------------
		case CLOUD_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = CloudTexture_ConstEvaluateFloat(hitPoint,
							tex->cloud.radius, tex->cloud.numspheres,
							tex->cloud.spheresize, tex->cloud.sharpness,
							tex->cloud.basefadedistance, tex->cloud.baseflatness,
							tex->cloud.variability, tex->cloud.omega,
							tex->cloud.noisescale, tex->cloud.noiseoffset,
							tex->cloud.turbulence, tex->cloud.octaves,
							&tex->cloud.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = CloudTexture_ConstEvaluateSpectrum(hitPoint,
							tex->cloud.radius, tex->cloud.numspheres,
							tex->cloud.spheresize, tex->cloud.sharpness,
							tex->cloud.basefadedistance, tex->cloud.baseflatness,
							tex->cloud.variability, tex->cloud.omega,
							tex->cloud.noisescale, tex->cloud.noiseoffset,
							tex->cloud.turbulence, tex->cloud.octaves,
							&tex->cloud.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// FBM_TEX
		//----------------------------------------------------------------------
		case FBM_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = FBMTexture_ConstEvaluateFloat(hitPoint,
							tex->fbm.omega, tex->fbm.octaves,
							&tex->fbm.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = FBMTexture_ConstEvaluateSpectrum(hitPoint,
							tex->fbm.omega, tex->fbm.octaves,
							&tex->fbm.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// MARBLE
		//----------------------------------------------------------------------
		case MARBLE: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = MarbleTexture_ConstEvaluateFloat(hitPoint,
							tex->marble.scale, tex->marble.omega,
							tex->marble.octaves, tex->marble.variation,
							&tex->marble.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = MarbleTexture_ConstEvaluateSpectrum(hitPoint,
							tex->marble.scale, tex->marble.omega,
							tex->marble.octaves, tex->marble.variation,
							&tex->marble.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// DOTS
		//----------------------------------------------------------------------
		case DOTS: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2;
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = DotsTexture_ConstEvaluateFloat(hitPoint,
							tex1, tex2, &tex->checkerBoard2D.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2;
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = DotsTexture_ConstEvaluateSpectrum(hitPoint,
							tex1, tex2, &tex->checkerBoard2D.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BRICK
		//----------------------------------------------------------------------
		case BRICK: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1, tex2, tex3;
					EvalStack_PopFloat(tex3);
					EvalStack_PopFloat(tex2);
					EvalStack_PopFloat(tex1);

					const float eval = BrickTexture_ConstEvaluateFloat(hitPoint,
							tex1, tex2, tex3,
							tex->brick.bond,
							tex->brick.brickwidth, tex->brick.brickheight,
							tex->brick.brickdepth, tex->brick.mortarsize,
							(float3)(tex->brick.offsetx, tex->brick.offsety, tex->brick.offsetz),
							tex->brick.run , tex->brick.mortarwidth,
							tex->brick.mortarheight, tex->brick.mortardepth,
							tex->brick.proportion, tex->brick.invproportion,
							&tex->brick.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float3 tex1, tex2, tex3;
					EvalStack_PopFloat3(tex3);
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);

					const float3 eval = BrickTexture_ConstEvaluateSpectrum(hitPoint,
							tex1, tex2, tex3,
							tex->brick.bond,
							tex->brick.brickwidth, tex->brick.brickheight,
							tex->brick.brickdepth, tex->brick.mortarsize,
							(float3)(tex->brick.offsetx, tex->brick.offsety, tex->brick.offsetz),
							tex->brick.run , tex->brick.mortarwidth,
							tex->brick.mortarheight, tex->brick.mortardepth,
							tex->brick.proportion, tex->brick.invproportion,
							&tex->brick.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// WINDY
		//----------------------------------------------------------------------
		case WINDY: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = WindyTexture_ConstEvaluateFloat(hitPoint,
							&tex->windy.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = WindyTexture_ConstEvaluateSpectrum(hitPoint,
							&tex->windy.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// WRINKLED
		//----------------------------------------------------------------------
		case WRINKLED: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = WrinkledTexture_ConstEvaluateFloat(hitPoint,
							tex->wrinkled.omega, tex->wrinkled.octaves,
							&tex->wrinkled.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = WrinkledTexture_ConstEvaluateSpectrum(hitPoint,
							tex->wrinkled.omega, tex->wrinkled.octaves,
							&tex->wrinkled.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// UV_TEX
		//----------------------------------------------------------------------
		case UV_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = UVTexture_ConstEvaluateFloat(hitPoint,
							&tex->uvTex.mapping);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = UVTexture_ConstEvaluateSpectrum(hitPoint,
							&tex->uvTex.mapping);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// BAND_TEX
		//----------------------------------------------------------------------
		case BAND_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					float tex1;
					EvalStack_PopFloat(tex1);

					const float eval = BandTexture_ConstEvaluateFloat(hitPoint,
							tex->band.interpType, tex->band.size,
							tex->band.offsets, tex->band.values,
							tex1);
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					float tex1;
					EvalStack_PopFloat(tex1);

					const float3 eval = BandTexture_ConstEvaluateSpectrum(hitPoint,
							tex->band.interpType, tex->band.size,
							tex->band.offsets, tex->band.values,
							tex1);
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					Texture_EvalOpGenericBump(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// For the very special case of Triplanar texture evaluation
		//
		// EVAL_TRIPLANAR_STEP_1, EVAL_TRIPLANAR_STEP_2, EVAL_TRIPLANAR_STEP_3
		// and EVAL_TRIPLANAR
		//----------------------------------------------------------------------
		case TRIPLANAR_TEX: {
			switch (evalType) {
				case EVAL_TRIPLANAR_STEP_1: {
					// Save original UV
					const uint uvIndex = tex->triplanarTex.uvIndex;
					EvalStack_PushFloat(hitPoint->uv[uvIndex].u);
					EvalStack_PushFloat(hitPoint->uv[uvIndex].v);

					// Compute localPoint
					float3 localShadeN;
					const float3 localPoint = TextureMapping3D_Map(&tex->triplanarTex.mapping,
							hitPoint, &localShadeN);

					// Compute the 3 weights
					float weightsX = Sqr(Sqr(localShadeN.x));
					float weightsY = Sqr(Sqr(localShadeN.y));
					float weightsZ = Sqr(Sqr(localShadeN.z));

					const float sum = weightsX + weightsY + weightsZ;
					weightsX = weightsX / sum;
					weightsY = weightsY / sum;
					weightsZ = weightsZ / sum;

					// Save 3 weights	
					EvalStack_PushFloat(weightsX);
					EvalStack_PushFloat(weightsY);
					EvalStack_PushFloat(weightsZ);
					// Save localPoint
					EvalStack_PushFloat3(localPoint);

					// Update HitPoint
					__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
					hitPointTmp->uv[uvIndex].u = localPoint.y;
					hitPointTmp->uv[uvIndex].v = localPoint.z;
					break;
				}
				case EVAL_TRIPLANAR_STEP_2: {
					// Read localPoint
					//
					// Note: -3 is there to skip the result of EVAL_TRIPLANAR_STEP_1
					const float3 localPoint = EvalStack_Read3(-3 - 3);

					// Update HitPoint
					__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
					const uint uvIndex = tex->triplanarTex.uvIndex;
					hitPointTmp->uv[uvIndex].u = localPoint.x;
					hitPointTmp->uv[uvIndex].v = localPoint.z;
					break;
				}
				case EVAL_TRIPLANAR_STEP_3: {
					// Read localPoint
					//
					// Note: -6 is there to skip the result of EVAL_TRIPLANAR_STEP_1 and EVAL_TRIPLANAR_STEP_2
					const float3 localPoint = EvalStack_Read3(-3 - 6);

					// Update HitPoint
					__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
					const uint uvIndex = tex->triplanarTex.uvIndex;
					hitPointTmp->uv[uvIndex].u = localPoint.x;
					hitPointTmp->uv[uvIndex].v = localPoint.y;
					break;
				}
				case EVAL_FLOAT:
				case EVAL_SPECTRUM: {
					// Read textures evaluation
					float3 tex1, tex2, tex3;
					EvalStack_PopFloat3(tex3);
					EvalStack_PopFloat3(tex2);
					EvalStack_PopFloat3(tex1);
					
					// Read localPoint
					float3 localPoint;
					EvalStack_PopFloat3(localPoint);

					// Read 3 weights
					float weightX, weightY, weightZ;
					EvalStack_PopFloat(weightZ);
					EvalStack_PopFloat(weightY);
					EvalStack_PopFloat(weightX);

					// Restore original UV
					const uint uvIndex = tex->triplanarTex.uvIndex;
					__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
					EvalStack_PopFloat(hitPointTmp->uv[uvIndex].v);
					EvalStack_PopFloat(hitPointTmp->uv[uvIndex].u);

					const float3 result = tex1 * weightX + tex2 * weightY + tex3 * weightZ;
					if (evalType == EVAL_FLOAT) {
						EvalStack_PushFloat(Spectrum_Y(result));
					} else {
						EvalStack_PushFloat3(result);
					}
					break;
				}
				case EVAL_BUMP_TRIPLANAR_STEP_1: {
					// Save original hit point
					const float3 p = VLOAD3F(&hitPoint->p.x);
					EvalStack_PushFloat3(p);

					// Update HitPoint
					__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
					hitPointTmp->p.x = p.x + sampleDistance;
					hitPointTmp->p.y = p.y;
					hitPointTmp->p.z = p.z;
					break;
				}
				case EVAL_BUMP_TRIPLANAR_STEP_2: {
					// Read original hit point
					//
					// Note: -1 is there to skip the result of EVAL_BUMP_TRIPLANAR_STEP_1
					const float3 p = EvalStack_Read3(-1 - 3);

					// Update HitPoint
					__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
					hitPointTmp->p.x = p.x;
					hitPointTmp->p.y = p.y + sampleDistance;
					hitPointTmp->p.z = p.z;
					break;
				}
				case EVAL_BUMP_TRIPLANAR_STEP_3: {
					// Read original hit point
					//
					// Note: -2 is there to skip the result of EVAL_BUMP_TRIPLANAR_STEP_2
					const float3 p = EvalStack_Read3(-2 - 3);

					// Update HitPoint
					__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
					hitPointTmp->p.x = p.x;
					hitPointTmp->p.y = p.y;
					hitPointTmp->p.z = p.z + sampleDistance;
					break;
				}
				case EVAL_BUMP_GENERIC_OFFSET_U:
					Texture_EvalOpGenericBumpOffsetU(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP_GENERIC_OFFSET_V:
					Texture_EvalOpGenericBumpOffsetV(evalStack, evalStackOffset,
							hitPoint, sampleDistance);
					break;
				case EVAL_BUMP:
					if (tex->triplanarTex.enableUVlessBumpMap) {
						float evalFloatTexBase, evalFloatTexOffsetX,
								evalFloatTexOffsetY, evalFloatTexOffsetZ;

						// Read X/Y/Z textures evaluation
						EvalStack_PopFloat(evalFloatTexOffsetZ);
						EvalStack_PopFloat(evalFloatTexOffsetY);
						EvalStack_PopFloat(evalFloatTexOffsetX);

						// Read and restore original hit point
						float3 p;
						EvalStack_PopFloat3(p);
						__global HitPoint *hitPointTmp = (__global HitPoint *)hitPoint;
						VSTORE3F(p, &hitPointTmp->p.x);

						// Read base textures evaluation
						EvalStack_PopFloat(evalFloatTexBase);

						const float3 shadeN = TriplanarTexture_BumpUVLess(hitPoint,
								sampleDistance, evalFloatTexBase, evalFloatTexOffsetX,
								evalFloatTexOffsetY, evalFloatTexOffsetZ);
						EvalStack_PushFloat3(shadeN);
					} else
						Texture_EvalOpGenericBump(evalStack, evalStackOffset,
								hitPoint, sampleDistance);
					break;
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// FRESNELCOLOR_TEX
		//----------------------------------------------------------------------
		case FRESNELCOLOR_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = FresnelColorTexture_ConstEvaluateFloat();
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = FresnelColorTexture_ConstEvaluateSpectrum();
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					const float3 shadeN = ConstTexture_Bump(hitPoint);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		// FRESNELCONST_TEX
		//----------------------------------------------------------------------
		case FRESNELCONST_TEX: {
			switch (evalType) {
				case EVAL_FLOAT: {
					const float eval = FresnelColorTexture_ConstEvaluateFloat();
					EvalStack_PushFloat(eval);
					break;
				}
				case EVAL_SPECTRUM: {
					const float3 eval = FresnelColorTexture_ConstEvaluateSpectrum();
					EvalStack_PushFloat3(eval);
					break;
				}
				case EVAL_BUMP: {
					const float3 shadeN = ConstTexture_Bump(hitPoint);
					EvalStack_PushFloat3(shadeN);
					break;
				}
				default:
					// Something wrong here
					break;
			}
			break;
		}
		//----------------------------------------------------------------------
		default:
			// Something wrong here
			break;
	}
}

//------------------------------------------------------------------------------
// Texture_GetFloatValue()
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE float Texture_GetFloatValue(const uint texIndex,
		__global const HitPoint *hitPoint
		TEXTURES_PARAM_DECL) {
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("===============================================================\n");
	printf("Texture_GetFloatValue()\n");
	printf("===============================================================\n");
#endif

	__global const Texture* restrict startTex = &texs[texIndex];
	const size_t gid = get_global_id(0);
	__global float *evalStack = &texEvalStacks[gid * maxTextureEvalStackSize];

	const uint evalFloatOpStartIndex = startTex->evalFloatOpStartIndex;
	const uint evalFloatOpLength = startTex->evalFloatOpLength;

#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("texIndex=%d evalFloatOpStartIndex=%d evalFloatOpLength=%d\n", texIndex, evalFloatOpStartIndex, evalFloatOpLength);
#endif

	uint evalStackOffsetVal = 0;
	uint *evalStackOffset = &evalStackOffsetVal; // Used by macros
	for (uint i = 0; i < evalFloatOpLength; ++i) {
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
		printf("EvalOp: #%d\n", i);
#endif

		__global const TextureEvalOp* restrict evalOp = &texEvalOps[evalFloatOpStartIndex + i];

		Texture_EvalOp(evalOp, evalStack, &evalStackOffsetVal, hitPoint, 0.f TEXTURES_PARAM);
	}
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("evalStackOffset=#%d\n", evalStackOffsetVal);
#endif

	float result;
	EvalStack_PopFloat(result);

#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("Result=%f\n", result);
#endif
	
	return result;
}

//------------------------------------------------------------------------------
// Texture_GetSpectrumValue()
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE float3 Texture_GetSpectrumValue(const uint texIndex,
		__global const HitPoint *hitPoint
		TEXTURES_PARAM_DECL) {
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("===============================================================\n");
	printf("Texture_GetSpectrumValue()\n");
	printf("===============================================================\n");
#endif

	__global const Texture* restrict startTex = &texs[texIndex];
	const size_t gid = get_global_id(0);
	__global float *evalStack = &texEvalStacks[gid * maxTextureEvalStackSize];

	const uint evalSpectrumOpStartIndex = startTex->evalSpectrumOpStartIndex;
	const uint evalSpectrumOpLength = startTex->evalSpectrumOpLength;

#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("texIndex=%d evalSpectrumOpStartIndex=%d evalSpectrumOpLength=%d\n", texIndex, evalSpectrumOpStartIndex, evalSpectrumOpLength);
#endif

	uint evalStackOffsetVal = 0;
	uint *evalStackOffset = &evalStackOffsetVal; // Used by macros
	for (uint i = 0; i < evalSpectrumOpLength; ++i) {
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
		printf("EvalOp: #%d\n", i);
#endif

		__global const TextureEvalOp* restrict evalOp = &texEvalOps[evalSpectrumOpStartIndex + i];

		Texture_EvalOp(evalOp, evalStack, &evalStackOffsetVal, hitPoint, 0.f TEXTURES_PARAM);
	}
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("evalStackOffset=#%d\n", evalStackOffsetVal);
#endif
	
	float3 result;
	EvalStack_PopFloat3(result);


#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("Result=(%f, %f, %f)\n", result.s0, result.s1, result.s2);
#endif

	return result;
}

//------------------------------------------------------------------------------
// Texture_Bump()
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE float3 Texture_Bump(const uint texIndex,
		__global HitPoint *hitPoint,
		const float sampleDistance
		TEXTURES_PARAM_DECL) {
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("===============================================================\n");
	printf("Texture_Bump()\n");
	printf("===============================================================\n");
#endif

	__global const Texture* restrict startTex = &texs[texIndex];
	const size_t gid = get_global_id(0);
	__global float *evalStack = &texEvalStacks[gid * maxTextureEvalStackSize];

	const uint evalBumpOpStartIndex = startTex->evalBumpOpStartIndex;
	const uint evalBumpOpLength = startTex->evalBumpOpLength;

#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("texIndex=%d evalSpectrumOpStartIndex=%d evalSpectrumOpLength=%d\n", texIndex, evalBumpOpStartIndex, evalBumpOpLength);
#endif

	uint evalStackOffsetVal = 0;
	uint *evalStackOffset = &evalStackOffsetVal; // Used by macros
	for (uint i = 0; i < evalBumpOpLength; ++i) {
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
		printf("EvalOp: #%d\n", i);
#endif

		__global const TextureEvalOp* restrict evalOp = &texEvalOps[evalBumpOpStartIndex + i];

		Texture_EvalOp(evalOp, evalStack, &evalStackOffsetVal, hitPoint, sampleDistance TEXTURES_PARAM);
	}
#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("evalStackOffset=#%d\n", evalStackOffsetVal);
#endif

	float3 result;
	EvalStack_PopFloat3(result);

#if defined(DEBUG_PRINTF_TEXTURE_EVAL)
	printf("Result=(%f, %f, %f)\n", result.x, result.y, result.z);
#endif

	return result;
}
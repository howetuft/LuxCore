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

#include <thread>
#include "luxrays/utils/thread.h"

#include "slg/engines/pathcpu/pathcpu.h"
#include "slg/volumes/volume.h"
#include "slg/utils/varianceclamping.h"
#include "slg/samplers/metropolis.h"

using namespace std;
using namespace luxrays;
using namespace slg;
using namespace std::literals::chrono_literals;

//------------------------------------------------------------------------------
// PathCPU RenderThread
//------------------------------------------------------------------------------

PathCPURenderThread::PathCPURenderThread(PathCPURenderEngine *engine,
		const u_int index, IntersectionDevice *device) :
		CPUNoTileRenderThread(engine, index, device) {
}

void PathCPURenderThread::RenderFunc(std::stop_token stop_token) {
	//SLG_LOG("[PathCPURenderEngine::" << threadIndex << "] Rendering thread started");

	//--------------------------------------------------------------------------
	// Initialization
	//--------------------------------------------------------------------------

	// This is really used only by Windows for 64+ threads support
	SetThreadGroupAffinity(threadIndex);

	PathCPURenderEngine *engine = (PathCPURenderEngine *)renderEngine;
	const PathTracer &pathTracer = engine->pathTracer;

	// (engine->seedBase + 1) seed is used for sharedRndGen
	RandomGenerator *rndGen = new RandomGenerator(engine->seedBase + 1 + threadIndex);

	// Setup the sampler(s)

	Sampler *eyeSampler = nullptr;
	Sampler *lightSampler = nullptr;

	eyeSampler = engine->renderConfig->AllocSampler(rndGen, engine->film,
			nullptr, engine->samplerSharedData, Properties());
	eyeSampler->SetThreadIndex(threadIndex);
	eyeSampler->RequestSamples(PIXEL_NORMALIZED_ONLY, pathTracer.eyeSampleSize);

	if (pathTracer.hybridBackForwardEnable) {
		// Light path sampler is always Metropolis
		Properties props;
		props <<
			Property("sampler.type")("METROPOLIS") <<
			// Disable image plane meaning for samples 0 and 1
			Property("sampler.imagesamples.enable")(false) <<
			Property("sampler.metropolis.addonlycaustics")(true);

		lightSampler = Sampler::FromProperties(props, rndGen, engine->film, engine->lightSampleSplatter,
				engine->lightSamplerSharedData);
		
		lightSampler->SetThreadIndex(threadIndex);
		lightSampler->RequestSamples(SCREEN_NORMALIZED_ONLY, pathTracer.lightSampleSize);
	}

	// Setup variance clamping
	VarianceClamping varianceClamping(pathTracer.sqrtVarianceClampMaxValue);

	// Setup PathTracer thread state
	PathTracerThreadState pathTracerThreadState(device,
			eyeSampler, lightSampler,
			engine->renderConfig->scene, engine->film,
			&varianceClamping);

	//--------------------------------------------------------------------------
	// Trace paths
	//--------------------------------------------------------------------------

	for (u_int steps = 0; !stop_token.stop_requested(); ++steps) {
		// Check if we are in pause mode
		if (engine->pauseMode) {
			// Check every 100ms if I have to continue the rendering
			while (!stop_token.stop_requested() && engine->pauseMode)
				std::this_thread::sleep_for(100ms);

			if (stop_token.stop_requested())
				break;
		}

		pathTracer.RenderSample(pathTracerThreadState);

#ifdef WIN32
		// Work around Windows bad scheduling
        std::this_thread::yield();
#endif

		// Check halt conditions
		if (engine->film->GetConvergence() == 1.f)
			break;
                if (stop_token.stop_requested())
                        break;

		if (engine->photonGICache) {
			const u_int spp = engine->film->GetTotalEyeSampleCount() / engine->film->GetPixelCount();
			engine->photonGICache->Update(threadIndex, spp);
		}
	}

	delete eyeSampler;
	delete lightSampler;
	delete rndGen;

	threadDone = true;

	// This is done to stop threads pending on barrier wait
	// inside engine->photonGICache->Update(). This can happen when an
	// halt condition is satisfied.
	if (engine->photonGICache)
		engine->photonGICache->FinishUpdate(threadIndex);

	//SLG_LOG("[PathCPURenderEngine::" << threadIndex << "] Rendering thread halted");
}

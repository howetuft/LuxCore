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

#include <boost/format.hpp>

#include <bcd/core/SamplesAccumulator.h>
#include <bcd/core/Denoiser.h>
#include <bcd/core/MultiscaleDenoiser.h>
#include <bcd/core/IDenoiser.h>
#include <bcd/core/Utils.h>

#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/bcddenoiser.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Background image plugin
//------------------------------------------------------------------------------

BOOST_CLASS_EXPORT_IMPLEMENT(slg::BCDDenoiserPlugin)

BCDDenoiserPlugin::BCDDenoiserPlugin(float histogramDistanceThreshold,
		int patchRadius,
		int searchWindowRadius,
		float minEigenValue,
		bool useRandomPixelOrder,
		float markedPixelsSkippingProbability,
		int threadCount,
		int scales)
	: histogramDistanceThreshold(histogramDistanceThreshold),
	  patchRadius(patchRadius),
	  searchWindowRadius(searchWindowRadius),
	  minEigenValue(minEigenValue),
	  useRandomPixelOrder(useRandomPixelOrder),
	  markedPixelsSkippingProbability(markedPixelsSkippingProbability),
	  threadCount(threadCount),
	  scales(scales)
{}
	
BCDDenoiserPlugin::BCDDenoiserPlugin() {
}

BCDDenoiserPlugin::~BCDDenoiserPlugin() {
}

ImagePipelinePlugin *BCDDenoiserPlugin::Copy() const {
	return new BCDDenoiserPlugin(histogramDistanceThreshold,
		patchRadius,
		searchWindowRadius,
		minEigenValue,
		useRandomPixelOrder,
		markedPixelsSkippingProbability,
		threadCount,
		scales);
}

//------------------------------------------------------------------------------
// CPU version
//------------------------------------------------------------------------------

static void ProgressCallBack(const float progress) {
	static double lastPrint = WallClockTime();
	
	const double now = WallClockTime();
	if (now - lastPrint > 1.0) {
		SLG_LOG("BCD progress: " << (boost::format("%.2f") % (100.0 * progress)) << "%");
		lastPrint = now;
	}
}

void BCDDenoiserPlugin::Apply(Film &film, const u_int index) {
	const double startTime = WallClockTime();

	const FilmDenoiser &filmDenoiser = film.GetDenoiser();

	Spectrum *pixels = (Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels();
	const u_int width = film.GetWidth();
	const u_int height = film.GetHeight();
	
	const bcd::SamplesStatisticsImages stats = filmDenoiser.GetSamplesStatistics();
	if (stats.m_nbOfSamplesImage.isEmpty()
			|| stats.m_histoImage.isEmpty()
			|| stats.m_covarImage.isEmpty()) {
		return;
	}
	
	// Init inputs
	
	bcd::DeepImage<float> inputColors(width, height, 3);

	const float sampleScale = filmDenoiser.GetSampleScale();
	SLG_LOG("BCD sample scale: " << sampleScale);
	const float sampleMaxValue = filmDenoiser.GetSampleMaxValue();
	SLG_LOG("BCD sample max. value: " << sampleMaxValue);
	// TODO alpha?
	for(u_int y = 0; y < height; ++y) {
		for(u_int x = 0; x < width; ++x) {
			const u_int i = (height - y - 1) * width + x;
			
			const Spectrum color = (pixels[i] *  sampleScale).Clamp(0.f, sampleMaxValue);
			inputColors.set(y, x, 0, color.c[0]);
			inputColors.set(y, x, 1, color.c[1]);
			inputColors.set(y, x, 2, color.c[2]);
		}
	}

	bcd::DenoiserInputs inputs;
	inputs.m_pColors = &inputColors;
	inputs.m_pNbOfSamples = &stats.m_nbOfSamplesImage;
	inputs.m_pHistograms = &stats.m_histoImage;
	inputs.m_pSampleCovariances = &stats.m_covarImage;

	// Init parameters
	
	bcd::DenoiserParameters parameters;
	parameters.m_histogramDistanceThreshold = histogramDistanceThreshold;
	parameters.m_patchRadius = patchRadius;
	parameters.m_searchWindowRadius = searchWindowRadius;
	parameters.m_minEigenValue = minEigenValue;
	parameters.m_useRandomPixelOrder = useRandomPixelOrder;
	parameters.m_markedPixelsSkippingProbability = markedPixelsSkippingProbability;
	parameters.m_nbOfCores = threadCount;
	parameters.m_useCuda = false;
	
	// Init outputs
	
	bcd::DeepImage<float> denoisedImg(width, height, 3);
	bcd::DenoiserOutputs outputs;
	outputs.m_pDenoisedColors = &denoisedImg;
	
	// Create denoiser and run denoising
	
	unique_ptr<bcd::IDenoiser> denoiser = nullptr;

	if (scales > 1)
		denoiser.reset(new bcd::MultiscaleDenoiser(scales));
	else
		denoiser.reset(new bcd::Denoiser());
	denoiser->setProgressCallback(ProgressCallBack);
		
	denoiser->setInputs(inputs);
	denoiser->setOutputs(outputs);
	denoiser->setParameters(parameters);
	
	denoiser->denoise();
	
	// Copy to output pixels
	const float invSampleScale = 1.f / sampleScale;
	for(u_int y = 0; y < height; ++y) {
		for(u_int x = 0; x < width; ++x) {
			const u_int i = (height - y - 1) * width + x;
			Spectrum *pixel = pixels + i;
			
			pixel->c[0] = denoisedImg.get(y, x, 0) * invSampleScale;
			pixel->c[1] = denoisedImg.get(y, x, 1) * invSampleScale;
			pixel->c[2] = denoisedImg.get(y, x, 2) * invSampleScale;
		}
	}
	
	SLG_LOG("BCD Apply took: " << (boost::format("%.1f") % (WallClockTime() - startTime)) << "secs");
}
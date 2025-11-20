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

#ifndef _LUXCOREIMPL_H
#define	_LUXCOREIMPL_H

#include <format>

#include <luxcore/luxcore.h>
#include <slg/renderconfig.h>
#include <slg/rendersession.h>
#include <slg/renderstate.h>
#include <slg/scene/scene.h>
#include <slg/film/film.h>

namespace luxcore {
namespace detail {

class RenderSessionImpl;
using RenderSessionImplPtr = std::shared_ptr<RenderSessionImpl>;
using RenderSessionImplConstPtr = std::shared_ptr<const RenderSessionImpl>;

class RenderConfigImpl;
using RenderConfigImplPtr = std::shared_ptr<RenderConfigImpl>;

class RenderStateImpl;
using RenderStateImplPtr = std::shared_ptr<RenderStateImpl>;

class SceneImpl;
using SceneImplConstRef = const SceneImpl &;

class CameraImpl;
using CameraImplPtr = std::shared_ptr<CameraImpl>;

class FilmImpl;
using FilmImplPtr = std::shared_ptr<FilmImpl>;

// Disambiguation: there are luxcore::Film and slg:Film...
using LuxFilm = luxcore::Film;
using LuxFilmPtr = std::shared_ptr<luxcore::Film>;
using LuxFilmConstPtr = std::shared_ptr<const luxcore::Film>;

// Disambiguation: there are luxcore::Camera and slg:Camera...
using LuxCamera = luxcore::Camera;
using LuxCameraConstRef = const luxcore::Camera &;


//------------------------------------------------------------------------------
// FilmImpl
//------------------------------------------------------------------------------

class FilmImpl : public luxcore::Film {
public:
	FilmImpl(const std::string &fileName);
	FilmImpl(
		const luxrays::Properties &props,
		const bool hasPixelNormalizedChannel,
		const bool hasScreenNormalizedChannel
	);
	FilmImpl(RenderSessionImplConstPtr session);
	FilmImpl(slg::FilmPtr film);
	~FilmImpl();

	unsigned int GetWidth() const;
	unsigned int GetHeight() const;
	luxrays::Properties GetStats() const;
	float GetFilmY(const unsigned int imagePipelineIndex = 0) const;

	void Clear();
	void AddFilm(LuxFilmConstPtr film);
	void AddFilm(
		LuxFilmConstPtr film,
		const unsigned int srcOffsetX, const unsigned int srcOffsetY,
		const unsigned int srcWidth, const unsigned int srcHeight,
		const unsigned int dstOffsetX, const unsigned int dstOffsetY
	);

	void SaveOutputs() const;
	void SaveOutput(
		const std::string &fileName,
		const FilmOutputType type,
		const luxrays::Properties &props
	) const;
	void SaveFilm(const std::string &fileName) const;

	double GetTotalSampleCount() const;

	size_t GetOutputSize(const FilmOutputType type) const;
	bool HasOutput(const FilmOutputType type) const;
	unsigned int GetOutputCount(const FilmOutputType type) const;

	unsigned int GetRadianceGroupCount() const;
	bool HasChannel(const FilmChannelType type) const;
	unsigned int GetChannelCount(const FilmChannelType type) const;

	void GetOutputFloat(const FilmOutputType type, float *buffer,
			const unsigned int index, const bool executeImagePipeline);
	void GetOutputUInt(const FilmOutputType type, unsigned int *buffer,
			const unsigned int index, const bool executeImagePipeline);
	void UpdateOutputFloat(const FilmOutputType type, const float *buffer,
			const unsigned int index, const bool executeImagePipeline);
	void UpdateOutputUInt(const FilmOutputType type, const unsigned int *buffer,
			const unsigned int index, const bool executeImagePipeline);

	const float *GetChannelFloat(const FilmChannelType type,
			const unsigned int index, const bool executeImagePipeline);
	const unsigned int *GetChannelUInt(const FilmChannelType type,
			const unsigned int index, const bool executeImagePipeline);
	float *UpdateChannelFloat(const FilmChannelType type,
			const unsigned int index, const bool executeImagePipeline);
	unsigned int *UpdateChannelUInt(const FilmChannelType type,
			const unsigned int index, const bool executeImagePipeline);

	void Parse(const luxrays::Properties &props);

	void DeleteAllImagePipelines();

	void ExecuteImagePipeline(const u_int index);
	void AsyncExecuteImagePipeline(const u_int index);
	void WaitAsyncExecuteImagePipeline();
	bool HasDoneAsyncExecuteImagePipeline();

	void ApplyOIDN(const u_int index);

	friend class RenderSessionImpl;

private:
	slg::FilmPtr GetSLGFilm() const;

	RenderSessionImplConstPtr renderSession;
	slg::FilmPtr standAloneFilm;
};

//------------------------------------------------------------------------------
// CameraImpl
//------------------------------------------------------------------------------


class CameraImpl : public luxcore::Camera {
public:
	CameraImpl(SceneImplConstRef scene);
	~CameraImpl();

	const CameraType GetType() const;

	void Translate(const float x, const float y, const float z) const;
	void TranslateLeft(const float t) const;
	void TranslateRight(const float t) const;
	void TranslateForward(const float t) const;
	void TranslateBackward(const float t) const;

	void Rotate(const float angle, const float x, const float y, const float z) const;
	void RotateLeft(const float angle) const;
	void RotateRight(const float angle) const;
	void RotateUp(const float angle) const;
	void RotateDown(const float angle) const;

	friend class SceneImpl;

private:
	SceneImplConstRef scene;
};

//------------------------------------------------------------------------------
// SceneImpl
//------------------------------------------------------------------------------

class SceneImpl : public luxcore::Scene {
public:
	SceneImpl(slg::ScenePtr scn);
	SceneImpl(const luxrays::Properties *resizePolicyProps = nullptr);
	SceneImpl(
		const luxrays::Properties &props,
		const luxrays::Properties *resizePolicyProps = nullptr
	);
	SceneImpl(
		const std::string &fileName,
		const luxrays::Properties *resizePolicyProps = nullptr
	);
	~SceneImpl();

	void GetBBox(float min[3], float max[3]) const;
	LuxCameraConstRef GetCamera() const;

	bool IsImageMapDefined(const std::string &imgMapName) const;

	void SetDeleteMeshData(const bool v);
	void SetMeshAppliedTransformation(const std::string &meshName,
			const float *appliedTransMat);

	void DefineMesh(const std::string &meshName,
		const long plyNbVerts, const long plyNbTris,
		float *p, unsigned int *vi, float *n,
		float *uvs,	float *cols, float *alphas);
	void DefineMeshExt(const std::string &meshName,
		const long plyNbVerts, const long plyNbTris,
		float *p, unsigned int *vi, float *n,
		std::array<float *, LC_MESH_MAX_DATA_COUNT> *uv,
		std::array<float *, LC_MESH_MAX_DATA_COUNT> *cols,
		std::array<float *, LC_MESH_MAX_DATA_COUNT> *alphas);
	void SetMeshVertexAOV(const std::string &meshName,
		const unsigned int index, float *data);
	void SetMeshTriangleAOV(const std::string &meshName,
		const unsigned int index, float *data);

	void SaveMesh(const std::string &meshName, const std::string &fileName);
	void DefineStrands(
		const std::string &shapeName,
		const luxrays::cyHairFile &strandsFile,
		const StrandsTessellationType tesselType,
		const unsigned int adaptiveMaxDepth,
		const float adaptiveError,
		const unsigned int solidSideCount,
		const bool solidCapBottom,
		const bool solidCapTop,
		const bool useCameraPosition
	);

	bool IsMeshDefined(const std::string &meshName) const;
	bool IsTextureDefined(const std::string &texName) const;
	bool IsMaterialDefined(const std::string &matName) const;

	const unsigned int GetLightCount() const;
	const unsigned int GetObjectCount() const;

	void Parse(const luxrays::Properties &props);

	void DuplicateObject(
		const std::string &srcObjName, const std::string &dstObjName,
		const float transMat[16], const unsigned int objectID
	);
	void DuplicateObject(
		const std::string &srcObjName, const std::string &dstObjNamePrefix,
		const unsigned int count, const float *transMat, const unsigned int *objectIDs
	);
	void DuplicateObject(
		const std::string &srcObjName, const std::string &dstObjName,
		const unsigned int steps, const float *times, const float *transMats,
		const unsigned int objectID);
	void DuplicateObject(
		const std::string &srcObjName, const std::string &dstObjNamePrefix,
		const unsigned int count, const unsigned int steps, const float *times,
		const float *transMats, const unsigned int *objectIDs);
	void UpdateObjectTransformation(
		const std::string &objName, const float transMat[16]
	);
	void UpdateObjectMaterial(
		const std::string &objName, const std::string &matName
	);

	void DeleteObject(const std::string &objName);
	void DeleteObjects(std::vector<std::string> &objNames);
	void DeleteLight(const std::string &lightName);
	void DeleteLights(std::vector<std::string> &lightNames);

	void RemoveUnusedImageMaps();
	void RemoveUnusedTextures();
	void RemoveUnusedMaterials();
	void RemoveUnusedMeshes();

	void DefineImageMapUChar(
		const std::string &imgMapName,
		unsigned char *pixels,
		const float gamma,
		const unsigned int channels,
		const unsigned int width,
		const unsigned int height,
		ChannelSelectionType selectionType,
		WrapType wrapType
	);
	void DefineImageMapHalf(const std::string &imgMapName,
			unsigned short *pixels, const float gamma, const unsigned int channels,
			const unsigned int width, const unsigned int height,
			ChannelSelectionType selectionType, WrapType wrapType);
	void DefineImageMapFloat(const std::string &imgMapName,
			float *pixels, const float gamma, const unsigned int channels,
			const unsigned int width, const unsigned int height,
			ChannelSelectionType selectionType, WrapType wrapType);

	const luxrays::Properties &ToProperties() const;
	void Save(const std::string &fileName) const;

	// Note: this method is not part of LuxCore API and it is used only internally
	void DefineMesh(std::shared_ptr<luxrays::ExtTriangleMesh> mesh);

	static luxrays::Point *AllocVerticesBuffer(const unsigned int meshVertCount);
	static luxrays::Triangle *AllocTrianglesBuffer(const unsigned int meshTriCount);

	friend class CameraImpl;
	friend class RenderConfigImpl;
	friend class RenderSessionImpl;

private:
	mutable luxrays::Properties scenePropertiesCache;

	slg::ScenePtr scene;
	CameraImplPtr camera;
	bool allocatedScene;
};

//------------------------------------------------------------------------------
// RenderConfigImpl
//------------------------------------------------------------------------------


class RenderConfigImpl : public luxcore::RenderConfig {
public:
	RenderConfigImpl(const luxrays::Properties &props, std::shared_ptr<SceneImpl> scene = NULL);
	RenderConfigImpl(const std::string &fileName);
	RenderConfigImpl(
		const std::string &fileName,
		std::shared_ptr<RenderStateImpl> * startState,
		std::shared_ptr<FilmImpl> * startFilm
	);
	~RenderConfigImpl();

	const luxrays::Properties &GetProperties() const;
	const luxrays::Property GetProperty(const std::string &name) const;
	const luxrays::Properties &ToProperties() const;

	Scene &GetScene() const;

	bool HasCachedKernels() const;

	void Parse(const luxrays::Properties &props);

	void Delete(const std::string &prefix);

	bool GetFilmSize(unsigned int *filmFullWidth, unsigned int *filmFullHeight,
		unsigned int *filmSubRegion) const;

	void DeleteSceneOnExit();

	void Save(const std::string &fileName) const;
	void Export(const std::string &dirName) const;
	void ExportGLTF(const std::string &fileName) const;

	static const luxrays::Properties &GetDefaultProperties();

	friend class RenderSessionImpl;

private:
	std::shared_ptr<slg::RenderConfig> renderConfig;

	std::shared_ptr<SceneImpl> scene;
	bool allocatedScene;
};

//------------------------------------------------------------------------------
// RenderStateImpl
//------------------------------------------------------------------------------

class RenderStateImpl : public RenderState {
public:
	RenderStateImpl(const std::string &fileName);
	RenderStateImpl(std::shared_ptr<slg::RenderState> state);
	~RenderStateImpl();

	void Save(const std::string &fileName) const;

	friend class RenderSessionImpl;

private:
	std::shared_ptr<slg::RenderState> renderState;
};

//------------------------------------------------------------------------------
// RenderSessionImpl
//------------------------------------------------------------------------------

class RenderSessionImpl : public RenderSession, public std::enable_shared_from_this<RenderSessionImpl> {

	// https://en.cppreference.com/w/cpp/memory/enable_shared_from_this.html
	struct Private{ explicit Private() = default; };

public:

	static RenderSessionImplPtr Create(
		std::shared_ptr<RenderConfigImpl> config,
		std::shared_ptr<RenderStateImpl>& startState,
		std::shared_ptr<FilmImpl>& startFilm
	);
	static RenderSessionImplPtr Create(
		std::shared_ptr<RenderConfigImpl> config,
		const std::string &startStateFileName,
		const std::string &startFilmFileName
	);

	// Public, but private, constructors
	// https://en.cppreference.com/w/cpp/memory/enable_shared_from_this.html
	RenderSessionImpl(
		Private priv,
		std::shared_ptr<RenderConfigImpl> config,
		std::shared_ptr<RenderStateImpl> startState = nullptr,
		std::shared_ptr<FilmImpl> startFilm = nullptr
	);
	RenderSessionImpl(
		Private priv,
		std::shared_ptr<RenderConfigImpl> config,
		const std::string &startStateFileName,
		const std::string &startFilmFileName
	);
	~RenderSessionImpl();


	std::shared_ptr<RenderConfig> GetRenderConfig();
	std::shared_ptr<RenderState> GetRenderState();

	void Start();
	void Stop();
	bool IsStarted() const;

	void BeginSceneEdit();
	void EndSceneEdit();
	bool IsInSceneEdit() const;

	void Pause();
	void Resume();
	bool IsInPause() const;

	bool HasDone() const;
	void WaitForDone() const;
	void WaitNewFrame();

	bool NeedPeriodicFilmSave();
	LuxFilmPtr GetFilm();

	void UpdateStats();
	const luxrays::Properties &GetStats() const;

	void Parse(const luxrays::Properties &props);

	void SaveResumeFile(const std::string &fileName);

	friend class FilmImpl;

private:
	std::shared_ptr<RenderConfigImpl> renderConfig;
	std::shared_ptr<FilmImpl> film;

	std::shared_ptr<slg::RenderSession> renderSession;
	luxrays::Properties stats;

	void InitFilm();

};

}
}

template <> struct std::formatter<luxcore::Camera::CameraType>: formatter<string_view> {

  auto format(luxcore::Camera::CameraType cam, std::format_context& ctx) const
    -> format_context::iterator;
};



#endif	/* _LUXCOREIMPL_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

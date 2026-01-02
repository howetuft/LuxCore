/***************************************************************************
 * Copyright 1998-2025 by authors (see AUTHORS.txt)                        *
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

// This file is intended to gather all usings (pointers, refs etc.) for
// slg classes

#pragma once

#include <memory>
#include "luxrays/usings.h"

namespace slg {

class Camera;
using CameraUPtr = std::unique_ptr<Camera>;
using CameraRef = Camera&;
using CameraConstRef = const Camera&;

class Scene;
using SceneUPtr = std::unique_ptr<Scene>;
using SceneConstUPtr = std::unique_ptr<const Scene>;
using SceneConstRef = const Scene&;
using SceneRef = Scene&;

class SceneObject;
using SceneObjectRef = SceneObject&;
using SceneObjectConstRef = const SceneObject&;
using SceneObjectUPtr = std::unique_ptr<SceneObject>;
using SceneObjectConstUPtr = std::unique_ptr<const SceneObject>;

class Shape;
using ShapeUPtr = std::unique_ptr<Shape>;

class Film;
using FilmPtr = std::shared_ptr<Film>;
using FilmConstPtr = std::shared_ptr<const Film>;
using FilmConstRef = const Film&;
using FilmRef = Film&;

class ImageMap;
using ImageMapConstRef = const ImageMap&;
using ImageMapRef = ImageMap&;
using ImageMapConstUPtr = std::unique_ptr<const ImageMap>;
using ImageMapUPtr = std::unique_ptr<ImageMap>;
using ImageMapConstSPtr = std::shared_ptr<const ImageMap>;
using ImageMapSPtr = std::shared_ptr<ImageMap>;

class ImageMapStorage;
using ImageMapStorageUPtr = std::unique_ptr<ImageMapStorage>;
using ImageMapStorageRef = ImageMapStorage&;
using ImageMapStorageConstRef = const ImageMapStorage&;

class LightSource;
using LightSourceConstPtr = std::shared_ptr<const LightSource>;
using LightSourcePtr = std::shared_ptr<LightSource>;
using LightSourceUPtr = std::unique_ptr<LightSource>;
using LightSourceRef = LightSource&;
using LightSourceConstRef = const LightSource&;

class TriangleLight;
using TriangleLightConstPtr = std::shared_ptr<const TriangleLight>;
using TriangleLightPtr = std::shared_ptr<TriangleLight>;
using TriangleLightConstUPtr = std::unique_ptr<const TriangleLight>;
using TriangleLightUPtr = std::unique_ptr<TriangleLight>;
using TriangleLightRef = TriangleLight&;
using TriangleLightConstRef = const TriangleLight&;

class EnvLightSource;
using EnvLightSourceConstPtr = std::shared_ptr<const EnvLightSource>;
using EnvLightSourcePtr = std::shared_ptr<EnvLightSource>;
using EnvLightSourceUPtr = std::unique_ptr<EnvLightSource>;
using EnvLightSourceRef = EnvLightSource&;

class LightStrategy;
using LightStrategyConstPtr = std::shared_ptr<const LightStrategy>;
using LightStrategyPtr = std::shared_ptr<LightStrategy>;
using LightStrategyUPtr = std::unique_ptr<LightStrategy>;
using LightStrategyConstRef = const LightStrategy&;

class Material;
using MaterialRef = Material&;
using MaterialConstRef = const Material&;
using MaterialUPtr = std::unique_ptr<Material>;
using MaterialConstUPtr = std::unique_ptr<const Material>;
using MatRef = OptionalPtr<const Material>;  // This is just for convenience

class RenderConfig;
using RenderConfigConstPtr = std::shared_ptr<const RenderConfig>;
using RenderConfigPtr = std::shared_ptr<RenderConfig>;
using RenderConfigUPtr = std::unique_ptr<RenderConfig>;
using RenderConfigConstUPtr = std::unique_ptr<const RenderConfig>;
using RenderConfigRef = RenderConfig &;
using RenderConfigConstRef = const RenderConfig &;

class RenderSession;
using RenderSessionConstPtr = std::shared_ptr<const RenderSession>;
using RenderSessionPtr = std::shared_ptr<RenderSession>;
using RenderSessionRef = RenderSession &;
using RenderSessionConstRef = const RenderSession &;

class RenderState;
using RenderStateConstPtr = std::shared_ptr<const RenderState>;
using RenderStatePtr = std::shared_ptr<RenderState>;

class RenderEngine;
using RenderEngineConstPtr = std::shared_ptr<const RenderEngine>;
using RenderEnginePtr = std::shared_ptr<RenderEngine>;

class RenderEngine;
using RenderEngineConstPtr = std::shared_ptr<const RenderEngine>;
using RenderEnginePtr = std::shared_ptr<RenderEngine>;
using RenderEngineUPtr = std::unique_ptr<RenderEngine>;

class Texture;
using TextureUPtr = std::unique_ptr<Texture>;
using TextureRef = Texture&;
using TextureConstRef = const Texture&;

class FresnelTexture;
using FresnelTextureUPtr = std::unique_ptr<FresnelTexture>;
using FresnelTextureConstRef = const FresnelTexture &;

class ImageMapTexture;
using ImageMapTextureUPtr = std::unique_ptr<ImageMapTexture>;

class TextureMapping2D;
using TextureMapping2DUPtr = std::unique_ptr<TextureMapping2D>;
using TextureMapping2DRef = TextureMapping2D&;
using TextureMapping2DConstRef = const TextureMapping2D&;

class TextureMapping3D;
using TextureMapping3DUPtr = std::unique_ptr<TextureMapping3D>;
using TextureMapping3DRef = TextureMapping3D&;
using TextureMapping3DConstRef = const TextureMapping3D&;

class Volume;
using VolumeUPtr = std::unique_ptr<Volume>;
//using VolumePtr = std::shared_ptr<Volume>;
//using VolumeConstPtr = std::shared_ptr<const Volume>;
using VolumeRef = Volume&;
using VolumeConstRef = const Volume&;

class Sampler;
using SamplerUPtr = std::unique_ptr<Sampler>;

class SamplerSharedData;
using SamplerSharedDataUPtr = std::unique_ptr<SamplerSharedData>;

}  // namespace slg

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

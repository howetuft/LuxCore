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

#include <iostream>
#include <fstream>
#include <cstring>

#include <boost/format.hpp>

#include "luxrays/core/exttrianglemesh.h"
#include "luxrays/utils/ply/rply.h"
#include "luxrays/utils/serializationutils.h"

using namespace std;
using namespace luxrays;

//------------------------------------------------------------------------------
// ExtMesh PLY reader
//------------------------------------------------------------------------------

// Generic handler for callback
template<typename T>
static float& getter(Buffer<T>&, long, long);

// Generic callback
template <typename T>
static int callback(p_ply_argument argument) {
	long userIndex = 0;
	void *userData = nullptr;
	ply_get_argument_user_data(argument, &userData, &userIndex);

	Buffer<T>& buffer = *static_cast<Buffer<T> *>(userData);

	// Buffer index
	long bufferIndex;
	ply_get_argument_element(argument, nullptr, &bufferIndex);

	// Data type
	p_ply_property property = nullptr;
	ply_get_argument_property(argument, &property, nullptr, nullptr);
	e_ply_type dataType;
	ply_get_property_info(property, nullptr, &dataType, nullptr, nullptr);

	// Argument value
	auto argvalue = static_cast<float>(ply_get_argument_value(argument));
	if (dataType == PLY_UCHAR) argvalue /= 255.0;

	getter<T>(buffer, bufferIndex, userIndex) = argvalue;

	return 1;
}

// Point handler specialization
template <>
float& getter(Buffer<Point>& buffer, long bufferIndex, long userIndex) {
	switch(userIndex) {
		case 0: return buffer[bufferIndex].x;
		case 1: return buffer[bufferIndex].y;
		case 2: return buffer[bufferIndex].z;
	};
	throw std::out_of_range("Getter index is out of range");
}

// Normal handler specialization
template <>
float& getter(Buffer<Normal>& buffer, long bufferIndex, long userIndex) {
	switch(userIndex) {
		case 0: return buffer[bufferIndex].x;
		case 1: return buffer[bufferIndex].y;
		case 2: return buffer[bufferIndex].z;
	};
	throw std::out_of_range("Getter index is out of range");
}

// UV handler specialization
template <>
float& getter(Buffer<UV>& buffer, long bufferIndex, long userIndex) {
	switch(userIndex) {
		case 0: return buffer[bufferIndex].u;
		case 1: return buffer[bufferIndex].v;
	};
	throw std::out_of_range("Getter index is out of range");
}

// Color handler specialization
template <>
float& getter(Buffer<Spectrum>& buffer, long bufferIndex, long userIndex) {
	return buffer[bufferIndex].c[userIndex];
}

// Alpha and AOVVertex (aka float) handler specialization
template <>
float& getter(Buffer<float>& buffer, long bufferIndex, long userIndex) {
	return buffer[bufferIndex];
}


// rply face callback
static int FaceCB(p_ply_argument argument) {
	void *userData = nullptr;
	ply_get_argument_user_data(argument, &userData, nullptr);

	Buffer<Triangle> *tris = static_cast<Buffer<Triangle> *> (userData);

	long length, valueIndex;
	ply_get_argument_property(argument, nullptr, &length, &valueIndex);

	if (length == 3) {
		if (valueIndex < 0)
			tris->push_back(Triangle());
		else if (valueIndex < 3)
			tris->back().v[valueIndex] =
					static_cast<u_int> (ply_get_argument_value(argument));
	} else if (length == 4) {
		// I have to split the quad in 2x triangles
		if (valueIndex < 0) {
			tris->push_back(Triangle());
		} else if (valueIndex < 3)
			tris->back().v[valueIndex] =
					static_cast<u_int> (ply_get_argument_value(argument));
		else if (valueIndex == 3) {
			const u_int i0 = tris->back().v[0];
			const u_int i1 = tris->back().v[2];
			const u_int i2 = static_cast<u_int> (ply_get_argument_value(argument));

			tris->push_back(Triangle(i0, i1, i2));
		}
	}

	return 1;
}


ExtTriangleMeshPtr ExtTriangleMesh::LoadPly(const string &fileName) {

	using ArrayOfSizes = std::array<size_t, EXTMESH_MAX_DATA_COUNT>;

	// Open file and read header
	p_ply plyfile = ply_open(fileName.c_str(), nullptr);
	if (!plyfile) {
		stringstream ss;
		ss << "Unable to read PLY mesh file '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	if (!ply_read_header(plyfile)) {
		stringstream ss;
		ss << "Unable to read PLY header from '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	// Set vertex callback
	Buffer<Point> p;
	const long plyNbVerts = ply_set_read_cb(
		plyfile, "vertex", "x", callback<Point>, &p, 0
	);
	ply_set_read_cb(plyfile, "vertex", "y", callback<Point>, &p, 1);
	ply_set_read_cb(plyfile, "vertex", "z", callback<Point>, &p, 2);
	if (plyNbVerts <= 0) {
		stringstream ss;
		ss << "No vertices found in '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	// Set triangle callback
	Buffer<Triangle> vi;
	const long plyNbFaces = ply_set_read_cb(
		plyfile,
		"face",
		"vertex_indices",
		FaceCB,
		&vi,
		0
	);
	if (plyNbFaces <= 0) {
		stringstream ss;
		ss << "No faces found in '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	// Check if the file includes normal information
	Buffer<Normal> n;
	const long plyNbNormals = ply_set_read_cb(
		plyfile,
		"vertex",
		"nx",
		callback<Normal>,
		&n,
		0
	);
	ply_set_read_cb(plyfile, "vertex", "ny", callback<Normal>, &n, 1);
	ply_set_read_cb(plyfile, "vertex", "nz", callback<Normal>, &n, 2);
	if ((plyNbNormals > 0) && (plyNbNormals != plyNbVerts)) {
		stringstream ss;
		ss << "Wrong count of normals in '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	// Check if the file includes triaov information
	ArrayOfBuffers<float> TriAOVs;
	ArrayOfSizes plyNbTriAOVs;
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		const std::string suffix = (i == 0) ? "" : ToString(i);

		plyNbTriAOVs[i] = ply_set_read_cb(
			plyfile,
			("faceaov" + suffix).c_str(),
			"triaov",
			callback<float>,
			&TriAOVs[i],
			0
		);
		if ((plyNbTriAOVs[i] > 0) && (plyNbTriAOVs[i] != plyNbFaces)) {
			stringstream ss;
			ss << "Wrong count of triangle AOV #" << i << " in '" << fileName << "'";
			throw runtime_error(ss.str());
		}
	}

	// This is our own extension to file PLY format in order to support multiple
	// UVs, Colors and Alphas for each vertex

	ArrayOfBuffers<UV> uvs;
	ArrayOfBuffers<Spectrum> cols;
	ArrayOfBuffers<float> alphas;
	ArrayOfBuffers<float> vertexAOVs;

	ArrayOfSizes plyNbUVs;
	ArrayOfSizes plyNbColors;
	ArrayOfSizes plyNbAlphas;
	ArrayOfSizes plyNbVertexAOVs;

	// Check if the file includes uv, color, alpha, vertex AOV information
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		const string suffix = (i == 0) ? "" : ToString(i);

		// Callback initializer
		auto setCB =
			[i, &plyfile, &suffix, &plyNbVerts, &fileName]
			<typename T, size_t N>
			(
				ArrayOfBuffers<T>& arrbuf,
				std::array<const char *, N> props,
				const char * name
			)
			-> long
		{
			// Set callbacks and get size
			long count;
			for (size_t j = 0; j < N; ++j) {
				const std::string prop = std::string(props[j]) + suffix;
				count = ply_set_read_cb(
					plyfile, "vertex", prop.c_str(), callback<T>, &arrbuf[i], j
				);
			}

			if ((count > 0) and (count != plyNbVerts)) {
				stringstream ss;
				ss  << "Wrong count of " << name << " #" << i
					<< " in '" << fileName << "'";
				throw runtime_error(ss.str());
			}

			return count;
		};

		// Check if the file includes information
		plyNbUVs[i] = setCB(uvs, std::array{"s", "s"}, "uv");
		plyNbColors[i] = setCB(cols, std::array{"red", "green", "blue"}, "colors");
		plyNbAlphas[i] = setCB(alphas, std::array{"alpha"}, "alphas");
		plyNbVertexAOVs[i] = setCB(vertexAOVs, std::array{"vertaov"}, "vertex AOV");

	}

	// Allocate memory
	p.resize(plyNbVerts);
	vi.resize(plyNbFaces);

	if (plyNbNormals) {
		n.resize(plyNbNormals);
	}

	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {

		// Helper to resize arrays of optional buffers
		auto resize_array = [i]<typename T> (
			ArrayOfBuffers<T>& arr,
			const array<size_t, EXTMESH_MAX_DATA_COUNT>& bufsize
		)
		{
			if(bufsize[i]) { arr[i].resize(bufsize[i]); }
		};

		resize_array(uvs, plyNbUVs);
		resize_array(cols, plyNbColors);
		resize_array(alphas, plyNbAlphas);
		resize_array(vertexAOVs, plyNbVertexAOVs);
		resize_array(TriAOVs, plyNbTriAOVs);
	}

	if (!ply_read(plyfile)) {
		stringstream ss;
		ss << "Unable to parse PLY file '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	ply_close(plyfile);

	auto mesh = std::make_shared<ExtTriangleMesh>(
		plyNbVerts,
		vi.size(),
		std::move(p),
		std::move(vi),
		Optionals<Normal>(std::move(n)),
		ArrayOfOptionals<UV>(std::move(uvs)),
		ArrayOfOptionals<Spectrum>(std::move(cols)),
		ArrayOfOptionals<float>(std::move(alphas))
	);
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (vertexAOVs[i]) {
			mesh->SetVertexAOV(i, Optionals<float>(std::move(vertexAOVs[i])));
		}

		if (TriAOVs[i]) {
			mesh->SetTriAOV(i, Optionals<float>(std::move(TriAOVs[i])));
		}
	}

	return mesh;
}

//------------------------------------------------------------------------------
// ExtTriangleMesh Load
//------------------------------------------------------------------------------

ExtTriangleMeshPtr ExtTriangleMesh::Load(const string &fileName) {
	const std::filesystem::path ext = std::filesystem::path(fileName).extension();
	if (ext == ".ply")
		return LoadPly(fileName);
	else if (ext == ".bpy")
		return LoadSerialized(fileName);
	else
		throw runtime_error(
			"Unknown file extension while loading a mesh from: " + fileName
		);
}

//------------------------------------------------------------------------------
// ExtTriangleMesh Save
//------------------------------------------------------------------------------

void ExtTriangleMesh::Save(const string &fileName) const {
	const std::filesystem::path ext = std::filesystem::path(fileName).extension();
	if (ext == ".ply")
		SavePly(fileName);
	else if (ext == ".bpy")
		SaveSerialized(fileName);
	else
		throw runtime_error("Unknown file extension while saving a mesh to: " + fileName);
}

void ExtTriangleMesh::SavePly(const string &fileName) const {
	// The use of std::filesystem::path is required for UNICODE support: fileName
	// is supposed to be UTF-8 encoded.
	std::ofstream plyFile(std::filesystem::path(fileName),
			std::ofstream::out |
			std::ofstream::binary |
			std::ofstream::trunc);
	if(!plyFile.is_open())
		throw runtime_error("Unable to open: " + fileName);

	plyFile.imbue(cLocale);
	
	// Write the PLY header
	plyFile << "ply\n"
			"format " + string(ply_storage_mode_list[ply_arch_endian()]) + " 1.0\n"
			"comment Created by LuxRays v" LUXRAYS_VERSION "\n"
			"element vertex " << vertCount << "\n"
			"property float x\n"
			"property float y\n"
			"property float z\n";

	if (HasNormals())
		plyFile << "property float nx\n"
				"property float ny\n"
				"property float nz\n";

	// This is our own extension to file PLY format in order to support multiple
	// UVs, Colors, Alphas and Vertex AOVs for each vertex
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		const string suffix = (i == 0) ? "" : ToString(i);

		if (HasUVs(i))
			plyFile << "property float s" << suffix << "\n"
					"property float t" << suffix << "\n";

		if (HasColors(i))
			plyFile << "property float red" << suffix << "\n"
					"property float green" << suffix << "\n"
					"property float blue" << suffix << "\n";

		if (HasAlphas(i))
			plyFile << "property float alpha" << suffix << "\n";	

		if (HasVertexAOV(i))
			plyFile << "property float vertaov" << suffix << "\n";	
	}

	plyFile << "element face " << triCount << "\n"
				"property list uchar uint vertex_indices\n";

	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		const string suffix = (i == 0) ? "" : ToString(i);

		if (HasTriAOV(i))
			plyFile << "element faceaov" << suffix << " " << triCount << "\n"
					"property float triaov\n";
	}

	plyFile << "end_header\n";

	if (!plyFile.good())
		throw runtime_error("Unable to write PLY header to: " + fileName);

	// Write all vertex data
	for (u_int i = 0; i < vertCount; ++i) {
		plyFile.write(reinterpret_cast<const char *>(&vertices[i]), sizeof(Point));
		if (HasNormals())
			plyFile.write(reinterpret_cast<const char *>(&normals[i]), sizeof(Normal));

		for (u_int j = 0; j < EXTMESH_MAX_DATA_COUNT; ++j) {
			if (HasUVs(j))
				plyFile.write(reinterpret_cast<const char *>(&(*uvs[j])[i]), sizeof(UV));
			if (HasColors(j))
				plyFile.write(reinterpret_cast<const char *>(&(*cols[j])[i]), sizeof(Spectrum));
			if (HasAlphas(j))
				plyFile.write(reinterpret_cast<const char *>(&(*alphas[j])[i]), sizeof(float));
			if (HasVertexAOV(j))
				plyFile.write(reinterpret_cast<const char *>(&(*vertAOV[j])[i]), sizeof(float));
		}
	}

	if (!plyFile.good())
		throw runtime_error("Unable to write PLY vertex data to: " + fileName);

	// Write all face data
	const u_char len = 3;
	for (u_int i = 0; i < triCount; ++i) {
		plyFile.write(reinterpret_cast<const char *>(&len), 1);
		plyFile.write(reinterpret_cast<const char *>(&tris[i]), sizeof(Triangle));
	}

	for (u_int j = 0; j < EXTMESH_MAX_DATA_COUNT; ++j) {
		if (HasTriAOV(j)) {
			for (u_int i = 0; i < triCount; ++i)
				plyFile.write(reinterpret_cast<const char *>(&(*triAOV[j])[i]), sizeof(float));
		}
	}

	if (!plyFile.good())
		throw runtime_error("Unable to write PLY face data to: " + fileName);

	plyFile.close();
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

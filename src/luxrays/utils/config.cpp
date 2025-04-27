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

#if defined(__APPLE__)
#include <sys/types.h>
#include <pwd.h>
#endif

#include "luxrays/utils/config.h"

using namespace std;
using namespace luxrays;

namespace luxrays {

string SanitizeFileName(const string &name) {
	string sanitizedName(name.size(), '_');
	
	for (u_int i = 0; i < sanitizedName.size(); ++i) {
		if ((name[i] >= 'A' && name[i] <= 'Z') || (name[i] >= 'a' && name[i] <= 'z') ||
				(name[i] >= '0' && name[i] <= '9'))
			sanitizedName[i] = name[i];
	}

	return sanitizedName;
}

std::filesystem::path GetConfigDir() {
#if defined(__linux__)
	// std::filesystem::temp_directory_path() is usually mapped to /tmp and
	// the content of the directory is often deleted at each reboot
	std::filesystem::path kernelConfigDir = getenv("HOME");
	kernelConfigDir = kernelConfigDir / ".config" / "luxcorerender.org";
#elif defined(__APPLE__)
	// std::filesystem::temp_directory_path() is usually mapped to /tmp and
	// the content of the directory is deleted at each reboot on MacOS
	
	// This may not work on MacOS for application started from the GUI
	//std::filesystem::path kernelConfigDir(getenv("HOME"));

	std::filesystem::path kernelConfigDir;

	const uid_t uid = getuid();
	const struct passwd *pwd = getpwuid(uid);
	if (!pwd)
		kernelConfigDir = std::filesystem::temp_directory_path();
	else
		kernelConfigDir = string(pwd->pw_dir);
	
	kernelConfigDir = kernelConfigDir / "luxcorerender.org";
#else
	std::filesystem::path kernelConfigDir= std::filesystem::temp_directory_path() / "luxcorerender.org";
#endif

	return kernelConfigDir;
}

}

#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void showGBuffer(uchar4* pbo, int interation, bool showGbuffer, bool showNormal);
void showImage(uchar4* pbo, int iter);

void denoiser(uchar4* pbo, int iter);

#ifndef LOAD_GEO2_H
#define LOAD_GEO2_H

#include <string>
#include "dist/json/json.h"
#include "volume_data.h"

void print(std::string s);
void print(int i);
/**
 * used to read and set the x,y,z dimensions
 * @param string info contains the line from the json file with key "volume_summary" 
 */
void loadDimensions(std::string info);
/**
 * 
 */
std::vector<std::string> getNameMapping(std::string s);
std::vector<volume_data> getEachVoxel(Json::Value allVoxelData, std::vector<std::string> vDtypes);
volume_data getData4Vox(Json::Value vData);
void setVolData(Json::Value tile, volume_data &v, int xOff, int yOff, int zOff, std::string compr);

#endif
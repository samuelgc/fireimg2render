#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "dist/json/json.h"


using namespace std;
int xDim = 0; // THe dim in x direction
int yDim = 0; // THe dim in y direction
int zDim = 0; // THe dim in z direction
void getDimensions(string info);
int main()
{
    ifstream jsonFile("./help.json",ifstream::binary);
    Json::Value root;
    jsonFile >> root; 
    string info =  root[11]["volume_summary"].asString();
    Json::Value allVoxelData = root[21];
    cout << allVoxelData;
    // cout << info << endl;
    getDimensions(info);
    return 0;
}

//gets the X,Y,Z Dimensions
void getDimensions(string info)
{
    int indexB1 = -1; // means index Bracket 1
    int indexB2 = -1; // means index Bracket 2
    for(unsigned int i = 0 ; i < info.length() ;i++)
    {
        if(info[i] == '[')
            indexB1 = i;
        if(info[i] == ']'){
            indexB2 = i;
            break;
        }
    }
    string num =  info.substr(indexB1 + 1,indexB2 - indexB1 - 1);
    num.erase(remove(num.begin(), num.end(),' '), num.end());
    int indexC1 = -1;
    int indexC2 = -1;
    //used to find commas
    for(unsigned int i = 0 ; i < num.length() ;i++){
        if(num[i] == ',')
        {
            if(indexC1 == -1)
                indexC1 = i;
            else{
                indexC2 = i;
                break;
            }
        }
    }
    string x = num.substr(0,indexC1);
    string y = num.substr(indexC1 + 1,indexC2 - indexC1 -1);
    string z = num.substr(indexC2 + 1,num.length() - indexC2 - 1);
    //convert to Integars
    xDim = stoi(x);
    yDim = stoi(y);
    zDim = stoi(z);
}
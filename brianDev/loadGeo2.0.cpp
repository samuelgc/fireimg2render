#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "dist/json/json.h"


using namespace std;

class volume 
{
    private:
        vector<vector<vector<double>>> data;
        int x;
        int y;
        int z;
    public:
    volume(int dx, int dy, int dz)
    {
        x = dx;
        y = dy;
        z = dz;
        data =  vector<vector<vector<double>>>(double(dx), vector<vector<double>>(double(dy), vector<double>(double(dz))));
    }
    void setValue(double val,int x, int y, int z)
    {
        data[x][y][z] = val;
    }
    int size()
    {
        return x * y * z;
    }
    void writeSlice()
    {
        int filesize = 54 + 3*x*z;
        unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0};
        unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
        // unsigned char bmppad[3] = {0,0,0};
        bmpfileheader[2] = (unsigned char)(filesize    );
        bmpfileheader[3] = (unsigned char)(filesize>> 8);
        bmpfileheader[4] = (unsigned char)(filesize>>16);
        bmpfileheader[5] = (unsigned char)(filesize>>24);
        bmpinfoheader[ 4] = (unsigned char)(z    );
        bmpinfoheader[ 5] = (unsigned char)(z>> 8);
        bmpinfoheader[ 6] = (unsigned char)(z>>16);
        bmpinfoheader[ 7] = (unsigned char)(z>>24);
        bmpinfoheader[ 8] = (unsigned char)(x    );
        bmpinfoheader[ 9] = (unsigned char)(x>> 8);
        bmpinfoheader[10] = (unsigned char)(x>>16);
        bmpinfoheader[11] = (unsigned char)(x>>24);
        FILE * fp = fopen("file.bmp","wb");
        fwrite(bmpfileheader,1,14,fp);
        fwrite(bmpinfoheader,1,40,fp);
        // unsigned char alpha = 127;
        cout << "size of char = " << sizeof(char) << " size of int = " << sizeof(int) << endl;
        int count = 0;
        for(int dx = 0 ; dx < x ;dx++){
            for(int dz = 0 ; dz < x ;dz++){
                if(data[dx][0][dz] * 255 > 255)
                    cout << "hmm" << endl;
                if(data[dx][0][dz] * 255 < 0)
                    cout << "hmm" << endl;
                char val = data[dx][10][dz] * 255; 
                fwrite(&val,sizeof(val),1,fp);
                fwrite(&val,sizeof(val),1,fp);
                fwrite(&val,sizeof(val),1,fp);
                count++;
                // fwrite(&alpha,sizeof(alpha),1,fp);
                // fwrite(&bmppad,sizeof(bmppad),1,fp);
            }
        }
        cout << count << endl;
        fclose(fp);
    }
};


void print(string s);
void print(int i);
void getDimensions(string info);
vector<string> getNameMapping(string s);
vector<volume> getEachVoxel(Json::Value allVoxelData, vector<string> vDtypes);
volume getData4Vox(Json::Value vData);
void setVolData(Json::Value tile,volume &v,int xOff,int yOff,int zOff,string compr);
int xDim = 0; // THe dim in x direction
int yDim = 0; // THe dim in y direction
int zDim = 0; // THe dim in z direction
int main()
{
    
    ifstream jsonFile("./temp.json",ifstream::binary);
    Json::Value root;
    jsonFile >> root; 
    string volume_info =  root[11]["volume_summary"].asString();
    getDimensions(volume_info); // get the dimensions
    vector<string> vDtypes = getNameMapping(root[11]["volume_summary"].asString()); // gets the different kind of data stored as voxels.
    Json::Value allVoxelData = root[21];
    vector<volume> allVoxData = getEachVoxel(allVoxelData,vDtypes);
    allVoxData[0].writeSlice();
    cout << vDtypes[0] << endl;
    // cout << root[20].asString();
    return 0;
}
void setVolData(Json::Value tile,volume &v,int xOff,int yOff,int zOff,string compr)
{
    //calc start x y z values for current tile
    int x_start = xOff * 16;
    int y_start = yOff * 16;
    int z_start = zOff * 16;
    //Calculate the lengths if on edge AKA where lengths < 16
    int xLen = xDim - xOff * 16;
    int yLen = yDim - yOff * 16;
    int zLen = zDim - zOff * 16;
    if(xLen > 16)
        xLen = 16;
    if(yLen > 16)
        yLen = 16;
    if(zLen > 16)
        zLen = 16;
    bool full = true; // if no compression is used.
    if(compr == "2")
    {
        full = false;
    }
    int count = 0;
    cout << "({" << xLen <<"},{" << yLen << "},{" << zLen << "})" << endl;
    for(int lx = 0 ; lx < xLen; lx++){
        for(int ly = 0 ; ly < yLen; ly++){
            for(int lz = 0 ; lz < zLen; lz++){
                if(full)
                {
                    v.setValue(tile[count].asDouble(),(x_start +lx),(y_start +ly),(z_start +lz));
                    count++;
                    // cout << "({" << x_start + lx <<"},{" << y_start + ly << "},{" << z_start + lz << "})" << endl;
                }
                else
                    v.setValue(tile.asDouble(),(x_start + lx),(y_start + ly),(z_start + lz));
            }
        }
    }
}
volume getData4Vox(Json::Value vData)
{
    volume vd1 = volume(xDim,yDim,zDim);
    Json::Value comprNames = vData[2][1][3];
    Json:: Value tiles = vData[2][1][5];
    int tileCount = 0;
    for(int tx = 0 ; tx < (xDim + 15) / 16 ;tx++){
        for(int ty = 0 ; ty < (yDim + 15) / 16 ;ty++){
            for(int tz = 0 ; tz < (zDim + 15) / 16 ;tz++){
                Json::Value tile = tiles[tileCount];
                setVolData(tile[3],vd1,tx,ty,tz,tile[1].asString());
                tileCount++;
            }
        }
    }
    return vd1;
}
vector<volume>  getEachVoxel(Json::Value allVoxelData,vector<string> vDtypes)
{   
    vector<volume> volData;
    
    for(unsigned i = 0 ; i < vDtypes.size() ; i++)
    {
        volData.push_back(getData4Vox(allVoxelData[(i * 2) + 1]));
    }
    // cout  << "size = " <<  volData.size() << endl;
    // for(size_t i = 0 ; i < volData.size(); i++)
    // {
    //     cout << vDtypes[i] << " " <<  volData[i].size() << endl;
    // }
    return volData;
}
void print(string s)
{
    cout << s << endl;
}
void print(int i)
{
    cout << i << endl;
}
vector<string> getNameMapping(string s)
{
    size_t found = s.find_first_of("(");
    size_t found2 = s.find_first_of(")");
    vector<string> names; 
    while (found!=string::npos)
    {  
        // cout << s.substr(found + 1,found2 - found - 1) << endl;
        names.push_back(s.substr(found + 1,found2 - found - 1));
        s = s.substr(found2 + 1,string::npos);
        found = s.find_first_of("(");
        found2 = s.find_first_of(")");
    }
    return names;    
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
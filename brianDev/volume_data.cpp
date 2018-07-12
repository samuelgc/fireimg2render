#include "volume_data.h"
#include <fstream>


using namespace std;

volume_data::volume_data(int dx, int dy, int dz)
{
    dimX = dx;
    dimY = dy;
    dimZ = dz;
    vol_data =  vector<vector<vector<double>>>(double(dx), vector<vector<double>>(double(dy), vector<double>(double(dz))));
}
void volume_data::setValue(double val,int dimX, int dimY, int dimZ)
{
    vol_data[dimX][dimY][dimZ] = val;
}
int volume_data::size()
{
    return dimX * dimY * dimZ;
}
void volume_data::writeSlice()
{
    int filesize = 54 + 3*dimX*dimZ;
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    // unsigned char bmppad[3] = {0,0,0};
    bmpfileheader[2] = (unsigned char)(filesize    );
    bmpfileheader[3] = (unsigned char)(filesize>> 8);
    bmpfileheader[4] = (unsigned char)(filesize>>16);
    bmpfileheader[5] = (unsigned char)(filesize>>24);
    bmpinfoheader[ 4] = (unsigned char)(dimZ    );
    bmpinfoheader[ 5] = (unsigned char)(dimZ>> 8);
    bmpinfoheader[ 6] = (unsigned char)(dimZ>>16);
    bmpinfoheader[ 7] = (unsigned char)(dimZ>>24);
    bmpinfoheader[ 8] = (unsigned char)(dimX    );
    bmpinfoheader[ 9] = (unsigned char)(dimX>> 8);
    bmpinfoheader[10] = (unsigned char)(dimX>>16);
    bmpinfoheader[11] = (unsigned char)(dimX>>24);
    FILE * fp = fopen("file.bmp","wb");
    fwrite(bmpfileheader,1,14,fp);
    fwrite(bmpinfoheader,1,40,fp);
    // unsigned char alpha = 127;
    cout << "size of char = " << sizeof(char) << " size of int = " << sizeof(int) << endl;
    ofstream lol;
    lol.open("testing.txt");
    for(int dx = 0 ; dx < dimX ;dx++){
        for(int dz = 0 ; dz < dimZ ;dz++){
            if(vol_data[dx][0][dz] * 255 > 255)
                cout << "hmm" << endl;
            if(vol_data[dx][0][dz] * 255 < 0)
                cout << "hmm" << endl;
            char val = vol_data[dx][0][dz] * 255; 
            fwrite(&val,sizeof(val),1,fp);
            fwrite(&val,sizeof(val),1,fp);
            fwrite(&val,sizeof(val),1,fp);
            unsigned char block = '+';
            if(val == 0)
                block = '_';
            if(dz != dimZ-1)
                lol << block << " ";
            else
                lol << block << endl;
        }
    }
    lol.close();
    fclose(fp);
}
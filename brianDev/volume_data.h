#ifndef VOLUME_DATA_H
#define VOLUME_DATA_H

#include <iostream>
#include <vector>

class volume_data
{
    private:
        // Used to store the values at each Voxel
        std::vector<std::vector<std::vector<double>>> vol_data;
        int dimX; // The dimension in the X Direction
        int dimY; // The dimension in the Y Direction
        int dimZ; // The dimension in the Z Direction
    public:
       
        /**
         * @return the number of data points in the volume
        */
        int size();
        
        /**
         * Used to set the value at a certin location to a the passed in Value
         * @param x = The X-coridinate
         * @param y = The Y-coridinate
         * @param z = The Z-coridinate
         * @param double val = The value to be set
         * @return
        */
        void setValue(double val,int x, int y, int z);
        
        /**
         * used for debuggin
         * write to a BMP <-- Still bugged
         * write to a txt file where if value at pixel = 0 then write _
         * if value of pixel != 0 write +
         * @return  
         */
        void writeSlice();
        
        /**
         * The constructor
         * @param dimX = the dimension in the X direction
         * @param dimY = the dimension in the Y direction
         * @param dimZ = the dimension in the Z direction
         */
        volume_data(int dimX, int dimY, int dimZ);
        // ~volume_data();
    
};
#endif

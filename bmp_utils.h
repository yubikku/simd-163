#ifndef BMP_UTILS_H
#define BMP_UTILS_H

// pragma prevents the compiler from padding the structs
#pragma pack(push, 1)
typedef struct {
    unsigned short type;      
    unsigned int size;        
    unsigned short res1, res2;
    unsigned int offset;      
} BMPHeader;

typedef struct {
    unsigned int size;        
    int width, height;        
    unsigned short planes;    
    unsigned short bits;      
    unsigned int compression; 
    unsigned int imagesize;   
    int xres, yres;           
    unsigned int ncolours;    
    unsigned int impcolours;  
} BMPInfoHeader;
#pragma pack(pop)

// Helper to clamp float values to 8-bit [0, 255]
static unsigned char clamp(float val) {
    if (val < 0.0f) return 0;
    if (val > 255.0f) return 255;
    return (unsigned char)val;
}

#endif // BMP_UTILS_H
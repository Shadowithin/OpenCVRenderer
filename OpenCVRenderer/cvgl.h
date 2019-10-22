#ifndef __CVGL_H__
#define __CVGL_H__
#include <opencv.hpp>

extern cv::Matx<float, 4, 4>    Viewport;
extern cv::Matx<float, 4, 4>  Projection;
extern cv::Matx<float, 4, 4>   ModelView;

void viewport(int x, int y, int w, int h);
void projection(float coeff);
void lookat(cv::Vec3f eye, cv::Vec3f center, cv::Vec3f up);

class cvglShader {
public:
	virtual ~cvglShader(){};
	virtual cv::Vec4f vertex(int iface, int nthvert) = 0;
	virtual bool fragment(cv::Vec3f bar, cv::Scalar &color) = 0;
};

void triangle(cv::Vec4f *pts, cvglShader* shader, cv::Mat &image, cv::Mat &zbuffer);

#endif //__CVGL_H__

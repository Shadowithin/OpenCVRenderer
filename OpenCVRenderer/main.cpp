#include<iostream>
#include<vector>
#include "model.h"
#include "cvgl.h"

#pragma warning(disable:4996)

using namespace std;
using namespace cv;

const int width = 800;
const int height = 800;

Model        *model = nullptr;
cvglShader  *shader = nullptr;
Mat           shadowbuffer;

Vec3f   light_dir(1, 1, 1);
Vec3f         eye(1.2, -.8, 3);
Vec3f      center(0, 0, 0);
Vec3f          up(0, 1, 0);

class ZBuffer : public cvglShader {
private:
	Vec4f  varying_tri[3];
public:
	virtual Vec4f vertex(int iface, int nthvert) {
		Vec3f v = model->vert(iface, nthvert);
		Vec4f cvgl_Vertex = Vec4f(1, 1, 1, 1);
		for (int i = 0; i < 3; i++) cvgl_Vertex[i] = v[i];
		cvgl_Vertex = Viewport * Projection * ModelView * cvgl_Vertex;
		varying_tri[nthvert] = cvgl_Vertex;
		return cvgl_Vertex;
	}
	virtual bool fragment(Vec3f bar, Scalar &color) {
		color = Scalar(0, 0, 0);
		return false;
	}
};

int main()
{
	Mat frame = Mat::zeros(height, width, CV_8UC4);
	Mat zbuffer = Mat::zeros(height, width, CV_8UC1);
	model = new Model("./obj/diablo3_pose/diablo3_pose.obj");
	
	clock_t start = clock();

	light_dir = normalize(light_dir);
	lookat(eye, center, up);
	viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
	//viewport(0, 0, width, height);
	projection(-1.f / norm(eye - center));

	shader = new ZBuffer();
	for (int i = 0; i < model->nfaces(); i++) {
		Vec4f screen_coords[3];
		for (int j = 0; j < 3; j++)
			screen_coords[j] = shader->vertex(i, j);
		triangle(screen_coords, shader, frame, zbuffer);
	}

	clock_t end = clock();
	cout << end - start << endl;

	Mat framebuffer;
	flip(zbuffer, framebuffer, 0);
	imshow("frame", framebuffer);
	imwrite("frame_diablo.jpg", framebuffer);
	delete model;
	delete shader;
	model = nullptr;
	shader = nullptr;
	waitKey(0);
	return 0;
}